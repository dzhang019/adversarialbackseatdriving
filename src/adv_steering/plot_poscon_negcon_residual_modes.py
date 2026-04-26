from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description="Plot multiple poscon/negcon residual files on one shared PCA basis.")
    parser.add_argument("--residual-files", nargs="+", required=True, help="One or more poscon_negcon_residuals.pt files.")
    parser.add_argument("--layer", type=int, default=None, help="Layer to plot. Defaults to best_layer from the first residual file.")
    parser.add_argument("--output", default="", help="Output PNG path. Defaults next to the first residual file.")
    parser.add_argument("--json-output", default="", help="Optional JSON output path with projected point metadata.")
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument("--no-pair-lines", action="store_true", help="Do not draw poscon-negcon pair lines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [torch.load(path, map_location="cpu") for path in args.residual_files]
    layer = args.layer if args.layer is not None else int(payloads[0]["best_layer"])

    point_rows, feature_matrix = collect_points(args.residual_files, payloads, layer)
    if feature_matrix.shape[0] < 2:
        raise ValueError("Need at least two residual points for PCA.")

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(feature_matrix)
    for row, xy in zip(point_rows, coordinates):
        row["pc1"] = float(xy[0])
        row["pc2"] = float(xy[1])

    output_path = Path(args.output) if args.output else default_output_path(args.residual_files[0], layer)
    json_output_path = Path(args.json_output) if args.json_output else output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    save_plot(point_rows, output_path, layer, args.title, draw_pair_lines=not args.no_pair_lines)
    json_output_path.write_text(
        json.dumps(
            {
                "layer": layer,
                "residual_files": args.residual_files,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "points": point_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"layer": layer, "plot_path": str(output_path), "json_path": str(json_output_path)}, indent=2))


def collect_points(residual_files: list[str], payloads: list[dict], layer: int) -> tuple[list[dict], np.ndarray]:
    rows = []
    features = []
    for residual_file, payload in zip(residual_files, payloads):
        residual_mode = payload.get("residual_mode") or Path(residual_file).parent.name
        poscon = payload["poscon_residuals"].float()
        negcon = payload["negcon_residuals"].float()
        if layer < 0 or layer >= poscon.shape[1]:
            raise ValueError(f"Layer {layer} is out of range for {residual_file}.")

        labels = payload.get("residual_labels") or synthesize_labels(payload, residual_mode)
        for side, tensor, class_id in [("poscon", poscon, 1), ("negcon", negcon, 0)]:
            side_labels = labels.get(side, [])
            for index in range(tensor.shape[0]):
                label = side_labels[index] if index < len(side_labels) else {}
                features.append(tensor[index, layer, :].numpy())
                rows.append(
                    {
                        "source_file": residual_file,
                        "residual_mode": label.get("residual_mode", residual_mode),
                        "side": label.get("side", side),
                        "class_id": int(label.get("class_id", class_id)),
                        "class_label": label.get("class_label", "positive" if side == "poscon" else "negative"),
                        "concept": label.get("concept", concept_at(payload, index)),
                        "index": int(label.get("index", index)),
                    }
                )

    return rows, np.stack(features, axis=0)


def synthesize_labels(payload: dict, residual_mode: str) -> dict[str, list[dict]]:
    concepts = payload.get("concepts", [])
    count = int(payload["poscon_residuals"].shape[0])
    labels = {"poscon": [], "negcon": []}
    for index in range(count):
        concept = concepts[index] if index < len(concepts) else ""
        labels["poscon"].append(
            {
                "index": index,
                "concept": concept,
                "side": "poscon",
                "class_id": 1,
                "class_label": "positive",
                "residual_mode": residual_mode,
            }
        )
        labels["negcon"].append(
            {
                "index": index,
                "concept": concept,
                "side": "negcon",
                "class_id": 0,
                "class_label": "negative",
                "residual_mode": residual_mode,
            }
        )
    return labels


def concept_at(payload: dict, index: int) -> str:
    concepts = payload.get("concepts", [])
    return concepts[index] if index < len(concepts) else ""


def save_plot(point_rows: list[dict], output_path: Path, layer: int, title: str, draw_pair_lines: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    class_colors = {1: "#1f77b4", 0: "#d62728"}
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    modes = sorted({row["residual_mode"] for row in point_rows})
    mode_markers = {mode: markers[index % len(markers)] for index, mode in enumerate(modes)}

    plt.figure(figsize=(11, 8))
    if draw_pair_lines:
        for mode in modes:
            by_index = {}
            for row in point_rows:
                if row["residual_mode"] == mode:
                    by_index.setdefault(row["index"], {})[row["side"]] = row
            for pair in by_index.values():
                if "poscon" in pair and "negcon" in pair:
                    plt.plot(
                        [pair["poscon"]["pc1"], pair["negcon"]["pc1"]],
                        [pair["poscon"]["pc2"], pair["negcon"]["pc2"]],
                        linestyle=":",
                        linewidth=0.65,
                        color="#9ca3af",
                        alpha=0.3,
                        zorder=1,
                    )

    for mode in modes:
        for class_id in [1, 0]:
            subset = [row for row in point_rows if row["residual_mode"] == mode and row["class_id"] == class_id]
            if not subset:
                continue
            plt.scatter(
                [row["pc1"] for row in subset],
                [row["pc2"] for row in subset],
                s=34,
                marker=mode_markers[mode],
                color=class_colors[class_id],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )

    class_handles = [
        Line2D([0], [0], marker="o", color="w", label="poscon", markerfacecolor=class_colors[1], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="negcon", markerfacecolor=class_colors[0], markersize=8),
    ]
    mode_handles = [
        Line2D([0], [0], marker=mode_markers[mode], color="#374151", label=mode, linestyle="None", markersize=8)
        for mode in modes
    ]
    first_legend = plt.legend(handles=class_handles, title="Class", loc="best", frameon=False)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=mode_handles, title="Residual mode", loc="upper right", frameon=False)
    plt.title(title or f"Poscon/Negcon Residual Modes PCA, Layer {layer}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def default_output_path(first_residual_file: str, layer: int) -> Path:
    return Path(first_residual_file).resolve().parent / f"layer_{layer}_residual_modes_pca.png"


if __name__ == "__main__":
    main()
