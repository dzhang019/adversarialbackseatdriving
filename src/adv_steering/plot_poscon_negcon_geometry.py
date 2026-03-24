from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import read_jsonl
else:
    from .backend import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze pairwise residual geometry and plot PCA for poscon/negcon residuals.")
    parser.add_argument("--dataset", default="data/llama_happy_sad_corpus.jsonl", help="Path to the generated corpus JSONL.")
    parser.add_argument("--residual-file", required=True, help="Path to poscon_negcon_residuals.pt.")
    parser.add_argument("--layer", type=int, default=None, help="Layer to analyze. Defaults to best_layer from the residual file.")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to the residual file directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = torch.load(args.residual_file, map_location="cpu")
    rows = read_jsonl(args.dataset)

    layer = args.layer if args.layer is not None else payload["best_layer"]
    poscon_tensor = payload["poscon_residuals"]
    negcon_tensor = payload["negcon_residuals"]
    steering_vectors = payload["steering_vectors"]

    poscon_layer = poscon_tensor[:, layer, :].float().numpy()
    negcon_layer = negcon_tensor[:, layer, :].float().numpy()
    diff_layer = poscon_layer - negcon_layer
    average_direction = steering_vectors[layer].float().numpy()
    average_direction = average_direction / np.linalg.norm(average_direction).clip(min=1e-8)

    similarity_rows = build_similarity_rows(rows, diff_layer, average_direction)

    all_points = np.concatenate([poscon_layer, negcon_layer], axis=0)
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(all_points)
    poscon_2d = points_2d[: len(rows)]
    negcon_2d = points_2d[len(rows) :]

    average_direction_2d = pca.components_ @ average_direction

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.residual_file).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    similarities_path = output_dir / f"layer_{layer}_pair_cosine_to_average.json"
    plot_path = output_dir / f"layer_{layer}_pair_pca.png"
    similarities_path.write_text(json.dumps(similarity_rows, indent=2), encoding="utf-8")
    save_plot(rows, poscon_2d, negcon_2d, average_direction_2d, plot_path, layer)

    print(json.dumps({"layer": layer, "similarities_path": str(similarities_path), "plot_path": str(plot_path)}, indent=2))


def build_similarity_rows(rows: list[dict], diff_layer: np.ndarray, average_direction: np.ndarray) -> list[dict]:
    diff_norms = np.linalg.norm(diff_layer, axis=1).clip(min=1e-8)
    cosines = (diff_layer @ average_direction) / diff_norms
    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                "index": index,
                "concept": row["concept"],
                "cosine_to_average": float(cosines[index]),
                "diff_norm": float(diff_norms[index]),
                "poscon_prompt": row.get("poscon_prompt") or row.get("positive_prompt") or row.get("truth_prompt"),
                "negcon_prompt": row.get("negcon_prompt") or row.get("negative_prompt") or row.get("lie_prompt"),
            }
        )
    result.sort(key=lambda item: item["cosine_to_average"], reverse=True)
    return result


def save_plot(
    rows: list[dict],
    poscon_2d: np.ndarray,
    negcon_2d: np.ndarray,
    average_direction_2d: np.ndarray,
    plot_path: Path,
    layer: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    for index, row in enumerate(rows):
        plt.plot(
            [poscon_2d[index, 0], negcon_2d[index, 0]],
            [poscon_2d[index, 1], negcon_2d[index, 1]],
            linestyle=":",
            linewidth=0.8,
            color="gray",
            alpha=0.45,
        )

    plt.scatter(poscon_2d[:, 0], poscon_2d[:, 1], s=24, c="#1f77b4", alpha=0.8, label="poscon")
    plt.scatter(negcon_2d[:, 0], negcon_2d[:, 1], s=24, c="#d62728", alpha=0.8, label="negcon")

    pos_center = poscon_2d.mean(axis=0)
    neg_center = negcon_2d.mean(axis=0)
    arrow_start = 0.5 * (pos_center + neg_center)
    arrow_scale = max(np.linalg.norm(pos_center - neg_center), 1.0)
    arrow_vec = average_direction_2d / np.linalg.norm(average_direction_2d).clip(min=1e-8) * arrow_scale
    plt.arrow(
        arrow_start[0],
        arrow_start[1],
        arrow_vec[0],
        arrow_vec[1],
        width=0.01 * arrow_scale,
        head_width=0.08 * arrow_scale,
        length_includes_head=True,
        color="#2ca02c",
        alpha=0.9,
    )
    plt.text(
        arrow_start[0] + arrow_vec[0],
        arrow_start[1] + arrow_vec[1],
        "avg steering",
        color="#2ca02c",
    )

    plt.title(f"Pairwise Residual Geometry, Layer {layer}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
