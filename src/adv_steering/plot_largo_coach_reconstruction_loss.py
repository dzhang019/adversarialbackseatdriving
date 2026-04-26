from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Largo coach reconstruction loss over inner and summary steps.")
    parser.add_argument("artifact", help="Path to coach_reconstruction_largo.json.")
    parser.add_argument(
        "--output",
        default="",
        help="Output image path. Defaults to <artifact stem>_loss.png next to the artifact.",
    )
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument("--no-baselines", action="store_true", help="Do not draw baseline CE reference lines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    output_path = Path(args.output) if args.output else artifact_path.with_name(f"{artifact_path.stem}_loss.png")

    inner_points, summary_points, summary_boundaries, selected_points = collect_points(payload)
    if not inner_points:
        raise ValueError(f"No inner loss points found in {artifact_path}.")

    save_plot(
        payload=payload,
        inner_points=inner_points,
        summary_points=summary_points,
        summary_boundaries=summary_boundaries,
        selected_points=selected_points,
        output_path=output_path,
        title=args.title,
        draw_baselines=not args.no_baselines,
    )
    print(f"Saved Largo loss plot to {output_path}")


def collect_points(payload: dict) -> tuple[list[tuple[float, float]], list[tuple[float, float, int]], list[float], list[tuple[float, float]]]:
    inner_points: list[tuple[float, float]] = []
    summary_points: list[tuple[float, float, int]] = []
    summary_boundaries: list[float] = []
    selected_points: list[tuple[float, float]] = []
    total_inner_steps = 0

    for outer_row in payload.get("trace", []):
        inner_losses = extract_inner_losses(outer_row)
        for loss in inner_losses:
            total_inner_steps += 1
            inner_points.append((float(total_inner_steps), float(loss)))

        summary_x = float(total_inner_steps)
        summary_boundaries.append(summary_x)
        summary_candidates = outer_row.get("summary_candidates", [])
        for candidate in summary_candidates:
            if "objective" not in candidate:
                continue
            summary_points.append((summary_x, float(candidate["objective"]), int(candidate.get("candidate_index", 0))))

        if "objective" in outer_row:
            selected_points.append((summary_x, float(outer_row["objective"])))

    return inner_points, summary_points, summary_boundaries, selected_points


def extract_inner_losses(outer_row: dict) -> list[float]:
    if "inner_trace" in outer_row:
        return [float(row["objective"]) for row in outer_row["inner_trace"] if "objective" in row]
    if "inner_step_losses" in outer_row:
        return [float(loss) for loss in outer_row["inner_step_losses"]]
    return []


def save_plot(
    payload: dict,
    inner_points: list[tuple[float, float]],
    summary_points: list[tuple[float, float, int]],
    summary_boundaries: list[float],
    selected_points: list[tuple[float, float]],
    output_path: Path,
    title: str,
    draw_baselines: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inner_x = [point[0] for point in inner_points]
    inner_y = [point[1] for point in inner_points]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(inner_x, inner_y, color="#2f6f9f", linewidth=1.8, marker="o", markersize=3.5, label="Inner optimization loss")

    if summary_points:
        summary_x = [x + summary_dot_offset(candidate_index) for x, _y, candidate_index in summary_points]
        summary_y = [y for _x, y, _candidate_index in summary_points]
        ax.scatter(
            summary_x,
            summary_y,
            color="#c65f2d",
            s=34,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
            label="Summary candidate losses",
            zorder=4,
        )

    if selected_points:
        selected_x = [point[0] for point in selected_points]
        selected_y = [point[1] for point in selected_points]
        ax.scatter(
            selected_x,
            selected_y,
            color="#252525",
            marker="*",
            s=130,
            label="Selected summary loss",
            zorder=5,
        )

    y_min, y_max = ax.get_ylim()
    for index, boundary in enumerate(summary_boundaries):
        ax.axvline(boundary, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.text(
            boundary,
            y_max,
            f"S{index}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4b5563",
        )

    if draw_baselines:
        baseline_ce = payload.get("baseline_mean_ce", {})
        baseline_styles = [
            ("full_prompt", "#4d7c0f", "Full prompt CE"),
            ("base_prompt_without_tokseq", "#7c3aed", "Base prompt CE"),
        ]
        for key, color, label in baseline_styles:
            if key in baseline_ce:
                ax.axhline(float(baseline_ce[key]), color=color, linestyle=":", linewidth=1.25, alpha=0.8, label=label)

    ax.set_xlabel("Total inner optimization steps")
    ax.set_ylabel("Objective / loss")
    ax.set_title(title or default_title(payload))
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_xlim(left=0, right=max(inner_x) + 1)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summary_dot_offset(candidate_index: int) -> float:
    return ((candidate_index % 24) - 11.5) * 0.018


def default_title(payload: dict) -> str:
    samples = payload.get("samples", "?")
    outer_steps = payload.get("outer_steps", "?")
    inner_steps = payload.get("inner_steps", "?")
    summary_samples = payload.get("summary_samples", "?")
    return f"Largo coach reconstruction loss ({outer_steps} outer x {inner_steps} inner, {summary_samples} summaries, {samples} samples)"


if __name__ == "__main__":
    main()
