from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import collect_last_token_residuals, collect_user_story_last_token_residuals, collect_user_story_mean_residuals, load_bundle, read_jsonl
else:
    from .backend import collect_last_token_residuals, collect_user_story_last_token_residuals, collect_user_story_mean_residuals, load_bundle, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze positive-concept and negative-concept residuals and rank predictive layers.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen_vl", "causal_lm"], help="Inference backend.")
    parser.add_argument("--dataset", default="data/llama_happy_sad_corpus.jsonl", help="Generated corpus.")
    parser.add_argument("--residual-file", default="", help="Optional existing poscon_negcon_residuals.pt to re-rank without recollecting residuals.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--output-dir", default="runs", help="Directory for analysis outputs.")
    parser.add_argument("--top-k", type=int, default=8, help="How many top layers to highlight in the summary.")
    parser.add_argument(
        "--residual-mode",
        default="generation_prompt",
        choices=["generation_prompt", "story_only", "story_mean"],
        help=(
            "Where to collect residuals: generation_prompt uses the assistant-generation boundary from the old pipeline; "
            "story_only puts only the story in a user message and reads the final story token; "
            "story_mean mean-pools over story tokens in that same story-only user message."
        ),
    )
    return parser.parse_args()


def analyze_poscon_negcon_residuals(
    model: str,
    backend: str,
    dataset: str,
    residual_file: str = "",
    dtype: str = "auto",
    device_map: str = "auto",
    output_dir: str = "runs",
    top_k: int = 8,
    residual_mode: str = "generation_prompt",
) -> dict[str, Any]:
    if residual_mode not in {"generation_prompt", "story_only", "story_mean"}:
        raise ValueError(f"Unsupported residual mode: {residual_mode}")
    if residual_file:
        residual_path = Path(residual_file)
        residual_payload = torch.load(residual_path, map_location="cpu")
        model = residual_payload.get("model", model)
        resolved_backend = residual_payload.get("backend", backend)
        residual_mode = residual_payload.get("residual_mode", residual_mode)
        concepts = residual_payload.get("concepts", [])
        poscon_tensor = residual_payload["poscon_residuals"].float()
        negcon_tensor = residual_payload["negcon_residuals"].float()
        difference_tensor = residual_payload.get("difference_vectors")
        if difference_tensor is None:
            difference_tensor = poscon_tensor - negcon_tensor
        else:
            difference_tensor = difference_tensor.float()
        steering_vectors = residual_payload.get("steering_vectors")
        if steering_vectors is None:
            steering_vectors = _normalize(difference_tensor.mean(dim=0))
        else:
            steering_vectors = steering_vectors.float()
        rows = read_jsonl(dataset) if dataset and Path(dataset).exists() else []
        residual_labels = residual_payload.get("residual_labels")
        if residual_labels is None:
            residual_labels = build_residual_labels(rows, concepts, residual_mode)
    else:
        rows = read_jsonl(dataset)
        if not rows:
            raise ValueError(f"No rows found in {dataset}")

        bundle, resolved_backend = load_bundle(model_name=model, backend=backend, dtype=dtype, device_map=device_map)

        poscon_residuals = []
        negcon_residuals = []
        concepts = []

        for row in tqdm(rows, desc="Collecting poscon/negcon residuals"):
            poscon_prompt = row.get("poscon_prompt") or row.get("positive_prompt") or row["truth_prompt"]
            poscon_response = row.get("poscon_response") or row.get("positive_response") or row["truth_response"]
            negcon_prompt = row.get("negcon_prompt") or row.get("negative_prompt") or row["lie_prompt"]
            negcon_response = row.get("negcon_response") or row.get("negative_response") or row["lie_response"]

            if residual_mode == "story_only":
                poscon_stack = collect_user_story_last_token_residuals(bundle, resolved_backend, poscon_response)
                negcon_stack = collect_user_story_last_token_residuals(bundle, resolved_backend, negcon_response)
            elif residual_mode == "story_mean":
                poscon_stack = collect_user_story_mean_residuals(bundle, resolved_backend, poscon_response)
                negcon_stack = collect_user_story_mean_residuals(bundle, resolved_backend, negcon_response)
            else:
                poscon_stack = collect_last_token_residuals(bundle, resolved_backend, poscon_prompt, poscon_response)
                negcon_stack = collect_last_token_residuals(bundle, resolved_backend, negcon_prompt, negcon_response)
            poscon_residuals.append(poscon_stack)
            negcon_residuals.append(negcon_stack)
            concepts.append(row["concept"])

        residual_labels = build_residual_labels(rows, concepts, residual_mode)
        poscon_tensor = torch.stack(poscon_residuals, dim=0)
        negcon_tensor = torch.stack(negcon_residuals, dim=0)
        difference_tensor = poscon_tensor - negcon_tensor
        steering_vectors = _normalize(difference_tensor.mean(dim=0))

    layer_summaries = []
    for layer_index in range(steering_vectors.shape[0]):
        centroid_accuracy, centroid_margin = _leave_one_out_centroid_accuracy(
            poscon_tensor[:, layer_index, :],
            negcon_tensor[:, layer_index, :],
        )
        logistic_accuracy, logistic_auc = _leave_one_out_logistic_regression(
            poscon_tensor[:, layer_index, :],
            negcon_tensor[:, layer_index, :],
        )
        layer_summaries.append(
            {
                "layer": layer_index,
                "centroid_accuracy": centroid_accuracy,
                "centroid_mean_margin": centroid_margin,
                "logistic_accuracy": logistic_accuracy,
                "logistic_auc": logistic_auc,
                "vector_norm": float(difference_tensor[:, layer_index, :].mean(dim=0).norm().item()),
            }
        )

    layer_summaries.sort(
        key=lambda row: (
            -row["logistic_accuracy"],
            -row["logistic_auc"],
            -row["centroid_accuracy"],
            -row["centroid_mean_margin"],
            row["layer"],
        )
    )
    best_layer = layer_summaries[0]["layer"]

    if residual_file:
        residual_path = Path(residual_file).resolve()
        run_dir = Path(output_dir).resolve() if output_dir else residual_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        residual_out_path = run_dir / residual_path.name
        if residual_out_path != residual_path:
            torch.save(
                {
                    "model": model,
                    "backend": resolved_backend,
                    "residual_mode": residual_mode,
                    "concepts": concepts,
                    "residual_labels": residual_labels,
                    "poscon_residuals": poscon_tensor,
                    "negcon_residuals": negcon_tensor,
                    "difference_vectors": difference_tensor,
                    "steering_vectors": steering_vectors,
                    "best_layer": best_layer,
                },
                residual_out_path,
            )
        else:
            residual_out_path = residual_path
    else:
        run_prefix = "poscon_negcon" if residual_mode == "generation_prompt" else f"poscon_negcon_{residual_mode}"
        run_dir = Path(output_dir) / datetime.now().strftime(f"{run_prefix}_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        residual_out_path = run_dir / "poscon_negcon_residuals.pt"
        torch.save(
            {
                "model": model,
                "backend": resolved_backend,
                "residual_mode": residual_mode,
                "concepts": concepts,
                "residual_labels": residual_labels,
                "poscon_residuals": poscon_tensor,
                "negcon_residuals": negcon_tensor,
                "difference_vectors": difference_tensor,
                "steering_vectors": steering_vectors,
                "best_layer": best_layer,
            },
            residual_out_path,
        )

    summary = {
        "model": model,
        "backend": resolved_backend,
        "residual_mode": residual_mode,
        "dataset": dataset,
        "num_examples": len(rows),
        "best_layer": best_layer,
        "top_layers": layer_summaries[: top_k],
        "all_layers": layer_summaries,
    }
    (run_dir / "layer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "residual_path": str(residual_out_path),
        "summary_path": str(run_dir / "layer_summary.json"),
        "summary": summary,
    }


def main():
    args = parse_args()
    result = analyze_poscon_negcon_residuals(
        model=args.model,
        backend=args.backend,
        dataset=args.dataset,
        residual_file=args.residual_file,
        dtype=args.dtype,
        device_map=args.device_map,
        output_dir=args.output_dir,
        top_k=args.top_k,
        residual_mode=args.residual_mode,
    )
    print(json.dumps(result["summary"]["top_layers"], indent=2))


def build_residual_labels(rows: list[dict], concepts: list[str], residual_mode: str) -> dict[str, list[dict]]:
    poscon_labels = []
    negcon_labels = []
    for index, concept in enumerate(concepts):
        row = rows[index] if index < len(rows) else {}
        poscon_labels.append(
            {
                "index": index,
                "concept": concept,
                "side": "poscon",
                "class_id": 1,
                "class_label": row.get("poscon_label") or row.get("positive_label") or "positive",
                "residual_mode": residual_mode,
            }
        )
        negcon_labels.append(
            {
                "index": index,
                "concept": concept,
                "side": "negcon",
                "class_id": 0,
                "class_label": row.get("negcon_label") or row.get("negative_label") or "negative",
                "residual_mode": residual_mode,
            }
        )
    return {
        "poscon": poscon_labels,
        "negcon": negcon_labels,
    }


def _leave_one_out_centroid_accuracy(poscon_layer: torch.Tensor, negcon_layer: torch.Tensor) -> tuple[float, float]:
    num_examples = poscon_layer.shape[0]
    correct = 0
    margins = []

    for index in range(num_examples):
        mask = torch.ones(num_examples, dtype=torch.bool)
        mask[index] = False

        poscon_mean = poscon_layer[mask].mean(dim=0)
        negcon_mean = negcon_layer[mask].mean(dim=0)
        direction = _normalize(poscon_mean - negcon_mean)
        midpoint = 0.5 * (poscon_mean + negcon_mean)

        poscon_score = torch.dot(poscon_layer[index] - midpoint, direction)
        negcon_score = torch.dot(negcon_layer[index] - midpoint, direction)

        correct += int(poscon_score > 0)
        correct += int(negcon_score < 0)
        margins.append(float((poscon_score - negcon_score).item()))

    return correct / (2 * num_examples), sum(margins) / len(margins)


def _leave_one_out_logistic_regression(poscon_layer: torch.Tensor, negcon_layer: torch.Tensor) -> tuple[float, float]:
    poscon_np = poscon_layer.float().cpu().numpy()
    negcon_np = negcon_layer.float().cpu().numpy()
    features = _stack_features(poscon_np, negcon_np)
    labels = _stack_labels(len(poscon_np), len(negcon_np))

    probabilities = [0.0] * len(labels)
    predictions = []
    for index in range(len(poscon_np)):
        holdout_indices = [index, len(poscon_np) + index]
        train_mask = [row_index not in holdout_indices for row_index in range(len(labels))]
        x_train = features[train_mask]
        y_train = labels[train_mask]
        x_test = features[holdout_indices]
        y_test = labels[holdout_indices]

        classifier = LogisticRegression(max_iter=1000, solver="liblinear")
        classifier.fit(x_train, y_train)
        holdout_probabilities = classifier.predict_proba(x_test)[:, 1].tolist()
        for holdout_index, probability in zip(holdout_indices, holdout_probabilities):
            probabilities[holdout_index] = probability
        predictions.extend((classifier.predict(x_test) == y_test).tolist())

    return float(sum(predictions) / len(predictions)), float(roc_auc_score(labels, probabilities))


def _stack_features(poscon_np, negcon_np):
    import numpy as np
    return np.concatenate([poscon_np, negcon_np], axis=0)


def _stack_labels(num_poscon: int, num_negcon: int):
    import numpy as np
    return np.concatenate([np.ones(num_poscon, dtype=int), np.zeros(num_negcon, dtype=int)], axis=0)


def _normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)


if __name__ == "__main__":
    main()
