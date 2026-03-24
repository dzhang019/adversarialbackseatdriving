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
    from adv_steering.backend import collect_last_token_residuals, load_bundle, read_jsonl
else:
    from .backend import collect_last_token_residuals, load_bundle, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze positive-concept and negative-concept residuals and rank predictive layers.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen_vl", "causal_lm"], help="Inference backend.")
    parser.add_argument("--dataset", default="data/llama_happy_sad_corpus.jsonl", help="Generated corpus.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--output-dir", default="runs", help="Directory for analysis outputs.")
    parser.add_argument("--top-k", type=int, default=8, help="How many top layers to highlight in the summary.")
    return parser.parse_args()


def analyze_poscon_negcon_residuals(
    model: str,
    backend: str,
    dataset: str,
    dtype: str = "auto",
    device_map: str = "auto",
    output_dir: str = "runs",
    top_k: int = 8,
) -> dict[str, Any]:
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

        poscon_stack = collect_last_token_residuals(bundle, resolved_backend, poscon_prompt, poscon_response)
        negcon_stack = collect_last_token_residuals(bundle, resolved_backend, negcon_prompt, negcon_response)
        poscon_residuals.append(poscon_stack)
        negcon_residuals.append(negcon_stack)
        concepts.append(row["concept"])

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

    run_dir = Path(output_dir) / datetime.now().strftime("poscon_negcon_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model,
            "backend": resolved_backend,
            "concepts": concepts,
            "poscon_residuals": poscon_tensor,
            "negcon_residuals": negcon_tensor,
            "difference_vectors": difference_tensor,
            "steering_vectors": steering_vectors,
            "best_layer": best_layer,
        },
        run_dir / "poscon_negcon_residuals.pt",
    )

    summary = {
        "model": model,
        "backend": resolved_backend,
        "dataset": dataset,
        "num_examples": len(rows),
        "best_layer": best_layer,
        "top_layers": layer_summaries[: top_k],
        "all_layers": layer_summaries,
    }
    (run_dir / "layer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "residual_path": str(run_dir / "poscon_negcon_residuals.pt"),
        "summary_path": str(run_dir / "layer_summary.json"),
        "summary": summary,
    }


def main():
    args = parse_args()
    result = analyze_poscon_negcon_residuals(
        model=args.model,
        backend=args.backend,
        dataset=args.dataset,
        dtype=args.dtype,
        device_map=args.device_map,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(result["summary"]["top_layers"], indent=2))


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

    probabilities = []
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
        probabilities.extend(classifier.predict_proba(x_test)[:, 1].tolist())
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
