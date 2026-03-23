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
    parser = argparse.ArgumentParser(description="Analyze truth/lie residuals and rank predictive layers.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen_vl", "causal_lm"], help="Inference backend.")
    parser.add_argument("--dataset", default="data/qwen_truth_lie_corpus.jsonl", help="Generated truth/lie corpus.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--output-dir", default="runs", help="Directory for analysis outputs.")
    parser.add_argument("--top-k", type=int, default=8, help="How many top layers to highlight in the summary.")
    return parser.parse_args()


def analyze_truth_lie_residuals(
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

    truth_residuals = []
    lie_residuals = []
    concepts = []

    for row in tqdm(rows, desc="Collecting residuals"):
        truth_stack = collect_last_token_residuals(bundle, resolved_backend, row["truth_prompt"], row["truth_response"])
        lie_stack = collect_last_token_residuals(bundle, resolved_backend, row["lie_prompt"], row["lie_response"])
        truth_residuals.append(truth_stack)
        lie_residuals.append(lie_stack)
        concepts.append(row["concept"])

    truth_tensor = torch.stack(truth_residuals, dim=0)
    lie_tensor = torch.stack(lie_residuals, dim=0)
    difference_tensor = truth_tensor - lie_tensor
    steering_vectors = _normalize(difference_tensor.mean(dim=0))

    layer_summaries = []
    for layer_index in range(steering_vectors.shape[0]):
        centroid_accuracy, centroid_margin = _leave_one_out_centroid_accuracy(
            truth_tensor[:, layer_index, :],
            lie_tensor[:, layer_index, :],
        )
        logistic_accuracy, logistic_auc = _leave_one_out_logistic_regression(
            truth_tensor[:, layer_index, :],
            lie_tensor[:, layer_index, :],
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

    run_dir = Path(output_dir) / datetime.now().strftime("truth_lie_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model,
            "backend": resolved_backend,
            "concepts": concepts,
            "truth_residuals": truth_tensor,
            "lie_residuals": lie_tensor,
            "difference_vectors": difference_tensor,
            "steering_vectors": steering_vectors,
            "best_layer": best_layer,
        },
        run_dir / "truth_lie_residuals.pt",
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
        "residual_path": str(run_dir / "truth_lie_residuals.pt"),
        "summary_path": str(run_dir / "layer_summary.json"),
        "summary": summary,
    }


def main():
    args = parse_args()
    result = analyze_truth_lie_residuals(
        model=args.model,
        backend=args.backend,
        dataset=args.dataset,
        dtype=args.dtype,
        device_map=args.device_map,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(result["summary"]["top_layers"], indent=2))


def _leave_one_out_centroid_accuracy(
    truth_layer: torch.Tensor,
    lie_layer: torch.Tensor,
) -> tuple[float, float]:
    num_examples = truth_layer.shape[0]
    correct = 0
    margins = []

    for index in range(num_examples):
        truth_mask = torch.ones(num_examples, dtype=torch.bool)
        truth_mask[index] = False

        truth_mean = truth_layer[truth_mask].mean(dim=0)
        lie_mean = lie_layer[truth_mask].mean(dim=0)
        direction = _normalize(truth_mean - lie_mean)
        midpoint = 0.5 * (truth_mean + lie_mean)

        truth_score = torch.dot(truth_layer[index] - midpoint, direction)
        lie_score = torch.dot(lie_layer[index] - midpoint, direction)

        correct += int(truth_score > 0)
        correct += int(lie_score < 0)
        margins.append(float((truth_score - lie_score).item()))

    accuracy = correct / (2 * num_examples)
    mean_margin = sum(margins) / len(margins)
    return accuracy, mean_margin


def _leave_one_out_logistic_regression(
    truth_layer: torch.Tensor,
    lie_layer: torch.Tensor,
) -> tuple[float, float]:
    truth_np = truth_layer.float().cpu().numpy()
    lie_np = lie_layer.float().cpu().numpy()
    features = _stack_features(truth_np, lie_np)
    labels = _stack_labels(len(truth_np), len(lie_np))

    probabilities = []
    predictions = []

    for truth_index in range(len(truth_np)):
        holdout_indices = [truth_index, len(truth_np) + truth_index]
        train_mask = [index not in holdout_indices for index in range(len(labels))]

        x_train = features[train_mask]
        y_train = labels[train_mask]
        x_test = features[holdout_indices]
        y_test = labels[holdout_indices]

        classifier = LogisticRegression(max_iter=1000, solver="liblinear")
        classifier.fit(x_train, y_train)
        test_probabilities = classifier.predict_proba(x_test)[:, 1]
        test_predictions = classifier.predict(x_test)

        probabilities.extend(test_probabilities.tolist())
        predictions.extend((test_predictions == y_test).tolist())

    accuracy = sum(predictions) / len(predictions)
    auc = float(roc_auc_score(labels, probabilities))
    return float(accuracy), auc


def _stack_features(truth_np, lie_np):
    import numpy as np

    return np.concatenate([truth_np, lie_np], axis=0)


def _stack_labels(num_truth: int, num_lie: int):
    import numpy as np

    return np.concatenate(
        [
            np.ones(num_truth, dtype=int),
            np.zeros(num_lie, dtype=int),
        ],
        axis=0,
    )


def _normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)


if __name__ == "__main__":
    main()
