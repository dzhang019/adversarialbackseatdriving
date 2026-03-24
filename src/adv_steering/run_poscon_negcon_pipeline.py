from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.analyze_poscon_negcon_residuals import analyze_poscon_negcon_residuals
    from adv_steering.generate_poscon_negcon_corpus import generate_corpus
else:
    from .analyze_poscon_negcon_residuals import analyze_poscon_negcon_residuals
    from .generate_poscon_negcon_corpus import generate_corpus


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full positive-concept vs negative-concept pipeline end to end.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen_vl", "causal_lm"], help="Inference backend.")
    parser.add_argument("--concepts", default="data/concepts_200.txt", help="Path to newline-delimited concepts.")
    parser.add_argument("--dataset", default="data/llama_happy_sad_corpus.jsonl", help="Generated corpus path.")
    parser.add_argument("--poscon-label", default="happy", help="Positive concept label.")
    parser.add_argument("--negcon-label", default="sad", help="Negative concept label.")
    parser.add_argument("--mode", default="story", choices=["story", "statement"], help="Prompt style to generate.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum number of generated tokens.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for more diverse generations.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--overwrite-dataset", action="store_true", help="Regenerate the dataset from scratch.")
    parser.add_argument("--output-dir", default="runs", help="Directory for analysis outputs.")
    parser.add_argument("--top-k", type=int, default=8, help="How many top layers to export.")
    return parser.parse_args()


def main():
    args = parse_args()

    generation_result = generate_corpus(
        model=args.model,
        backend=args.backend,
        concepts_path=args.concepts,
        output_path=args.dataset,
        poscon_label=args.poscon_label,
        negcon_label=args.negcon_label,
        mode=args.mode,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        overwrite=args.overwrite_dataset,
    )

    analysis_result = analyze_poscon_negcon_residuals(
        model=args.model,
        backend=args.backend,
        dataset=args.dataset,
        dtype=args.dtype,
        device_map=args.device_map,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    residual_payload = torch.load(analysis_result["residual_path"], map_location="cpu")
    steering_vectors = residual_payload["steering_vectors"]
    layer_summary = analysis_result["summary"]["top_layers"]

    steering_candidates = []
    for layer_row in layer_summary:
        layer_index = layer_row["layer"]
        steering_candidates.append(
            {
                "layer": layer_index,
                "centroid_accuracy": layer_row["centroid_accuracy"],
                "centroid_mean_margin": layer_row["centroid_mean_margin"],
                "logistic_accuracy": layer_row["logistic_accuracy"],
                "logistic_auc": layer_row["logistic_auc"],
                "vector_norm": layer_row["vector_norm"],
                "vector": steering_vectors[layer_index].tolist(),
            }
        )

    run_dir = Path(analysis_result["run_dir"])
    candidates_path = run_dir / "steering_candidates.json"
    candidates_path.write_text(json.dumps(steering_candidates, indent=2), encoding="utf-8")

    steering_bundle = {
        "model": args.model,
        "backend": args.backend,
        "dataset": args.dataset,
        "best_layer": analysis_result["summary"]["best_layer"],
        "top_k": args.top_k,
        "candidates": steering_candidates,
    }
    torch.save(steering_bundle, run_dir / "steering_candidates.pt")

    summary = {
        "generation": generation_result,
        "analysis_run_dir": analysis_result["run_dir"],
        "best_layer": analysis_result["summary"]["best_layer"],
        "top_layers": analysis_result["summary"]["top_layers"],
        "steering_candidates_json": str(candidates_path),
        "steering_candidates_pt": str(run_dir / "steering_candidates.pt"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
