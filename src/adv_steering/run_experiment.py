from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from .data import load_examples
from .evaluate import evaluate_vector
from .gcg import optimize_prefix_for_example
from .modeling import load_model_bundle
from .steering import estimate_steering_vector


def parse_args():
    parser = argparse.ArgumentParser(description="Run steering-vector estimation and adversarial prefix search.")
    parser.add_argument("--model", required=True, help="Hugging Face model name or path.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--layer", type=int, required=True, help="Zero-based transformer layer index.")
    parser.add_argument("--token-index", type=int, default=-1, help="Token index used for hidden-state extraction.")
    parser.add_argument("--steering-scale", type=float, default=4.0, help="Multiplier applied to the steering vector.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length for evaluation.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="auto", help="Torch device, for example cuda or cpu.")
    parser.add_argument("--gcg-steps", type=int, default=40, help="Number of discrete optimization steps.")
    parser.add_argument("--gcg-prefix-length", type=int, default=12, help="Number of prefix tokens to optimize.")
    parser.add_argument("--gcg-search-width", type=int, default=32, help="How many candidate tokens to test per step.")
    parser.add_argument("--output-dir", default="runs", help="Directory where outputs are written.")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_examples(args.dataset)
    bundle = load_model_bundle(args.model, dtype=args.dtype, device=args.device)

    steering_result = estimate_steering_vector(
        bundle=bundle,
        examples=examples,
        layer=args.layer,
        token_index=args.token_index,
    )

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    vector_payload = {
        "model": args.model,
        "layer": steering_result.layer,
        "token_index": steering_result.token_index,
    }
    torch.save(
        {
            "metadata": vector_payload,
            "vector": steering_result.vector,
            "example_vectors": steering_result.example_vectors,
        },
        run_dir / "vector.pt",
    )

    evaluation = evaluate_vector(
        bundle=bundle,
        examples=examples,
        steering_vector=steering_result.vector,
        layer=args.layer,
        scale=args.steering_scale,
        max_new_tokens=args.max_new_tokens,
    )
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    attack_results = []
    for example in examples:
        result = optimize_prefix_for_example(
            bundle=bundle,
            example=example,
            steering_vector=steering_result.vector,
            layer=args.layer,
            scale=args.steering_scale,
            prefix_length=args.gcg_prefix_length,
            steps=args.gcg_steps,
            search_width=args.gcg_search_width,
        )
        attack_results.append(
            {
                "example_id": result.example_id,
                "prefix_token_ids": result.prefix_token_ids,
                "prefix_text": result.prefix_text,
                "loss": result.loss,
                "trace": result.trace,
            }
        )
    (run_dir / "attack_results.json").write_text(json.dumps(attack_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
