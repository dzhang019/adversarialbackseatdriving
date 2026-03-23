from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import generate_text, generate_text_with_steering, load_bundle
else:
    from .backend import generate_text, generate_text_with_steering, load_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="Run qualitative truth-steering tests.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen_vl", "causal_lm"], help="Inference backend.")
    parser.add_argument("--steering-file", default="runs/truth_lie_20260322_225028/truth_lie_residuals.pt", help="Path to truth_lie_residuals.pt or steering_candidates.pt.")
    parser.add_argument("--layer", type=int, default=19, help="Layer to steer.")
    parser.add_argument("--scale", type=float, default=3.0, help="Scale for truth steering.")
    parser.add_argument("--anti-scale", type=float, default=-3.0, help="Scale for anti-truth steering.")
    parser.add_argument("--prompts", default="data/qualitative_truth_prompts.txt", help="Path to newline-delimited prompts.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum number of new tokens.")
    parser.add_argument("--output", default="", help="Optional path to save results as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)
    steering_vector = load_steering_vector(args.steering_file, args.layer)
    bundle, resolved_backend = load_bundle(model_name=args.model, backend=args.backend, dtype=args.dtype, device_map=args.device_map)

    results = []
    for prompt in prompts:
        baseline = generate_text(bundle, resolved_backend, prompt, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=1.0)
        truth_steered = generate_text_with_steering(
            bundle,
            resolved_backend,
            prompt,
            layer=args.layer,
            steering_vector=steering_vector,
            scale=args.scale,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        anti_truth_steered = generate_text_with_steering(
            bundle,
            resolved_backend,
            prompt,
            layer=args.layer,
            steering_vector=steering_vector,
            scale=args.anti_scale,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        results.append(
            {
                "prompt": prompt,
                "baseline": baseline,
                "truth_steered": truth_steered,
                "anti_truth_steered": anti_truth_steered,
            }
        )

    print_results(results, args.layer, args.scale, args.anti_scale)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


def load_prompts(path: str) -> list[str]:
    prompts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            prompt = raw_line.strip()
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_steering_vector(path: str, layer: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if "steering_vectors" in payload:
        return payload["steering_vectors"][layer]
    if "candidates" in payload:
        for row in payload["candidates"]:
            if row["layer"] == layer:
                return torch.tensor(row["vector"], dtype=torch.float32)
        raise ValueError(f"Layer {layer} not found in {path}")
    raise ValueError(f"Unsupported steering file format: {path}")


def print_results(results: list[dict], layer: int, scale: float, anti_scale: float) -> None:
    print(f"Layer {layer} qualitative steering test")
    print(f"Truth scale: {scale}")
    print(f"Anti-truth scale: {anti_scale}")
    print("")
    for index, row in enumerate(results, start=1):
        print(f"[{index}] Prompt: {row['prompt']}")
        print(f"Baseline: {row['baseline']}")
        print(f"Truth-steered: {row['truth_steered']}")
        print(f"Anti-truth-steered: {row['anti_truth_steered']}")
        print("")


if __name__ == "__main__":
    main()
