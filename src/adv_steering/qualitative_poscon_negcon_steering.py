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
    parser = argparse.ArgumentParser(description="Run qualitative positive-concept vs negative-concept steering tests.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen_vl", "causal_lm"], help="Inference backend.")
    parser.add_argument("--steering-file", default="runs/poscon_negcon/latest/poscon_negcon_residuals.pt", help="Path to poscon_negcon_residuals.pt or steering_candidates.pt.")
    parser.add_argument("--layer", type=int, default=0, help="Layer to steer.")
    parser.add_argument("--poscon-scale", type=float, default=3.0, help="Scale for positive-concept steering.")
    parser.add_argument("--negcon-scale", type=float, default=-3.0, help="Scale for negative-concept steering.")
    parser.add_argument("--prompts", default="data/qualitative_happy_sad_prompts.txt", help="Path to newline-delimited prompts.")
    parser.add_argument("--suffix", default="", help="Optional path to a rep_suffix_*.json file whose suffix should be appended to every prompt.")
    parser.add_argument("--step", type=int, default=-1, help="Which optimization step from the suffix trace to use. Defaults to the final trace entry.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum number of new tokens.")
    parser.add_argument("--output", default="", help="Optional path to save results as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)
    suffix_text = load_suffix_text(args.suffix, args.step) if args.suffix else ""
    steering_vector = load_steering_vector(args.steering_file, args.layer)
    bundle, resolved_backend = load_bundle(model_name=args.model, backend=args.backend, dtype=args.dtype, device_map=args.device_map)

    results = []
    for index, prompt in enumerate(prompts, start=1):
        full_prompt = append_suffix(prompt, suffix_text)
        print(f"[{index}/{len(prompts)}] Prompt: {prompt}", flush=True)
        if suffix_text:
            print(f"  suffix: {suffix_text}", flush=True)
            print(f"  full_prompt: {full_prompt}", flush=True)
        baseline = generate_text(bundle, resolved_backend, full_prompt, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=1.0)
        print("  baseline done", flush=True)
        poscon_steered = generate_text_with_steering(
            bundle, resolved_backend, full_prompt, layer=args.layer, steering_vector=steering_vector, scale=args.poscon_scale,
            max_new_tokens=args.max_new_tokens, do_sample=False, temperature=1.0,
        )
        print("  poscon-steered done", flush=True)
        negcon_steered = generate_text_with_steering(
            bundle, resolved_backend, full_prompt, layer=args.layer, steering_vector=steering_vector, scale=args.negcon_scale,
            max_new_tokens=args.max_new_tokens, do_sample=False, temperature=1.0,
        )
        print("  negcon-steered done", flush=True)
        results.append(
            {
                "prompt": prompt,
                "suffix_text": suffix_text,
                "full_prompt": full_prompt,
                "baseline": baseline,
                "poscon_steered": poscon_steered,
                "negcon_steered": negcon_steered,
            }
        )
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_results(results, args.layer, args.poscon_scale, args.negcon_scale)
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


def load_suffix_text(path: str, step: int) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trace = payload.get("trace", [])
    if not trace and isinstance(payload.get("result"), dict):
        trace = payload["result"].get("trace", [])
    if trace:
        index = step if step >= 0 else len(trace) - 1
        if index < 0 or index >= len(trace):
            raise ValueError(f"step={step} is out of range for trace length {len(trace)}")
        return trace[index].get("suffix_text", "")
    if "suffix_text" in payload:
        return payload.get("suffix_text", "")
    if isinstance(payload.get("result"), dict):
        return payload["result"].get("suffix_text", "")
    return ""


def append_suffix(prompt: str, suffix_text: str) -> str:
    if not suffix_text:
        return prompt
    return f"{prompt} {suffix_text}".strip()


def print_results(results: list[dict], layer: int, poscon_scale: float, negcon_scale: float) -> None:
    print(f"Layer {layer} qualitative steering test")
    print(f"Positive concept scale: {poscon_scale}")
    print(f"Negative concept scale: {negcon_scale}")
    print("")
    for index, row in enumerate(results, start=1):
        print(f"[{index}] Prompt: {row['prompt']}")
        if row["suffix_text"]:
            print(f"Suffix: {row['suffix_text']}")
            print(f"Full prompt: {row['full_prompt']}")
        print(f"Baseline: {row['baseline']}")
        print(f"Poscon-steered: {row['poscon_steered']}")
        print(f"Negcon-steered: {row['negcon_steered']}")
        print("")


if __name__ == "__main__":
    main()
