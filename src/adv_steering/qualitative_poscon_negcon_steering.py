from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch.nn.functional as F
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import generate_text, generate_text_with_steering, generate_text_with_top_logits, load_bundle
    from adv_steering.text_backend import encode_chat_prompt, steering_hook
else:
    from .backend import generate_text, generate_text_with_steering, generate_text_with_top_logits, load_bundle
    from .text_backend import encode_chat_prompt, steering_hook


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
    parser.add_argument("--show-top-logits", action="store_true", help="Print the top next-token logits at each generation step.")
    parser.add_argument("--top-logits-k", type=int, default=10, help="How many logits to print per generation step when --show-top-logits is enabled.")
    parser.add_argument("--response-targets", default="", help="Optional JSON file with fixed positive/neutral/negative responses for teacher-forced CE scoring.")
    parser.add_argument("--output", default="", help="Optional path to save results as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)
    output_path = resolve_output_path(args.output, args.steering_file)
    suffix_path = resolve_suffix_path(args.suffix, output_path.parent if output_path else None) if args.suffix else ""
    suffix_text = load_suffix_text(suffix_path, args.step) if suffix_path else ""
    response_targets = load_response_targets(args.response_targets) if args.response_targets else None
    steering_vector = load_steering_vector(args.steering_file, args.layer)
    bundle, resolved_backend = load_bundle(model_name=args.model, backend=args.backend, dtype=args.dtype, device_map=args.device_map)

    results = []
    for index, prompt in enumerate(prompts, start=1):
        full_prompt = append_suffix(prompt, suffix_text)
        print(f"[{index}/{len(prompts)}] Prompt: {prompt}", flush=True)
        if suffix_text:
            print(f"  suffix: {suffix_text}", flush=True)
            print(f"  full_prompt: {full_prompt}", flush=True)
        baseline, baseline_trace = generate_generation_result(
            bundle=bundle,
            backend=resolved_backend,
            prompt=full_prompt,
            max_new_tokens=args.max_new_tokens,
            show_top_logits=args.show_top_logits,
            top_logits_k=args.top_logits_k,
        )
        print("  baseline done", flush=True)
        poscon_steered, poscon_trace = generate_generation_result(
            bundle=bundle,
            backend=resolved_backend,
            prompt=full_prompt,
            max_new_tokens=args.max_new_tokens,
            show_top_logits=args.show_top_logits,
            top_logits_k=args.top_logits_k,
            steering_vector=steering_vector,
            layer=args.layer,
            scale=args.poscon_scale,
        )
        print("  poscon-steered done", flush=True)
        negcon_steered, negcon_trace = generate_generation_result(
            bundle=bundle,
            backend=resolved_backend,
            prompt=full_prompt,
            max_new_tokens=args.max_new_tokens,
            show_top_logits=args.show_top_logits,
            top_logits_k=args.top_logits_k,
            steering_vector=steering_vector,
            layer=args.layer,
            scale=args.negcon_scale,
        )
        print("  negcon-steered done", flush=True)
        if args.show_top_logits:
            print_token_trace("baseline", baseline_trace)
            print_token_trace("poscon-steered", poscon_trace)
            print_token_trace("negcon-steered", negcon_trace)
        target_cross_entropy = {}
        if response_targets is not None:
            target_cross_entropy = {
                "baseline": compute_response_target_cross_entropy(
                    bundle=bundle,
                    backend=resolved_backend,
                    prompt=full_prompt,
                    response_targets=response_targets,
                ),
                "poscon_steered": compute_response_target_cross_entropy(
                    bundle=bundle,
                    backend=resolved_backend,
                    prompt=full_prompt,
                    response_targets=response_targets,
                    steering_vector=steering_vector,
                    layer=args.layer,
                    scale=args.poscon_scale,
                ),
                "negcon_steered": compute_response_target_cross_entropy(
                    bundle=bundle,
                    backend=resolved_backend,
                    prompt=full_prompt,
                    response_targets=response_targets,
                    steering_vector=steering_vector,
                    layer=args.layer,
                    scale=args.negcon_scale,
                ),
            }
            print_target_cross_entropy(target_cross_entropy)
        results.append(
            {
                "prompt": prompt,
                "suffix_text": suffix_text,
                "full_prompt": full_prompt,
                "baseline": baseline,
                "poscon_steered": poscon_steered,
                "negcon_steered": negcon_steered,
                "baseline_token_trace": baseline_trace,
                "poscon_token_trace": poscon_trace,
                "negcon_token_trace": negcon_trace,
                "baseline_top_logits_table": token_trace_to_table(baseline_trace),
                "poscon_top_logits_table": token_trace_to_table(poscon_trace),
                "negcon_top_logits_table": token_trace_to_table(negcon_trace),
                "response_targets": response_targets,
                "target_cross_entropy": target_cross_entropy,
            }
        )
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_results(results, args.layer, args.poscon_scale, args.negcon_scale)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved qualitative outputs to {output_path}", flush=True)


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


def load_response_targets(path: str) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"No response targets found in {path}")
        payload = payload[0]
    positive = payload.get("positive_response") or payload.get("poscon_response") or payload.get("positive")
    neutral = payload.get("neutral_response") or payload.get("neutral")
    negative = payload.get("negative_response") or payload.get("negcon_response") or payload.get("negative")
    if not (positive and neutral and negative):
        raise ValueError("response-targets JSON must contain positive_response, neutral_response, and negative_response (or equivalent aliases).")
    return {
        "positive_response": positive,
        "neutral_response": neutral,
        "negative_response": negative,
    }


def load_suffix_text(path: str, step: int) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            return ""
        index = step if step >= 0 else len(payload) - 1
        if index < 0 or index >= len(payload):
            raise ValueError(f"step={step} is out of range for trace length {len(payload)}")
        row = payload[index]
        if isinstance(row, dict):
            return row.get("suffix_text", "")
        return str(row)
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


def resolve_output_path(output: str, steering_file: str) -> Path | None:
    if output:
        return Path(output)
    run_dir = infer_run_dir(steering_file)
    if run_dir is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "steering_generations" / f"qualitative_{timestamp}.json"


def resolve_suffix_path(suffix: str, output_dir: Path | None) -> str:
    suffix_path = Path(suffix)
    if suffix_path.exists():
        return str(suffix_path)
    if output_dir is None:
        return str(suffix_path)
    qualitative_dir = output_dir if output_dir.name == "steering_generations" else None
    run_dir = qualitative_dir.parent if qualitative_dir is not None else output_dir
    candidate = run_dir / "suffixes" / suffix_path.name
    if candidate.exists():
        return str(candidate)
    return str(suffix_path)


def infer_run_dir(artifact_path: str) -> Path | None:
    path = Path(artifact_path)
    if not path.parts:
        return None
    if path.parent.name in {"suffixes", "steering_generations"}:
        return path.parent.parent
    return path.parent


def append_suffix(prompt: str, suffix_text: str) -> str:
    if not suffix_text:
        return prompt
    return f"{prompt} {suffix_text}".strip()


@torch.no_grad()
def compute_response_target_cross_entropy(
    bundle,
    backend: str,
    prompt: str,
    response_targets: dict[str, str],
    steering_vector: torch.Tensor | None = None,
    layer: int | None = None,
    scale: float = 0.0,
) -> dict[str, float]:
    if backend != "causal_lm":
        raise ValueError("Fixed-response cross-entropy scoring is currently only supported for the causal_lm backend.")
    scores = {}
    for key, response_text in response_targets.items():
        scores[key] = compute_teacher_forced_cross_entropy(
            bundle=bundle,
            prompt=prompt,
            response_text=response_text,
            steering_vector=steering_vector,
            layer=layer,
            scale=scale,
        )
    return scores


@torch.no_grad()
def compute_teacher_forced_cross_entropy(
    bundle,
    prompt: str,
    response_text: str,
    steering_vector: torch.Tensor | None = None,
    layer: int | None = None,
    scale: float = 0.0,
) -> float:
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    prompt_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]
    target_ids = bundle.tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(bundle.device)
    full_input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    full_attention_mask = torch.cat(
        [attention_mask, torch.ones_like(target_ids, device=bundle.device, dtype=attention_mask.dtype)],
        dim=1,
    )
    context = (
        steering_hook(bundle.model, layer, steering_vector.to(bundle.device), scale)
        if steering_vector is not None and layer is not None
        else null_hook()
    )
    with context:
        outputs = bundle.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            use_cache=False,
        )
    prompt_length = prompt_ids.shape[1]
    target_length = target_ids.shape[1]
    logits = outputs.logits[:, prompt_length - 1 : prompt_length + target_length - 1, :]
    labels = target_ids
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(1, target_length)
    return float(token_losses.mean().item())


def generate_generation_result(
    bundle,
    backend: str,
    prompt: str,
    max_new_tokens: int,
    show_top_logits: bool,
    top_logits_k: int,
    steering_vector: torch.Tensor | None = None,
    layer: int | None = None,
    scale: float = 0.0,
) -> tuple[str, list[dict]]:
    if show_top_logits:
        traced = generate_text_with_top_logits(
            bundle=bundle,
            backend=backend,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_logits_k,
            steering_vector=steering_vector,
            layer=layer,
            scale=scale,
        )
        return traced["text"], traced["token_trace"]
    if steering_vector is None or layer is None:
        return generate_text(bundle, backend, prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0), []
    return (
        generate_text_with_steering(
            bundle,
            backend,
            prompt,
            layer=layer,
            steering_vector=steering_vector,
            scale=scale,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        ),
        [],
    )


def print_token_trace(label: str, token_trace: list[dict]) -> None:
    if not token_trace:
        return
    print(f"  {label} top logits:", flush=True)
    for row in token_trace:
        print(
            f"    step={row['position']} generated={row['generated_token_text']!r} "
            f"(id={row['generated_token_id']})",
            flush=True,
        )
        for candidate in row["top_logits"]:
            marker = "*" if candidate["is_generated"] else " "
            print(
                f"      {marker} rank={candidate['rank']} "
                f"token={candidate['token_text']!r} "
                f"id={candidate['token_id']} "
                f"logit={candidate['logit']:.6f}",
                flush=True,
            )


def print_target_cross_entropy(target_cross_entropy: dict[str, dict[str, float]]) -> None:
    print("  target cross-entropy:", flush=True)
    for branch_name, branch_scores in target_cross_entropy.items():
        print(
            f"    {branch_name}: "
            f"positive={branch_scores['positive_response']:.6f} "
            f"neutral={branch_scores['neutral_response']:.6f} "
            f"negative={branch_scores['negative_response']:.6f}",
            flush=True,
        )


def token_trace_to_table(token_trace: list[dict]) -> dict[str, dict[str, dict]]:
    table: dict[str, dict[str, dict]] = {}
    for row in token_trace:
        step_key = f"step_{row['position']}"
        table[step_key] = {}
        generated_id = row["generated_token_id"]
        logits = [candidate["logit"] for candidate in row["top_logits"]]
        probabilities = softmax(logits)
        for candidate, probability in zip(row["top_logits"], probabilities):
            rank_key = f"rank_{candidate['rank']}"
            table[step_key][rank_key] = {
                "token": candidate["token_text"],
                "id": candidate["token_id"],
                "logit": candidate["logit"],
                "probability": probability,
                "is_generated": candidate["token_id"] == generated_id,
            }
    return table


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [pow(2.718281828459045, value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


class null_hook:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


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
