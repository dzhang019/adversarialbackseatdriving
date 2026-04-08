from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.rep_suffix_attack import _single_prompt_objective_and_terms, load_prompt_pairs, load_steering_vector
    from adv_steering.rep_suffix_attack_largo import build_steered_ce_inputs
    from adv_steering.text_backend import encode_chat_prompt, load_text_model_bundle, split_chat_prompt_for_user_suffix
else:
    from .rep_suffix_attack import _single_prompt_objective_and_terms, load_prompt_pairs, load_steering_vector
    from .rep_suffix_attack_largo import build_steered_ce_inputs
    from .text_backend import encode_chat_prompt, load_text_model_bundle, split_chat_prompt_for_user_suffix


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize a continuous soft prompt without projecting back to discrete tokens.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--steering-file", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--objective-type", default="dot", choices=["dot", "cosine", "steered_ce"])
    parser.add_argument("--n-plus", default="")
    parser.add_argument("--n-minus", default="")
    parser.add_argument("--positive-prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--prompt-pairs-file", default="")
    parser.add_argument("--neutral-prompt", default="")
    parser.add_argument("--neutral-target", default="", help="Retained for compatibility; unused by current steered_ce.")
    parser.add_argument("--positive-target", default="")
    parser.add_argument("--negative-target", default="")
    parser.add_argument("--steering-scale", type=float, default=8.0)
    parser.add_argument("--suffix-length", type=int, default=200)
    parser.add_argument("--inner-steps", type=int, default=20, help="Number of optimization steps for the soft prompt matrix.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random_tokens"])
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_text_model_bundle(args.model, dtype=args.dtype, device_map=args.device_map)
    steering_vector = load_steering_vector(args.steering_file, args.layer).to(bundle.device)
    prompt_pairs = load_prompt_pairs(
        prompt_pairs_file=args.prompt_pairs_file,
        n_plus=args.n_plus,
        n_minus=args.n_minus,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
    )
    prompt_pair_segments = [
        (
            split_chat_prompt_for_user_suffix(bundle, n_plus),
            split_chat_prompt_for_user_suffix(bundle, n_minus),
        )
        for n_plus, n_minus in prompt_pairs
    ]
    neutral_prompt_ids, target_ids = build_steered_ce_inputs(
        bundle=bundle,
        objective_type=args.objective_type,
        neutral_prompt=args.neutral_prompt,
        positive_target=args.positive_target,
        negative_target=args.negative_target,
    )
    neutral_prompt_segments = (
        split_chat_prompt_for_user_suffix(bundle, args.neutral_prompt)
        if args.objective_type == "steered_ce"
        else None
    )

    soft_prompt = initialize_soft_prompt(
        bundle=bundle,
        suffix_length=args.suffix_length,
        init_mode=args.init_mode,
    )

    final_soft_prompt, trace, best = optimize_soft_prompt(
        bundle=bundle,
        soft_prompt=soft_prompt,
        prompt_pair_segments=prompt_pair_segments,
        steering_vector=steering_vector,
        layer=args.layer,
        objective_type=args.objective_type,
        target_ids=target_ids,
        neutral_prompt_segments=neutral_prompt_segments,
        steering_scale=args.steering_scale,
        inner_steps=args.inner_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    payload = {
        "model": args.model,
        "objective_type": args.objective_type,
        "layer": args.layer,
        "steering_scale": args.steering_scale,
        "neutral_prompt": args.neutral_prompt,
        "positive_target": args.positive_target,
        "negative_target": args.negative_target,
        "suffix_length": args.suffix_length,
        "init_mode": args.init_mode,
        "inner_steps": args.inner_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best_objective": best["objective"],
        "best_step": best["step"],
        "final_objective": trace[-1]["objective"] if trace else None,
        "soft_prompt_shape": list(final_soft_prompt.shape),
        "soft_prompt_dtype": str(final_soft_prompt.dtype),
        "soft_prompt": final_soft_prompt.detach().cpu().tolist(),
        "trace": trace,
        "prompt_pairs": [{"n_plus": n_plus, "n_minus": n_minus} for n_plus, n_minus in prompt_pairs],
    }
    print(
        json.dumps(
            {
                "model": payload["model"],
                "objective_type": payload["objective_type"],
                "layer": payload["layer"],
                "best_objective": payload["best_objective"],
                "best_step": payload["best_step"],
                "final_objective": payload["final_objective"],
                "soft_prompt_shape": payload["soft_prompt_shape"],
                "soft_prompt_dtype": payload["soft_prompt_dtype"],
                "output": str(resolve_output_path(args.output, args.steering_file)),
            },
            indent=2,
        )
    )

    output_path = resolve_output_path(args.output, args.steering_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved soft prompt artifact to {output_path}", flush=True)


def initialize_soft_prompt(bundle, suffix_length: int, init_mode: str) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    hidden_size = embedding_layer.weight.shape[1]
    if init_mode == "zeros":
        return torch.zeros((1, suffix_length, hidden_size), device=bundle.device, dtype=torch.float32)
    vocab_size = embedding_layer.weight.shape[0]
    token_ids = torch.randint(low=0, high=vocab_size, size=(suffix_length,), device=bundle.device)
    return embedding_layer(token_ids.unsqueeze(0)).detach().float()


def optimize_soft_prompt(
    bundle,
    soft_prompt: torch.Tensor,
    prompt_pair_segments,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids,
    neutral_prompt_segments,
    steering_scale: float,
    inner_steps: int,
    learning_rate: float,
    weight_decay: float,
):
    embedding_layer = bundle.model.get_input_embeddings()
    soft_prompt_parameter = torch.nn.Parameter(soft_prompt.detach().clone().float())
    optimizer = torch.optim.Adam([soft_prompt_parameter], lr=learning_rate, weight_decay=weight_decay)
    objective_prompt_pair_segments = prompt_pair_segments if objective_type != "steered_ce" else prompt_pair_segments[:1]
    trace = []
    best = {"objective": float("inf"), "step": -1}

    for inner_step in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), device=bundle.device, dtype=torch.float32)
        soft_prompt_embeds = soft_prompt_parameter.to(embedding_layer.weight.dtype)
        totals = None
        neutral_prompt_embeds = None if neutral_prompt_segments is None else build_prompt_embeds_with_soft_prompt(bundle, neutral_prompt_segments, soft_prompt_embeds)
        for plus_segments, minus_segments in objective_prompt_pair_segments:
            plus_prompt_embeds = build_prompt_embeds_with_soft_prompt(bundle, plus_segments, soft_prompt_embeds)
            minus_prompt_embeds = build_prompt_embeds_with_soft_prompt(bundle, minus_segments, soft_prompt_embeds)
            prompt_objective, terms = _single_prompt_objective_and_terms(
                bundle=bundle,
                plus_prompt_embeds=plus_prompt_embeds,
                minus_prompt_embeds=minus_prompt_embeds,
                steering_vector=steering_vector,
                layer=layer,
                objective_type=objective_type,
                target_ids=target_ids,
                neutral_prompt_embeds=neutral_prompt_embeds,
                steering_scale=steering_scale,
            )
            objective = objective + prompt_objective.float()
            if totals is None:
                totals = {name: float(value.detach().float().item()) for name, value in terms.items()}
            else:
                for name, value in terms.items():
                    totals[name] += float(value.detach().float().item())
        objective.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([soft_prompt_parameter], max_norm=1.0).item())
        optimizer.step()
        if not torch.isfinite(soft_prompt_parameter).all():
            raise ValueError(f"Non-finite soft prompt after step {inner_step + 1}/{inner_steps}.")

        objective_value = float(objective.detach().item())
        if objective_value < best["objective"]:
            best = {"objective": objective_value, "step": inner_step}
        trace.append(
            {
                "step": inner_step,
                "objective": objective_value,
                "term_breakdown": totals or {},
                "grad_norm": grad_norm,
                "soft_prompt_norm": float(soft_prompt_parameter.detach().norm().item()),
            }
        )
        print(
            f"[step {inner_step + 1}/{inner_steps}] objective={objective_value:.6f} "
            f"grad_norm={grad_norm:.6f} soft_prompt_norm={soft_prompt_parameter.detach().norm().item():.6f}",
            flush=True,
        )

    return soft_prompt_parameter.detach(), trace, best


def build_prompt_embeds_with_soft_prompt(bundle, prompt_segments: dict[str, torch.Tensor], soft_prompt_embeds: torch.Tensor) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    user_embeds = embedding_layer(prompt_segments["user_input_ids"]).detach()
    assistant_prefix_embeds = embedding_layer(prompt_segments["assistant_prefix_ids"]).detach()
    return torch.cat([user_embeds, soft_prompt_embeds, assistant_prefix_embeds], dim=1)


def resolve_output_path(output: str, steering_file: str) -> Path:
    if output:
        return Path(output)
    run_dir = infer_run_dir(steering_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "suffixes" / f"rep_suffix_soft_prompt_{timestamp}.json"


def infer_run_dir(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.parent.name in {"suffixes", "steering_generations"}:
        return path.parent.parent
    return path.parent


if __name__ == "__main__":
    main()
