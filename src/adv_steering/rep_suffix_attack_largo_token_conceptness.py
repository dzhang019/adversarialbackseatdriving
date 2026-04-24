from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import load_bundle
    from adv_steering.rep_prefix_attack_soft_prompt import (
        inversion_loss_sign,
        load_examples,
        load_weighted_steering_entries,
        select_tokens_for_label,
        selection_rule_for_label,
    )
    from adv_steering.rep_suffix_attack_largo import (
        build_prompt_embeds_with_tokseq,
        build_summary_prompt_variants,
        initialize_suffix_embeds,
        nearest_token_ids_for_suffix_embeds,
        prepare_prompt_segments,
        suffix_embeds_from_summary_token_ids,
        summarize_suffix_matrix,
        truncate_suffix_token_ids,
    )
    from adv_steering.text_backend import encode_chat_prompt, split_chat_prompt_for_user_suffix
else:
    from .backend import load_bundle
    from .rep_prefix_attack_soft_prompt import (
        inversion_loss_sign,
        load_examples,
        load_weighted_steering_entries,
        select_tokens_for_label,
        selection_rule_for_label,
    )
    from .rep_suffix_attack_largo import (
        build_prompt_embeds_with_tokseq,
        build_summary_prompt_variants,
        initialize_suffix_embeds,
        nearest_token_ids_for_suffix_embeds,
        prepare_prompt_segments,
        suffix_embeds_from_summary_token_ids,
        summarize_suffix_matrix,
        truncate_suffix_token_ids,
    )
    from .text_backend import encode_chat_prompt, split_chat_prompt_for_user_suffix


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "LARGO-style continuous tokseq optimization for token-level conceptness: "
            "optimize a soft prefix/suffix token sequence, verbalize it into candidate text suffixes, "
            "and keep the candidate that best inverts selected story-token probe dots."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="causal_lm", choices=["auto", "causal_lm"], help="Only causal_lm is supported.")
    parser.add_argument("--steering-file", required=True, help="Path to poscon_negcon_residuals.pt or steering_candidates.pt.")
    parser.add_argument("--steering-config", required=True, help="JSON config with entries containing layer and scale/coefficient/weight.")
    parser.add_argument("--dataset", default="data/llama_happy_sad_corpus_long.jsonl", help="JSONL corpus with prompts and stories.")
    parser.add_argument("--max-examples", type=int, default=4, help="Maximum corpus rows to optimize over.")
    parser.add_argument("--story-side", default="both", choices=["positive", "negative", "both"], help="Which responses from the corpus to use.")
    parser.add_argument("--prompt", default="", help="Optional single prompt for custom optimization.")
    parser.add_argument("--story", default="", help="Optional single story/response for custom optimization.")
    parser.add_argument("--custom-label", default="positive", choices=["positive", "negative"], help="Label used with --prompt/--story.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--tokseq-position",
        default="suffix",
        choices=["prefix", "suffix"],
        help="Insert the optimized token sequence after the assistant generation prefix (prefix) or inside the user message (suffix).",
    )
    parser.add_argument("--suffix-length", type=int, default=20, help="Number of continuous suffix vectors inserted inside the user message.")
    parser.add_argument("--outer-steps", type=int, default=15, help="How many optimize/verbalize rounds to run.")
    parser.add_argument("--inner-steps", type=int, default=20, help="How many gradient steps to take on the soft suffix per outer round.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random_tokens"])
    parser.add_argument("--dot-threshold", type=float, default=0.1, help="Positive examples select dot >= threshold; negative examples select dot <= -threshold.")
    parser.add_argument("--preserve-weight", type=float, default=1.0, help="Weight on preserving non-selected token dot products.")
    parser.add_argument("--inversion-weight", type=float, default=1.0, help="Weight on inverting selected token dot products.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm. Use <=0 to disable clipping.")
    parser.add_argument("--summary-prompt", default="Summarize the following: ")
    parser.add_argument("--summary-samples", type=int, default=16, help="How many sampled verbalizations to evaluate per outer round.")
    parser.add_argument("--summary-temperature", type=float, default=1.0, help="Sampling temperature used when verbalizing the soft suffix.")
    parser.add_argument("--output", default="", help="Optional artifact output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt or args.story:
        if not (args.prompt and args.story):
            raise ValueError("Provide both --prompt and --story for custom optimization.")

    bundle, resolved_backend = load_bundle(args.model, args.backend, args.dtype, args.device_map)
    if resolved_backend != "causal_lm":
        raise ValueError("rep_suffix_attack_largo_token_conceptness.py currently supports only causal_lm.")

    steering_entries = load_weighted_steering_entries(args.steering_config, args.steering_file)
    raw_examples = load_examples(args)
    if not raw_examples:
        raise ValueError("No examples found for optimization.")
    examples = [
        prepare_example(
            bundle=bundle,
            example=example,
            steering_entries=steering_entries,
            dot_threshold=args.dot_threshold,
            tokseq_position=args.tokseq_position,
        )
        for example in raw_examples
    ]

    suffix_embeds = initialize_suffix_embeds(bundle=bundle, suffix_length=args.suffix_length, init_mode=args.init_mode)
    initial_suffix_token_ids = nearest_token_ids_for_suffix_embeds(bundle, suffix_embeds)
    initial_suffix_text = bundle.tokenizer.decode(initial_suffix_token_ids.tolist(), skip_special_tokens=True)
    trace = []
    best = {
        "objective": float("inf"),
        "suffix_token_ids": None,
        "suffix_text": "",
        "summary_text": "",
    }

    summary_prompts = build_summary_prompt_variants(args.summary_prompt)
    for outer_step in range(args.outer_steps):
        suffix_embeds, inner_trace = optimize_soft_suffix(
            bundle=bundle,
            suffix_embeds=suffix_embeds,
            examples=examples,
            steering_entries=steering_entries,
            inner_steps=args.inner_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
            grad_clip=args.grad_clip,
        )
        summary_candidates, selected = sample_and_select_token_conceptness_summary(
            bundle=bundle,
            suffix_embeds=suffix_embeds,
            model_name=args.model,
            summary_prompts=summary_prompts,
            summary_samples=args.summary_samples,
            summary_temperature=args.summary_temperature,
            max_new_tokens=args.suffix_length,
            examples=examples,
            steering_entries=steering_entries,
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        )
        suffix_token_ids = selected["suffix_token_ids_tensor"]
        suffix_embeds = suffix_embeds_from_summary_token_ids(
            bundle=bundle,
            summary_token_ids=suffix_token_ids,
            suffix_length=args.suffix_length,
        )

        suffix_text = bundle.tokenizer.decode(suffix_token_ids.tolist(), skip_special_tokens=True)
        trace_row = {
            "outer_step": outer_step,
            "objective": selected["objective"],
            "loss_breakdown": selected["loss_breakdown"],
            "inner_trace": inner_trace,
            "summary_candidates": [
                {key: value for key, value in candidate.items() if key != "suffix_token_ids_tensor"}
                for candidate in summary_candidates
            ],
            "selected_summary_candidate": selected["candidate_index"],
            "selected_summary_prompt": selected["summary_prompt"],
            "selected_assistant_prefill": selected["assistant_prefill"],
            "summary_token_ids": suffix_token_ids.tolist(),
            "summary_text": selected["summary_text"],
            "suffix_token_ids": suffix_token_ids.tolist(),
            "suffix_text": suffix_text,
        }
        trace.append(trace_row)
        print(
            f"[outer {outer_step + 1}/{args.outer_steps}] objective={selected['objective']:.6f} "
            f"inversion={selected['loss_breakdown']['inversion_loss']:.6f} "
            f"preserve={selected['loss_breakdown']['preserve_loss']:.6f} suffix={suffix_text!r}",
            flush=True,
        )

        if selected["objective"] < best["objective"]:
            best = {
                "objective": selected["objective"],
                "suffix_token_ids": suffix_token_ids.tolist(),
                "suffix_text": suffix_text,
                "summary_text": selected["summary_text"],
            }

    output_path = resolve_output_path(args.output, args.steering_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "backend": resolved_backend,
        "steering_file": args.steering_file,
        "steering_config": args.steering_config,
        "dataset": args.dataset,
        "story_side": args.story_side,
        "max_examples": args.max_examples,
        "custom_prompt": args.prompt,
        "custom_story": args.story,
        "custom_label": args.custom_label,
        "tokseq_position": args.tokseq_position,
        "suffix_length": args.suffix_length,
        "outer_steps": args.outer_steps,
        "inner_steps": args.inner_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "init_mode": args.init_mode,
        "dot_threshold": args.dot_threshold,
        "inversion_weight": args.inversion_weight,
        "preserve_weight": args.preserve_weight,
        "grad_clip": args.grad_clip,
        "summary_prompt": args.summary_prompt,
        "summary_samples": args.summary_samples,
        "summary_temperature": args.summary_temperature,
        "summary_prompt_variants": summary_prompts,
        "initial_suffix_token_ids": initial_suffix_token_ids.tolist(),
        "initial_suffix_text": initial_suffix_text,
        "suffix_token_ids": best["suffix_token_ids"],
        "suffix_text": best["suffix_text"],
        "summary_text": best["summary_text"],
        "best": best,
        "steering_entries": [
            {
                "layer": entry["layer"],
                "scale": entry["scale"],
                "weight": entry["weight"],
                "steering_file": entry["steering_file"],
            }
            for entry in steering_entries
        ],
        "examples": [summarize_prepared_example(example) for example in examples],
        "trace": trace,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "examples": len(examples),
                "best_objective": best["objective"],
                "best_suffix_text": best["suffix_text"],
            },
            indent=2,
        ),
        flush=True,
    )


@torch.no_grad()
def prepare_example(
    bundle,
    example: dict,
    steering_entries: list[dict],
    dot_threshold: float,
    tokseq_position: str,
) -> dict:
    segments = prepare_prompt_segments(bundle, example["prompt"], tokseq_position)
    story_ids = torch.tensor(
        bundle.tokenizer.encode(example["story"], add_special_tokens=False),
        dtype=torch.long,
        device=bundle.device,
    ).unsqueeze(0)
    if story_ids.shape[1] == 0:
        raise ValueError("Story tokenized to zero tokens.")
    baseline_scores = compute_story_probe_scores(
        bundle=bundle,
        prompt_segments=segments,
        story_ids=story_ids,
        suffix_embeds=None,
        suffix_token_ids=None,
        steering_entries=steering_entries,
    )
    entry_stats = {}
    for entry in steering_entries:
        layer = entry["layer"]
        selected_mask = select_tokens_for_label(baseline_scores[layer], example["label"], dot_threshold)
        entry_stats[str(layer)] = {
            "selection_rule": selection_rule_for_label(example["label"], dot_threshold),
            "selected_tokens": int(selected_mask.sum().item()),
            "preserved_tokens": int((~selected_mask).sum().item()),
            "baseline_mean_dot": float(baseline_scores[layer].mean().item()),
            "baseline_max_dot": float(baseline_scores[layer].max().item()),
            "baseline_min_dot": float(baseline_scores[layer].min().item()),
        }
    return {
        **example,
        "prompt_segments": detach_prompt_segments(segments),
        "story_ids": story_ids.detach(),
        "baseline_dots": {layer: values.detach() for layer, values in baseline_scores.items()},
        "entry_stats": entry_stats,
    }


def detach_prompt_segments(prompt_segments: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor | str]:
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in prompt_segments.items()
    }


def optimize_soft_suffix(
    bundle,
    suffix_embeds: torch.Tensor,
    examples: list[dict],
    steering_entries: list[dict],
    inner_steps: int,
    learning_rate: float,
    weight_decay: float,
    inversion_weight: float,
    preserve_weight: float,
    dot_threshold: float,
    grad_clip: float,
) -> tuple[torch.Tensor, list[dict]]:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_parameter = torch.nn.Parameter(suffix_embeds.detach().clone().float())
    optimizer = torch.optim.Adam([suffix_parameter], lr=learning_rate, weight_decay=weight_decay)
    trace = []

    for inner_step in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        typed_suffix = suffix_parameter.to(embedding_layer.weight.dtype)
        objective, breakdown = token_conceptness_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            suffix_embeds=typed_suffix,
            suffix_token_ids=None,
            inversion_weight=inversion_weight,
            preserve_weight=preserve_weight,
            dot_threshold=dot_threshold,
        )
        objective.backward()
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_([suffix_parameter], max_norm=grad_clip).item())
        else:
            grad_norm = float(suffix_parameter.grad.detach().norm().item()) if suffix_parameter.grad is not None else 0.0
        optimizer.step()
        if not torch.isfinite(suffix_parameter).all():
            raise ValueError(f"Non-finite suffix parameter after inner step {inner_step + 1}/{inner_steps}.")

        row = {
            "inner_step": inner_step,
            "objective": float(objective.detach().item()),
            **breakdown,
            "grad_norm": grad_norm,
            "suffix_norm": float(suffix_parameter.detach().norm().item()),
        }
        trace.append(row)
        print(
            f"  [inner {inner_step + 1}/{inner_steps}] objective={row['objective']:.6f} "
            f"inversion={row['inversion_loss']:.6f} preserve={row['preserve_loss']:.6f} "
            f"grad_norm={grad_norm:.6f}",
            flush=True,
        )

    return suffix_parameter.detach(), trace


@torch.no_grad()
def sample_and_select_token_conceptness_summary(
    bundle,
    suffix_embeds: torch.Tensor,
    model_name: str,
    summary_prompts: list[dict[str, str]],
    summary_samples: int,
    summary_temperature: float,
    max_new_tokens: int,
    examples: list[dict],
    steering_entries: list[dict],
    inversion_weight: float,
    preserve_weight: float,
    dot_threshold: float,
) -> tuple[list[dict], dict]:
    if summary_samples <= 0:
        raise ValueError("--summary-samples must be positive.")
    candidates = []
    for candidate_index in range(summary_samples):
        prompt_variant = summary_prompts[candidate_index % len(summary_prompts)]
        raw_summary_token_ids, summary_text = summarize_suffix_matrix(
            bundle=bundle,
            suffix_embeds=suffix_embeds,
            model_name=model_name,
            summary_prompt=prompt_variant["summary_prompt"],
            assistant_prefill=prompt_variant["assistant_prefill"],
            max_new_tokens=max_new_tokens,
            temperature=summary_temperature,
        )
        suffix_token_ids = truncate_suffix_token_ids(bundle, raw_summary_token_ids, max_new_tokens)
        objective, breakdown = token_conceptness_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            suffix_embeds=None,
            suffix_token_ids=suffix_token_ids,
            inversion_weight=inversion_weight,
            preserve_weight=preserve_weight,
            dot_threshold=dot_threshold,
        )
        suffix_text = bundle.tokenizer.decode(suffix_token_ids.tolist(), skip_special_tokens=True)
        candidate = {
            "candidate_index": candidate_index,
            "summary_prompt": prompt_variant["summary_prompt"],
            "assistant_prefill": prompt_variant["assistant_prefill"],
            "objective": float(objective.item()),
            "loss_breakdown": breakdown,
            "raw_summary_token_ids": raw_summary_token_ids,
            "summary_token_ids": suffix_token_ids.tolist(),
            "summary_text": summary_text,
            "suffix_token_ids": suffix_token_ids.tolist(),
            "suffix_text": suffix_text,
            "suffix_token_ids_tensor": suffix_token_ids,
        }
        candidates.append(candidate)
        print(
            f"  [summary {candidate_index + 1}/{summary_samples}] "
            f"objective={candidate['objective']:.6f} prompt={prompt_variant['name']!r} suffix={suffix_text!r}",
            flush=True,
        )

    selected = min(candidates, key=lambda candidate: candidate["objective"])
    print(
        f"  [summary selected] index={selected['candidate_index']} "
        f"objective={selected['objective']:.6f} suffix={selected['suffix_text']!r}",
        flush=True,
    )
    return candidates, selected


def token_conceptness_objective(
    bundle,
    examples: list[dict],
    steering_entries: list[dict],
    suffix_embeds: torch.Tensor | None,
    suffix_token_ids: torch.Tensor | None,
    inversion_weight: float,
    preserve_weight: float,
    dot_threshold: float,
) -> tuple[torch.Tensor, dict]:
    objective = torch.zeros((), device=bundle.device, dtype=torch.float32)
    totals = {
        "inversion_loss": 0.0,
        "preserve_loss": 0.0,
        "selected_tokens": 0,
        "preserved_tokens": 0,
    }
    for example in examples:
        current_scores = compute_story_probe_scores(
            bundle=bundle,
            prompt_segments=example["prompt_segments"],
            story_ids=example["story_ids"],
            suffix_embeds=suffix_embeds,
            suffix_token_ids=suffix_token_ids,
            steering_entries=steering_entries,
        )
        for entry in steering_entries:
            layer = entry["layer"]
            weight = float(entry["weight"])
            baseline_dots = example["baseline_dots"][layer].to(bundle.device)
            current_dots = current_scores[layer]
            selected_mask = select_tokens_for_label(baseline_dots, example["label"], dot_threshold)
            preserved_mask = ~selected_mask

            sign = inversion_loss_sign(example["label"])
            inversion_loss = sign * current_dots[selected_mask].mean() if selected_mask.any() else current_dots.sum() * 0.0
            preserve_loss = (
                F.mse_loss(current_dots[preserved_mask], baseline_dots[preserved_mask])
                if preserved_mask.any()
                else current_dots.sum() * 0.0
            )
            objective = objective + weight * (inversion_weight * inversion_loss + preserve_weight * preserve_loss)
            totals["inversion_loss"] += float((weight * inversion_loss).detach().item())
            totals["preserve_loss"] += float((weight * preserve_loss).detach().item())
            totals["selected_tokens"] += int(selected_mask.sum().item())
            totals["preserved_tokens"] += int(preserved_mask.sum().item())

    example_count = max(len(examples), 1)
    objective = objective / example_count
    return objective, {
        "inversion_loss": totals["inversion_loss"] / example_count,
        "preserve_loss": totals["preserve_loss"] / example_count,
        "selected_tokens": totals["selected_tokens"],
        "preserved_tokens": totals["preserved_tokens"],
    }


def compute_story_probe_scores(
    bundle,
    prompt_segments: dict[str, torch.Tensor | str],
    story_ids: torch.Tensor,
    suffix_embeds: torch.Tensor | None,
    suffix_token_ids: torch.Tensor | None,
    steering_entries: list[dict],
) -> dict[int, torch.Tensor]:
    embedding_layer = bundle.model.get_input_embeddings()
    user_embeds = embedding_layer(prompt_segments["user_input_ids"]).detach()
    assistant_prefix_embeds = embedding_layer(prompt_segments["assistant_prefix_ids"]).detach()
    story_embeds = embedding_layer(story_ids).detach()
    if suffix_embeds is not None and suffix_token_ids is not None:
        raise ValueError("Use either suffix_embeds or suffix_token_ids, not both.")
    if suffix_embeds is not None:
        suffix_part_embeds = suffix_embeds.to(bundle.device).to(embedding_layer.weight.dtype)
    elif suffix_token_ids is not None:
        suffix_part_embeds = embedding_layer(suffix_token_ids.to(bundle.device).unsqueeze(0)).detach()
    else:
        suffix_part_embeds = None

    if suffix_part_embeds is not None:
        prompt_embeds = build_prompt_embeds_with_tokseq(bundle, prompt_segments, suffix_part_embeds)
    else:
        if prompt_segments["tokseq_position"] == "prefix":
            prompt_embeds = embedding_layer(prompt_segments["prompt_with_generation_ids"]).detach()
        else:
            prompt_embeds = torch.cat([user_embeds, assistant_prefix_embeds], dim=1)
    input_embeds = torch.cat([prompt_embeds, story_embeds], dim=1)
    story_start = input_embeds.shape[1] - story_embeds.shape[1]
    attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    scores = {}
    story_length = story_ids.shape[1]
    for entry in steering_entries:
        vector = entry["vector"].to(bundle.device).to(torch.float32)
        states = outputs.hidden_states[entry["layer"] + 1][0, story_start : story_start + story_length, :].float()
        scores[entry["layer"]] = torch.matmul(states, vector)
    return scores


def summarize_prepared_example(example: dict) -> dict:
    return {
        "concept": example["concept"],
        "label": example["label"],
        "prompt": example["prompt"],
        "story": example["story"],
        "tokseq_position": str(example["prompt_segments"]["tokseq_position"]),
        "prompt_tokens": int(
            (
                example["prompt_segments"]["prompt_with_generation_ids"].shape[1]
                if example["prompt_segments"]["tokseq_position"] == "prefix"
                else example["prompt_segments"]["user_input_ids"].shape[1] + example["prompt_segments"]["assistant_prefix_ids"].shape[1]
            )
        ),
        "user_prompt_tokens": int(example["prompt_segments"]["user_input_ids"].shape[1])
        if example["prompt_segments"]["tokseq_position"] == "suffix"
        else 0,
        "assistant_prefix_tokens": int(example["prompt_segments"]["assistant_prefix_ids"].shape[1])
        if example["prompt_segments"]["tokseq_position"] == "suffix"
        else 0,
        "story_tokens": int(example["story_ids"].shape[1]),
        "entry_stats": example["entry_stats"],
    }


def resolve_output_path(output: str, steering_file: str) -> Path:
    if output:
        return Path(output)
    run_dir = infer_run_dir(steering_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "suffixes" / f"rep_suffix_largo_token_conceptness_{timestamp}.json"


def infer_run_dir(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.parent.name in {"suffixes", "steering_generations", "token_conceptness", "prefixes"}:
        return path.parent.parent
    return path.parent


if __name__ == "__main__":
    main()
