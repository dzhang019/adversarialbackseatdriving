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
        build_summary_prompt_variants,
        initialize_tokseq_embeds,
        nearest_token_ids_for_tokseq_embeds,
        tokseq_embeds_from_summary_token_ids,
        summarize_tokseq_matrix,
        truncate_tokseq_token_ids,
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
        build_summary_prompt_variants,
        initialize_tokseq_embeds,
        nearest_token_ids_for_tokseq_embeds,
        tokseq_embeds_from_summary_token_ids,
        summarize_tokseq_matrix,
        truncate_tokseq_token_ids,
    )
    from .text_backend import encode_chat_prompt, split_chat_prompt_for_user_suffix


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "LARGO-style continuous tokseq optimization for tokenwise probe: "
            "optimize a soft token sequence, verbalize it into candidate text tokseqs, "
            "and keep the candidate that best inverts selected story-token probe dots."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="causal_lm", choices=["auto", "causal_lm"], help="Only causal_lm is supported.")
    parser.add_argument("--steering-file", required=True, help="Path to poscon_negcon_residuals.pt or steering_candidates.pt.")
    parser.add_argument("--steering-config", default="", help="Optional JSON config with entries containing layer and scale/coefficient/weight.")
    parser.add_argument("--layer", type=int, default=None, help="Probe layer to optimize when --steering-config is omitted.")
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
        help="Insert the optimized token sequence before the user prompt text (prefix) or after it inside the user message (suffix).",
    )
    parser.add_argument(
        "--tokseq-length",
        dest="tokseq_length",
        type=int,
        default=20,
        help="Number of continuous tokseq vectors to optimize.",
    )
    parser.add_argument("--suffix-length", dest="tokseq_length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--outer-steps", type=int, default=15, help="How many optimize/verbalize rounds to run.")
    parser.add_argument("--inner-steps", type=int, default=20, help="How many gradient steps to take on the soft tokseq per outer round.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random_tokens"])
    parser.add_argument("--dot-threshold", type=float, default=0.1, help="Positive examples select dot >= threshold; negative examples select dot <= -threshold.")
    parser.add_argument("--preserve-weight", type=float, default=1.0, help="Weight on preserving non-selected token dot products.")
    parser.add_argument("--inversion-weight", type=float, default=1.0, help="Weight on inverting selected token dot products.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm. Use <=0 to disable clipping.")
    parser.add_argument("--summary-prompt", default="Summarize the following: ")
    parser.add_argument("--summary-samples", type=int, default=16, help="How many sampled verbalizations to evaluate per outer round.")
    parser.add_argument("--summary-temperature", type=float, default=1.0, help="Sampling temperature used when verbalizing the soft tokseq.")
    parser.add_argument("--output", default="", help="Optional artifact output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt or args.story:
        if not (args.prompt and args.story):
            raise ValueError("Provide both --prompt and --story for custom optimization.")

    bundle, resolved_backend = load_bundle(args.model, args.backend, args.dtype, args.device_map)
    if resolved_backend != "causal_lm":
        raise ValueError("ps_tokenwise_probe_attack_largo.py currently supports only causal_lm.")
    print(
        f"[setup] loaded model={args.model!r} backend={resolved_backend} device={bundle.device}",
        flush=True,
    )

    steering_entries = load_probe_entries(args.steering_config, args.steering_file, args.layer)
    print(
        f"[setup] loaded {len(steering_entries)} steering entr{'y' if len(steering_entries) == 1 else 'ies'} "
        f"objective={steering_entries[0]['objective_type']!r}",
        flush=True,
    )
    raw_examples = load_examples(args)
    if not raw_examples:
        raise ValueError("No examples found for optimization.")
    print(f"[setup] loaded {len(raw_examples)} raw example(s); preparing baseline token scores", flush=True)
    examples = []
    for example_index, example in enumerate(raw_examples, start=1):
        print(
            f"[setup] preparing example {example_index}/{len(raw_examples)} "
            f"label={example['label']!r} concept={example.get('concept', '')!r}",
            flush=True,
        )
        prepared = prepare_example(
            bundle=bundle,
            example=example,
            steering_entries=steering_entries,
            dot_threshold=args.dot_threshold,
            tokseq_position=args.tokseq_position,
        )
        examples.append(prepared)
        story_tokens = int(prepared["story_ids"].shape[1])
        selected_tokens = sum(int(row["selected_tokens"]) for row in prepared["entry_stats"].values())
        print(
            f"[setup] prepared example {example_index}/{len(raw_examples)} "
            f"story_tokens={story_tokens} selected_tokens={selected_tokens}",
            flush=True,
        )

    tokseq_embeds = initialize_tokseq_embeds(bundle=bundle, tokseq_length=args.tokseq_length, init_mode=args.init_mode)
    print(
        f"[setup] initialized tokseq embeds shape={list(tokseq_embeds.shape)} init_mode={args.init_mode!r}",
        flush=True,
    )
    initial_tokseq_token_ids = nearest_token_ids_for_tokseq_embeds(bundle, tokseq_embeds)
    initial_tokseq_text = bundle.tokenizer.decode(initial_tokseq_token_ids.tolist(), skip_special_tokens=True)
    print(f"[setup] initial nearest-token tokseq={initial_tokseq_text!r}", flush=True)
    trace = []
    best = {
        "objective": float("inf"),
        "tokseq_token_ids": None,
        "tokseq_text": "",
        "summary_text": "",
    }

    summary_prompts = build_summary_prompt_variants(args.summary_prompt)
    for outer_step in range(args.outer_steps):
        tokseq_embeds, inner_trace = optimize_soft_tokseq(
            bundle=bundle,
            tokseq_embeds=tokseq_embeds,
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
        summary_candidates, selected = sample_and_select_tokenwise_probe_summary(
            bundle=bundle,
            tokseq_embeds=tokseq_embeds,
            model_name=args.model,
            summary_prompts=summary_prompts,
            summary_samples=args.summary_samples,
            summary_temperature=args.summary_temperature,
            max_new_tokens=args.tokseq_length,
            examples=examples,
            steering_entries=steering_entries,
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        )
        tokseq_token_ids = selected["tokseq_token_ids_tensor"]
        tokseq_embeds = tokseq_embeds_from_summary_token_ids(
            bundle=bundle,
            summary_token_ids=tokseq_token_ids,
            tokseq_length=args.tokseq_length,
        )

        tokseq_text = bundle.tokenizer.decode(tokseq_token_ids.tolist(), skip_special_tokens=True)
        trace_row = {
            "outer_step": outer_step,
            "objective": selected["objective"],
            "loss_breakdown": selected["loss_breakdown"],
            "inner_trace": inner_trace,
            "summary_candidates": [
                {key: value for key, value in candidate.items() if key != "tokseq_token_ids_tensor"}
                for candidate in summary_candidates
            ],
            "selected_summary_candidate": selected["candidate_index"],
            "selected_summary_prompt": selected["summary_prompt"],
            "selected_assistant_prefill": selected["assistant_prefill"],
            "summary_token_ids": tokseq_token_ids.tolist(),
            "summary_text": selected["summary_text"],
            "tokseq_token_ids": tokseq_token_ids.tolist(),
            "tokseq_text": tokseq_text,
        }
        trace.append(trace_row)
        print(
            f"[outer {outer_step + 1}/{args.outer_steps}] objective={selected['objective']:.6f} "
            f"inversion={selected['loss_breakdown']['inversion_loss']:.6f} "
            f"preserve={selected['loss_breakdown']['preserve_loss']:.6f} tokseq={tokseq_text!r}",
            flush=True,
        )

        if selected["objective"] < best["objective"]:
            best = {
                "objective": selected["objective"],
                "tokseq_token_ids": tokseq_token_ids.tolist(),
                "tokseq_text": tokseq_text,
                "summary_text": selected["summary_text"],
            }

    output_path = resolve_output_path(args.output, args.steering_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "backend": resolved_backend,
        "steering_file": args.steering_file,
        "steering_config": args.steering_config,
        "layer": args.layer,
        "dataset": args.dataset,
        "story_side": args.story_side,
        "max_examples": args.max_examples,
        "custom_prompt": args.prompt,
        "custom_story": args.story,
        "custom_label": args.custom_label,
        "tokseq_position": args.tokseq_position,
        "tokseq_length": args.tokseq_length,
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
        "initial_tokseq_token_ids": initial_tokseq_token_ids.tolist(),
        "initial_tokseq_text": initial_tokseq_text,
        "tokseq_token_ids": best["tokseq_token_ids"],
        "tokseq_text": best["tokseq_text"],
        "summary_text": best["summary_text"],
        "best": best,
        "steering_entries": [
            {
                "layer": entry["layer"],
                "scale": entry["scale"],
                "weight": entry["weight"],
                "steering_file": entry["steering_file"],
                "objective_type": entry["objective_type"],
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
                "best_tokseq_text": best["tokseq_text"],
            },
            indent=2,
        ),
        flush=True,
    )


def load_probe_entries(config_path: str, steering_file: str, layer: int | None) -> list[dict]:
    if config_path:
        entries = load_weighted_steering_entries(config_path, steering_file)
        for entry in entries:
            entry["objective_type"] = "label_inversion"
        return entries

    if layer is None:
        raise ValueError("Provide either --steering-config or --layer.")
    return [
        {
            "layer": int(layer),
            "scale": 1.0,
            "weight": 1.0,
            "steering_file": steering_file,
            "vector": load_steering_vector(steering_file, int(layer)).float(),
            "objective_type": "label_inversion",
        }
    ]


def load_steering_vector(path: str, layer: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if "steering_vectors" in payload:
        return payload["steering_vectors"][layer].float()
    if "candidates" in payload:
        for row in payload["candidates"]:
            if int(row["layer"]) == int(layer):
                return torch.tensor(row["vector"], dtype=torch.float32)
        raise ValueError(f"Layer {layer} not found in {path}")
    raise ValueError(f"Unsupported steering file format: {path}")


def select_tokens_for_objective(entry: dict, dots: torch.Tensor, label: str, dot_threshold: float) -> torch.Tensor:
    return select_tokens_for_label(dots, label, dot_threshold)


def selection_rule_for_objective(entry: dict, label: str, dot_threshold: float) -> str:
    return selection_rule_for_label(label, dot_threshold)


def objective_loss_sign(entry: dict, label: str) -> float:
    return inversion_loss_sign(label)


def should_preserve_non_selected(entry: dict) -> bool:
    return True


@torch.no_grad()
def prepare_example(
    bundle,
    example: dict,
    steering_entries: list[dict],
    dot_threshold: float,
    tokseq_position: str,
) -> dict:
    segments = prepare_tokseq_prompt_segments(bundle, example["prompt"], tokseq_position)
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
        tokseq_embeds=None,
        tokseq_token_ids=None,
        steering_entries=steering_entries,
    )
    entry_stats = {}
    for entry in steering_entries:
        layer = entry["layer"]
        selected_mask = select_tokens_for_objective(entry, baseline_scores[layer], example["label"], dot_threshold)
        entry_stats[str(layer)] = {
            "selection_rule": selection_rule_for_objective(entry, example["label"], dot_threshold),
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


def prepare_tokseq_prompt_segments(bundle, prompt: str, tokseq_position: str) -> dict[str, torch.Tensor | str]:
    if tokseq_position == "suffix":
        segments = split_chat_prompt_for_user_suffix(bundle, prompt)
        return {
            "tokseq_position": tokseq_position,
            "user_input_ids": segments["user_input_ids"],
            "assistant_prefix_ids": segments["assistant_prefix_ids"],
        }
    if tokseq_position != "prefix":
        raise ValueError(f"Unsupported tokseq position: {tokseq_position}")

    prompt_without_generation = encode_chat_prompt(bundle, prompt, add_generation_prompt=False)
    prompt_with_generation = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    user_turn_ids = prompt_without_generation["input_ids"][0]
    full_ids = prompt_with_generation["input_ids"][0]
    user_turn_length = user_turn_ids.shape[0]
    if full_ids.shape[0] < user_turn_length or not torch.equal(full_ids[:user_turn_length], user_turn_ids):
        raise ValueError(
            "Could not split the chat template into user turn and assistant generation prefix. "
            "This tokenizer's generation prompt is not a simple token suffix."
        )

    content_start = find_user_content_start_index(bundle, prompt, user_turn_ids)
    return {
        "tokseq_position": tokseq_position,
        "user_prefix_ids": user_turn_ids[:content_start].unsqueeze(0),
        "user_content_and_suffix_ids": user_turn_ids[content_start:].unsqueeze(0),
        "assistant_prefix_ids": full_ids[user_turn_length:].unsqueeze(0),
    }


def find_user_content_start_index(bundle, prompt: str, user_turn_ids: torch.Tensor) -> int:
    sentinel = "<LARGO_TOKSEQ_CONTENT_BOUNDARY_6b7f5c5a>"
    extended_user_turn_ids = encode_chat_prompt(bundle, sentinel + prompt, add_generation_prompt=False)["input_ids"][0]
    base_tokens = user_turn_ids.tolist()
    extended_tokens = extended_user_turn_ids.tolist()

    prefix_length = 0
    max_prefix_length = min(len(base_tokens), len(extended_tokens))
    while prefix_length < max_prefix_length and base_tokens[prefix_length] == extended_tokens[prefix_length]:
        prefix_length += 1

    return prefix_length


def build_prompt_embeds_with_tokseq(bundle, prompt_segments: dict[str, torch.Tensor | str], tokseq_embeds: torch.Tensor) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    if prompt_segments["tokseq_position"] == "prefix":
        user_prefix_embeds = embedding_layer(prompt_segments["user_prefix_ids"]).detach()
        user_content_embeds = embedding_layer(prompt_segments["user_content_and_suffix_ids"]).detach()
        assistant_prefix_embeds = embedding_layer(prompt_segments["assistant_prefix_ids"]).detach()
        return torch.cat([user_prefix_embeds, tokseq_embeds, user_content_embeds, assistant_prefix_embeds], dim=1)

    user_embeds = embedding_layer(prompt_segments["user_input_ids"]).detach()
    assistant_prefix_embeds = embedding_layer(prompt_segments["assistant_prefix_ids"]).detach()
    return torch.cat([user_embeds, tokseq_embeds, assistant_prefix_embeds], dim=1)


def build_prompt_embeds_without_tokseq(bundle, prompt_segments: dict[str, torch.Tensor | str]) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    if prompt_segments["tokseq_position"] == "prefix":
        user_prefix_embeds = embedding_layer(prompt_segments["user_prefix_ids"]).detach()
        user_content_embeds = embedding_layer(prompt_segments["user_content_and_suffix_ids"]).detach()
        assistant_prefix_embeds = embedding_layer(prompt_segments["assistant_prefix_ids"]).detach()
        return torch.cat([user_prefix_embeds, user_content_embeds, assistant_prefix_embeds], dim=1)

    user_embeds = embedding_layer(prompt_segments["user_input_ids"]).detach()
    assistant_prefix_embeds = embedding_layer(prompt_segments["assistant_prefix_ids"]).detach()
    return torch.cat([user_embeds, assistant_prefix_embeds], dim=1)


def detach_prompt_segments(prompt_segments: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor | str]:
    return {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in prompt_segments.items()
    }


def optimize_soft_tokseq(
    bundle,
    tokseq_embeds: torch.Tensor,
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
    tokseq_parameter = torch.nn.Parameter(tokseq_embeds.detach().clone().float())
    optimizer = torch.optim.Adam([tokseq_parameter], lr=learning_rate, weight_decay=weight_decay)
    trace = []

    for inner_step in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        typed_tokseq = tokseq_parameter.to(embedding_layer.weight.dtype)
        objective, breakdown = tokenwise_probe_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            tokseq_embeds=typed_tokseq,
            tokseq_token_ids=None,
            inversion_weight=inversion_weight,
            preserve_weight=preserve_weight,
            dot_threshold=dot_threshold,
        )
        objective.backward()
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_([tokseq_parameter], max_norm=grad_clip).item())
        else:
            grad_norm = float(tokseq_parameter.grad.detach().norm().item()) if tokseq_parameter.grad is not None else 0.0
        optimizer.step()
        if not torch.isfinite(tokseq_parameter).all():
            raise ValueError(f"Non-finite tokseq parameter after inner step {inner_step + 1}/{inner_steps}.")

        row = {
            "inner_step": inner_step,
            "objective": float(objective.detach().item()),
            **breakdown,
            "grad_norm": grad_norm,
            "tokseq_norm": float(tokseq_parameter.detach().norm().item()),
        }
        trace.append(row)
        print(
            f"  [inner {inner_step + 1}/{inner_steps}] objective={row['objective']:.6f} "
            f"inversion={row['inversion_loss']:.6f} preserve={row['preserve_loss']:.6f} "
            f"grad_norm={grad_norm:.6f}",
            flush=True,
        )

    return tokseq_parameter.detach(), trace


@torch.no_grad()
def sample_and_select_tokenwise_probe_summary(
    bundle,
    tokseq_embeds: torch.Tensor,
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
        raw_summary_token_ids, summary_text = summarize_tokseq_matrix(
            bundle=bundle,
            tokseq_embeds=tokseq_embeds,
            model_name=model_name,
            summary_prompt=prompt_variant["summary_prompt"],
            assistant_prefill=prompt_variant["assistant_prefill"],
            max_new_tokens=max_new_tokens,
            temperature=summary_temperature,
        )
        tokseq_token_ids = truncate_tokseq_token_ids(bundle, raw_summary_token_ids, max_new_tokens)
        objective, breakdown = tokenwise_probe_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            tokseq_embeds=None,
            tokseq_token_ids=tokseq_token_ids,
            inversion_weight=inversion_weight,
            preserve_weight=preserve_weight,
            dot_threshold=dot_threshold,
        )
        tokseq_text = bundle.tokenizer.decode(tokseq_token_ids.tolist(), skip_special_tokens=True)
        candidate = {
            "candidate_index": candidate_index,
            "summary_prompt": prompt_variant["summary_prompt"],
            "assistant_prefill": prompt_variant["assistant_prefill"],
            "objective": float(objective.item()),
            "loss_breakdown": breakdown,
            "raw_summary_token_ids": raw_summary_token_ids,
            "summary_token_ids": tokseq_token_ids.tolist(),
            "summary_text": summary_text,
            "tokseq_token_ids": tokseq_token_ids.tolist(),
            "tokseq_text": tokseq_text,
            "tokseq_token_ids_tensor": tokseq_token_ids,
        }
        candidates.append(candidate)
        print(
            f"  [summary {candidate_index + 1}/{summary_samples}] "
            f"objective={candidate['objective']:.6f} prompt={prompt_variant['name']!r} tokseq={tokseq_text!r}",
            flush=True,
        )

    selected = min(candidates, key=lambda candidate: candidate["objective"])
    print(
        f"  [summary selected] index={selected['candidate_index']} "
        f"objective={selected['objective']:.6f} tokseq={selected['tokseq_text']!r}",
        flush=True,
    )
    return candidates, selected


def tokenwise_probe_objective(
    bundle,
    examples: list[dict],
    steering_entries: list[dict],
    tokseq_embeds: torch.Tensor | None,
    tokseq_token_ids: torch.Tensor | None,
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
            tokseq_embeds=tokseq_embeds,
            tokseq_token_ids=tokseq_token_ids,
            steering_entries=steering_entries,
        )
        for entry in steering_entries:
            layer = entry["layer"]
            weight = float(entry["weight"])
            baseline_dots = example["baseline_dots"][layer].to(bundle.device)
            current_dots = current_scores[layer]
            selected_mask = select_tokens_for_objective(entry, baseline_dots, example["label"], dot_threshold)
            preserved_mask = ~selected_mask

            sign = objective_loss_sign(entry, example["label"])
            inversion_loss = sign * current_dots[selected_mask].mean() if selected_mask.any() else current_dots.sum() * 0.0
            preserve_loss = (
                F.mse_loss(current_dots[preserved_mask], baseline_dots[preserved_mask])
                if should_preserve_non_selected(entry) and preserved_mask.any()
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
    tokseq_embeds: torch.Tensor | None,
    tokseq_token_ids: torch.Tensor | None,
    steering_entries: list[dict],
) -> dict[int, torch.Tensor]:
    embedding_layer = bundle.model.get_input_embeddings()
    story_embeds = embedding_layer(story_ids).detach()
    if tokseq_embeds is not None and tokseq_token_ids is not None:
        raise ValueError("Use either tokseq_embeds or tokseq_token_ids, not both.")
    if tokseq_embeds is not None:
        tokseq_part_embeds = tokseq_embeds.to(bundle.device).to(embedding_layer.weight.dtype)
    elif tokseq_token_ids is not None:
        tokseq_part_embeds = embedding_layer(tokseq_token_ids.to(bundle.device).unsqueeze(0)).detach()
    else:
        tokseq_part_embeds = None

    if tokseq_part_embeds is not None:
        prompt_embeds = build_prompt_embeds_with_tokseq(bundle, prompt_segments, tokseq_part_embeds)
    else:
        prompt_embeds = build_prompt_embeds_without_tokseq(bundle, prompt_segments)
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
        "prompt_tokens": int(build_prompt_length(example["prompt_segments"])),
        "user_prompt_tokens": int(build_user_prompt_length(example["prompt_segments"])),
        "assistant_prefix_tokens": int(example["prompt_segments"]["assistant_prefix_ids"].shape[1]),
        "story_tokens": int(example["story_ids"].shape[1]),
        "entry_stats": example["entry_stats"],
    }


def build_prompt_length(prompt_segments: dict[str, torch.Tensor | str]) -> int:
    if prompt_segments["tokseq_position"] == "prefix":
        return int(
            prompt_segments["user_prefix_ids"].shape[1]
            + prompt_segments["user_content_and_suffix_ids"].shape[1]
            + prompt_segments["assistant_prefix_ids"].shape[1]
        )
    return int(prompt_segments["user_input_ids"].shape[1] + prompt_segments["assistant_prefix_ids"].shape[1])


def build_user_prompt_length(prompt_segments: dict[str, torch.Tensor | str]) -> int:
    if prompt_segments["tokseq_position"] == "prefix":
        return int(prompt_segments["user_prefix_ids"].shape[1] + prompt_segments["user_content_and_suffix_ids"].shape[1])
    return int(prompt_segments["user_input_ids"].shape[1])


def resolve_output_path(output: str, steering_file: str) -> Path:
    if output:
        return Path(output)
    run_dir = infer_run_dir(steering_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "tokenwise_probe" / f"ps_tokenwise_probe_largo_{timestamp}.json"


def infer_run_dir(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.parent.name in {"suffixes", "steering_generations", "tokenwise_probe", "prefixes", "tokseqs"}:
        return path.parent.parent
    return path.parent


if __name__ == "__main__":
    main()
