from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
    from adv_steering.rep_suffix_attack import _default_fill_token_id
    from adv_steering.text_backend import encode_chat_prompt
else:
    from .backend import load_bundle
    from .rep_prefix_attack_soft_prompt import (
        inversion_loss_sign,
        load_examples,
        load_weighted_steering_entries,
        select_tokens_for_label,
        selection_rule_for_label,
    )
    from .rep_suffix_attack import _default_fill_token_id
    from .text_backend import encode_chat_prompt


@dataclass
class GcgPrefixStep:
    step: int
    objective: float
    inversion_loss: float
    preserve_loss: float
    selected_tokens: int
    preserved_tokens: int
    prefix_text: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "GCG discrete-token prefix attack for token conceptness. The optimized prefix is inserted "
            "after the assistant generation prefix and before fixed story tokens."
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
    parser.add_argument("--prefix-length", type=int, default=20, help="Number of discrete prefix tokens to optimize.")
    parser.add_argument("--init-mode", default="random", choices=["random", "fill"], help="How to initialize prefix tokens.")
    parser.add_argument("--steps", type=int, default=500, help="Number of GCG iterations.")
    parser.add_argument("--top-k", type=int, default=256, help="How many replacement tokens to keep per position.")
    parser.add_argument("--batch-size", type=int, default=256, help="How many one-token edits to evaluate per iteration.")
    parser.add_argument("--dot-threshold", type=float, default=0.1, help="Positive examples select dot >= threshold; negative examples select dot <= -threshold.")
    parser.add_argument("--preserve-weight", type=float, default=1.0, help="Weight on preserving non-selected token dot products.")
    parser.add_argument("--inversion-weight", type=float, default=1.0, help="Weight on inverting selected token dot products.")
    parser.add_argument("--resume-from", default="", help="Optional previous prefix artifact JSON to resume from.")
    parser.add_argument("--output", default="", help="Optional artifact output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt or args.story:
        if not (args.prompt and args.story):
            raise ValueError("Provide both --prompt and --story for custom optimization.")

    bundle, resolved_backend = load_bundle(args.model, args.backend, args.dtype, args.device_map)
    if resolved_backend != "causal_lm":
        raise ValueError("rep_prefix_attack_gcg_token_conceptness.py currently supports only causal_lm.")

    steering_entries = load_weighted_steering_entries(args.steering_config, args.steering_file)
    raw_examples = load_examples(args)
    if not raw_examples:
        raise ValueError("No examples found for optimization.")
    examples = [
        prepare_example(bundle=bundle, example=example, steering_entries=steering_entries, dot_threshold=args.dot_threshold)
        for example in raw_examples
    ]

    forbidden_token_ids = default_forbidden_token_ids(bundle.tokenizer)
    resumed = load_resume_artifact(args.resume_from) if args.resume_from else None
    if resumed is not None:
        prefix_token_ids = torch.tensor(resumed["prefix_token_ids"], dtype=torch.long, device=bundle.device)
        if prefix_token_ids.shape[0] != args.prefix_length:
            raise ValueError(
                f"Resumed prefix length {prefix_token_ids.shape[0]} does not match --prefix-length {args.prefix_length}."
            )
        trace = list(resumed.get("trace", []))
        best_objective = float(resumed.get("objective", float("inf")))
        best_prefix_token_ids = prefix_token_ids.clone()
    else:
        prefix_token_ids = initialize_prefix_token_ids(bundle, args.prefix_length, args.init_mode, forbidden_token_ids)
        trace = []
        best_objective = float("inf")
        best_prefix_token_ids = prefix_token_ids.clone()

    baselines = {
        "no_prefix": summarize_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            prefix_token_ids=torch.empty((0,), dtype=torch.long, device=bundle.device),
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        ),
        "fill_prefix": summarize_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            prefix_token_ids=initialize_prefix_token_ids(bundle, args.prefix_length, "fill", forbidden_token_ids),
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        ),
    }
    print(
        f"[baseline] no_prefix_objective={baselines['no_prefix']['objective']:.6f} "
        f"fill_prefix_objective={baselines['fill_prefix']['objective']:.6f} "
        f"fill_prefix={baselines['fill_prefix']['prefix_text']!r}",
        flush=True,
    )

    step_offset = trace[-1]["step"] + 1 if trace else 0
    plateau_prefix_signature: tuple[int, ...] | None = None
    plateau_tried_edits: set[tuple[int, int]] = set()
    completed_steps = 0
    early_stopped = False
    early_stop_reason = None

    for step_index in range(args.steps):
        current_objective, grad = objective_and_prefix_grad(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            prefix_token_ids=prefix_token_ids,
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        )
        logged = summarize_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            prefix_token_ids=prefix_token_ids,
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        )
        if logged["objective"] < best_objective:
            best_objective = logged["objective"]
            best_prefix_token_ids = prefix_token_ids.clone()

        candidate_token_sets = top_replacement_token_sets(
            bundle=bundle,
            grad=grad,
            top_k=args.top_k,
            forbidden_token_ids=forbidden_token_ids,
        )
        current_signature = tuple(int(token_id) for token_id in prefix_token_ids.tolist())
        if current_signature != plateau_prefix_signature:
            plateau_prefix_signature = current_signature
            plateau_tried_edits.clear()

        candidate_edits = build_candidate_edits(prefix_token_ids, candidate_token_sets, plateau_tried_edits)
        if not candidate_edits:
            completed_steps = step_index
            early_stopped = True
            early_stop_reason = "exhausted_local_one_token_neighborhood"
            print(
                f"[early stop] requested_steps={args.steps} completed_steps={completed_steps} "
                f"reason={early_stop_reason}",
                flush=True,
            )
            break

        sampled_edits = sample_edits(candidate_edits, args.batch_size, bundle.device)
        candidate_batch = prefix_token_ids.unsqueeze(0).repeat(len(sampled_edits), 1)
        for batch_index, (position, token_id) in enumerate(sampled_edits):
            candidate_batch[batch_index, position] = token_id

        candidate_objectives = batched_prefix_objectives(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            candidate_prefix_token_ids=candidate_batch,
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        )
        best_batch_objective, best_batch_index = torch.min(candidate_objectives, dim=0)
        proposed_prefix = candidate_batch[int(best_batch_index.item())].clone()
        proposed_logged = summarize_objective(
            bundle=bundle,
            examples=examples,
            steering_entries=steering_entries,
            prefix_token_ids=proposed_prefix,
            inversion_weight=args.inversion_weight,
            preserve_weight=args.preserve_weight,
            dot_threshold=args.dot_threshold,
        )
        if proposed_logged["objective"] < logged["objective"]:
            prefix_token_ids = proposed_prefix
            logged = proposed_logged

        new_signature = tuple(int(token_id) for token_id in prefix_token_ids.tolist())
        if new_signature == current_signature:
            plateau_tried_edits.update(sampled_edits)
        else:
            plateau_prefix_signature = new_signature
            plateau_tried_edits.clear()

        if logged["objective"] < best_objective:
            best_objective = logged["objective"]
            best_prefix_token_ids = prefix_token_ids.clone()

        step_row = asdict(
            GcgPrefixStep(
                step=step_offset + step_index,
                objective=float(logged["objective"]),
                inversion_loss=float(logged["inversion_loss"]),
                preserve_loss=float(logged["preserve_loss"]),
                selected_tokens=int(logged["selected_tokens"]),
                preserved_tokens=int(logged["preserved_tokens"]),
                prefix_text=logged["prefix_text"],
            )
        )
        trace.append(step_row)
        print(
            f"[step {step_offset + step_index + 1}/{step_offset + args.steps}] "
            f"objective={logged['objective']:.6f} "
            f"inversion={logged['inversion_loss']:.6f} preserve={logged['preserve_loss']:.6f} "
            f"selected={logged['selected_tokens']} preserved={logged['preserved_tokens']} "
            f"prefix={logged['prefix_text']!r}",
            flush=True,
        )
        completed_steps = step_index + 1

    best_summary = summarize_objective(
        bundle=bundle,
        examples=examples,
        steering_entries=steering_entries,
        prefix_token_ids=best_prefix_token_ids,
        inversion_weight=args.inversion_weight,
        preserve_weight=args.preserve_weight,
        dot_threshold=args.dot_threshold,
    )
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
        "prefix_length": args.prefix_length,
        "init_mode": args.init_mode,
        "steps": args.steps,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "dot_threshold": args.dot_threshold,
        "inversion_weight": args.inversion_weight,
        "preserve_weight": args.preserve_weight,
        "requested_steps": args.steps,
        "completed_steps": completed_steps,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "objective": best_summary["objective"],
        "inversion_loss": best_summary["inversion_loss"],
        "preserve_loss": best_summary["preserve_loss"],
        "prefix_token_ids": best_prefix_token_ids.tolist(),
        "prefix_text": best_summary["prefix_text"],
        "baselines": baselines,
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
    print(f"Saved GCG prefix artifact to {output_path}", flush=True)


@torch.no_grad()
def prepare_example(bundle, example: dict, steering_entries: list[dict], dot_threshold: float) -> dict:
    prompt_inputs = encode_chat_prompt(bundle, example["prompt"], add_generation_prompt=True)
    prompt_ids = prompt_inputs["input_ids"].detach()
    story_ids = torch.tensor(
        bundle.tokenizer.encode(example["story"], add_special_tokens=False),
        dtype=torch.long,
        device=bundle.device,
    ).unsqueeze(0)
    if story_ids.shape[1] == 0:
        raise ValueError("Story tokenized to zero tokens.")
    baseline_scores = compute_story_probe_scores(
        bundle=bundle,
        prompt_ids=prompt_ids,
        story_ids=story_ids,
        prefix_embeds=None,
        prefix_token_ids=None,
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
        "prompt_ids": prompt_ids,
        "story_ids": story_ids,
        "baseline_dots": {layer: scores.detach() for layer, scores in baseline_scores.items()},
        "entry_stats": entry_stats,
    }


def objective_and_prefix_grad(
    bundle,
    examples: list[dict],
    steering_entries: list[dict],
    prefix_token_ids: torch.Tensor,
    inversion_weight: float,
    preserve_weight: float,
    dot_threshold: float,
) -> tuple[float, torch.Tensor]:
    embedding_layer = bundle.model.get_input_embeddings()
    prefix_embeds = embedding_layer(prefix_token_ids.unsqueeze(0)).detach().clone().requires_grad_(True)
    objective, _ = token_conceptness_objective(
        bundle=bundle,
        examples=examples,
        steering_entries=steering_entries,
        prefix_embeds=prefix_embeds,
        prefix_token_ids=None,
        inversion_weight=inversion_weight,
        preserve_weight=preserve_weight,
        dot_threshold=dot_threshold,
    )
    objective.backward()
    grad = prefix_embeds.grad[0].detach()
    return float(objective.detach().item()), grad


@torch.no_grad()
def summarize_objective(
    bundle,
    examples: list[dict],
    steering_entries: list[dict],
    prefix_token_ids: torch.Tensor,
    inversion_weight: float,
    preserve_weight: float,
    dot_threshold: float,
) -> dict:
    objective, breakdown = token_conceptness_objective(
        bundle=bundle,
        examples=examples,
        steering_entries=steering_entries,
        prefix_embeds=None,
        prefix_token_ids=prefix_token_ids,
        inversion_weight=inversion_weight,
        preserve_weight=preserve_weight,
        dot_threshold=dot_threshold,
    )
    return {
        "objective": float(objective.item()),
        **breakdown,
        "prefix_token_ids": prefix_token_ids.tolist(),
        "prefix_text": bundle.tokenizer.decode(prefix_token_ids.tolist(), skip_special_tokens=True),
    }


def token_conceptness_objective(
    bundle,
    examples: list[dict],
    steering_entries: list[dict],
    prefix_embeds: torch.Tensor | None,
    prefix_token_ids: torch.Tensor | None,
    inversion_weight: float,
    preserve_weight: float,
    dot_threshold: float,
) -> tuple[torch.Tensor, dict]:
    objective = torch.zeros((), device=bundle.device, dtype=torch.float32)
    totals = {"inversion_loss": 0.0, "preserve_loss": 0.0, "selected_tokens": 0, "preserved_tokens": 0}
    for example in examples:
        scores = compute_story_probe_scores(
            bundle=bundle,
            prompt_ids=example["prompt_ids"],
            story_ids=example["story_ids"],
            prefix_embeds=prefix_embeds,
            prefix_token_ids=prefix_token_ids,
            steering_entries=steering_entries,
        )
        for entry in steering_entries:
            layer = entry["layer"]
            weight = float(entry["weight"])
            baseline_dots = example["baseline_dots"][layer].to(bundle.device)
            current_dots = scores[layer]
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
    return objective / example_count, {
        "inversion_loss": totals["inversion_loss"] / example_count,
        "preserve_loss": totals["preserve_loss"] / example_count,
        "selected_tokens": totals["selected_tokens"],
        "preserved_tokens": totals["preserved_tokens"],
    }


def compute_story_probe_scores(
    bundle,
    prompt_ids: torch.Tensor,
    story_ids: torch.Tensor,
    prefix_embeds: torch.Tensor | None,
    prefix_token_ids: torch.Tensor | None,
    steering_entries: list[dict],
) -> dict[int, torch.Tensor]:
    embedding_layer = bundle.model.get_input_embeddings()
    prompt_embeds = embedding_layer(prompt_ids).detach()
    story_embeds = embedding_layer(story_ids).detach()
    if prefix_embeds is not None and prefix_token_ids is not None:
        raise ValueError("Use either prefix_embeds or prefix_token_ids, not both.")
    if prefix_embeds is None and prefix_token_ids is not None and prefix_token_ids.numel() > 0:
        prefix_embeds = embedding_layer(prefix_token_ids.to(bundle.device).unsqueeze(0)).detach()
    pieces = [prompt_embeds]
    if prefix_embeds is not None and prefix_embeds.shape[1] > 0:
        pieces.append(prefix_embeds.to(bundle.device).to(embedding_layer.weight.dtype))
    pieces.append(story_embeds)
    input_embeds = torch.cat(pieces, dim=1)
    story_start = input_embeds.shape[1] - story_embeds.shape[1]
    attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    story_length = story_ids.shape[1]
    scores = {}
    for entry in steering_entries:
        vector = entry["vector"].to(bundle.device, dtype=torch.float32)
        states = outputs.hidden_states[entry["layer"] + 1][0, story_start : story_start + story_length, :].float()
        scores[entry["layer"]] = torch.matmul(states, vector)
    return scores


@torch.no_grad()
def batched_prefix_objectives(
    bundle,
    examples: list[dict],
    steering_entries: list[dict],
    candidate_prefix_token_ids: torch.Tensor,
    inversion_weight: float,
    preserve_weight: float,
    dot_threshold: float,
) -> torch.Tensor:
    batch_size = candidate_prefix_token_ids.shape[0]
    total = torch.zeros(batch_size, dtype=torch.float32, device=bundle.device)
    totals_count = max(len(examples), 1)
    for example in examples:
        scores = compute_story_probe_scores_batched(
            bundle=bundle,
            prompt_ids=example["prompt_ids"],
            story_ids=example["story_ids"],
            candidate_prefix_token_ids=candidate_prefix_token_ids,
            steering_entries=steering_entries,
        )
        for entry in steering_entries:
            layer = entry["layer"]
            weight = float(entry["weight"])
            baseline_dots = example["baseline_dots"][layer].to(bundle.device)
            current_dots = scores[layer]
            selected_mask = select_tokens_for_label(baseline_dots, example["label"], dot_threshold)
            preserved_mask = ~selected_mask
            sign = inversion_loss_sign(example["label"])
            if selected_mask.any():
                inversion_loss = sign * current_dots[:, selected_mask].mean(dim=1)
            else:
                inversion_loss = current_dots.sum(dim=1) * 0.0
            if preserved_mask.any():
                preserve_loss = ((current_dots[:, preserved_mask] - baseline_dots[preserved_mask].unsqueeze(0)) ** 2).mean(dim=1)
            else:
                preserve_loss = current_dots.sum(dim=1) * 0.0
            total = total + weight * (inversion_weight * inversion_loss + preserve_weight * preserve_loss)
    return total / totals_count


def compute_story_probe_scores_batched(
    bundle,
    prompt_ids: torch.Tensor,
    story_ids: torch.Tensor,
    candidate_prefix_token_ids: torch.Tensor,
    steering_entries: list[dict],
) -> dict[int, torch.Tensor]:
    embedding_layer = bundle.model.get_input_embeddings()
    batch_size = candidate_prefix_token_ids.shape[0]
    prompt_embeds = embedding_layer(prompt_ids).detach().expand(batch_size, -1, -1)
    prefix_embeds = embedding_layer(candidate_prefix_token_ids.to(bundle.device)).detach()
    story_embeds = embedding_layer(story_ids).detach().expand(batch_size, -1, -1)
    input_embeds = torch.cat([prompt_embeds, prefix_embeds, story_embeds], dim=1)
    story_start = input_embeds.shape[1] - story_embeds.shape[1]
    attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    story_length = story_ids.shape[1]
    scores = {}
    for entry in steering_entries:
        vector = entry["vector"].to(bundle.device, dtype=torch.float32)
        states = outputs.hidden_states[entry["layer"] + 1][:, story_start : story_start + story_length, :].float()
        scores[entry["layer"]] = torch.matmul(states, vector)
    return scores


def top_replacement_token_sets(bundle, grad: torch.Tensor, top_k: int, forbidden_token_ids: set[int]) -> list[list[int]]:
    vocab_matrix = bundle.model.get_input_embeddings().weight.detach()
    candidate_sets = []
    for position in range(grad.shape[0]):
        typed_vocab = vocab_matrix.to(grad[position].dtype)
        scores = torch.matmul(typed_vocab, -grad[position])
        candidate_ids = torch.topk(scores, k=min(top_k, scores.shape[0])).indices.tolist()
        filtered = [int(token_id) for token_id in candidate_ids if int(token_id) not in forbidden_token_ids]
        candidate_sets.append(filtered)
    return candidate_sets


def build_candidate_edits(
    prefix_token_ids: torch.Tensor,
    candidate_token_sets: list[list[int]],
    tried_edits: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    edits = []
    for position, token_choices in enumerate(candidate_token_sets):
        current_token_id = int(prefix_token_ids[position].item())
        seen = set()
        for token_id in token_choices:
            if token_id == current_token_id or token_id in seen:
                continue
            seen.add(token_id)
            edit = (position, int(token_id))
            if edit not in tried_edits:
                edits.append(edit)
    return edits


def sample_edits(candidate_edits: list[tuple[int, int]], batch_size: int, device: torch.device) -> list[tuple[int, int]]:
    if len(candidate_edits) <= batch_size:
        return candidate_edits
    indices = torch.randperm(len(candidate_edits), device=device)[:batch_size].tolist()
    return [candidate_edits[index] for index in indices]


def initialize_prefix_token_ids(bundle, prefix_length: int, init_mode: str, forbidden_token_ids: set[int]) -> torch.Tensor:
    if init_mode == "fill":
        fill_token_id = _default_fill_token_id(bundle.tokenizer)
        if fill_token_id in forbidden_token_ids:
            fill_token_id = 0
        return torch.full((prefix_length,), fill_token_id, dtype=torch.long, device=bundle.device)
    vocab_size = len(bundle.tokenizer)
    token_ids = torch.randint(low=0, high=vocab_size, size=(prefix_length,), dtype=torch.long, device=bundle.device)
    if forbidden_token_ids:
        forbidden = torch.tensor(sorted(forbidden_token_ids), dtype=torch.long, device=bundle.device)
        while torch.isin(token_ids, forbidden).any():
            mask = torch.isin(token_ids, forbidden)
            token_ids[mask] = torch.randint(low=0, high=vocab_size, size=(int(mask.sum().item()),), dtype=torch.long, device=bundle.device)
    return token_ids


def default_forbidden_token_ids(tokenizer) -> set[int]:
    forbidden = {tokenizer.pad_token_id}
    if tokenizer.bos_token_id is not None:
        forbidden.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        forbidden.add(tokenizer.eos_token_id)
    return {int(token_id) for token_id in forbidden if token_id is not None}


def load_resume_artifact(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "prefix_token_ids" not in payload:
        raise ValueError(f"Resume artifact has no prefix_token_ids: {path}")
    return payload


def summarize_prepared_example(example: dict) -> dict:
    return {
        "concept": example["concept"],
        "label": example["label"],
        "prompt": example["prompt"],
        "story": example["story"],
        "prompt_tokens": int(example["prompt_ids"].shape[1]),
        "story_tokens": int(example["story_ids"].shape[1]),
        "entry_stats": example["entry_stats"],
    }


def resolve_output_path(output: str, steering_file: str) -> Path:
    if output:
        return Path(output)
    run_dir = infer_run_dir(steering_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "prefixes" / f"rep_prefix_gcg_token_conceptness_{timestamp}.json"


def infer_run_dir(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.parent.name in {"suffixes", "steering_generations", "token_conceptness", "prefixes"}:
        return path.parent.parent
    return path.parent


if __name__ == "__main__":
    main()
