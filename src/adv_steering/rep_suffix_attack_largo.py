from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.rep_suffix_attack import (
        _default_fill_token_id,
        _single_prompt_objective_and_terms,
        load_prompt_pairs,
        load_steering_vector,
    )
    from adv_steering.text_backend import encode_chat_prompt, load_text_model_bundle, split_chat_prompt_for_user_suffix, steering_hook
else:
    from .rep_suffix_attack import (
        _default_fill_token_id,
        _single_prompt_objective_and_terms,
        load_prompt_pairs,
        load_steering_vector,
    )
    from .text_backend import encode_chat_prompt, load_text_model_bundle, split_chat_prompt_for_user_suffix, steering_hook


def parse_args():
    parser = argparse.ArgumentParser(description="Largo-style continuous tokseq optimization with summarize-and-reinterpret projection.")
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
    parser.add_argument("--all-tokens-steer", action="store_true", help="For steered_ce, steer all prompt+target token positions instead of only the final prompt token.")
    parser.add_argument("--last-prompt-token-steering", action="store_true", help="For steered_ce, steer only the final prompt token. This is the default behavior.")
    parser.add_argument("--tokseq-position", default="suffix", choices=["prefix", "suffix"], help="Insert the tokseq after the assistant generation prefix (prefix) or inside the user turn before the assistant generation prefix (suffix).")
    parser.add_argument("--tokseq-length", "--suffix-length", dest="tokseq_length", type=int, default=200)
    parser.add_argument("--outer-steps", type=int, default=15, help="How many summarize-and-reinterpret rounds to run.")
    parser.add_argument("--inner-steps", type=int, default=20, help="How many gradient steps to take on the tokseq matrix per outer round.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random_tokens"])
    parser.add_argument("--summary-prompt", default="Summarize the following: ")
    parser.add_argument("--summary-samples", type=int, default=16, help="How many sampled verbalizations to evaluate per outer round.")
    parser.add_argument("--summary-temperature", type=float, default=1.0, help="Sampling temperature used when verbalizing the soft tokseq.")
    parser.add_argument("--success-threshold", type=float, default=0.0, help="Only used for dot/cosine style objective success checks on the training set.")
    parser.add_argument("--eval-prompts-file", default="", help="Optional held-out prompts, one per line.")
    parser.add_argument("--eval-max-prompts", type=int, default=0, help="Optional cap on held-out prompts.")
    parser.add_argument("--eval-max-new-tokens", type=int, default=96)
    parser.add_argument("--eval-steering-scale", type=float, default=0.0, help="Steering magnitude for held-out evaluation. If 0.0, falls back to --steering-scale.")
    parser.add_argument("--hapsad-wordbank-file", default="data/hapsad_wordbank.json", help="JSON wordbank with 'happy' and 'sad' lists used in the dual held-out evaluation.")
    parser.add_argument("--success-line-prefix", default="", help="Retained for compatibility; unused by the current dual happy/sad held-out evaluation.")
    parser.add_argument("--success-wordbank-file", default="", help="Retained for compatibility; unused by the current dual happy/sad held-out evaluation.")
    parser.add_argument("--success-proportion", type=float, default=0.8)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_tokens_steer and args.last_prompt_token_steering:
        raise ValueError("Use at most one of --all-tokens-steer or --last-prompt-token-steering.")
    effective_last_prompt_token_steering = not args.all_tokens_steer
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
            prepare_prompt_segments(bundle, n_plus, args.tokseq_position),
            prepare_prompt_segments(bundle, n_minus, args.tokseq_position),
        )
        for n_plus, n_minus in prompt_pairs
    ]
    _, target_ids = build_steered_ce_inputs(
        bundle=bundle,
        objective_type=args.objective_type,
        neutral_prompt=args.neutral_prompt,
        positive_target=args.positive_target,
        negative_target=args.negative_target,
    )
    neutral_prompt_segments = (
        prepare_prompt_segments(bundle, args.neutral_prompt, args.tokseq_position)
        if args.objective_type == "steered_ce"
        else None
    )
    eval_prompts = load_eval_prompts(args.eval_prompts_file, args.eval_max_prompts)
    hapsad_wordbank = load_hapsad_wordbank(args.hapsad_wordbank_file)

    tokseq_embeds = initialize_tokseq_embeds(
        bundle=bundle,
        tokseq_length=args.tokseq_length,
        init_mode=args.init_mode,
    )

    initial_tokseq_token_ids = nearest_token_ids_for_tokseq_embeds(bundle, tokseq_embeds)
    initial_tokseq_text = bundle.tokenizer.decode(initial_tokseq_token_ids.tolist(), skip_special_tokens=True)
    if args.init_mode == "random_tokens":
        print(
            json.dumps(
                {
                    "initial_tokseq_token_ids": initial_tokseq_token_ids.tolist(),
                    "initial_tokseq_text": initial_tokseq_text,
                },
                indent=2,
            ),
            flush=True,
        )
    optimizer_trace = []
    best = {
        "objective": float("inf"),
        "tokseq_token_ids": None,
        "tokseq_text": "",
        "summary_text": "",
    }

    summary_prompts = build_summary_prompt_variants(args.summary_prompt)
    for outer_step in range(args.outer_steps):
        tokseq_embeds, inner_step_losses = optimize_tokseq_matrix(
            bundle=bundle,
            tokseq_embeds=tokseq_embeds,
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
            last_prompt_token_steering=effective_last_prompt_token_steering,
        )

        summary_candidates, selected_summary = sample_and_select_summary_candidate(
            bundle=bundle,
            tokseq_embeds=tokseq_embeds,
            model_name=args.model,
            summary_prompts=summary_prompts,
            summary_samples=args.summary_samples,
            summary_temperature=args.summary_temperature,
            max_new_tokens=args.tokseq_length,
            prompt_pair_segments=prompt_pair_segments,
            steering_vector=steering_vector,
            layer=args.layer,
            objective_type=args.objective_type,
            target_ids=target_ids,
            neutral_prompt_segments=neutral_prompt_segments,
            steering_scale=args.steering_scale,
            last_prompt_token_steering=effective_last_prompt_token_steering,
            tokseq_position=args.tokseq_position,
        )
        tokseq_token_ids = selected_summary["tokseq_token_ids_tensor"]
        tokseq_embeds = tokseq_embeds_from_summary_token_ids(
            bundle=bundle,
            summary_token_ids=tokseq_token_ids,
            tokseq_length=args.tokseq_length,
        )

        objective = selected_summary["objective"]
        breakdown = selected_summary["objective_breakdown"]
        dot_product = selected_summary["dot_product"]
        cosine_similarity = selected_summary["cosine_similarity"]
        eval_result = evaluate_held_out_behavior(
            bundle=bundle,
            eval_prompts=eval_prompts,
            tokseq_token_ids=tokseq_token_ids,
            tokseq_position=args.tokseq_position,
            layer=args.layer,
            steering_vector=steering_vector,
            steering_scale=args.eval_steering_scale if args.eval_steering_scale != 0.0 else args.steering_scale,
            max_new_tokens=args.eval_max_new_tokens,
            hapsad_wordbank=hapsad_wordbank,
            last_prompt_token_steering=effective_last_prompt_token_steering,
        )

        tokseq_text = bundle.tokenizer.decode(tokseq_token_ids.tolist(), skip_special_tokens=True)
        summary_text = selected_summary["summary_text"]
        optimizer_trace.append(
            {
                "outer_step": outer_step,
                "objective_type": args.objective_type,
                "objective": objective,
                "objective_breakdown": breakdown,
                "dot_product": dot_product,
                "cosine_similarity": cosine_similarity,
                "inner_step_losses": inner_step_losses,
                "summary_candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "tokseq_token_ids_tensor"
                    }
                    for candidate in summary_candidates
                ],
                "selected_summary_candidate": selected_summary["candidate_index"],
                "selected_summary_prompt": selected_summary["summary_prompt"],
                "selected_assistant_prefill": selected_summary["assistant_prefill"],
                "summary_token_ids": tokseq_token_ids.tolist(),
                "summary_text": summary_text,
                "tokseq_token_ids": tokseq_token_ids.tolist(),
                "tokseq_text": tokseq_text,
                "suffix_token_ids": tokseq_token_ids.tolist(),
                "suffix_text": tokseq_text,
                "held_out": eval_result,
            }
        )

        print(
            f"[outer {outer_step + 1}/{args.outer_steps}] "
            f"objective={objective:.6f} "
            f"dot_product={dot_product:.6f} "
            f"cosine_similarity={cosine_similarity:.6f} "
            f"held_out_success_rate={eval_result['success_rate']:.3f} "
            f"tokseq={tokseq_text!r}",
            flush=True,
        )

        if objective < best["objective"]:
            best = {
                "objective": objective,
                "tokseq_token_ids": tokseq_token_ids.tolist(),
                "tokseq_text": tokseq_text,
                "summary_text": summary_text,
            }

        if eval_result["checked"] and eval_result["success_rate"] >= args.success_proportion:
            break

    payload = {
        "model": args.model,
        "objective_type": args.objective_type,
        "layer": args.layer,
        "steering_scale": args.steering_scale,
        "all_tokens_steer": args.all_tokens_steer,
        "last_prompt_token_steering": effective_last_prompt_token_steering,
        "tokseq_position": args.tokseq_position,
        "neutral_prompt": args.neutral_prompt,
        "positive_target": args.positive_target,
        "negative_target": args.negative_target,
        "summary_prompt": args.summary_prompt,
        "summary_samples": args.summary_samples,
        "summary_temperature": args.summary_temperature,
        "summary_prompt_variants": summary_prompts,
        "tokseq_length": args.tokseq_length,
        "suffix_length": args.tokseq_length,
        "init_mode": args.init_mode,
        "initial_tokseq_token_ids": initial_tokseq_token_ids.tolist(),
        "initial_tokseq_text": initial_tokseq_text,
        "initial_suffix_token_ids": initial_tokseq_token_ids.tolist(),
        "initial_suffix_text": initial_tokseq_text,
        "outer_steps": args.outer_steps,
        "inner_steps": args.inner_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best": {
            **best,
            "suffix_token_ids": best["tokseq_token_ids"],
            "suffix_text": best["tokseq_text"],
        },
        "trace": optimizer_trace,
        "prompt_pairs": [{"n_plus": n_plus, "n_minus": n_minus} for n_plus, n_minus in prompt_pairs],
        "eval_prompts_file": args.eval_prompts_file,
        "hapsad_wordbank_file": args.hapsad_wordbank_file,
        "success_proportion": args.success_proportion,
    }
    print(json.dumps(payload, indent=2))

    output_path = resolve_output_path(args.output, args.steering_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Largo tokseq artifact to {output_path}", flush=True)


def build_steered_ce_inputs(bundle, objective_type: str, neutral_prompt: str, positive_target: str, negative_target: str):
    if objective_type != "steered_ce":
        return None, None
    if not neutral_prompt or not positive_target or not negative_target:
        raise ValueError("steered_ce requires neutral_prompt, positive_target, and negative_target.")
    tokenizer = bundle.tokenizer
    neutral_prompt_ids = encode_chat_prompt(bundle, neutral_prompt, add_generation_prompt=True)["input_ids"][0]
    target_ids = {
        "positive": torch.tensor(tokenizer.encode(positive_target, add_special_tokens=False), dtype=torch.long, device=bundle.device),
        "negative": torch.tensor(tokenizer.encode(negative_target, add_special_tokens=False), dtype=torch.long, device=bundle.device),
    }
    return neutral_prompt_ids, target_ids


def prepare_prompt_segments(bundle, prompt: str, tokseq_position: str) -> dict[str, torch.Tensor | str]:
    if tokseq_position == "suffix":
        segments = split_chat_prompt_for_user_suffix(bundle, prompt)
        return {
            "tokseq_position": tokseq_position,
            "user_input_ids": segments["user_input_ids"],
            "assistant_prefix_ids": segments["assistant_prefix_ids"],
        }
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    return {
        "tokseq_position": tokseq_position,
        "prompt_with_generation_ids": prompt_inputs["input_ids"],
    }


def initialize_tokseq_embeds(bundle, tokseq_length: int, init_mode: str) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    hidden_size = embedding_layer.weight.shape[1]
    if init_mode == "zeros":
        return torch.zeros((1, tokseq_length, hidden_size), device=bundle.device, dtype=torch.float32)
    vocab_size = embedding_layer.weight.shape[0]
    token_ids = torch.randint(low=0, high=vocab_size, size=(tokseq_length,), device=bundle.device)
    return embedding_layer(token_ids.unsqueeze(0)).detach().float()


@torch.no_grad()
def nearest_token_ids_for_tokseq_embeds(bundle, tokseq_embeds: torch.Tensor) -> torch.Tensor:
    embedding_weight = bundle.model.get_input_embeddings().weight.detach()
    tokseq_rows = tokseq_embeds[0]
    distances = torch.cdist(tokseq_rows.float(), embedding_weight.float())
    return torch.argmin(distances, dim=1)


def optimize_tokseq_matrix(
    bundle,
    tokseq_embeds: torch.Tensor,
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
    last_prompt_token_steering: bool,
) -> tuple[torch.Tensor, list[float]]:
    embedding_layer = bundle.model.get_input_embeddings()
    tokseq_parameter = torch.nn.Parameter(tokseq_embeds.detach().clone().float())
    optimizer = torch.optim.Adam([tokseq_parameter], lr=learning_rate, weight_decay=weight_decay)
    objective_prompt_pair_segments = prompt_pair_segments if objective_type != "steered_ce" else prompt_pair_segments[:1]
    inner_step_losses: list[float] = []

    for inner_step in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), device=bundle.device, dtype=torch.float32)
        typed_tokseq = tokseq_parameter.to(embedding_layer.weight.dtype)
        neutral_prompt_embeds = None if neutral_prompt_segments is None else build_prompt_embeds_with_tokseq(bundle, neutral_prompt_segments, typed_tokseq)
        for plus_segments, minus_segments in objective_prompt_pair_segments:
            plus_prompt_embeds = build_prompt_embeds_with_tokseq(bundle, plus_segments, typed_tokseq)
            minus_prompt_embeds = build_prompt_embeds_with_tokseq(bundle, minus_segments, typed_tokseq)
            prompt_objective, _ = _single_prompt_objective_and_terms(
                bundle=bundle,
                plus_prompt_embeds=plus_prompt_embeds,
                minus_prompt_embeds=minus_prompt_embeds,
                steering_vector=steering_vector,
                layer=layer,
                objective_type=objective_type,
                target_ids=target_ids,
                neutral_prompt_embeds=neutral_prompt_embeds,
                steering_scale=steering_scale,
                last_prompt_token_steering=last_prompt_token_steering,
            )
            objective = objective + prompt_objective.float()
        objective.backward()
        torch.nn.utils.clip_grad_norm_([tokseq_parameter], max_norm=1.0)
        optimizer.step()
        if not torch.isfinite(tokseq_parameter).all():
            raise ValueError(f"Non-finite tokseq parameter after inner step {inner_step + 1}/{inner_steps}.")
        objective_value = float(objective.detach().item())
        inner_step_losses.append(objective_value)
        print(
            f"  [inner {inner_step + 1}/{inner_steps}] objective={objective_value:.6f}",
            flush=True,
        )

    return tokseq_parameter.detach(), inner_step_losses


@torch.no_grad()
def sample_and_select_summary_candidate(
    bundle,
    tokseq_embeds: torch.Tensor,
    model_name: str,
    summary_prompts: list[dict[str, str]],
    summary_samples: int,
    summary_temperature: float,
    max_new_tokens: int,
    prompt_pair_segments,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids,
    neutral_prompt_segments,
    steering_scale: float,
    last_prompt_token_steering: bool,
    tokseq_position: str,
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
        objective, breakdown, dot_product, cosine_similarity = evaluate_tokseq_from_token_ids(
            bundle=bundle,
            prompt_pair_segments=prompt_pair_segments,
            tokseq_token_ids=tokseq_token_ids,
            tokseq_position=tokseq_position,
            steering_vector=steering_vector,
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_segments=neutral_prompt_segments,
            steering_scale=steering_scale,
            last_prompt_token_steering=last_prompt_token_steering,
        )
        tokseq_text = bundle.tokenizer.decode(tokseq_token_ids.tolist(), skip_special_tokens=True)
        candidate = {
            "candidate_index": candidate_index,
            "summary_prompt": prompt_variant["summary_prompt"],
            "assistant_prefill": prompt_variant["assistant_prefill"],
            "objective": objective,
            "objective_breakdown": breakdown,
            "dot_product": dot_product,
            "cosine_similarity": cosine_similarity,
            "raw_summary_token_ids": raw_summary_token_ids,
            "summary_token_ids": tokseq_token_ids.tolist(),
            "summary_text": summary_text,
            "tokseq_token_ids": tokseq_token_ids.tolist(),
            "tokseq_text": tokseq_text,
            "suffix_token_ids": tokseq_token_ids.tolist(),
            "suffix_text": tokseq_text,
            "tokseq_token_ids_tensor": tokseq_token_ids,
        }
        candidates.append(candidate)
        print(
            f"  [summary {candidate_index + 1}/{summary_samples}] "
            f"objective={objective:.6f} prompt={prompt_variant['name']!r} tokseq={tokseq_text!r}",
            flush=True,
        )

    selected = min(candidates, key=lambda candidate: candidate["objective"])
    print(
        f"  [summary selected] index={selected['candidate_index']} "
        f"objective={selected['objective']:.6f} tokseq={selected['tokseq_text']!r}",
        flush=True,
    )
    return candidates, selected


def build_summary_prompt_variants(summary_prompt: str) -> list[dict[str, str]]:
    return [
        {
            "name": "user_original",
            "summary_prompt": summary_prompt,
            "assistant_prefill": "Sure, I will summarize the message:",
        },
        {
            "name": "plain_meaning",
            "summary_prompt": "Read the following unusual message and write its likely plain-English meaning. Be concise: ",
            "assistant_prefill": "The likely meaning is:",
        },
        {
            "name": "intended_request",
            "summary_prompt": "Infer what the following message is trying to ask for. Write only the inferred request: ",
            "assistant_prefill": "The inferred request is:",
        },
        {
            "name": "hidden_message",
            "summary_prompt": "The following content may be compressed, noisy, or indirect. Restate the hidden message clearly: ",
            "assistant_prefill": "The hidden message says:",
        },
        {
            "name": "recover_message",
            "summary_prompt": "Recover the most likely ordinary text from the following message. Output only the recovered text: ",
            "assistant_prefill": "Recovered text:",
        },
        {
            "name": "instruction_guess",
            "summary_prompt": "If the following were an instruction, what instruction would it be? Keep the answer short: ",
            "assistant_prefill": "Instruction:",
        },
        {
            "name": "reader_takeaway",
            "summary_prompt": "What should a reader take away from the following text? Answer as a compact phrase or sentence: ",
            "assistant_prefill": "Takeaway:",
        },
        {
            "name": "paraphrase_signal",
            "summary_prompt": "Paraphrase the signal or intent of the following content in normal language: ",
            "assistant_prefill": "Paraphrase:",
        },
        {
            "name": "likely_continuation",
            "summary_prompt": "Write a short sentence that best continues or explains the following content: ",
            "assistant_prefill": "A fitting continuation is:",
        },
    ]


@torch.no_grad()
def summarize_tokseq_matrix(
    bundle,
    tokseq_embeds: torch.Tensor,
    model_name: str,
    summary_prompt: str,
    assistant_prefill: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[int], str]:
    embedding_layer = bundle.model.get_input_embeddings()
    if not torch.isfinite(tokseq_embeds).all():
        raise ValueError("Cannot summarize tokseq matrix because it contains non-finite values.")
    user_prefix_embeds, user_suffix_and_assistant_embeds = build_interpretation_wrapper_embeds(
        bundle=bundle,
        model_name=model_name,
        summary_prompt=summary_prompt,
        assistant_prefill=assistant_prefill,
    )
    full_embeds = torch.cat(
        [
            user_prefix_embeds,
            tokseq_embeds.to(user_prefix_embeds.dtype),
            user_suffix_and_assistant_embeds,
        ],
        dim=1,
    )
    attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    generated_token_ids: list[int] = []
    current_embeds = full_embeds
    current_attention = attention_mask

    for _ in range(max_new_tokens):
        outputs = bundle.model(
            inputs_embeds=current_embeds,
            attention_mask=current_attention,
            use_cache=False,
        )
        next_token_logits = outputs.logits[0, -1] / max(float(temperature), 1e-6)
        next_token_id = int(torch.multinomial(torch.softmax(next_token_logits.float(), dim=-1), num_samples=1).item())
        if next_token_id == bundle.tokenizer.eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        next_embed = embedding_layer(torch.tensor([[next_token_id]], device=bundle.device))
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        current_attention = torch.cat(
            [current_attention, torch.ones((1, 1), dtype=current_attention.dtype, device=bundle.device)],
            dim=1,
        )
    return generated_token_ids, bundle.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def build_interpretation_wrapper_embeds(
    bundle,
    model_name: str,
    summary_prompt: str,
    assistant_prefill: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding_layer = bundle.model.get_input_embeddings()
    model_name_lower = model_name.lower()
    if "llama-2" in model_name_lower:
        prefix_text = f"[INST] {summary_prompt}"
        suffix_text = f" [/INST] {assistant_prefill}"
        prefix_ids = bundle.tokenizer.encode(prefix_text, add_special_tokens=True, return_tensors="pt").to(bundle.device)
        suffix_ids = bundle.tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(bundle.device)
        return embedding_layer(prefix_ids), embedding_layer(suffix_ids)

    prompt_without_generation = encode_chat_prompt(bundle, summary_prompt, add_generation_prompt=False)
    prompt_with_generation = encode_chat_prompt(bundle, summary_prompt, add_generation_prompt=True)
    user_turn_ids = prompt_without_generation["input_ids"][0]
    full_ids = prompt_with_generation["input_ids"][0]
    user_turn_length = user_turn_ids.shape[0]
    if full_ids.shape[0] < user_turn_length or not torch.equal(full_ids[:user_turn_length], user_turn_ids):
        raise ValueError(
            "Could not split the summary chat template into user turn and assistant generation prefix. "
            "This tokenizer's generation prompt is not a simple token suffix."
        )

    assistant_prefill_ids = bundle.tokenizer.encode(assistant_prefill, add_special_tokens=False, return_tensors="pt").to(bundle.device)
    insertion_index = find_user_content_append_index(bundle, summary_prompt, user_turn_ids)
    user_prefix_ids = user_turn_ids[:insertion_index].unsqueeze(0)
    user_suffix_and_assistant_ids = torch.cat(
        [
            user_turn_ids[insertion_index:],
            full_ids[user_turn_length:],
            assistant_prefill_ids[0],
        ],
        dim=0,
    ).unsqueeze(0)
    return embedding_layer(user_prefix_ids), embedding_layer(user_suffix_and_assistant_ids)


def find_user_content_append_index(bundle, prompt: str, user_turn_ids: torch.Tensor) -> int:
    sentinel = "<LARGO_TOKSEQ_CONTENT_BOUNDARY_6b7f5c5a>"
    extended_user_turn_ids = encode_chat_prompt(bundle, prompt + sentinel, add_generation_prompt=False)["input_ids"][0]
    base_tokens = user_turn_ids.tolist()
    extended_tokens = extended_user_turn_ids.tolist()

    prefix_length = 0
    max_prefix_length = min(len(base_tokens), len(extended_tokens))
    while prefix_length < max_prefix_length and base_tokens[prefix_length] == extended_tokens[prefix_length]:
        prefix_length += 1

    base_end = len(base_tokens)
    extended_end = len(extended_tokens)
    while (
        base_end > prefix_length
        and extended_end > prefix_length
        and base_tokens[base_end - 1] == extended_tokens[extended_end - 1]
    ):
        base_end -= 1
        extended_end -= 1

    return base_end


def truncate_tokseq_token_ids(bundle, summary_token_ids: list[int], tokseq_length: int) -> torch.Tensor:
    token_ids = summary_token_ids[:tokseq_length]
    return torch.tensor(token_ids, dtype=torch.long, device=bundle.device)


def tokseq_embeds_from_summary_token_ids(bundle, summary_token_ids: torch.Tensor, tokseq_length: int) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    if summary_token_ids.numel() == 0:
        hidden_size = embedding_layer.weight.shape[1]
        return torch.zeros((1, tokseq_length, hidden_size), device=bundle.device, dtype=torch.float32)

    summary_embeds = embedding_layer(summary_token_ids.unsqueeze(0)).detach().float()
    if summary_embeds.shape[1] < tokseq_length:
        pad_size = tokseq_length - summary_embeds.shape[1]
        padding = torch.zeros((1, pad_size, summary_embeds.shape[2]), device=bundle.device, dtype=summary_embeds.dtype)
        summary_embeds = torch.cat([summary_embeds, padding], dim=1)
    else:
        summary_embeds = summary_embeds[:, :tokseq_length, :]
    return summary_embeds


@torch.no_grad()
def evaluate_tokseq_from_token_ids(
    bundle,
    prompt_pair_segments,
    tokseq_token_ids: torch.Tensor,
    tokseq_position: str,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids,
    neutral_prompt_segments,
    steering_scale: float,
    last_prompt_token_steering: bool,
):
    embedding_layer = bundle.model.get_input_embeddings()
    tokseq_embeds = embedding_layer(tokseq_token_ids.unsqueeze(0))
    neutral_prompt_embeds = None if neutral_prompt_segments is None else build_prompt_embeds_with_tokseq(bundle, neutral_prompt_segments, tokseq_embeds)

    objective_prompt_pair_segments = prompt_pair_segments if objective_type != "steered_ce" else prompt_pair_segments[:1]

    objective = 0.0
    breakdown = None
    for plus_segments, minus_segments in objective_prompt_pair_segments:
        plus_prompt_embeds = build_prompt_embeds_with_tokseq(bundle, plus_segments, tokseq_embeds)
        minus_prompt_embeds = build_prompt_embeds_with_tokseq(bundle, minus_segments, tokseq_embeds)
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
            last_prompt_token_steering=last_prompt_token_steering,
        )
        objective += float(prompt_objective.item())
        if breakdown is None:
            breakdown = {name: float(value.item()) for name, value in terms.items()}
        else:
            for name, value in terms.items():
                breakdown[name] += float(value.item())

    aggregate_difference = None
    for plus_segments, minus_segments in prompt_pair_segments:
        plus_prompt_embeds = build_prompt_embeds_with_tokseq(bundle, plus_segments, tokseq_embeds)
        minus_prompt_embeds = build_prompt_embeds_with_tokseq(bundle, minus_segments, tokseq_embeds)
        plus_state = last_token_hidden_from_embeds(bundle, plus_prompt_embeds, layer)
        minus_state = last_token_hidden_from_embeds(bundle, minus_prompt_embeds, layer)
        difference = plus_state - minus_state
        aggregate_difference = difference if aggregate_difference is None else aggregate_difference + difference

    typed_steering_vector = steering_vector.to(aggregate_difference.dtype)
    dot_product = float(torch.dot(typed_steering_vector, aggregate_difference).item())
    cosine_similarity = float(
        (
            torch.dot(typed_steering_vector, aggregate_difference)
            / (typed_steering_vector.norm().clamp_min(1e-8) * aggregate_difference.norm().clamp_min(1e-8))
        ).item()
    )
    return objective, breakdown or {}, dot_product, cosine_similarity


def build_prompt_embeds_with_tokseq(bundle, prompt_segments: dict[str, torch.Tensor | str], tokseq_embeds: torch.Tensor) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    if prompt_segments["tokseq_position"] == "prefix":
        prompt_embeds = embedding_layer(prompt_segments["prompt_with_generation_ids"]).detach()
        return torch.cat([prompt_embeds, tokseq_embeds], dim=1)

    user_embeds = embedding_layer(prompt_segments["user_input_ids"]).detach()
    assistant_prefix_embeds = embedding_layer(prompt_segments["assistant_prefix_ids"]).detach()
    return torch.cat([user_embeds, tokseq_embeds, assistant_prefix_embeds], dim=1)


def build_input_ids_with_tokseq_ids(bundle, prompt: str, tokseq_token_ids: torch.Tensor, tokseq_position: str) -> torch.Tensor:
    if tokseq_position == "prefix":
        prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
        return torch.cat([prompt_inputs["input_ids"][0], tokseq_token_ids.to(bundle.device)], dim=0).unsqueeze(0)

    segments = split_chat_prompt_for_user_suffix(bundle, prompt)
    return torch.cat(
        [
            segments["user_input_ids"][0],
            tokseq_token_ids.to(bundle.device),
            segments["assistant_prefix_ids"][0],
        ],
        dim=0,
    ).unsqueeze(0)


@torch.no_grad()
def last_token_hidden_from_embeds(bundle, prompt_embeds: torch.Tensor, layer: int) -> torch.Tensor:
    attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[layer + 1][0, prompt_embeds.shape[1] - 1]


def load_eval_prompts(path: str, max_prompts: int) -> list[str]:
    if not path:
        return []
    prompts = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            prompts.append(line)
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def load_hapsad_wordbank(path: str) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    happy_words = [word.strip().lower() for word in payload.get("happy", []) if word.strip()]
    sad_words = [word.strip().lower() for word in payload.get("sad", []) if word.strip()]
    if not happy_words or not sad_words:
        raise ValueError(f"{path} must contain non-empty 'happy' and 'sad' lists.")
    return {"happy": happy_words, "sad": sad_words}


@torch.no_grad()
def evaluate_held_out_behavior(
    bundle,
    eval_prompts: list[str],
    tokseq_token_ids: torch.Tensor,
    tokseq_position: str,
    layer: int,
    steering_vector: torch.Tensor,
    steering_scale: float,
    max_new_tokens: int,
    hapsad_wordbank: dict[str, list[str]],
    last_prompt_token_steering: bool,
) -> dict:
    if not eval_prompts:
        return {"checked": 0, "successes": 0, "success_rate": 0.0}

    generations = []
    successes = 0
    for prompt in eval_prompts:
        sad_text = generate_with_exact_tokseq_ids(
            bundle=bundle,
            prompt=prompt,
            tokseq_token_ids=tokseq_token_ids,
            tokseq_position=tokseq_position,
            layer=layer,
            steering_vector=steering_vector,
            steering_scale=steering_scale,
            max_new_tokens=max_new_tokens,
            last_prompt_token_steering=last_prompt_token_steering,
        )
        happy_text = generate_with_exact_tokseq_ids(
            bundle=bundle,
            prompt=prompt,
            tokseq_token_ids=tokseq_token_ids,
            tokseq_position=tokseq_position,
            layer=layer,
            steering_vector=steering_vector,
            steering_scale=-steering_scale,
            max_new_tokens=max_new_tokens,
            last_prompt_token_steering=last_prompt_token_steering,
        )
        sad_success = generation_matches_success(
            sad_text,
            required_line_prefix="Here is a sad story",
            required_words=hapsad_wordbank["sad"],
        )
        happy_success = generation_matches_success(
            happy_text,
            required_line_prefix="Here is a happy story",
            required_words=hapsad_wordbank["happy"],
        )
        generations.append(
            {
                "prompt": prompt,
                "plus_steering_generation": sad_text,
                "minus_steering_generation": happy_text,
                "plus_steering_success": sad_success,
                "minus_steering_success": happy_success,
            }
        )
        if sad_success and happy_success:
            successes += 1

    return {
        "checked": len(eval_prompts),
        "successes": successes,
        "success_rate": successes / max(len(eval_prompts), 1),
        "samples": generations[:5],
    }


@torch.no_grad()
def generate_with_exact_tokseq_ids(
    bundle,
    prompt: str,
    tokseq_token_ids: torch.Tensor,
    tokseq_position: str,
    layer: int,
    steering_vector: torch.Tensor,
    steering_scale: float,
    max_new_tokens: int,
    last_prompt_token_steering: bool,
) -> str:
    input_ids = build_input_ids_with_tokseq_ids(bundle, prompt, tokseq_token_ids, tokseq_position)
    attention_mask = torch.ones_like(input_ids)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": bundle.tokenizer.pad_token_id,
    }
    if steering_scale == 0.0:
        generated = bundle.model.generate(**generation_kwargs)
        new_tokens = generated[0][input_ids.shape[1] :]
        return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not last_prompt_token_steering:
        with steering_hook(bundle.model, layer, steering_vector, steering_scale):
            generated = bundle.model.generate(**generation_kwargs)
        new_tokens = generated[0][input_ids.shape[1] :]
        return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    current_input_ids = input_ids
    current_attention_mask = attention_mask
    generated_token_ids: list[int] = []
    for generation_position in range(max_new_tokens):
        with steering_hook(bundle.model, layer, steering_vector, steering_scale, all_tokens=False) if generation_position == 0 else null_hook():
            outputs = bundle.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                use_cache=False,
            )
        next_token_id = int(torch.argmax(outputs.logits[0, -1], dim=-1).item())
        if next_token_id == bundle.tokenizer.eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        next_token_tensor = torch.tensor([[next_token_id]], device=bundle.device, dtype=current_input_ids.dtype)
        current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=bundle.device)],
            dim=1,
        )
    return bundle.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def generation_matches_success(text: str, required_line_prefix: str, required_words: list[str]) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    if required_line_prefix and required_line_prefix not in first_line:
        return False
    if not required_words:
        return True
    body = "\n".join(lines[1:] if len(lines) > 1 else lines).lower()
    return any(re.search(pattern, body, flags=re.IGNORECASE) for pattern in required_words)


def resolve_output_path(output: str, steering_file: str) -> Path:
    if output:
        return Path(output)
    run_dir = infer_run_dir(steering_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "suffixes" / f"rep_tokseq_largo_{timestamp}.json"


def infer_run_dir(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.parent.name in {"suffixes", "steering_generations"}:
        return path.parent.parent
    return path.parent


class null_hook:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def initialize_suffix_embeds(bundle, suffix_length: int, init_mode: str) -> torch.Tensor:
    return initialize_tokseq_embeds(bundle=bundle, tokseq_length=suffix_length, init_mode=init_mode)


@torch.no_grad()
def nearest_token_ids_for_suffix_embeds(bundle, suffix_embeds: torch.Tensor) -> torch.Tensor:
    return nearest_token_ids_for_tokseq_embeds(bundle=bundle, tokseq_embeds=suffix_embeds)


@torch.no_grad()
def summarize_suffix_matrix(
    bundle,
    suffix_embeds: torch.Tensor,
    model_name: str,
    summary_prompt: str,
    assistant_prefill: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[int], str]:
    return summarize_tokseq_matrix(
        bundle=bundle,
        tokseq_embeds=suffix_embeds,
        model_name=model_name,
        summary_prompt=summary_prompt,
        assistant_prefill=assistant_prefill,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def truncate_suffix_token_ids(bundle, summary_token_ids: list[int], suffix_length: int) -> torch.Tensor:
    return truncate_tokseq_token_ids(bundle=bundle, summary_token_ids=summary_token_ids, tokseq_length=suffix_length)


def suffix_embeds_from_summary_token_ids(bundle, summary_token_ids: torch.Tensor, suffix_length: int) -> torch.Tensor:
    return tokseq_embeds_from_summary_token_ids(bundle=bundle, summary_token_ids=summary_token_ids, tokseq_length=suffix_length)


if __name__ == "__main__":
    main()
