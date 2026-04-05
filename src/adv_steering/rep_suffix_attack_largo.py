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
    from adv_steering.text_backend import encode_chat_prompt, load_text_model_bundle, steering_hook
else:
    from .rep_suffix_attack import (
        _default_fill_token_id,
        _single_prompt_objective_and_terms,
        load_prompt_pairs,
        load_steering_vector,
    )
    from .text_backend import encode_chat_prompt, load_text_model_bundle, steering_hook


def parse_args():
    parser = argparse.ArgumentParser(description="Largo-style continuous suffix optimization with summarize-and-reinterpret projection.")
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
    parser.add_argument("--suffix-length", type=int, default=20)
    parser.add_argument("--outer-steps", type=int, default=15, help="How many summarize-and-reinterpret rounds to run.")
    parser.add_argument("--inner-steps", type=int, default=10, help="How many gradient steps to take on the suffix matrix per outer round.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--init-mode", default="random_tokens", choices=["zeros", "random_tokens"])
    parser.add_argument("--summary-prompt", default="Summarize the following for me:")
    parser.add_argument("--summary-max-new-tokens", type=int, default=64)
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
    bundle = load_text_model_bundle(args.model, dtype=args.dtype, device_map=args.device_map)
    steering_vector = load_steering_vector(args.steering_file, args.layer).to(bundle.device)
    prompt_pairs = load_prompt_pairs(
        prompt_pairs_file=args.prompt_pairs_file,
        n_plus=args.n_plus,
        n_minus=args.n_minus,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
    )
    prompt_pair_ids = [
        (
            encode_chat_prompt(bundle, n_plus, add_generation_prompt=True)["input_ids"][0],
            encode_chat_prompt(bundle, n_minus, add_generation_prompt=True)["input_ids"][0],
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
    eval_prompts = load_eval_prompts(args.eval_prompts_file, args.eval_max_prompts)
    hapsad_wordbank = load_hapsad_wordbank(args.hapsad_wordbank_file)

    suffix_embeds = initialize_suffix_embeds(
        bundle=bundle,
        suffix_length=args.suffix_length,
        init_mode=args.init_mode,
    )

    embedding_layer = bundle.model.get_input_embeddings()
    initial_suffix_token_ids = nearest_token_ids_for_suffix_embeds(bundle, suffix_embeds)
    initial_suffix_text = bundle.tokenizer.decode(initial_suffix_token_ids.tolist(), skip_special_tokens=True)
    if args.init_mode == "random_tokens":
        print(
            json.dumps(
                {
                    "initial_suffix_token_ids": initial_suffix_token_ids.tolist(),
                    "initial_suffix_text": initial_suffix_text,
                },
                indent=2,
            ),
            flush=True,
        )
    optimizer_trace = []
    best = {
        "objective": float("inf"),
        "suffix_token_ids": None,
        "suffix_text": "",
        "summary_text": "",
    }

    for outer_step in range(args.outer_steps):
        suffix_embeds = optimize_suffix_matrix(
            bundle=bundle,
            suffix_embeds=suffix_embeds,
            prompt_pair_ids=prompt_pair_ids,
            steering_vector=steering_vector,
            layer=args.layer,
            objective_type=args.objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=args.steering_scale,
            inner_steps=args.inner_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        summary_text = summarize_suffix_matrix(
            bundle=bundle,
            suffix_embeds=suffix_embeds,
            summary_prompt=args.summary_prompt,
            max_new_tokens=args.summary_max_new_tokens,
        )
        suffix_token_ids = normalize_summary_to_suffix_ids(
            bundle=bundle,
            summary_text=summary_text,
            suffix_length=args.suffix_length,
        )
        suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0)).detach()

        objective, breakdown, dot_product, cosine_similarity = evaluate_suffix_from_token_ids(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector,
            layer=args.layer,
            objective_type=args.objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=args.steering_scale,
        )
        eval_result = evaluate_held_out_behavior(
            bundle=bundle,
            eval_prompts=eval_prompts,
            suffix_token_ids=suffix_token_ids,
            layer=args.layer,
            steering_vector=steering_vector,
            steering_scale=args.eval_steering_scale if args.eval_steering_scale != 0.0 else args.steering_scale,
            max_new_tokens=args.eval_max_new_tokens,
            hapsad_wordbank=hapsad_wordbank,
        )

        suffix_text = bundle.tokenizer.decode(suffix_token_ids.tolist(), skip_special_tokens=True)
        optimizer_trace.append(
            {
                "outer_step": outer_step,
                "objective_type": args.objective_type,
                "objective": objective,
                "objective_breakdown": breakdown,
                "dot_product": dot_product,
                "cosine_similarity": cosine_similarity,
                "summary_text": summary_text,
                "suffix_token_ids": suffix_token_ids.tolist(),
                "suffix_text": suffix_text,
                "held_out": eval_result,
            }
        )

        print(
            f"[outer {outer_step + 1}/{args.outer_steps}] "
            f"objective={objective:.6f} "
            f"dot_product={dot_product:.6f} "
            f"cosine_similarity={cosine_similarity:.6f} "
            f"held_out_success_rate={eval_result['success_rate']:.3f} "
            f"suffix={suffix_text!r}",
            flush=True,
        )

        if objective < best["objective"]:
            best = {
                "objective": objective,
                "suffix_token_ids": suffix_token_ids.tolist(),
                "suffix_text": suffix_text,
                "summary_text": summary_text,
            }

        if eval_result["checked"] and eval_result["success_rate"] >= args.success_proportion:
            break

    payload = {
        "model": args.model,
        "objective_type": args.objective_type,
        "layer": args.layer,
        "steering_scale": args.steering_scale,
        "neutral_prompt": args.neutral_prompt,
        "positive_target": args.positive_target,
        "negative_target": args.negative_target,
        "summary_prompt": args.summary_prompt,
        "suffix_length": args.suffix_length,
        "init_mode": args.init_mode,
        "initial_suffix_token_ids": initial_suffix_token_ids.tolist(),
        "initial_suffix_text": initial_suffix_text,
        "outer_steps": args.outer_steps,
        "inner_steps": args.inner_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best": best,
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
    print(f"Saved Largo suffix artifact to {output_path}", flush=True)


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


def initialize_suffix_embeds(bundle, suffix_length: int, init_mode: str) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    hidden_size = embedding_layer.weight.shape[1]
    if init_mode == "zeros":
        return torch.zeros((1, suffix_length, hidden_size), device=bundle.device, dtype=embedding_layer.weight.dtype)
    vocab_size = embedding_layer.weight.shape[0]
    token_ids = torch.randint(low=0, high=vocab_size, size=(suffix_length,), device=bundle.device)
    return embedding_layer(token_ids.unsqueeze(0)).detach()


@torch.no_grad()
def nearest_token_ids_for_suffix_embeds(bundle, suffix_embeds: torch.Tensor) -> torch.Tensor:
    embedding_weight = bundle.model.get_input_embeddings().weight.detach()
    suffix_rows = suffix_embeds[0]
    distances = torch.cdist(suffix_rows.float(), embedding_weight.float())
    return torch.argmin(distances, dim=1)


def optimize_suffix_matrix(
    bundle,
    suffix_embeds: torch.Tensor,
    prompt_pair_ids,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids,
    neutral_prompt_ids,
    steering_scale: float,
    inner_steps: int,
    learning_rate: float,
    weight_decay: float,
) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_parameter = torch.nn.Parameter(suffix_embeds.detach().clone())
    optimizer = torch.optim.Adam([suffix_parameter], lr=learning_rate, weight_decay=weight_decay)
    neutral_prompt_embeds = None
    if neutral_prompt_ids is not None:
        neutral_prompt_embeds = embedding_layer(neutral_prompt_ids.unsqueeze(0)).detach()

    objective_prompt_pair_ids = prompt_pair_ids if objective_type != "steered_ce" else prompt_pair_ids[:1]

    for _ in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), device=bundle.device, dtype=torch.float32)
        for n_plus_ids, n_minus_ids in objective_prompt_pair_ids:
            plus_prompt_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)).detach(), suffix_parameter], dim=1)
            minus_prompt_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)).detach(), suffix_parameter], dim=1)
            prompt_objective, _ = _single_prompt_objective_and_terms(
                bundle=bundle,
                plus_prompt_embeds=plus_prompt_embeds,
                minus_prompt_embeds=minus_prompt_embeds,
                steering_vector=steering_vector,
                layer=layer,
                objective_type=objective_type,
                target_ids=target_ids,
                neutral_prompt_embeds=None if neutral_prompt_embeds is None else torch.cat([neutral_prompt_embeds, suffix_parameter], dim=1),
                steering_scale=steering_scale,
            )
            objective = objective + prompt_objective.float()
        objective.backward()
        optimizer.step()

    return suffix_parameter.detach()


@torch.no_grad()
def summarize_suffix_matrix(bundle, suffix_embeds: torch.Tensor, summary_prompt: str, max_new_tokens: int) -> str:
    prompt_ids = encode_chat_prompt(bundle, summary_prompt, add_generation_prompt=True)["input_ids"][0]
    embedding_layer = bundle.model.get_input_embeddings()
    prompt_embeds = embedding_layer(prompt_ids.unsqueeze(0))
    full_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1)
    attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    generated_token_ids = []
    current_embeds = full_embeds
    current_attention = attention_mask

    for _ in range(max_new_tokens):
        outputs = bundle.model(
            inputs_embeds=current_embeds,
            attention_mask=current_attention,
            use_cache=False,
        )
        next_token_id = int(torch.argmax(outputs.logits[0, -1]).item())
        if next_token_id == bundle.tokenizer.eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        next_embed = embedding_layer(torch.tensor([[next_token_id]], device=bundle.device))
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        current_attention = torch.cat(
            [current_attention, torch.ones((1, 1), dtype=current_attention.dtype, device=bundle.device)],
            dim=1,
        )
    return bundle.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def normalize_summary_to_suffix_ids(bundle, summary_text: str, suffix_length: int) -> torch.Tensor:
    tokenizer = bundle.tokenizer
    token_ids = tokenizer.encode(summary_text, add_special_tokens=False)
    fill_token_id = space_or_fill_token_id(tokenizer)
    if len(token_ids) < suffix_length:
        token_ids = token_ids + [fill_token_id] * (suffix_length - len(token_ids))
    else:
        token_ids = token_ids[:suffix_length]
    return torch.tensor(token_ids, dtype=torch.long, device=bundle.device)


def space_or_fill_token_id(tokenizer) -> int:
    token_ids = tokenizer.encode(" ", add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]
    return _default_fill_token_id(tokenizer)


@torch.no_grad()
def evaluate_suffix_from_token_ids(
    bundle,
    prompt_pair_ids,
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids,
    neutral_prompt_ids,
    steering_scale: float,
):
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    neutral_prompt_embeds = None
    if neutral_prompt_ids is not None:
        neutral_prompt_embeds = torch.cat([embedding_layer(neutral_prompt_ids.unsqueeze(0)), suffix_embeds], dim=1)

    objective_prompt_pair_ids = prompt_pair_ids if objective_type != "steered_ce" else prompt_pair_ids[:1]

    objective = 0.0
    breakdown = None
    for n_plus_ids, n_minus_ids in objective_prompt_pair_ids:
        plus_prompt_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_prompt_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
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
        objective += float(prompt_objective.item())
        if breakdown is None:
            breakdown = {name: float(value.item()) for name, value in terms.items()}
        else:
            for name, value in terms.items():
                breakdown[name] += float(value.item())

    aggregate_difference = None
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_prompt_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_prompt_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
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
    suffix_token_ids: torch.Tensor,
    layer: int,
    steering_vector: torch.Tensor,
    steering_scale: float,
    max_new_tokens: int,
    hapsad_wordbank: dict[str, list[str]],
) -> dict:
    if not eval_prompts:
        return {"checked": 0, "successes": 0, "success_rate": 0.0}

    generations = []
    successes = 0
    for prompt in eval_prompts:
        sad_text = generate_with_exact_suffix_ids(
            bundle=bundle,
            prompt=prompt,
            suffix_token_ids=suffix_token_ids,
            layer=layer,
            steering_vector=steering_vector,
            steering_scale=steering_scale,
            max_new_tokens=max_new_tokens,
        )
        happy_text = generate_with_exact_suffix_ids(
            bundle=bundle,
            prompt=prompt,
            suffix_token_ids=suffix_token_ids,
            layer=layer,
            steering_vector=steering_vector,
            steering_scale=-steering_scale,
            max_new_tokens=max_new_tokens,
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
def generate_with_exact_suffix_ids(
    bundle,
    prompt: str,
    suffix_token_ids: torch.Tensor,
    layer: int,
    steering_vector: torch.Tensor,
    steering_scale: float,
    max_new_tokens: int,
) -> str:
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    input_ids = torch.cat([prompt_inputs["input_ids"][0], suffix_token_ids.to(bundle.device)], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": bundle.tokenizer.pad_token_id,
    }
    context = (
        steering_hook(bundle.model, layer, steering_vector, steering_scale)
        if steering_scale != 0.0
        else null_hook()
    )
    with context:
        generated = bundle.model.generate(**generation_kwargs)
    new_tokens = generated[0][input_ids.shape[1] :]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
    return run_dir / "suffixes" / f"rep_suffix_largo_{timestamp}.json"


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


if __name__ == "__main__":
    main()
