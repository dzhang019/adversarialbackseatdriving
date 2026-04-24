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
    from adv_steering.backend import load_bundle, read_jsonl
    from adv_steering.qualitative_poscon_negcon_steering import load_steering_vector
    from adv_steering.text_backend import encode_chat_prompt
else:
    from .backend import load_bundle, read_jsonl
    from .qualitative_poscon_negcon_steering import load_steering_vector
    from .text_backend import encode_chat_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a continuous soft prefix prepended to fixed story tokens so positive-label "
            "high-probe tokens and negative-label low-probe tokens are inverted while other token "
            "probe scores stay near baseline."
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
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prefix-length", type=int, default=20, help="Number of continuous prefix vectors inserted before the story.")
    parser.add_argument("--steps", type=int, default=20, help="Number of optimization steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random_tokens"])
    parser.add_argument("--custom-label", default="positive", choices=["positive", "negative"], help="Label used with --prompt/--story.")
    parser.add_argument(
        "--dot-threshold",
        type=float,
        default=0.1,
        help="Positive examples select baseline dot >= threshold; negative examples select baseline dot <= -threshold.",
    )
    parser.add_argument("--preserve-weight", type=float, default=1.0, help="Weight on the non-selected-token dot preservation MSE.")
    parser.add_argument(
        "--suppress-weight",
        "--inversion-weight",
        dest="suppress_weight",
        type=float,
        default=1.0,
        help="Weight on the selected-token dot inversion term.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm. Use <=0 to disable clipping.")
    parser.add_argument("--save-all-steps", action="store_true", help="Store the full soft prefix matrix at every optimization step.")
    parser.add_argument("--output", default="", help="Optional artifact output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt or args.story:
        if not (args.prompt and args.story):
            raise ValueError("Provide both --prompt and --story for custom optimization.")

    bundle, resolved_backend = load_bundle(args.model, args.backend, args.dtype, args.device_map)
    if resolved_backend != "causal_lm":
        raise ValueError("rep_prefix_attack_soft_prompt.py currently supports only causal_lm.")

    steering_entries = load_weighted_steering_entries(args.steering_config, args.steering_file)
    examples = load_examples(args)
    if not examples:
        raise ValueError("No examples found for optimization.")

    prepared_examples = [
        prepare_example(bundle=bundle, example=example, steering_entries=steering_entries, dot_threshold=args.dot_threshold)
        for example in examples
    ]
    soft_prefix = initialize_soft_prefix(bundle=bundle, prefix_length=args.prefix_length, init_mode=args.init_mode)
    final_prefix, trace, best = optimize_soft_prefix(
        bundle=bundle,
        soft_prefix=soft_prefix,
        examples=prepared_examples,
        steering_entries=steering_entries,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        suppress_weight=args.suppress_weight,
        preserve_weight=args.preserve_weight,
        dot_threshold=args.dot_threshold,
        grad_clip=args.grad_clip,
        save_all_steps=args.save_all_steps,
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
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "init_mode": args.init_mode,
        "dot_threshold": args.dot_threshold,
        "suppress_weight": args.suppress_weight,
        "preserve_weight": args.preserve_weight,
        "grad_clip": args.grad_clip,
        "save_all_steps": args.save_all_steps,
        "best_objective": best["objective"],
        "best_step": best["step"],
        "final_objective": trace[-1]["objective"] if trace else None,
        "soft_prefix_shape": list(final_prefix.shape),
        "soft_prefix_dtype": str(final_prefix.dtype),
        "soft_prefix": final_prefix.detach().cpu().tolist(),
        "steering_entries": [
            {
                "layer": entry["layer"],
                "scale": entry["scale"],
                "weight": entry["weight"],
                "steering_file": entry["steering_file"],
            }
            for entry in steering_entries
        ],
        "examples": [summarize_prepared_example(example) for example in prepared_examples],
        "trace": trace,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "examples": len(prepared_examples),
                "prefix_shape": payload["soft_prefix_shape"],
                "best_objective": payload["best_objective"],
                "best_step": payload["best_step"],
                "final_objective": payload["final_objective"],
            },
            indent=2,
        ),
        flush=True,
    )


def load_weighted_steering_entries(config_path: str, default_steering_file: str) -> list[dict]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("entries") or payload.get("layers") or payload.get("steering") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("steering-config must be a JSON list or dict containing entries/layers/steering.")
    if not rows:
        raise ValueError(f"No steering entries found in {config_path}")

    raw_entries = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"steering-config entry {index} must be an object.")
        if "layer" not in row:
            raise ValueError(f"steering-config entry {index} is missing 'layer'.")
        scale = row.get("scale", row.get("coefficient", row.get("weight")))
        if scale is None:
            raise ValueError(f"steering-config entry {index} must provide scale/coefficient/weight.")
        steering_file = row.get("steering_file") or row.get("path") or default_steering_file
        raw_entries.append(
            {
                "layer": int(row["layer"]),
                "scale": float(scale),
                "steering_file": steering_file,
            }
        )

    total_abs_scale = sum(abs(entry["scale"]) for entry in raw_entries)
    if total_abs_scale <= 0:
        raise ValueError("At least one steering-config scale must be non-zero.")
    for entry in raw_entries:
        vector = load_steering_vector(entry["steering_file"], entry["layer"]).float()
        if entry["scale"] < 0:
            vector = -vector
        entry["vector"] = vector
        entry["weight"] = len(raw_entries) * abs(entry["scale"]) / total_abs_scale
    return raw_entries


def load_examples(args) -> list[dict]:
    if args.prompt and args.story:
        return [{"concept": "custom", "label": args.custom_label, "prompt": args.prompt, "story": args.story}]

    rows = read_jsonl(args.dataset)
    examples = []
    for row in rows[: args.max_examples]:
        concept = row.get("concept", "")
        if args.story_side in {"positive", "both"}:
            prompt = row.get("poscon_prompt") or row.get("positive_prompt") or row.get("truth_prompt")
            story = row.get("poscon_response") or row.get("positive_response") or row.get("truth_response")
            if prompt and story:
                examples.append({"concept": concept, "label": "positive", "prompt": prompt, "story": story})
        if args.story_side in {"negative", "both"}:
            prompt = row.get("negcon_prompt") or row.get("negative_prompt") or row.get("lie_prompt")
            story = row.get("negcon_response") or row.get("negative_response") or row.get("lie_response")
            if prompt and story:
                examples.append({"concept": concept, "label": "negative", "prompt": prompt, "story": story})
    return examples


@torch.no_grad()
def prepare_example(bundle, example: dict, steering_entries: list[dict], dot_threshold: float) -> dict:
    prompt_ids, story_ids = build_prompt_story_ids(bundle, example["prompt"], example["story"])
    baseline = compute_story_probe_scores(
        bundle=bundle,
        prompt_ids=prompt_ids,
        story_ids=story_ids,
        soft_prefix_embeds=None,
        steering_entries=steering_entries,
    )
    entry_stats = {}
    for entry in steering_entries:
        layer = entry["layer"]
        selected_mask = select_tokens_for_label(baseline[layer], example["label"], dot_threshold)
        entry_stats[str(layer)] = {
            "selection_rule": selection_rule_for_label(example["label"], dot_threshold),
            "selected_tokens": int(selected_mask.sum().item()),
            "preserved_tokens": int((~selected_mask).sum().item()),
            "baseline_mean_dot": float(baseline[layer].mean().item()),
            "baseline_max_dot": float(baseline[layer].max().item()),
        }
    return {
        **example,
        "prompt_ids": prompt_ids.detach(),
        "story_ids": story_ids.detach(),
        "baseline_dots": {layer: values.detach() for layer, values in baseline.items()},
        "entry_stats": entry_stats,
    }


def select_tokens_for_label(dots: torch.Tensor, label: str, dot_threshold: float) -> torch.Tensor:
    threshold = abs(float(dot_threshold))
    if label == "negative":
        return dots <= -threshold
    return dots >= threshold


def inversion_loss_sign(label: str) -> float:
    # Minimize happy/positive token dots, maximize sad/negative token dots.
    return -1.0 if label == "negative" else 1.0


def selection_rule_for_label(label: str, dot_threshold: float) -> str:
    threshold = abs(float(dot_threshold))
    if label == "negative":
        return f"baseline_dot <= -{threshold:g}; maximize current_dot"
    return f"baseline_dot >= {threshold:g}; minimize current_dot"


def initialize_soft_prefix(bundle, prefix_length: int, init_mode: str) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    hidden_size = embedding_layer.weight.shape[1]
    if init_mode == "zeros":
        return torch.zeros((1, prefix_length, hidden_size), device=bundle.device, dtype=torch.float32)
    vocab_size = embedding_layer.weight.shape[0]
    token_ids = torch.randint(low=0, high=vocab_size, size=(prefix_length,), device=bundle.device)
    return embedding_layer(token_ids.unsqueeze(0)).detach().float()


def optimize_soft_prefix(
    bundle,
    soft_prefix: torch.Tensor,
    examples: list[dict],
    steering_entries: list[dict],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    suppress_weight: float,
    preserve_weight: float,
    dot_threshold: float,
    grad_clip: float,
    save_all_steps: bool,
):
    embedding_layer = bundle.model.get_input_embeddings()
    soft_prefix_parameter = torch.nn.Parameter(soft_prefix.detach().clone().float())
    optimizer = torch.optim.Adam([soft_prefix_parameter], lr=learning_rate, weight_decay=weight_decay)
    trace = []
    best = {"objective": float("inf"), "step": -1}

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        soft_prefix_embeds = soft_prefix_parameter.to(embedding_layer.weight.dtype)
        objective = torch.zeros((), device=bundle.device, dtype=torch.float32)
        totals = {
            "inversion_loss": 0.0,
            "preserve_loss": 0.0,
            "selected_tokens": 0,
            "preserved_tokens": 0,
        }

        for example in examples:
            current = compute_story_probe_scores(
                bundle=bundle,
                prompt_ids=example["prompt_ids"],
                story_ids=example["story_ids"],
                soft_prefix_embeds=soft_prefix_embeds,
                steering_entries=steering_entries,
            )
            for entry in steering_entries:
                layer = entry["layer"]
                weight = float(entry["weight"])
                baseline_dots = example["baseline_dots"][layer].to(bundle.device)
                current_dots = current[layer]
                selected_mask = select_tokens_for_label(baseline_dots, example["label"], dot_threshold)
                preserved_mask = ~selected_mask

                inversion_sign = inversion_loss_sign(example["label"])
                inversion_loss = (
                    inversion_sign * current_dots[selected_mask].mean()
                    if selected_mask.any()
                    else current_dots.sum() * 0.0
                )
                preserve_loss = (
                    F.mse_loss(current_dots[preserved_mask], baseline_dots[preserved_mask])
                    if preserved_mask.any()
                    else current_dots.sum() * 0.0
                )
                objective = objective + weight * (suppress_weight * inversion_loss + preserve_weight * preserve_loss)
                totals["inversion_loss"] += float((weight * inversion_loss).detach().item())
                totals["preserve_loss"] += float((weight * preserve_loss).detach().item())
                totals["selected_tokens"] += int(selected_mask.sum().item())
                totals["preserved_tokens"] += int(preserved_mask.sum().item())

        objective = objective / max(len(examples), 1)
        objective.backward()
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_([soft_prefix_parameter], max_norm=grad_clip).item())
        else:
            grad_norm = float(soft_prefix_parameter.grad.detach().norm().item()) if soft_prefix_parameter.grad is not None else 0.0
        optimizer.step()
        if not torch.isfinite(soft_prefix_parameter).all():
            raise ValueError(f"Non-finite soft prefix after step {step + 1}/{steps}.")

        objective_value = float(objective.detach().item())
        if objective_value < best["objective"]:
            best = {"objective": objective_value, "step": step}
        trace_row = {
            "step": step,
            "objective": objective_value,
            "inversion_loss": totals["inversion_loss"] / max(len(examples), 1),
            "preserve_loss": totals["preserve_loss"] / max(len(examples), 1),
            "selected_tokens": totals["selected_tokens"],
            "preserved_tokens": totals["preserved_tokens"],
            "grad_norm": grad_norm,
            "soft_prefix_norm": float(soft_prefix_parameter.detach().norm().item()),
            "soft_prefix": soft_prefix_parameter.detach().cpu().tolist() if save_all_steps else None,
        }
        trace.append(trace_row)
        print(
            f"[step {step + 1}/{steps}] objective={objective_value:.6f} "
            f"inversion={trace_row['inversion_loss']:.6f} preserve={trace_row['preserve_loss']:.6f} "
            f"grad_norm={grad_norm:.6f} soft_prefix_norm={trace_row['soft_prefix_norm']:.6f}",
            flush=True,
        )

    return soft_prefix_parameter.detach(), trace, best


def compute_story_probe_scores(
    bundle,
    prompt_ids: torch.Tensor,
    story_ids: torch.Tensor,
    soft_prefix_embeds: torch.Tensor | None,
    steering_entries: list[dict],
) -> dict[int, torch.Tensor]:
    embedding_layer = bundle.model.get_input_embeddings()
    prompt_embeds = embedding_layer(prompt_ids).detach()
    story_embeds = embedding_layer(story_ids).detach()
    if soft_prefix_embeds is None:
        input_embeds = torch.cat([prompt_embeds, story_embeds], dim=1)
        story_start = prompt_embeds.shape[1]
    else:
        input_embeds = torch.cat([prompt_embeds, soft_prefix_embeds, story_embeds], dim=1)
        story_start = prompt_embeds.shape[1] + soft_prefix_embeds.shape[1]
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


def build_prompt_story_ids(bundle, prompt: str, story: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    prompt_ids = prompt_inputs["input_ids"].detach()
    story_ids = torch.tensor(
        bundle.tokenizer.encode(story, add_special_tokens=False),
        dtype=torch.long,
        device=bundle.device,
    ).unsqueeze(0)
    if story_ids.shape[1] == 0:
        raise ValueError("Story tokenized to zero tokens.")
    return prompt_ids, story_ids


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
    return run_dir / "prefixes" / f"rep_prefix_soft_prompt_{timestamp}.json"


def infer_run_dir(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.parent.name in {"suffixes", "steering_generations", "token_conceptness", "prefixes"}:
        return path.parent.parent
    return path.parent


if __name__ == "__main__":
    main()
