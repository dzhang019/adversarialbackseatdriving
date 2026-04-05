from __future__ import annotations

import argparse
from datetime import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.text_backend import TextModelBundle, encode_chat_prompt, load_text_model_bundle, steering_hook
else:
    from .text_backend import TextModelBundle, encode_chat_prompt, load_text_model_bundle, steering_hook


@dataclass
class RepSuffixStep:
    step: int
    objective_type: str
    objective: float
    objective_term_1_name: str
    objective_term_1: float
    objective_term_2_name: str
    objective_term_2: float
    objective_term_3_name: str
    objective_term_3: float
    dot_product: float
    cosine_similarity: float
    next_token_kl: float | None
    suffix_text: str
    active_examples: int


@dataclass
class RepSuffixResult:
    layer: int
    objective_type: str
    suffix_token_ids: list[int]
    suffix_text: str
    objective: float
    baselines: dict
    trace: list[dict]


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize a suffix against a representation-space contrastive objective.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Causal LM model name or local path.")
    parser.add_argument("--n-plus", default="", help="Positive-concept prompt n+.")
    parser.add_argument("--n-minus", default="", help="Negative-concept prompt n-.")
    parser.add_argument("--neutral-prompt", default="", help="Neutral prompt for the steered cross-entropy objective.")
    parser.add_argument("--positive-prompt", default="", help="Positive-concept prompt for the steered cross-entropy objective.")
    parser.add_argument("--negative-prompt", default="", help="Negative-concept prompt for the steered cross-entropy objective.")
    parser.add_argument("--neutral-target", default="", help="Retained for compatibility; not used by the current steered_ce objective.")
    parser.add_argument("--positive-target", default="", help="Teacher-forced target completion for neutral_prompt under negative steering in steered_ce.")
    parser.add_argument("--negative-target", default="", help="Teacher-forced target completion for neutral_prompt under positive steering in steered_ce.")
    parser.add_argument("--steering-scale", type=float, default=8.0, help="Scale of the steering vector used inside the steered cross-entropy objective.")
    parser.add_argument("--prompt-pairs-file", default="", help="Optional JSONL file with prompt pairs. Each row should contain n_plus/n_minus or poscon_prompt/negcon_prompt.")
    parser.add_argument("--steering-file", required=True, help="Path to poscon_negcon_residuals.pt or steering_candidates.pt.")
    parser.add_argument("--layer", type=int, required=True, help="Layer to use for both the steering vector and prompt-state objective.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--suffix-length", type=int, default=20, help="Number of suffix tokens to optimize.")
    parser.add_argument("--init-mode", default="random", choices=["random", "fill"], help="How to initialize the suffix tokens.")
    parser.add_argument("--steps", type=int, default=500, help="Number of optimization iterations.")
    parser.add_argument("--top-k", type=int, default=256, help="How many promising replacement tokens to keep per position.")
    parser.add_argument("--batch-size", type=int, default=512, help="How many one-edit candidate prompts to sample per iteration.")
    parser.add_argument("--objective-type", default="dot", choices=["dot", "cosine", "steered_ce"], help="Which objective to minimize.")
    parser.add_argument("--success-threshold", type=float, default=0.0, help="A prompt pair is considered solved when its objective is below this threshold.")
    parser.add_argument("--kl-interval", type=int, default=50, help="Log average next-token KL(base || suffixed) every N steps. Set <= 0 to disable.")
    parser.add_argument("--resume-from", default="", help="Optional suffix artifact JSON to resume from. Reuses its suffix_token_ids and appends to its trace.")
    parser.add_argument("--output", default="", help="Optional path to save the result as JSON.")
    return parser.parse_args()


def optimize_suffix_against_direction(
    bundle: TextModelBundle,
    prompt_pairs: list[tuple[str, str]],
    steering_vector: torch.Tensor,
    layer: int,
    suffix_length: int,
    steps: int,
    top_k: int = 256,
    batch_size: int = 512,
    objective_type: str = "dot",
    success_threshold: float = 0.0,
    kl_interval: int = 50,
    init_mode: str = "random",
    neutral_prompt: str = "",
    neutral_target: str = "",
    positive_target: str = "",
    negative_target: str = "",
    steering_scale: float = 8.0,
    initial_suffix_token_ids: torch.Tensor | None = None,
    existing_trace: list[dict] | None = None,
    existing_best_objective: float | None = None,
    forbidden_token_ids: set[int] | None = None,
) -> RepSuffixResult:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    if forbidden_token_ids is None:
        forbidden_token_ids = {tokenizer.pad_token_id}
        if tokenizer.bos_token_id is not None:
            forbidden_token_ids.add(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            forbidden_token_ids.add(tokenizer.eos_token_id)

    prompt_pair_ids = [
        (
            encode_chat_prompt(bundle, n_plus, add_generation_prompt=True)["input_ids"][0],
            encode_chat_prompt(bundle, n_minus, add_generation_prompt=True)["input_ids"][0],
        )
        for n_plus, n_minus in prompt_pairs
    ]
    target_ids = None
    neutral_prompt_ids = None
    if objective_type == "steered_ce":
        if not neutral_prompt or not positive_target or not negative_target:
            raise ValueError("steered_ce requires neutral_prompt, positive_target, and negative_target.")
        neutral_prompt_ids = encode_chat_prompt(bundle, neutral_prompt, add_generation_prompt=True)["input_ids"][0]
        target_ids = {
            "positive": torch.tensor(tokenizer.encode(positive_target, add_special_tokens=False), dtype=torch.long, device=device),
            "negative": torch.tensor(tokenizer.encode(negative_target, add_special_tokens=False), dtype=torch.long, device=device),
        }
        if target_ids["positive"].numel() == 0 or target_ids["negative"].numel() == 0:
            raise ValueError("steered_ce targets must tokenize to at least one token.")

    initial_prompt_pair_ids = prompt_pair_ids[:1]
    no_suffix_token_ids = torch.empty((0,), dtype=torch.long, device=device)
    fill_suffix_token_ids = _initialize_suffix_token_ids(
        tokenizer=tokenizer,
        suffix_length=suffix_length,
        device=device,
        init_mode="fill",
        forbidden_token_ids=forbidden_token_ids,
    )
    baselines = {
        "no_suffix": _compute_suffix_metrics(
            bundle=bundle,
            prompt_pair_ids=initial_prompt_pair_ids,
            suffix_token_ids=no_suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
            suffix_text="",
        ),
        "fill_suffix": _compute_suffix_metrics(
            bundle=bundle,
            prompt_pair_ids=initial_prompt_pair_ids,
            suffix_token_ids=fill_suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
            suffix_text=tokenizer.decode(fill_suffix_token_ids, skip_special_tokens=True),
        ),
    }
    _print_baselines(objective_type=objective_type, baselines=baselines)

    if initial_suffix_token_ids is not None:
        if initial_suffix_token_ids.shape[0] != suffix_length:
            raise ValueError(
                f"Resumed suffix length {initial_suffix_token_ids.shape[0]} does not match requested suffix_length {suffix_length}."
            )
        suffix_token_ids = initial_suffix_token_ids.to(device=device, dtype=torch.long).clone()
    else:
        suffix_token_ids = _initialize_suffix_token_ids(
            tokenizer=tokenizer,
            suffix_length=suffix_length,
            device=device,
            init_mode=init_mode,
            forbidden_token_ids=forbidden_token_ids,
        )

    embedding_layer = model.get_input_embeddings()
    vocab_matrix = embedding_layer.weight.detach()
    trace: list[dict] = list(existing_trace or [])
    step_offset = trace[-1]["step"] + 1 if trace else 0
    best_objective = float(existing_best_objective) if existing_best_objective is not None else float("inf")
    best_suffix = suffix_token_ids.clone()
    active_examples = 1

    for step_index in range(steps):
        model.zero_grad(set_to_none=True)
        active_prompt_pair_ids = prompt_pair_ids[:active_examples]
        current_objective, grad = _aggregate_objective_and_suffix_grad(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
        )

        if current_objective < best_objective:
            best_objective = current_objective
            best_suffix = suffix_token_ids.clone()

        candidate_token_sets = []
        for position in range(suffix_length):
            typed_vocab_matrix = vocab_matrix.to(grad[position].dtype)
            scores = torch.matmul(typed_vocab_matrix, -grad[position])
            candidate_ids = torch.topk(scores, k=min(top_k, scores.shape[0])).indices.tolist()
            filtered_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in forbidden_token_ids]
            if not filtered_ids:
                filtered_ids = [int(suffix_token_ids[position].item())]
            candidate_token_sets.append(filtered_ids)

        candidate_best_objective = current_objective
        candidate_best_suffix = suffix_token_ids.clone()
        candidate_batch = suffix_token_ids.unsqueeze(0).repeat(batch_size, 1)
        sampled_positions = torch.randint(low=0, high=suffix_length, size=(batch_size,), device=device)

        for batch_index in range(batch_size):
            position = int(sampled_positions[batch_index].item())
            token_choices = candidate_token_sets[position]
            choice_index = int(torch.randint(low=0, high=len(token_choices), size=(1,), device=device).item())
            candidate_batch[batch_index, position] = token_choices[choice_index]

        batch_objectives = _batched_suffix_objectives(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            candidate_suffix_token_ids=candidate_batch,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
        )
        best_batch_objective, best_batch_index = torch.min(batch_objectives, dim=0)
        if float(best_batch_objective.item()) < candidate_best_objective:
            candidate_best_objective = float(best_batch_objective.item())
            candidate_best_suffix = candidate_batch[int(best_batch_index.item())].clone()

        suffix_token_ids = candidate_best_suffix
        if candidate_best_objective < best_objective:
            best_objective = candidate_best_objective
            best_suffix = candidate_best_suffix.clone()

        logged_objective = _aggregate_suffix_objective(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
        )
        per_prompt_objectives = _per_prompt_objectives(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
        )
        objective_breakdown = _aggregate_objective_breakdown(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
        )
        dot_product = _aggregate_dot_product(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
        )
        cosine_similarity = _aggregate_cosine_similarity(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
        )
        next_token_kl = None
        if kl_interval > 0 and step_index % kl_interval == 0:
            next_token_kl = _average_next_token_kl(
                bundle=bundle,
                prompt_pair_ids=active_prompt_pair_ids,
                suffix_token_ids=suffix_token_ids,
            )
        if all(objective < success_threshold for objective in per_prompt_objectives) and active_examples < len(prompt_pair_ids):
            active_examples += 1

        trace.append(
            asdict(
                RepSuffixStep(
                    step=step_offset + step_index,
                    objective_type=objective_type,
                    objective=float(logged_objective),
                    objective_term_1_name=objective_breakdown["term_1_name"],
                    objective_term_1=float(objective_breakdown["term_1"]),
                    objective_term_2_name=objective_breakdown["term_2_name"],
                    objective_term_2=float(objective_breakdown["term_2"]),
                    objective_term_3_name=objective_breakdown["term_3_name"],
                    objective_term_3=float(objective_breakdown["term_3"]),
                    dot_product=float(dot_product),
                    cosine_similarity=float(cosine_similarity),
                    next_token_kl=next_token_kl,
                    suffix_text=tokenizer.decode(suffix_token_ids, skip_special_tokens=True),
                    active_examples=active_examples,
                )
            )
        )
        print(
            f"[step {step_offset + step_index + 1}/{step_offset + steps}] "
            f"objective_type={objective_type} "
            f"objective={logged_objective:.6f} "
            f"{objective_breakdown['term_1_name']}={objective_breakdown['term_1']:.6f} "
            f"{objective_breakdown['term_2_name']}={objective_breakdown['term_2']:.6f} "
            f"{objective_breakdown['term_3_name']}={objective_breakdown['term_3']:.6f} "
            f"dot_product={dot_product:.6f} "
            f"cosine_similarity={cosine_similarity:.6f} "
            f"next_token_kl={next_token_kl if next_token_kl is not None else 'NA'} "
            f"active_examples={active_examples} "
            f"suffix={tokenizer.decode(suffix_token_ids, skip_special_tokens=True)!r}",
            flush=True,
        )

    return RepSuffixResult(
        layer=layer,
        objective_type=objective_type,
        suffix_token_ids=best_suffix.tolist(),
        suffix_text=tokenizer.decode(best_suffix, skip_special_tokens=True),
        objective=float(best_objective),
        baselines=baselines,
        trace=trace,
    )

def _aggregate_objective_and_suffix_grad(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None = None,
    neutral_prompt_ids: torch.Tensor | None = None,
    steering_scale: float = 8.0,
) -> tuple[float, torch.Tensor]:
    model = bundle.model
    embedding_layer = model.get_input_embeddings()
    prefix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0)).detach().clone().requires_grad_(True)
    objective = torch.zeros((), device=bundle.device, dtype=torch.float32)
    neutral_prompt_embeds = None
    if neutral_prompt_ids is not None:
        neutral_prompt_embeds = embedding_layer(neutral_prompt_ids.unsqueeze(0)).detach()
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        n_plus_embeds = embedding_layer(n_plus_ids.unsqueeze(0)).detach()
        n_minus_embeds = embedding_layer(n_minus_ids.unsqueeze(0)).detach()
        plus_prompt_embeds = torch.cat([n_plus_embeds, prefix_embeds], dim=1)
        minus_prompt_embeds = torch.cat([n_minus_embeds, prefix_embeds], dim=1)
        prompt_objective, _ = _single_prompt_objective_and_terms(
            bundle=bundle,
            plus_prompt_embeds=plus_prompt_embeds,
            minus_prompt_embeds=minus_prompt_embeds,
            steering_vector=steering_vector,
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_embeds=None if neutral_prompt_embeds is None else torch.cat([neutral_prompt_embeds, prefix_embeds], dim=1),
            steering_scale=steering_scale,
        )
        objective = objective + prompt_objective.float()
    objective.backward()
    grad = prefix_embeds.grad[0].detach()
    return float(objective.detach().item()), grad


@torch.no_grad()
def _aggregate_suffix_objective(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None = None,
    neutral_prompt_ids: torch.Tensor | None = None,
    steering_scale: float = 8.0,
) -> float:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    total_objective = 0.0
    neutral_prompt_embeds = None
    if neutral_prompt_ids is not None:
        neutral_prompt_embeds = torch.cat([embedding_layer(neutral_prompt_ids.unsqueeze(0)), suffix_embeds], dim=1)
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        prompt_objective, _ = _single_prompt_objective_and_terms(
            bundle=bundle,
            plus_prompt_embeds=plus_embeds,
            minus_prompt_embeds=minus_embeds,
            steering_vector=steering_vector,
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_embeds=neutral_prompt_embeds,
            steering_scale=steering_scale,
        )
        total_objective += float(prompt_objective.item())
    return total_objective


@torch.no_grad()
def _batched_suffix_objectives(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    candidate_suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None = None,
    neutral_prompt_ids: torch.Tensor | None = None,
    steering_scale: float = 8.0,
) -> torch.Tensor:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(candidate_suffix_token_ids)
    batch_size = candidate_suffix_token_ids.shape[0]
    total_objectives = torch.zeros(batch_size, dtype=torch.float32, device=bundle.device)
    neutral_prompt_embeds = None
    if neutral_prompt_ids is not None:
        neutral_prefix = embedding_layer(neutral_prompt_ids.unsqueeze(0)).expand(batch_size, -1, -1)
        neutral_prompt_embeds = torch.cat([neutral_prefix, suffix_embeds], dim=1)

    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_prefix = embedding_layer(n_plus_ids.unsqueeze(0)).expand(batch_size, -1, -1)
        minus_prefix = embedding_layer(n_minus_ids.unsqueeze(0)).expand(batch_size, -1, -1)
        plus_embeds = torch.cat([plus_prefix, suffix_embeds], dim=1)
        minus_embeds = torch.cat([minus_prefix, suffix_embeds], dim=1)
        prompt_objective, _ = _batched_prompt_objective_and_terms(
            bundle=bundle,
            plus_prompt_embeds=plus_embeds,
            minus_prompt_embeds=minus_embeds,
            steering_vector=steering_vector,
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_embeds=neutral_prompt_embeds,
            steering_scale=steering_scale,
        )
        total_objectives += prompt_objective.float()

    return total_objectives


@torch.no_grad()
def _aggregate_objective_breakdown(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None = None,
    neutral_prompt_ids: torch.Tensor | None = None,
    steering_scale: float = 8.0,
) -> dict[str, float | str]:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    totals = None
    neutral_prompt_embeds = None
    if neutral_prompt_ids is not None:
        neutral_prompt_embeds = torch.cat([embedding_layer(neutral_prompt_ids.unsqueeze(0)), suffix_embeds], dim=1)

    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        _, terms = _single_prompt_objective_and_terms(
            bundle=bundle,
            plus_prompt_embeds=plus_embeds,
            minus_prompt_embeds=minus_embeds,
            steering_vector=steering_vector,
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_embeds=neutral_prompt_embeds,
            steering_scale=steering_scale,
        )
        if totals is None:
            totals = {name: float(value.item()) for name, value in terms.items()}
        else:
            for name, value in terms.items():
                totals[name] += float(value.item())

    if totals is None:
        totals = _empty_term_totals(objective_type)
    return _term_totals_to_breakdown(objective_type, totals)


@torch.no_grad()
def _per_prompt_objectives(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None = None,
    neutral_prompt_ids: torch.Tensor | None = None,
    steering_scale: float = 8.0,
) -> list[float]:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    objectives = []
    neutral_prompt_embeds = None
    if neutral_prompt_ids is not None:
        neutral_prompt_embeds = torch.cat([embedding_layer(neutral_prompt_ids.unsqueeze(0)), suffix_embeds], dim=1)
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        prompt_objective, _ = _single_prompt_objective_and_terms(
            bundle=bundle,
            plus_prompt_embeds=plus_embeds,
            minus_prompt_embeds=minus_embeds,
            steering_vector=steering_vector,
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_embeds=neutral_prompt_embeds,
            steering_scale=steering_scale,
        )
        objectives.append(float(prompt_objective.item()))
    return objectives


@torch.no_grad()
def _aggregate_cosine_similarity(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
) -> float:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    aggregate_difference = None
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        plus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=plus_embeds, layer=layer)
        minus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=minus_embeds, layer=layer)
        difference = plus_state - minus_state
        aggregate_difference = difference if aggregate_difference is None else aggregate_difference + difference

    typed_steering_vector = steering_vector.to(aggregate_difference.dtype)
    numerator = torch.dot(typed_steering_vector, aggregate_difference)
    denominator = typed_steering_vector.norm().clamp_min(1e-8) * aggregate_difference.norm().clamp_min(1e-8)
    return float((numerator / denominator).item())


@torch.no_grad()
def _aggregate_dot_product(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
) -> float:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    aggregate_difference = None
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        plus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=plus_embeds, layer=layer)
        minus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=minus_embeds, layer=layer)
        difference = plus_state - minus_state
        aggregate_difference = difference if aggregate_difference is None else aggregate_difference + difference

    typed_steering_vector = steering_vector.to(aggregate_difference.dtype)
    return float(torch.dot(typed_steering_vector, aggregate_difference).item())


@torch.no_grad()
def _average_next_token_kl(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    neutral_prompt_ids: torch.Tensor | None = None,
) -> float:
    kls = []
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        kls.append(_prompt_next_token_kl(bundle, n_plus_ids, suffix_token_ids))
        kls.append(_prompt_next_token_kl(bundle, n_minus_ids, suffix_token_ids))
    if neutral_prompt_ids is not None:
        kls.append(_prompt_next_token_kl(bundle, neutral_prompt_ids, suffix_token_ids))
    return float(sum(kls) / len(kls))


@torch.no_grad()
def _prompt_next_token_kl(
    bundle: TextModelBundle,
    prompt_ids: torch.Tensor,
    suffix_token_ids: torch.Tensor,
) -> float:
    embedding_layer = bundle.model.get_input_embeddings()

    base_embeds = embedding_layer(prompt_ids.unsqueeze(0))
    suffixed_embeds = torch.cat([base_embeds, embedding_layer(suffix_token_ids.unsqueeze(0))], dim=1)

    base_logits = _next_token_logits_from_embeds(bundle, base_embeds)
    suffixed_logits = _next_token_logits_from_embeds(bundle, suffixed_embeds)

    base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
    suffixed_log_probs = F.log_softmax(suffixed_logits.float(), dim=-1)
    base_probs = base_log_probs.exp()
    kl = torch.sum(base_probs * (base_log_probs - suffixed_log_probs), dim=-1)
    return float(kl.item())


def _compute_suffix_metrics(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None,
    neutral_prompt_ids: torch.Tensor | None,
    steering_scale: float,
    suffix_text: str,
) -> dict:
    breakdown = _aggregate_objective_breakdown(
        bundle=bundle,
        prompt_pair_ids=prompt_pair_ids,
        suffix_token_ids=suffix_token_ids,
        steering_vector=steering_vector,
        layer=layer,
        objective_type=objective_type,
        target_ids=target_ids,
        neutral_prompt_ids=neutral_prompt_ids,
        steering_scale=steering_scale,
    )
    return {
        "suffix_text": suffix_text,
        "objective": _aggregate_suffix_objective(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector,
            layer=layer,
            objective_type=objective_type,
            target_ids=target_ids,
            neutral_prompt_ids=neutral_prompt_ids,
            steering_scale=steering_scale,
        ),
        "objective_term_1_name": breakdown["term_1_name"],
        "objective_term_1": breakdown["term_1"],
        "objective_term_2_name": breakdown["term_2_name"],
        "objective_term_2": breakdown["term_2"],
        "objective_term_3_name": breakdown["term_3_name"],
        "objective_term_3": breakdown["term_3"],
        "dot_product": _aggregate_dot_product(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector,
            layer=layer,
        ),
        "cosine_similarity": _aggregate_cosine_similarity(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector,
            layer=layer,
        ),
        "next_token_kl": _average_next_token_kl(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            neutral_prompt_ids=neutral_prompt_ids,
        ),
    }


def _print_baselines(objective_type: str, baselines: dict) -> None:
    no_suffix = baselines["no_suffix"]
    fill_suffix = baselines["fill_suffix"]
    print(
        "[baseline] "
        f"objective_type={objective_type} "
        f"no_suffix_objective={no_suffix['objective']:.6f} "
        f"{no_suffix['objective_term_1_name']}={no_suffix['objective_term_1']:.6f} "
        f"{no_suffix['objective_term_2_name']}={no_suffix['objective_term_2']:.6f} "
        f"{no_suffix['objective_term_3_name']}={no_suffix['objective_term_3']:.6f} "
        f"no_suffix_dot_product={no_suffix['dot_product']:.6f} "
        f"no_suffix_cosine_similarity={no_suffix['cosine_similarity']:.6f}",
        flush=True,
    )
    print(
        "[baseline] "
        f"fill_suffix_objective={fill_suffix['objective']:.6f} "
        f"{fill_suffix['objective_term_1_name']}={fill_suffix['objective_term_1']:.6f} "
        f"{fill_suffix['objective_term_2_name']}={fill_suffix['objective_term_2']:.6f} "
        f"{fill_suffix['objective_term_3_name']}={fill_suffix['objective_term_3']:.6f} "
        f"fill_suffix_dot_product={fill_suffix['dot_product']:.6f} "
        f"fill_suffix_cosine_similarity={fill_suffix['cosine_similarity']:.6f} "
        f"fill_suffix={fill_suffix['suffix_text']!r}",
        flush=True,
    )


def _pair_objective(
    steering_vector: torch.Tensor,
    plus_state: torch.Tensor,
    minus_state: torch.Tensor,
    objective_type: str,
) -> torch.Tensor:
    difference = plus_state - minus_state
    typed_steering_vector = steering_vector.to(difference.dtype)
    if objective_type == "dot":
        return torch.dot(typed_steering_vector, difference)
    if objective_type == "cosine":
        numerator = torch.dot(typed_steering_vector, difference)
        denominator = typed_steering_vector.norm().clamp_min(1e-8) * difference.norm().clamp_min(1e-8)
        return numerator / denominator
    raise ValueError(f"Unsupported objective_type: {objective_type}")


def _batched_pair_objective(
    steering_vector: torch.Tensor,
    plus_states: torch.Tensor,
    minus_states: torch.Tensor,
    objective_type: str,
) -> torch.Tensor:
    difference = plus_states - minus_states
    typed_steering_vector = steering_vector.to(difference.dtype)
    if objective_type == "dot":
        return torch.matmul(difference, typed_steering_vector)
    if objective_type == "cosine":
        numerator = torch.matmul(difference, typed_steering_vector)
        denominator = typed_steering_vector.norm().clamp_min(1e-8) * difference.norm(dim=-1).clamp_min(1e-8)
        return numerator / denominator
    raise ValueError(f"Unsupported objective_type: {objective_type}")


def _single_prompt_objective_and_terms(
    bundle: TextModelBundle,
    plus_prompt_embeds: torch.Tensor,
    minus_prompt_embeds: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None = None,
    neutral_prompt_embeds: torch.Tensor | None = None,
    steering_scale: float = 8.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if objective_type == "steered_ce":
        if target_ids is None or neutral_prompt_embeds is None:
            raise ValueError("steered_ce requires target_ids and neutral_prompt_embeds.")
        return _steered_ce_terms_from_neutral_prompt_embeds(
            bundle=bundle,
            neutral_prompt_embeds=neutral_prompt_embeds,
            steering_vector=steering_vector,
            layer=layer,
            target_ids=target_ids,
            steering_scale=steering_scale,
            batched=False,
        )

    plus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=plus_prompt_embeds, layer=layer)
    minus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=minus_prompt_embeds, layer=layer)
    typed_steering_vector = steering_vector.to(plus_state.dtype)
    objective = _pair_objective(
        steering_vector=typed_steering_vector,
        plus_state=plus_state,
        minus_state=minus_state,
        objective_type=objective_type,
    )
    return objective, {
        "plus_projection": torch.dot(typed_steering_vector, plus_state),
        "minus_projection_negated": -torch.dot(typed_steering_vector, minus_state),
        "unused_term": torch.zeros((), device=plus_state.device, dtype=plus_state.dtype),
    }


def _batched_prompt_objective_and_terms(
    bundle: TextModelBundle,
    plus_prompt_embeds: torch.Tensor,
    minus_prompt_embeds: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
    target_ids: dict[str, torch.Tensor] | None = None,
    neutral_prompt_embeds: torch.Tensor | None = None,
    steering_scale: float = 8.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if objective_type == "steered_ce":
        if target_ids is None or neutral_prompt_embeds is None:
            raise ValueError("steered_ce requires target_ids and neutral_prompt_embeds.")
        return _steered_ce_terms_from_neutral_prompt_embeds(
            bundle=bundle,
            neutral_prompt_embeds=neutral_prompt_embeds,
            steering_vector=steering_vector,
            layer=layer,
            target_ids=target_ids,
            steering_scale=steering_scale,
            batched=True,
        )

    plus_states = _last_token_hidden_from_embeds_batched(bundle=bundle, prompt_embeds=plus_prompt_embeds, layer=layer)
    minus_states = _last_token_hidden_from_embeds_batched(bundle=bundle, prompt_embeds=minus_prompt_embeds, layer=layer)
    typed_steering_vector = steering_vector.to(plus_states.dtype)
    objective = _batched_pair_objective(
        steering_vector=typed_steering_vector,
        plus_states=plus_states,
        minus_states=minus_states,
        objective_type=objective_type,
    )
    return objective, {
        "plus_projection": torch.matmul(plus_states, typed_steering_vector),
        "minus_projection_negated": -torch.matmul(minus_states, typed_steering_vector),
        "unused_term": torch.zeros_like(objective),
    }


def _empty_term_totals(objective_type: str) -> dict[str, float]:
    if objective_type == "steered_ce":
        return {
            "neutral_prompt_positive_steering_negative_target_ce": 0.0,
            "neutral_prompt_negative_steering_positive_target_ce": 0.0,
            "unused_term": 0.0,
        }
    return {
        "plus_projection": 0.0,
        "minus_projection_negated": 0.0,
        "unused_term": 0.0,
    }


def _term_totals_to_breakdown(objective_type: str, totals: dict[str, float]) -> dict[str, float | str]:
    if objective_type == "steered_ce":
        return {
            "term_1_name": "neutral_prompt_positive_steering_negative_target_ce",
            "term_1": totals["neutral_prompt_positive_steering_negative_target_ce"],
            "term_2_name": "neutral_prompt_negative_steering_positive_target_ce",
            "term_2": totals["neutral_prompt_negative_steering_positive_target_ce"],
            "term_3_name": "unused_term",
            "term_3": totals["unused_term"],
        }
    return {
        "term_1_name": "plus_projection",
        "term_1": totals["plus_projection"],
        "term_2_name": "minus_projection_negated",
        "term_2": totals["minus_projection_negated"],
        "term_3_name": "unused_term",
        "term_3": totals["unused_term"],
    }


def _steered_ce_terms_from_neutral_prompt_embeds(
    bundle: TextModelBundle,
    neutral_prompt_embeds: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    target_ids: dict[str, torch.Tensor],
    steering_scale: float,
    batched: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    negative_target_ce = _teacher_forced_ce_from_prompt_embeds(
        bundle=bundle,
        prompt_embeds=neutral_prompt_embeds,
        target_token_ids=target_ids["negative"],
        steering_layer=layer,
        steering_vector=steering_vector,
        steering_scale=steering_scale,
    )
    positive_target_ce = _teacher_forced_ce_from_prompt_embeds(
        bundle=bundle,
        prompt_embeds=neutral_prompt_embeds,
        target_token_ids=target_ids["positive"],
        steering_layer=layer,
        steering_vector=steering_vector,
        steering_scale=-steering_scale,
    )
    objective = negative_target_ce + positive_target_ce
    unused_term = torch.zeros_like(objective) if batched else torch.zeros((), device=objective.device, dtype=objective.dtype)
    if not batched:
        objective = objective[0]
        negative_target_ce = negative_target_ce[0]
        positive_target_ce = positive_target_ce[0]
    return objective, {
        "neutral_prompt_positive_steering_negative_target_ce": negative_target_ce,
        "neutral_prompt_negative_steering_positive_target_ce": positive_target_ce,
        "unused_term": unused_term,
    }


def _teacher_forced_ce_from_prompt_embeds(
    bundle: TextModelBundle,
    prompt_embeds: torch.Tensor,
    target_token_ids: torch.Tensor,
    steering_layer: int | None = None,
    steering_vector: torch.Tensor | None = None,
    steering_scale: float = 0.0,
) -> torch.Tensor:
    batch_size = prompt_embeds.shape[0]
    embedding_layer = bundle.model.get_input_embeddings()
    target_token_ids = target_token_ids.to(bundle.device)
    target_embeds = embedding_layer(target_token_ids.unsqueeze(0)).expand(batch_size, -1, -1)
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
    attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=bundle.device)

    if steering_layer is not None and steering_vector is not None and steering_scale != 0.0:
        with steering_hook(bundle.model, steering_layer, steering_vector.to(bundle.device), steering_scale):
            outputs = bundle.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
    else:
        outputs = bundle.model(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )

    prompt_length = prompt_embeds.shape[1]
    target_length = target_token_ids.shape[0]
    logits = outputs.logits[:, prompt_length - 1 : prompt_length + target_length - 1, :]
    labels = target_token_ids.unsqueeze(0).expand(batch_size, -1)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="none",
    ).view(batch_size, target_length)
    return token_losses.mean(dim=1)


def _last_token_hidden_from_embeds(
    bundle: TextModelBundle,
    prompt_embeds: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[layer + 1][0, prompt_embeds.shape[1] - 1]


def _last_token_hidden_from_embeds_batched(
    bundle: TextModelBundle,
    prompt_embeds: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[layer + 1][:, prompt_embeds.shape[1] - 1, :]


def _next_token_logits_from_embeds(
    bundle: TextModelBundle,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=prompt_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return outputs.logits[0, prompt_embeds.shape[1] - 1]


def load_steering_vector(path: str, layer: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if "steering_vectors" in payload:
        return payload["steering_vectors"][layer].float()
    if "candidates" in payload:
        for row in payload["candidates"]:
            if row["layer"] == layer:
                return torch.tensor(row["vector"], dtype=torch.float32)
    raise ValueError(f"Unsupported steering file format or missing layer {layer}: {path}")


def load_resume_artifact(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "suffix_token_ids" not in payload:
        raise ValueError(f"Resume artifact is missing suffix_token_ids: {path}")
    return payload


def _default_fill_token_id(tokenizer) -> int:
    for token in [" and", " the", ".", ","]:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
    return 0


def _initialize_suffix_token_ids(
    tokenizer,
    suffix_length: int,
    device: torch.device,
    init_mode: str,
    forbidden_token_ids: set[int],
) -> torch.Tensor:
    if init_mode == "fill":
        return torch.full(
            (suffix_length,),
            fill_value=_default_fill_token_id(tokenizer),
            dtype=torch.long,
            device=device,
        )
    if init_mode == "random":
        vocab_size = len(tokenizer)
        allowed_token_ids = [token_id for token_id in range(vocab_size) if token_id not in forbidden_token_ids]
        if not allowed_token_ids:
            raise ValueError("No allowed token ids available for random suffix initialization.")
        allowed_tensor = torch.tensor(allowed_token_ids, dtype=torch.long, device=device)
        sampled_indices = torch.randint(low=0, high=allowed_tensor.shape[0], size=(suffix_length,), device=device)
        return allowed_tensor[sampled_indices]
    raise ValueError(f"Unsupported init_mode: {init_mode}")


def main():
    args = parse_args()
    resume_payload = load_resume_artifact(args.resume_from) if args.resume_from else None
    bundle = load_text_model_bundle(args.model, dtype=args.dtype, device_map=args.device_map)
    steering_vector = load_steering_vector(args.steering_file, args.layer)
    prompt_pairs = load_prompt_pairs(
        prompt_pairs_file=args.prompt_pairs_file,
        n_plus=args.n_plus,
        n_minus=args.n_minus,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
    )
    result = optimize_suffix_against_direction(
        bundle=bundle,
        prompt_pairs=prompt_pairs,
        steering_vector=steering_vector,
        layer=args.layer,
        suffix_length=args.suffix_length,
        steps=args.steps,
        top_k=args.top_k,
        batch_size=args.batch_size,
        objective_type=args.objective_type,
        success_threshold=args.success_threshold,
        kl_interval=args.kl_interval,
        init_mode=args.init_mode,
        neutral_prompt=args.neutral_prompt,
        neutral_target=args.neutral_target,
        positive_target=args.positive_target,
        negative_target=args.negative_target,
        steering_scale=args.steering_scale,
        initial_suffix_token_ids=None if resume_payload is None else torch.tensor(resume_payload["suffix_token_ids"], dtype=torch.long),
        existing_trace=None if resume_payload is None else resume_payload.get("trace", []),
        existing_best_objective=None if resume_payload is None else resume_payload.get("objective"),
    )

    payload = {
        "layer": result.layer,
        "objective_type": result.objective_type,
        "neutral_prompt": args.neutral_prompt,
        "positive_prompt": args.positive_prompt,
        "negative_prompt": args.negative_prompt,
        "neutral_target": args.neutral_target,
        "positive_target": args.positive_target,
        "negative_target": args.negative_target,
        "steering_scale": args.steering_scale,
        "suffix_token_ids": result.suffix_token_ids,
        "suffix_text": result.suffix_text,
        "objective": result.objective,
        "baselines": result.baselines,
        "trace": result.trace,
        "prompt_pairs": [{"n_plus": n_plus, "n_minus": n_minus} for n_plus, n_minus in prompt_pairs],
        "resume_from": args.resume_from,
    }
    print(json.dumps(payload, indent=2))
    output_path = resolve_output_path(args.output, args.steering_file, args.resume_from)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved suffix artifact to {output_path}", flush=True)


def load_prompt_pairs(
    prompt_pairs_file: str,
    n_plus: str,
    n_minus: str,
    positive_prompt: str = "",
    negative_prompt: str = "",
) -> list[tuple[str, str]]:
    if prompt_pairs_file:
        rows = []
        with Path(prompt_pairs_file).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                plus = (
                    row.get("n_plus")
                    or row.get("neutral_prompt")
                    or row.get("poscon_prompt")
                    or row.get("positive_prompt")
                    or row.get("truth_prompt")
                )
                minus = (
                    row.get("n_minus")
                    or row.get("steered_positive_prompt")
                    or row.get("negcon_prompt")
                    or row.get("negative_prompt")
                    or row.get("lie_prompt")
                )
                if plus and minus:
                    rows.append((plus, minus))
        if not rows:
            raise ValueError(f"No prompt pairs found in {prompt_pairs_file}")
        return rows

    if positive_prompt and negative_prompt:
        return [(positive_prompt, negative_prompt)]
    if not n_plus or not n_minus:
        raise ValueError("Provide either --prompt-pairs-file, both --positive-prompt and --negative-prompt, or both --n-plus and --n-minus.")
    return [(n_plus, n_minus)]


def resolve_output_path(output: str, steering_file: str, resume_from: str = "") -> Path | None:
    if output:
        return Path(output)
    if resume_from:
        return Path(resume_from)
    run_dir = infer_run_dir(steering_file)
    if run_dir is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "suffixes" / f"rep_suffix_{timestamp}.json"


def infer_run_dir(artifact_path: str) -> Path | None:
    path = Path(artifact_path)
    if not path.parts:
        return None
    if path.parent.name in {"suffixes", "steering_generations"}:
        return path.parent.parent
    return path.parent


if __name__ == "__main__":
    main()
