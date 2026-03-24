from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.text_backend import TextModelBundle, encode_chat_prompt, load_text_model_bundle
else:
    from .text_backend import TextModelBundle, encode_chat_prompt, load_text_model_bundle


@dataclass
class RepSuffixStep:
    step: int
    objective_type: str
    objective: float
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
    trace: list[dict]


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize a suffix against a representation-space contrastive objective.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Causal LM model name or local path.")
    parser.add_argument("--n-plus", default="", help="Positive-concept prompt n+.")
    parser.add_argument("--n-minus", default="", help="Negative-concept prompt n-.")
    parser.add_argument("--prompt-pairs-file", default="", help="Optional JSONL file with prompt pairs. Each row should contain n_plus/n_minus or poscon_prompt/negcon_prompt.")
    parser.add_argument("--steering-file", required=True, help="Path to poscon_negcon_residuals.pt or steering_candidates.pt.")
    parser.add_argument("--layer", type=int, required=True, help="Layer to use for both the steering vector and prompt-state objective.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--suffix-length", type=int, default=20, help="Number of suffix tokens to optimize.")
    parser.add_argument("--steps", type=int, default=500, help="Number of optimization iterations.")
    parser.add_argument("--top-k", type=int, default=256, help="How many promising replacement tokens to keep per position.")
    parser.add_argument("--batch-size", type=int, default=512, help="How many one-edit candidate prompts to sample per iteration.")
    parser.add_argument("--objective-type", default="dot", choices=["dot", "cosine"], help="Which objective to minimize.")
    parser.add_argument("--success-threshold", type=float, default=0.0, help="A prompt pair is considered solved when its objective is below this threshold.")
    parser.add_argument("--kl-interval", type=int, default=50, help="Log average next-token KL(base || suffixed) every N steps. Set <= 0 to disable.")
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
    suffix_token_ids = torch.full(
        (suffix_length,),
        fill_value=_default_fill_token_id(tokenizer),
        dtype=torch.long,
        device=device,
    )

    embedding_layer = model.get_input_embeddings()
    vocab_matrix = embedding_layer.weight.detach()
    trace: list[dict] = []
    best_objective = float("inf")
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
        )

        if current_objective < best_objective:
            best_objective = current_objective
            best_suffix = suffix_token_ids.clone()

        candidate_token_sets = []
        for position in range(suffix_length):
            scores = torch.matmul(vocab_matrix, -grad[position])
            candidate_ids = torch.topk(scores, k=min(top_k, scores.shape[0])).indices.tolist()
            filtered_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in forbidden_token_ids]
            if not filtered_ids:
                filtered_ids = [int(suffix_token_ids[position].item())]
            candidate_token_sets.append(filtered_ids)

        candidate_best_objective = current_objective
        candidate_best_suffix = suffix_token_ids.clone()
        sampled_positions = torch.randint(low=0, high=suffix_length, size=(batch_size,), device=device)

        for batch_index in range(batch_size):
            position = int(sampled_positions[batch_index].item())
            token_choices = candidate_token_sets[position]
            choice_index = int(torch.randint(low=0, high=len(token_choices), size=(1,), device=device).item())
            candidate_id = token_choices[choice_index]

            trial = suffix_token_ids.clone()
            trial[position] = candidate_id
            trial_objective = _aggregate_suffix_objective(
                bundle=bundle,
                prompt_pair_ids=active_prompt_pair_ids,
                suffix_token_ids=trial,
                steering_vector=steering_vector.to(device),
                layer=layer,
                objective_type=objective_type,
            )
            if trial_objective < candidate_best_objective:
                candidate_best_objective = trial_objective
                candidate_best_suffix = trial

        suffix_token_ids = candidate_best_suffix
        if candidate_best_objective < best_objective:
            best_objective = candidate_best_objective
            best_suffix = candidate_best_suffix.clone()

        per_prompt_objectives = _per_prompt_objectives(
            bundle=bundle,
            prompt_pair_ids=active_prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            objective_type=objective_type,
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
                    step=step_index,
                    objective_type=objective_type,
                    objective=float(candidate_best_objective),
                    dot_product=float(dot_product),
                    cosine_similarity=float(cosine_similarity),
                    next_token_kl=next_token_kl,
                    suffix_text=tokenizer.decode(suffix_token_ids, skip_special_tokens=True),
                    active_examples=active_examples,
                )
            )
        )
        print(
            f"[step {step_index + 1}/{steps}] "
            f"objective_type={objective_type} "
            f"objective={candidate_best_objective:.6f} "
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
        trace=trace,
    )

def _aggregate_objective_and_suffix_grad(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
) -> tuple[float, torch.Tensor]:
    model = bundle.model
    embedding_layer = model.get_input_embeddings()
    prefix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0)).detach().clone().requires_grad_(True)
    objective = 0.0
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        n_plus_embeds = embedding_layer(n_plus_ids.unsqueeze(0)).detach()
        n_minus_embeds = embedding_layer(n_minus_ids.unsqueeze(0)).detach()

        plus_state = _last_token_hidden_from_embeds(
            bundle=bundle,
            prompt_embeds=torch.cat([n_plus_embeds, prefix_embeds], dim=1),
            layer=layer,
        )
        minus_state = _last_token_hidden_from_embeds(
            bundle=bundle,
            prompt_embeds=torch.cat([n_minus_embeds, prefix_embeds], dim=1),
            layer=layer,
        )
        typed_steering_vector = steering_vector.to(plus_state.dtype)
        objective = objective + _pair_objective(
            steering_vector=typed_steering_vector,
            plus_state=plus_state,
            minus_state=minus_state,
            objective_type=objective_type,
        )
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
) -> float:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    total_objective = 0.0
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        plus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=plus_embeds, layer=layer)
        minus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=minus_embeds, layer=layer)
        typed_steering_vector = steering_vector.to(plus_state.dtype)
        total_objective += float(
            _pair_objective(
                steering_vector=typed_steering_vector,
                plus_state=plus_state,
                minus_state=minus_state,
                objective_type=objective_type,
            ).item()
        )
    return total_objective


@torch.no_grad()
def _per_prompt_objectives(
    bundle: TextModelBundle,
    prompt_pair_ids: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_token_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    objective_type: str,
) -> list[float]:
    embedding_layer = bundle.model.get_input_embeddings()
    suffix_embeds = embedding_layer(suffix_token_ids.unsqueeze(0))
    objectives = []
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        plus_embeds = torch.cat([embedding_layer(n_plus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        minus_embeds = torch.cat([embedding_layer(n_minus_ids.unsqueeze(0)), suffix_embeds], dim=1)
        plus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=plus_embeds, layer=layer)
        minus_state = _last_token_hidden_from_embeds(bundle=bundle, prompt_embeds=minus_embeds, layer=layer)
        typed_steering_vector = steering_vector.to(plus_state.dtype)
        objectives.append(
            float(
                _pair_objective(
                    steering_vector=typed_steering_vector,
                    plus_state=plus_state,
                    minus_state=minus_state,
                    objective_type=objective_type,
                ).item()
            )
        )
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
) -> float:
    kls = []
    for n_plus_ids, n_minus_ids in prompt_pair_ids:
        kls.append(_prompt_next_token_kl(bundle, n_plus_ids, suffix_token_ids))
        kls.append(_prompt_next_token_kl(bundle, n_minus_ids, suffix_token_ids))
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


def _pair_objective(
    steering_vector: torch.Tensor,
    plus_state: torch.Tensor,
    minus_state: torch.Tensor,
    objective_type: str,
) -> torch.Tensor:
    difference = plus_state - minus_state
    if objective_type == "dot":
        return torch.dot(steering_vector, difference)
    if objective_type == "cosine":
        numerator = torch.dot(steering_vector, difference)
        denominator = steering_vector.norm().clamp_min(1e-8) * difference.norm().clamp_min(1e-8)
        return numerator / denominator
    raise ValueError(f"Unsupported objective_type: {objective_type}")


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


def _default_fill_token_id(tokenizer) -> int:
    for token in [" and", " the", ".", ","]:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
    return 0


def main():
    args = parse_args()
    bundle = load_text_model_bundle(args.model, dtype=args.dtype, device_map=args.device_map)
    steering_vector = load_steering_vector(args.steering_file, args.layer)
    prompt_pairs = load_prompt_pairs(args.prompt_pairs_file, args.n_plus, args.n_minus)
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
    )

    payload = {
        "layer": result.layer,
        "objective_type": result.objective_type,
        "suffix_token_ids": result.suffix_token_ids,
        "suffix_text": result.suffix_text,
        "objective": result.objective,
        "trace": result.trace,
        "prompt_pairs": [{"n_plus": n_plus, "n_minus": n_minus} for n_plus, n_minus in prompt_pairs],
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_prompt_pairs(prompt_pairs_file: str, n_plus: str, n_minus: str) -> list[tuple[str, str]]:
    if prompt_pairs_file:
        rows = []
        with Path(prompt_pairs_file).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                plus = row.get("n_plus") or row.get("poscon_prompt") or row.get("positive_prompt") or row.get("truth_prompt")
                minus = row.get("n_minus") or row.get("negcon_prompt") or row.get("negative_prompt") or row.get("lie_prompt")
                if plus and minus:
                    rows.append((plus, minus))
        if not rows:
            raise ValueError(f"No prompt pairs found in {prompt_pairs_file}")
        return rows

    if not n_plus or not n_minus:
        raise ValueError("Provide either --prompt-pairs-file or both --n-plus and --n-minus.")
    return [(n_plus, n_minus)]


if __name__ == "__main__":
    main()
