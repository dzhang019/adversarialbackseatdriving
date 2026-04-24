from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.rep_suffix_attack_largo import (
        build_prompt_embeds_with_tokseq,
        build_summary_prompt_variants,
        build_input_ids_with_tokseq_ids,
        generate_with_exact_tokseq_ids,
        initialize_tokseq_embeds,
        prepare_prompt_segments,
        summarize_tokseq_matrix,
        tokseq_embeds_from_summary_token_ids,
        truncate_tokseq_token_ids,
    )
    from adv_steering.text_backend import encode_chat_prompt, load_text_model_bundle
else:
    from .rep_suffix_attack_largo import (
        build_prompt_embeds_with_tokseq,
        build_summary_prompt_variants,
        build_input_ids_with_tokseq_ids,
        generate_with_exact_tokseq_ids,
        initialize_tokseq_embeds,
        prepare_prompt_segments,
        summarize_tokseq_matrix,
        tokseq_embeds_from_summary_token_ids,
        truncate_tokseq_token_ids,
    )
    from .text_backend import encode_chat_prompt, load_text_model_bundle


DEFAULT_COACH_PROMPT = (
    "You are a ruthless self-improvement coach. Be brutally honest, demanding, concise, and relentlessly "
    "action-oriented. Do not comfort me, do not reassure me, and do not soften the message. Point out excuses, "
    "identify weaknesses, and tell me exactly what I need to do next."
)
DEFAULT_QUESTION = "How can I stop wasting time and become genuinely disciplined?"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a ruthless-coach dataset and run a Largo tokseq reconstruction experiment.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--coach-prompt", default=DEFAULT_COACH_PROMPT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--samples", type=int, default=25, help="How many sampled responses to generate for the training set.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="How many response tokens to generate per sampled response.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokseq-position", default="prefix", choices=["prefix", "suffix"])
    parser.add_argument("--tokseq-length", type=int, default=64)
    parser.add_argument("--outer-steps", type=int, default=12)
    parser.add_argument("--inner-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--init-mode", default="zeros", choices=["zeros", "random_tokens"])
    parser.add_argument("--summary-prompt", default="Summarize the following: ")
    parser.add_argument("--summary-samples", type=int, default=24, help="Best-of-n candidate verbalizations per outer step.")
    parser.add_argument("--summary-temperature", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1, help="Print progress every N items in long loops.")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--dataset-output", default="")
    parser.add_argument("--artifact-output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    print(f"[setup] loading model {args.model!r}", flush=True)
    bundle = load_text_model_bundle(args.model, dtype=args.dtype, device_map=args.device_map)
    print("[setup] model loaded", flush=True)

    full_prompt = build_full_prompt(args.coach_prompt, args.question)
    base_prompt = build_base_prompt(args.question)
    print(
        f"[dataset] generating {args.samples} sampled responses with max_new_tokens={args.max_new_tokens} temperature={args.temperature}",
        flush=True,
    )
    dataset = generate_dataset(
        bundle=bundle,
        full_prompt=full_prompt,
        num_samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        log_every=args.log_every,
    )
    print(f"[dataset] finished generating {len(dataset)} responses", flush=True)

    run_dir = resolve_run_dir(args.output_dir)
    dataset_path = Path(args.dataset_output) if args.dataset_output else run_dir / "coach_reconstruction_dataset.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_payload = {
        "model": args.model,
        "full_prompt": full_prompt,
        "base_prompt": base_prompt,
        "samples": len(dataset),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "responses": dataset,
    }
    dataset_path.write_text(json.dumps(dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dataset] saved dataset to {dataset_path}", flush=True)

    prompt_segments = prepare_prompt_segments(bundle, base_prompt, args.tokseq_position)
    print("[baseline] scoring teacher-forced CE for full prompt", flush=True)
    full_prompt_ce = mean_teacher_forced_ce(bundle, full_prompt, dataset, label="full_prompt", log_every=args.log_every)
    print("[baseline] scoring teacher-forced CE for base prompt without tokseq", flush=True)
    base_prompt_ce = mean_teacher_forced_ce(bundle, base_prompt, dataset, label="base_prompt_without_tokseq", log_every=args.log_every)
    print(
        f"[baseline] full_prompt_ce={full_prompt_ce:.6f} base_prompt_ce={base_prompt_ce:.6f}",
        flush=True,
    )

    tokseq_embeds = initialize_tokseq_embeds(bundle, args.tokseq_length, args.init_mode)
    summary_prompts = build_summary_prompt_variants(args.summary_prompt)
    trace = []
    best = {
        "objective": float("inf"),
        "tokseq_text": "",
        "tokseq_token_ids": [],
        "summary_text": "",
    }

    for outer_step in range(args.outer_steps):
        print(f"[outer {outer_step + 1}/{args.outer_steps}] starting inner optimization", flush=True)
        tokseq_embeds, inner_trace = optimize_tokseq_matrix(
            bundle=bundle,
            tokseq_embeds=tokseq_embeds,
            prompt_segments=prompt_segments,
            responses=dataset,
            inner_steps=args.inner_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print(f"[outer {outer_step + 1}/{args.outer_steps}] starting best-of-{args.summary_samples} summarization search", flush=True)
        summary_candidates, selected = sample_and_select_tokseq_summary(
            bundle=bundle,
            tokseq_embeds=tokseq_embeds,
            model_name=args.model,
            summary_prompts=summary_prompts,
            summary_samples=args.summary_samples,
            summary_temperature=args.summary_temperature,
            tokseq_length=args.tokseq_length,
            prompt_segments=prompt_segments,
            responses=dataset,
            tokseq_position=args.tokseq_position,
            base_prompt=base_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        tokseq_token_ids = selected["tokseq_token_ids_tensor"]
        tokseq_embeds = tokseq_embeds_from_summary_token_ids(bundle, tokseq_token_ids, args.tokseq_length)
        trace_row = {
            "outer_step": outer_step,
            "inner_trace": inner_trace,
            "summary_candidates": [
                {key: value for key, value in candidate.items() if key != "tokseq_token_ids_tensor"}
                for candidate in summary_candidates
            ],
            "selected_summary_candidate": selected["candidate_index"],
            "selected_summary_prompt": selected["summary_prompt"],
            "selected_assistant_prefill": selected["assistant_prefill"],
            "objective": selected["objective"],
            "summary_text": selected["summary_text"],
            "tokseq_text": selected["tokseq_text"],
            "tokseq_token_ids": selected["tokseq_token_ids"],
            "sample_generations": selected["sample_generations"],
        }
        trace.append(trace_row)
        if selected["objective"] < best["objective"]:
            best = {
                "objective": selected["objective"],
                "tokseq_text": selected["tokseq_text"],
                "tokseq_token_ids": selected["tokseq_token_ids"],
                "summary_text": selected["summary_text"],
                "sample_generations": selected["sample_generations"],
            }
        print(
            f"[outer {outer_step + 1}/{args.outer_steps}] objective={selected['objective']:.6f} "
            f"tokseq={selected['tokseq_text']!r}",
            flush=True,
        )

    artifact_path = Path(args.artifact_output) if args.artifact_output else run_dir / "coach_reconstruction_largo.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "seed": args.seed,
        "coach_prompt": args.coach_prompt,
        "question": args.question,
        "full_prompt": full_prompt,
        "base_prompt": base_prompt,
        "tokseq_position": args.tokseq_position,
        "tokseq_length": args.tokseq_length,
        "outer_steps": args.outer_steps,
        "inner_steps": args.inner_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "summary_prompt": args.summary_prompt,
        "summary_samples": args.summary_samples,
        "summary_temperature": args.summary_temperature,
        "dataset_path": str(dataset_path),
        "samples": len(dataset),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "baseline_mean_ce": {
            "full_prompt": full_prompt_ce,
            "base_prompt_without_tokseq": base_prompt_ce,
        },
        "best": best,
        "trace": trace,
    }
    artifact_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "artifact_path": str(artifact_path),
                "full_prompt_ce": full_prompt_ce,
                "base_prompt_ce": base_prompt_ce,
                "best_objective": best["objective"],
                "best_tokseq_text": best["tokseq_text"],
            },
            indent=2,
        ),
        flush=True,
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_full_prompt(coach_prompt: str, question: str) -> str:
    return f"{coach_prompt.strip()}\n\nQuestion: {question.strip()}"


def build_base_prompt(question: str) -> str:
    return f"Question: {question.strip()}"


@torch.no_grad()
def generate_dataset(
    bundle,
    full_prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    log_every: int,
) -> list[str]:
    prompt_inputs = encode_chat_prompt(bundle, full_prompt, add_generation_prompt=True)
    generations = []
    for sample_index in range(num_samples):
        generated = bundle.model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=bundle.tokenizer.pad_token_id,
        )
        new_tokens = generated[0][prompt_inputs["input_ids"].shape[1] :]
        generations.append(bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        if log_every > 0 and (sample_index + 1) % log_every == 0:
            print(f"  [dataset {sample_index + 1}/{num_samples}] generated response", flush=True)
    return generations


@torch.no_grad()
def mean_teacher_forced_ce(bundle, prompt: str, responses: list[str], label: str, log_every: int) -> float:
    if not responses:
        raise ValueError("Need at least one response to score.")
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    embedding_layer = bundle.model.get_input_embeddings()
    prompt_embeds = embedding_layer(prompt_inputs["input_ids"])
    total = len(responses)
    if log_every > 0:
        for response_index in range(total):
            if (response_index + 1) % log_every == 0:
                print(
                    f"  [baseline:{label} {response_index + 1}/{total}] queued for batched CE",
                    flush=True,
                )
    return float(teacher_forced_ce_from_prompt_embeds_batch(bundle, prompt_embeds, responses).mean().item())


@torch.no_grad()
def teacher_forced_ce_from_prompt_text(bundle, prompt: str, response_text: str) -> float:
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    embedding_layer = bundle.model.get_input_embeddings()
    prompt_embeds = embedding_layer(prompt_inputs["input_ids"])
    return float(teacher_forced_ce_from_prompt_embeds(bundle, prompt_embeds, response_text).item())


def optimize_tokseq_matrix(
    bundle,
    tokseq_embeds: torch.Tensor,
    prompt_segments: dict,
    responses: list[str],
    inner_steps: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.Tensor, list[dict]]:
    embedding_layer = bundle.model.get_input_embeddings()
    tokseq_parameter = torch.nn.Parameter(tokseq_embeds.detach().clone().float())
    optimizer = torch.optim.Adam([tokseq_parameter], lr=learning_rate, weight_decay=weight_decay)
    trace = []

    for inner_step in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        typed_tokseq = tokseq_parameter.to(embedding_layer.weight.dtype)
        prompt_embeds = build_prompt_embeds_with_tokseq(bundle, prompt_segments, typed_tokseq)
        objective = teacher_forced_ce_from_prompt_embeds_batch(bundle, prompt_embeds, responses).mean()
        objective.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([tokseq_parameter], max_norm=1.0).item())
        optimizer.step()
        if not torch.isfinite(tokseq_parameter).all():
            raise ValueError(f"Non-finite tokseq parameter after inner step {inner_step + 1}/{inner_steps}.")
        row = {
            "inner_step": inner_step,
            "objective": float(objective.detach().item()),
            "grad_norm": grad_norm,
            "tokseq_norm": float(tokseq_parameter.detach().norm().item()),
        }
        trace.append(row)
        print(
            f"  [inner {inner_step + 1}/{inner_steps}] objective={row['objective']:.6f} grad_norm={grad_norm:.6f}",
            flush=True,
        )
    return tokseq_parameter.detach(), trace


@torch.no_grad()
def sample_and_select_tokseq_summary(
    bundle,
    tokseq_embeds: torch.Tensor,
    model_name: str,
    summary_prompts: list[dict[str, str]],
    summary_samples: int,
    summary_temperature: float,
    tokseq_length: int,
    prompt_segments: dict,
    responses: list[str],
    tokseq_position: str,
    base_prompt: str,
    max_new_tokens: int,
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
            max_new_tokens=tokseq_length,
            temperature=summary_temperature,
        )
        tokseq_token_ids = truncate_tokseq_token_ids(bundle, raw_summary_token_ids, tokseq_length)
        objective = evaluate_dataset_nll_from_tokseq_ids(bundle, prompt_segments, tokseq_token_ids, responses)
        tokseq_text = bundle.tokenizer.decode(tokseq_token_ids.tolist(), skip_special_tokens=True)
        sample_generations = sample_tokseq_generations(
            bundle=bundle,
            prompt=base_prompt,
            tokseq_token_ids=tokseq_token_ids,
            tokseq_position=tokseq_position,
            max_new_tokens=max_new_tokens,
        )
        candidate = {
            "candidate_index": candidate_index,
            "summary_prompt": prompt_variant["summary_prompt"],
            "assistant_prefill": prompt_variant["assistant_prefill"],
            "objective": objective,
            "raw_summary_token_ids": raw_summary_token_ids,
            "summary_token_ids": tokseq_token_ids.tolist(),
            "summary_text": summary_text,
            "tokseq_token_ids": tokseq_token_ids.tolist(),
            "tokseq_text": tokseq_text,
            "tokseq_token_ids_tensor": tokseq_token_ids,
            "sample_generations": sample_generations,
        }
        candidates.append(candidate)
        print(
            f"  [summary {candidate_index + 1}/{summary_samples}] objective={objective:.6f} "
            f"prompt={prompt_variant['name']!r} tokseq={tokseq_text!r}",
            flush=True,
        )
    selected = min(candidates, key=lambda candidate: candidate["objective"])
    print(
        f"  [summary selected] index={selected['candidate_index']} objective={selected['objective']:.6f} "
        f"tokseq={selected['tokseq_text']!r}",
        flush=True,
    )
    return candidates, selected


@torch.no_grad()
def evaluate_dataset_nll_from_tokseq_ids(bundle, prompt_segments: dict, tokseq_token_ids: torch.Tensor, responses: list[str]) -> float:
    embedding_layer = bundle.model.get_input_embeddings()
    tokseq_embeds = embedding_layer(tokseq_token_ids.unsqueeze(0))
    prompt_embeds = build_prompt_embeds_with_tokseq(bundle, prompt_segments, tokseq_embeds)
    return float(teacher_forced_ce_from_prompt_embeds_batch(bundle, prompt_embeds, responses).mean().item())


def teacher_forced_ce_from_prompt_embeds(bundle, prompt_embeds: torch.Tensor, response_text: str) -> torch.Tensor:
    return teacher_forced_ce_from_prompt_embeds_batch(bundle, prompt_embeds, [response_text])[0]


def teacher_forced_ce_from_prompt_embeds_batch(
    bundle,
    prompt_embeds: torch.Tensor,
    response_texts: list[str],
) -> torch.Tensor:
    if not response_texts:
        raise ValueError("Need at least one response to score.")

    response_token_lists = [
        bundle.tokenizer.encode(response_text, add_special_tokens=False)
        for response_text in response_texts
    ]
    if any(len(token_ids) == 0 for token_ids in response_token_lists):
        raise ValueError("At least one response tokenized to zero tokens.")

    batch_size = len(response_token_lists)
    prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
    embedding_layer = bundle.model.get_input_embeddings()
    max_target_length = max(len(token_ids) for token_ids in response_token_lists)

    target_token_ids = torch.full(
        (batch_size, max_target_length),
        fill_value=bundle.tokenizer.pad_token_id,
        dtype=torch.long,
        device=bundle.device,
    )
    target_mask = torch.zeros((batch_size, max_target_length), dtype=torch.bool, device=bundle.device)
    for row_index, token_ids in enumerate(response_token_lists):
        row = torch.tensor(token_ids, dtype=torch.long, device=bundle.device)
        target_token_ids[row_index, : row.shape[0]] = row
        target_mask[row_index, : row.shape[0]] = True

    target_embeds = embedding_layer(target_token_ids)
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
    attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=bundle.device)
    outputs = bundle.model(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    prompt_length = prompt_embeds.shape[1]
    logits = outputs.logits[:, prompt_length - 1 : prompt_length + max_target_length - 1, :]
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target_token_ids.reshape(-1),
        reduction="none",
    ).view(batch_size, max_target_length)
    masked_token_losses = token_losses * target_mask.float()
    per_example_token_counts = target_mask.sum(dim=1).clamp_min(1)
    return masked_token_losses.sum(dim=1) / per_example_token_counts


@torch.no_grad()
def sample_tokseq_generations(
    bundle,
    prompt: str,
    tokseq_token_ids: torch.Tensor,
    tokseq_position: str,
    max_new_tokens: int,
) -> list[str]:
    input_ids = build_input_ids_with_tokseq_ids(bundle, prompt, tokseq_token_ids, tokseq_position)
    attention_mask = torch.ones_like(input_ids)
    samples = []
    for _ in range(3):
        generated = bundle.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            pad_token_id=bundle.tokenizer.pad_token_id,
        )
        new_tokens = generated[0][input_ids.shape[1] :]
        samples.append(bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return samples


def resolve_run_dir(output_root: str) -> Path:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"coach_reconstruction_{timestamp}"


if __name__ == "__main__":
    main()
