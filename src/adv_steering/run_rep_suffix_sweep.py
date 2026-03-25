from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.rep_suffix_attack import (
        _aggregate_cosine_similarity,
        _aggregate_dot_product,
        _average_next_token_kl,
        _default_fill_token_id,
        load_prompt_pairs,
        load_steering_vector,
        optimize_suffix_against_direction,
    )
    from adv_steering.text_backend import encode_chat_prompt, load_text_model_bundle
else:
    from .rep_suffix_attack import (
        _aggregate_cosine_similarity,
        _aggregate_dot_product,
        _average_next_token_kl,
        _default_fill_token_id,
        load_prompt_pairs,
        load_steering_vector,
        optimize_suffix_against_direction,
    )
    from .text_backend import encode_chat_prompt, load_text_model_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="Run a controlled suffix-attack sweep.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Causal LM model name or local path.")
    parser.add_argument("--steering-file", required=True, help="Path to poscon_negcon_residuals.pt or steering_candidates.pt.")
    parser.add_argument("--layer", type=int, required=True, help="Layer to use for the steering vector and prompt-state objective.")
    parser.add_argument("--n-plus", default="", help="Positive-concept prompt n+.")
    parser.add_argument("--n-minus", default="", help="Negative-concept prompt n-.")
    parser.add_argument("--prompt-pairs-file", default="", help="Optional JSONL file with prompt pairs.")
    parser.add_argument("--max-pairs", type=int, default=1, help="How many prompt pairs to include from the source file.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--suffix-lengths", default="10,20,40", help="Comma-separated suffix lengths to test.")
    parser.add_argument("--init-mode", default="random", choices=["random", "fill"], help="How to initialize the optimized suffix.")
    parser.add_argument("--objectives", default="dot,cosine", help="Comma-separated objective types to test.")
    parser.add_argument("--steps-list", default="100,500", help="Comma-separated step counts to test.")
    parser.add_argument("--top-k-list", default="64,256", help="Comma-separated top-k values to test.")
    parser.add_argument("--batch-size-list", default="128,512", help="Comma-separated batch sizes to test.")
    parser.add_argument("--restarts", type=int, default=5, help="How many random restarts per configuration.")
    parser.add_argument("--success-threshold", type=float, default=0.0, help="Passed through to the optimizer.")
    parser.add_argument("--kl-interval", type=int, default=50, help="Passed through to the optimizer.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for restart generation.")
    parser.add_argument("--output-dir", default="runs", help="Directory under which to create the sweep run folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_pairs = load_prompt_pairs(args.prompt_pairs_file, args.n_plus, args.n_minus)
    if args.max_pairs > 0:
        prompt_pairs = prompt_pairs[: args.max_pairs]
    if not prompt_pairs:
        raise ValueError("No prompt pairs available for the sweep.")

    bundle = load_text_model_bundle(args.model, dtype=args.dtype, device_map=args.device_map)
    steering_vector = load_steering_vector(args.steering_file, args.layer)
    steering_vector_device = steering_vector.to(bundle.device)
    prompt_pair_ids = [
        (
            encode_chat_prompt(bundle, n_plus, add_generation_prompt=True)["input_ids"][0],
            encode_chat_prompt(bundle, n_minus, add_generation_prompt=True)["input_ids"][0],
        )
        for n_plus, n_minus in prompt_pairs
    ]

    run_dir = Path(args.output_dir) / datetime.now().strftime("rep_suffix_sweep_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    config_grid = build_config_grid(
        suffix_lengths=parse_int_csv(args.suffix_lengths),
        objectives=parse_str_csv(args.objectives),
        steps_list=parse_int_csv(args.steps_list),
        top_k_list=parse_int_csv(args.top_k_list),
        batch_size_list=parse_int_csv(args.batch_size_list),
        restarts=args.restarts,
        base_seed=args.seed,
    )

    rows = []
    for config_index, config in enumerate(config_grid, start=1):
        print(
            f"[config {config_index}/{len(config_grid)}] "
            f"objective={config['objective_type']} "
            f"suffix_length={config['suffix_length']} "
            f"steps={config['steps']} "
            f"top_k={config['top_k']} "
            f"batch_size={config['batch_size']} "
            f"restart={config['restart_index']} "
            f"seed={config['seed']}",
            flush=True,
        )

        set_all_seeds(config["seed"])
        result = optimize_suffix_against_direction(
            bundle=bundle,
            prompt_pairs=prompt_pairs,
            steering_vector=steering_vector,
            layer=args.layer,
            suffix_length=config["suffix_length"],
            steps=config["steps"],
            top_k=config["top_k"],
            batch_size=config["batch_size"],
            objective_type=config["objective_type"],
            success_threshold=args.success_threshold,
            kl_interval=args.kl_interval,
            init_mode=args.init_mode,
        )

        suffix_token_ids = torch.tensor(result.suffix_token_ids, dtype=torch.long, device=bundle.device)
        baseline_suffix_token_ids = torch.full(
            (config["suffix_length"],),
            fill_value=_default_fill_token_id(bundle.tokenizer),
            dtype=torch.long,
            device=bundle.device,
        )

        final_dot = _aggregate_dot_product(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector_device,
            layer=args.layer,
        )
        final_cosine = _aggregate_cosine_similarity(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
            steering_vector=steering_vector_device,
            layer=args.layer,
        )
        final_kl = _average_next_token_kl(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=suffix_token_ids,
        )
        baseline_dot = _aggregate_dot_product(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=baseline_suffix_token_ids,
            steering_vector=steering_vector_device,
            layer=args.layer,
        )
        baseline_cosine = _aggregate_cosine_similarity(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=baseline_suffix_token_ids,
            steering_vector=steering_vector_device,
            layer=args.layer,
        )
        baseline_kl = _average_next_token_kl(
            bundle=bundle,
            prompt_pair_ids=prompt_pair_ids,
            suffix_token_ids=baseline_suffix_token_ids,
        )
        best_trace_cosine = min(step["cosine_similarity"] for step in result.trace) if result.trace else float("inf")
        best_trace_objective = min(step["objective"] for step in result.trace) if result.trace else float("inf")

        artifact_path = run_dir / artifact_name_for_config(config)
        artifact_payload = {
            "model": args.model,
            "layer": args.layer,
            "prompt_pairs": [{"n_plus": plus, "n_minus": minus} for plus, minus in prompt_pairs],
            "config": config,
            "init_mode": args.init_mode,
            "baseline": {
                "dot_product": baseline_dot,
                "cosine_similarity": baseline_cosine,
                "next_token_kl": baseline_kl,
                "suffix_text": bundle.tokenizer.decode(baseline_suffix_token_ids, skip_special_tokens=True),
            },
            "result": {
                "layer": result.layer,
                "objective_type": result.objective_type,
                "suffix_token_ids": result.suffix_token_ids,
                "suffix_text": result.suffix_text,
                "objective": result.objective,
                "trace": result.trace,
                "final_dot_product": final_dot,
                "final_cosine_similarity": final_cosine,
                "final_next_token_kl": final_kl,
                "best_trace_cosine": best_trace_cosine,
                "best_trace_objective": best_trace_objective,
            },
        }
        artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

        rows.append(
            {
                "artifact_path": str(artifact_path),
                "objective_type": config["objective_type"],
                "suffix_length": config["suffix_length"],
                "steps": config["steps"],
                "top_k": config["top_k"],
                "batch_size": config["batch_size"],
                "restart_index": config["restart_index"],
                "seed": config["seed"],
                "num_prompt_pairs": len(prompt_pairs),
                "init_mode": args.init_mode,
                "baseline_dot_product": baseline_dot,
                "baseline_cosine_similarity": baseline_cosine,
                "baseline_next_token_kl": baseline_kl,
                "final_objective": result.objective,
                "final_dot_product": final_dot,
                "final_cosine_similarity": final_cosine,
                "final_next_token_kl": final_kl,
                "best_trace_objective": best_trace_objective,
                "best_trace_cosine": best_trace_cosine,
                "suffix_text": result.suffix_text,
            }
        )

    leaderboard = sorted(rows, key=lambda row: (row["final_cosine_similarity"], row["final_objective"]))
    summary = {
        "model": args.model,
        "layer": args.layer,
        "steering_file": args.steering_file,
        "init_mode": args.init_mode,
        "prompt_pairs": [{"n_plus": plus, "n_minus": minus} for plus, minus in prompt_pairs],
        "num_runs": len(rows),
        "runs": rows,
        "leaderboard_by_final_cosine": leaderboard,
    }
    (run_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(leaderboard[: min(10, len(leaderboard))], indent=2))


def artifact_name_for_config(config: dict) -> str:
    return (
        f"objective_{config['objective_type']}"
        f"_len_{config['suffix_length']}"
        f"_steps_{config['steps']}"
        f"_topk_{config['top_k']}"
        f"_batch_{config['batch_size']}"
        f"_restart_{config['restart_index']}.json"
    )


def build_config_grid(
    suffix_lengths: list[int],
    objectives: list[str],
    steps_list: list[int],
    top_k_list: list[int],
    batch_size_list: list[int],
    restarts: int,
    base_seed: int,
) -> list[dict]:
    grid = []
    seed_counter = 0
    for objective_type in objectives:
        for suffix_length in suffix_lengths:
            for steps in steps_list:
                for top_k in top_k_list:
                    for batch_size in batch_size_list:
                        for restart_index in range(restarts):
                            grid.append(
                                {
                                    "objective_type": objective_type,
                                    "suffix_length": suffix_length,
                                    "steps": steps,
                                    "top_k": top_k,
                                    "batch_size": batch_size,
                                    "restart_index": restart_index,
                                    "seed": base_seed + seed_counter,
                                }
                            )
                            seed_counter += 1
    return grid


def parse_int_csv(raw_value: str) -> list[int]:
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_str_csv(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
