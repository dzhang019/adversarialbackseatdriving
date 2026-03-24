from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import append_jsonl, generate_text, load_bundle, load_concepts, read_jsonl
else:
    from .backend import append_jsonl, generate_text, load_bundle, load_concepts, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Generate positive-concept and negative-concept examples for a list of concepts.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="auto", choices=["auto", "qwen_vl", "causal_lm"], help="Inference backend.")
    parser.add_argument("--concepts", default="data/concepts_200.txt", help="Path to newline-delimited concepts.")
    parser.add_argument("--output", default="data/llama_happy_sad_corpus.jsonl", help="Where to save generated rows.")
    parser.add_argument("--poscon-label", default="happy", help="Positive concept label.")
    parser.add_argument("--negcon-label", default="sad", help="Negative concept label.")
    parser.add_argument("--mode", default="story", choices=["story", "statement"], help="Prompt style to generate.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum number of generated tokens.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for more diverse generations.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file instead of resuming.")
    return parser.parse_args()


def generate_corpus(
    model: str,
    backend: str,
    concepts_path: str,
    output_path: str,
    poscon_label: str = "happy",
    negcon_label: str = "sad",
    mode: str = "story",
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 48,
    do_sample: bool = False,
    temperature: float = 1.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    concepts = load_concepts(concepts_path)
    output_path_obj = Path(output_path)

    existing_concepts = set()
    if output_path_obj.exists() and not overwrite:
        for row in read_jsonl(output_path_obj):
            existing_concepts.add(row["concept"])
    elif output_path_obj.exists() and overwrite:
        output_path_obj.unlink()

    bundle, resolved_backend = load_bundle(model_name=model, backend=backend, dtype=dtype, device_map=device_map)

    generated_count = 0
    for concept in tqdm(concepts, desc=f"Generating {poscon_label}/{negcon_label} pairs"):
        if concept in existing_concepts:
            continue

        poscon_prompt, negcon_prompt = build_prompts(
            concept=concept,
            poscon_label=poscon_label,
            negcon_label=negcon_label,
            mode=mode,
        )

        poscon_response = generate_text(
            bundle,
            resolved_backend,
            poscon_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        negcon_response = generate_text(
            bundle,
            resolved_backend,
            negcon_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        append_jsonl(
            output_path_obj,
            {
                "concept": concept,
                "poscon_label": poscon_label,
                "negcon_label": negcon_label,
                "mode": mode,
                "poscon_prompt": poscon_prompt,
                "poscon_response": poscon_response,
                "negcon_prompt": negcon_prompt,
                "negcon_response": negcon_response,
                "positive_label": poscon_label,
                "negative_label": negcon_label,
                "positive_prompt": poscon_prompt,
                "positive_response": poscon_response,
                "negative_prompt": negcon_prompt,
                "negative_response": negcon_response,
                "truth_prompt": poscon_prompt,
                "truth_response": poscon_response,
                "lie_prompt": negcon_prompt,
                "lie_response": negcon_response,
                "attack_prompt": poscon_prompt,
                "target_completion": default_target_completion(negcon_label, concept, mode),
            },
        )
        generated_count += 1

    return {
        "model": model,
        "backend": resolved_backend,
        "concepts_path": concepts_path,
        "output_path": str(output_path_obj),
        "poscon_label": poscon_label,
        "negcon_label": negcon_label,
        "mode": mode,
        "total_concepts": len(concepts),
        "generated_count": generated_count,
        "existing_count": len(existing_concepts),
    }


def build_prompts(concept: str, poscon_label: str, negcon_label: str, mode: str) -> tuple[str, str]:
    if mode == "story":
        return (
            f"Please tell me a {poscon_label} story about {concept}.",
            f"Please tell me a {negcon_label} story about {concept}.",
        )
    return (
        f"Please tell me something {poscon_label} about {concept}.",
        f"Please tell me something {negcon_label} about {concept}.",
    )


def default_target_completion(negcon_label: str, concept: str, mode: str) -> str:
    noun = "story" if mode == "story" else "response"
    article = "an" if negcon_label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"Sure, here is {article} {negcon_label} {noun} about {concept}."


def main():
    args = parse_args()
    generate_corpus(
        model=args.model,
        backend=args.backend,
        concepts_path=args.concepts,
        output_path=args.output,
        poscon_label=args.poscon_label,
        negcon_label=args.negcon_label,
        mode=args.mode,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
