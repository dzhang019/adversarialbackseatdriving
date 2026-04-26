from __future__ import annotations

import json
from pathlib import Path

from . import qwen_vl, text_backend
from .env import load_project_env


def resolve_backend_name(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    lowered = model_name.lower()
    if "qwen" in lowered and "vl" in lowered:
        return "qwen_vl"
    return "causal_lm"


def load_bundle(model_name: str, backend: str, dtype: str, device_map: str):
    load_project_env()
    resolved = resolve_backend_name(model_name, backend)
    if resolved == "qwen_vl":
        return qwen_vl.load_qwen3_vl_bundle(model_name=model_name, dtype=dtype, device_map=device_map), resolved
    if resolved == "causal_lm":
        return text_backend.load_text_model_bundle(model_name=model_name, dtype=dtype, device_map=device_map), resolved
    raise ValueError(f"Unsupported backend: {backend}")


def generate_text(bundle, backend: str, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    if backend == "qwen_vl":
        return qwen_vl.generate_text(bundle, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
    return text_backend.generate_text(bundle, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)


def generate_text_with_steering(bundle, backend: str, prompt: str, layer: int, steering_vector, scale: float, max_new_tokens: int, do_sample: bool, temperature: float, all_tokens_steer: bool = True) -> str:
    if backend == "qwen_vl":
        return qwen_vl.generate_text_with_steering(bundle, prompt, layer=layer, steering_vector=steering_vector, scale=scale, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
    return text_backend.generate_text_with_steering(
        bundle,
        prompt,
        layer=layer,
        steering_vector=steering_vector,
        scale=scale,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        all_tokens_steer=all_tokens_steer,
    )


def generate_text_with_top_logits(bundle, backend: str, prompt: str, max_new_tokens: int, top_k: int, steering_vector=None, layer: int | None = None, scale: float = 0.0, all_tokens_steer: bool = True):
    if backend == "qwen_vl":
        raise ValueError("Top-logit tracing is currently only supported for the causal_lm backend.")
    return text_backend.generate_text_with_top_logits(
        bundle,
        prompt,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        steering_vector=steering_vector,
        layer=layer,
        scale=scale,
        all_tokens_steer=all_tokens_steer,
    )


def collect_last_token_residuals(bundle, backend: str, prompt: str, response: str):
    if backend == "qwen_vl":
        return qwen_vl.collect_last_token_residuals(bundle, prompt, response)
    return text_backend.collect_last_token_residuals(bundle, prompt, response)


def collect_user_story_last_token_residuals(bundle, backend: str, story: str):
    if backend == "qwen_vl":
        return qwen_vl.collect_user_story_last_token_residuals(bundle, story)
    return text_backend.collect_user_story_last_token_residuals(bundle, story)


def collect_user_story_mean_residuals(bundle, backend: str, story: str):
    if backend == "qwen_vl":
        return qwen_vl.collect_user_story_mean_residuals(bundle, story)
    return text_backend.collect_user_story_mean_residuals(bundle, story)


def load_concepts(path: str | Path) -> list[str]:
    return qwen_vl.load_concepts(path)


def append_jsonl(path: str | Path, row: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
