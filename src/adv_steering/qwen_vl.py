from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .env import get_hf_token, load_project_env


@dataclass
class QwenVLBundle:
    processor: AutoProcessor
    tokenizer: Any
    model: Qwen3VLForConditionalGeneration
    device: torch.device


def load_qwen3_vl_bundle(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    dtype: str = "auto",
    device_map: str | None = "auto",
) -> QwenVLBundle:
    load_project_env()
    token = get_hf_token()
    processor = AutoProcessor.from_pretrained(model_name, token=token)
    torch_dtype = _resolve_dtype(dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=token,
    )
    model.eval()

    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return QwenVLBundle(processor=processor, tokenizer=tokenizer, model=model, device=device)


def generate_text(
    bundle: QwenVLBundle,
    prompt: str,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    return _generate_from_messages(
        bundle=bundle,
        messages=[_user_message(prompt)],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )


def generate_text_with_steering(
    bundle: QwenVLBundle,
    prompt: str,
    layer: int,
    steering_vector: torch.Tensor,
    scale: float,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    with steering_hook(bundle.model, layer, steering_vector.to(bundle.device), scale):
        return _generate_from_messages(
            bundle=bundle,
            messages=[_user_message(prompt)],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )


def _generate_from_messages(
    bundle: QwenVLBundle,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    inputs = bundle.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = _move_batch_to_device(inputs, bundle.device)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": bundle.tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    generated_ids = bundle.model.generate(**generation_kwargs)
    trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
    decoded = bundle.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


@torch.no_grad()
def collect_last_token_residuals(
    bundle: QwenVLBundle,
    prompt: str,
    response: str,
) -> torch.Tensor:
    prompt_inputs = bundle.processor.apply_chat_template(
        [_user_message(prompt)],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_inputs = _move_batch_to_device(prompt_inputs, bundle.device)

    response_ids = bundle.tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(bundle.device)

    input_ids = torch.cat([prompt_inputs["input_ids"], response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=bundle.device)

    outputs = bundle.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    last_index = input_ids.shape[1] - 1
    return torch.stack(
        [hidden_state[0, last_index].detach().cpu() for hidden_state in outputs.hidden_states[1:]],
        dim=0,
    )


def load_concepts(path: str | Path) -> list[str]:
    concepts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            concept = raw_line.strip()
            if concept:
                concepts.append(concept)
    if not concepts:
        raise ValueError(f"No concepts found in {path}")
    return concepts


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _user_message(prompt: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_qwen_transformer_layers(model: Qwen3VLForConditionalGeneration):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(getattr(model, "language_model", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", candidate)
        if layers is not None:
            return layers
    raise ValueError("Unsupported Qwen3-VL architecture: could not locate transformer layers.")


@contextmanager
def steering_hook(model, layer_index: int, steering_vector: torch.Tensor, scale: float):
    layers = get_qwen_transformer_layers(model)
    layer = layers[layer_index]
    vector = steering_vector.to(next(model.parameters()).device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            updated = hidden_states + (scale * vector).view(1, 1, -1)
            return (updated, *output[1:])
        return output + (scale * vector).view(1, 1, -1)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
