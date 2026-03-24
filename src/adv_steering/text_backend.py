from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import get_hf_token, load_project_env


@dataclass
class TextModelBundle:
    tokenizer: Any
    model: AutoModelForCausalLM
    device: torch.device


def load_text_model_bundle(
    model_name: str,
    dtype: str = "auto",
    device_map: str | None = "auto",
) -> TextModelBundle:
    load_project_env()
    token = get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=token,
    )
    model.eval()
    device = next(model.parameters()).device
    return TextModelBundle(tokenizer=tokenizer, model=model, device=device)


def generate_text(
    bundle: TextModelBundle,
    prompt: str,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    encoded = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    generation_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": bundle.tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    generated = bundle.model.generate(**generation_kwargs)
    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_text_with_steering(
    bundle: TextModelBundle,
    prompt: str,
    layer: int,
    steering_vector: torch.Tensor,
    scale: float,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    with steering_hook(bundle.model, layer, steering_vector.to(bundle.device), scale):
        return generate_text(
            bundle=bundle,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )


@torch.no_grad()
def collect_last_token_residuals(
    bundle: TextModelBundle,
    prompt: str,
    response: str,
) -> torch.Tensor:
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    outputs = bundle.model(
        input_ids=prompt_inputs["input_ids"],
        attention_mask=prompt_inputs["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    # Use the final assistant-prefill token state, which predicts the first assistant response token.
    last_index = prompt_inputs["input_ids"].shape[1] - 1
    return torch.stack(
        [hidden_state[0, last_index].detach().cpu() for hidden_state in outputs.hidden_states[1:]],
        dim=0,
    )


@torch.no_grad()
def collect_prompt_last_token_residuals(
    bundle: TextModelBundle,
    prompt: str,
) -> torch.Tensor:
    prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    outputs = bundle.model(
        input_ids=prompt_inputs["input_ids"],
        attention_mask=prompt_inputs["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    last_index = prompt_inputs["input_ids"].shape[1] - 1
    return torch.stack(
        [hidden_state[0, last_index].detach().cpu() for hidden_state in outputs.hidden_states[1:]],
        dim=0,
    )


def encode_chat_prompt(
    bundle: TextModelBundle,
    prompt: str,
    add_generation_prompt: bool = True,
) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(bundle.tokenizer, "apply_chat_template"):
        encoded = bundle.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        return {key: value.to(bundle.device) for key, value in encoded.items()}

    encoded = bundle.tokenizer(prompt, return_tensors="pt")
    return {key: value.to(bundle.device) for key, value in encoded.items()}


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


def get_transformer_layers(model: AutoModelForCausalLM):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("Unsupported text architecture: could not locate transformer layers.")


@contextmanager
def steering_hook(model, layer_index: int, steering_vector: torch.Tensor, scale: float):
    layers = get_transformer_layers(model)
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
