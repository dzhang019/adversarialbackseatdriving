from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


def load_model_bundle(model_name: str, dtype: str = "auto", device: str = "auto") -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)

    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()

    return ModelBundle(tokenizer=tokenizer, model=model, device=resolved_device)


def get_transformer_layers(model: AutoModelForCausalLM):
    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
    ]
    for _, layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("Unsupported architecture: could not locate transformer layers.")


def encode_text(bundle: ModelBundle, text: str) -> dict[str, torch.Tensor]:
    encoded = bundle.tokenizer(text, return_tensors="pt")
    return {key: value.to(bundle.device) for key, value in encoded.items()}


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


@contextmanager
def steering_hook(model, layer_index: int, steering_vector: torch.Tensor, scale: float) -> Iterator[None]:
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
