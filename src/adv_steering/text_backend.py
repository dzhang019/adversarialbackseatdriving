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
    all_tokens_steer: bool = True,
) -> str:
    with steering_hook(bundle.model, layer, steering_vector.to(bundle.device), scale, all_tokens=all_tokens_steer):
        return generate_text(
            bundle=bundle,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )


@torch.no_grad()
def generate_text_with_top_logits(
    bundle: TextModelBundle,
    prompt: str,
    max_new_tokens: int = 64,
    top_k: int = 10,
    steering_vector: torch.Tensor | None = None,
    layer: int | None = None,
    scale: float = 0.0,
    all_tokens_steer: bool = True,
) -> dict[str, Any]:
    encoded = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    generated_token_ids: list[int] = []
    trace: list[dict[str, Any]] = []

    context = (
        steering_hook(bundle.model, layer, steering_vector.to(bundle.device), scale, all_tokens=all_tokens_steer)
        if steering_vector is not None and layer is not None
        else null_hook()
    )
    with context:
        for position in range(max_new_tokens):
            outputs = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[0, -1]
            top_values, top_indices = torch.topk(logits.float(), k=min(top_k, logits.shape[-1]))
            top_probabilities = torch.softmax(top_values, dim=-1)
            next_token_id = int(torch.argmax(logits).item())
            generated_token_ids.append(next_token_id)
            trace.append(
                {
                    "position": position,
                    "generated_token_id": next_token_id,
                    "generated_token_text": bundle.tokenizer.decode([next_token_id], skip_special_tokens=False),
                    "generated_token_in_top_k": bool((top_indices == next_token_id).any().item()),
                    "top_logits": [
                        {
                            "rank": rank + 1,
                            "token_id": int(token_id.item()),
                            "token_text": bundle.tokenizer.decode([int(token_id.item())], skip_special_tokens=False),
                            "logit": float(logit.item()),
                            "probability": float(probability.item()),
                            "is_generated": int(token_id.item()) == next_token_id,
                        }
                        for rank, (token_id, logit, probability) in enumerate(zip(top_indices, top_values, top_probabilities))
                    ],
                }
            )
            next_token_tensor = torch.tensor([[next_token_id]], device=bundle.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=bundle.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            if next_token_id == bundle.tokenizer.eos_token_id:
                break

    return {
        "text": bundle.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip(),
        "token_trace": trace,
    }


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
    if hasattr(bundle.tokenizer, "apply_chat_template") and getattr(bundle.tokenizer, "chat_template", None):
        encoded = bundle.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        return {key: value.to(bundle.device) for key, value in encoded.items()}

    fallback_prompt = _fallback_chat_prompt_text(bundle.tokenizer, prompt, add_generation_prompt)
    encoded = bundle.tokenizer(fallback_prompt, return_tensors="pt")
    return {key: value.to(bundle.device) for key, value in encoded.items()}


def split_chat_prompt_for_user_suffix(
    bundle: TextModelBundle,
    prompt: str,
) -> dict[str, torch.Tensor]:
    prompt_without_generation = encode_chat_prompt(bundle, prompt, add_generation_prompt=False)
    prompt_with_generation = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)

    prefix_ids = prompt_without_generation["input_ids"]
    full_ids = prompt_with_generation["input_ids"]
    prefix_attention_mask = prompt_without_generation["attention_mask"]
    full_attention_mask = prompt_with_generation["attention_mask"]

    prefix_length = prefix_ids.shape[1]
    if full_ids.shape[1] < prefix_length or not torch.equal(full_ids[:, :prefix_length], prefix_ids):
        raise ValueError(
            "Could not split the chat template into user-prefix and assistant-generation suffix. "
            "This tokenizer's generation prompt is not a simple token suffix."
        )

    return {
        "user_input_ids": prefix_ids,
        "user_attention_mask": prefix_attention_mask,
        "assistant_prefix_ids": full_ids[:, prefix_length:],
        "assistant_prefix_attention_mask": full_attention_mask[:, prefix_length:],
    }


def _fallback_chat_prompt_text(tokenizer, prompt: str, add_generation_prompt: bool) -> str:
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "llama-2" in name and "chat" in name:
        return f"[INST] {prompt.strip()} [/INST]"
    return prompt


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
def steering_hook(model, layer_index: int, steering_vector: torch.Tensor, scale: float, all_tokens: bool = True):
    layers = get_transformer_layers(model)
    layer = layers[layer_index]
    vector = steering_vector.to(next(model.parameters()).device)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            typed_vector = vector.to(hidden_states.dtype)
            if all_tokens:
                updated = hidden_states + (scale * typed_vector).view(1, 1, -1)
            else:
                updated = hidden_states.clone()
                updated[:, -1, :] = hidden_states[:, -1, :] + (scale * typed_vector)
            return (updated, *output[1:])
        typed_vector = vector.to(output.dtype)
        if all_tokens:
            return output + (scale * typed_vector).view(1, 1, -1)
        updated = output.clone()
        updated[:, -1, :] = output[:, -1, :] + (scale * typed_vector)
        return updated

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def null_hook():
    yield
