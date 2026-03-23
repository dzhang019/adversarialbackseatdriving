from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import ContrastiveExample
from .modeling import ModelBundle, encode_text


@dataclass
class SteeringVectorResult:
    vector: torch.Tensor
    layer: int
    token_index: int
    example_vectors: list[torch.Tensor]


@torch.no_grad()
def estimate_steering_vector(
    bundle: ModelBundle,
    examples: list[ContrastiveExample],
    layer: int,
    token_index: int = -1,
) -> SteeringVectorResult:
    model = bundle.model
    example_vectors: list[torch.Tensor] = []

    for example in examples:
        positive = _hidden_state_at_token(bundle, example.positive_prompt, layer, token_index)
        negative = _hidden_state_at_token(bundle, example.negative_prompt, layer, token_index)
        example_vectors.append(positive - negative)

    stacked = torch.stack(example_vectors, dim=0)
    vector = stacked.mean(dim=0)
    vector = vector / vector.norm().clamp_min(1e-8)
    return SteeringVectorResult(
        vector=vector.detach().cpu(),
        layer=layer,
        token_index=token_index,
        example_vectors=[item.detach().cpu() for item in example_vectors],
    )


@torch.no_grad()
def _hidden_state_at_token(
    bundle: ModelBundle,
    text: str,
    layer: int,
    token_index: int,
) -> torch.Tensor:
    encoded = encode_text(bundle, text)
    outputs = bundle.model(**encoded, output_hidden_states=True, use_cache=False)
    hidden_state = outputs.hidden_states[layer + 1][0]
    index = token_index if token_index >= 0 else hidden_state.shape[0] + token_index
    return hidden_state[index].detach()
