from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .data import ContrastiveExample
from .metrics import lie_affirmation_score, truth_affirmation_score
from .modeling import ModelBundle, encode_text, steering_hook


@dataclass
class EvaluationRecord:
    id: str
    prompt: str
    baseline_text: str
    steered_text: str
    baseline_truth_affirmation_score: float
    steered_truth_affirmation_score: float
    baseline_lie_affirmation_score: float
    steered_lie_affirmation_score: float


@torch.no_grad()
def evaluate_vector(
    bundle: ModelBundle,
    examples: list[ContrastiveExample],
    steering_vector: torch.Tensor,
    layer: int,
    scale: float,
    max_new_tokens: int,
) -> list[dict]:
    results: list[dict] = []
    for example in examples:
        prompt = example.evaluation_prompt
        baseline_text = _generate(bundle, prompt, max_new_tokens)
        steered_text = _generate(
            bundle,
            prompt,
            max_new_tokens,
            steering_vector=steering_vector,
            layer=layer,
            scale=scale,
        )
        record = EvaluationRecord(
            id=example.id,
            prompt=prompt,
            baseline_text=baseline_text,
            steered_text=steered_text,
            baseline_truth_affirmation_score=truth_affirmation_score(baseline_text),
            steered_truth_affirmation_score=truth_affirmation_score(steered_text),
            baseline_lie_affirmation_score=lie_affirmation_score(baseline_text),
            steered_lie_affirmation_score=lie_affirmation_score(steered_text),
        )
        results.append(asdict(record))
    return results


@torch.no_grad()
def _generate(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int,
    steering_vector: torch.Tensor | None = None,
    layer: int | None = None,
    scale: float = 0.0,
) -> str:
    encoded = encode_text(bundle, prompt)
    generation_kwargs = dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=bundle.tokenizer.pad_token_id,
    )
    if steering_vector is None:
        output_tokens = bundle.model.generate(**generation_kwargs)
    else:
        if layer is None:
            raise ValueError("layer must be provided when steering is enabled")
        with steering_hook(bundle.model, layer, steering_vector.to(bundle.device), scale):
            output_tokens = bundle.model.generate(**generation_kwargs)
    return bundle.tokenizer.decode(
        output_tokens[0][encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
