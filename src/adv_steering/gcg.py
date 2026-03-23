from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .data import ContrastiveExample
from .modeling import ModelBundle, steering_hook


@dataclass
class GCGStep:
    step: int
    loss: float
    prefix_text: str


@dataclass
class GCGResult:
    example_id: str
    prefix_token_ids: list[int]
    prefix_text: str
    loss: float
    trace: list[dict]


def optimize_prefix_for_example(
    bundle: ModelBundle,
    example: ContrastiveExample,
    steering_vector: torch.Tensor,
    layer: int,
    scale: float,
    prefix_length: int,
    steps: int,
    search_width: int = 32,
    forbidden_token_ids: set[int] | None = None,
) -> GCGResult:
    device = bundle.device
    tokenizer = bundle.tokenizer
    model = bundle.model

    if forbidden_token_ids is None:
        forbidden_token_ids = {tokenizer.pad_token_id}
        if tokenizer.bos_token_id is not None:
            forbidden_token_ids.add(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            forbidden_token_ids.add(tokenizer.eos_token_id)

    prefix_token_ids = torch.full(
        (prefix_length,),
        fill_value=_default_fill_token_id(tokenizer),
        dtype=torch.long,
        device=device,
    )

    prompt_ids = tokenizer(example.attack_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    target_ids = tokenizer(example.target_completion, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    embedding_layer = model.get_input_embeddings()
    vocab_matrix = embedding_layer.weight.detach()

    trace: list[dict] = []
    best_loss = float("inf")
    best_prefix = prefix_token_ids.clone()

    for step_index in range(steps):
        model.zero_grad(set_to_none=True)
        current_loss, grad = _loss_and_prefix_grad(
            bundle=bundle,
            prefix_token_ids=prefix_token_ids,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
            steering_vector=steering_vector.to(device),
            layer=layer,
            scale=scale,
        )

        if current_loss < best_loss:
            best_loss = current_loss
            best_prefix = prefix_token_ids.clone()

        coordinate = step_index % prefix_length
        scores = torch.matmul(vocab_matrix, -grad[coordinate])
        candidate_ids = torch.topk(scores, k=min(search_width, scores.shape[0])).indices.tolist()

        candidate_best_loss = current_loss
        candidate_best_prefix = prefix_token_ids.clone()

        for candidate_id in candidate_ids:
            if candidate_id in forbidden_token_ids:
                continue
            trial = prefix_token_ids.clone()
            trial[coordinate] = candidate_id
            trial_loss = _teacher_forced_loss(
                bundle=bundle,
                prefix_token_ids=trial,
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                steering_vector=steering_vector.to(device),
                layer=layer,
                scale=scale,
            )
            if trial_loss < candidate_best_loss:
                candidate_best_loss = trial_loss
                candidate_best_prefix = trial

        prefix_token_ids = candidate_best_prefix
        if candidate_best_loss < best_loss:
            best_loss = candidate_best_loss
            best_prefix = candidate_best_prefix.clone()
        trace.append(
            asdict(
                GCGStep(
                    step=step_index,
                    loss=float(candidate_best_loss),
                    prefix_text=tokenizer.decode(prefix_token_ids, skip_special_tokens=True),
                )
            )
        )

    final_prefix = best_prefix
    return GCGResult(
        example_id=example.id,
        prefix_token_ids=final_prefix.tolist(),
        prefix_text=tokenizer.decode(final_prefix, skip_special_tokens=True),
        loss=float(best_loss),
        trace=trace,
    )


def _loss_and_prefix_grad(
    bundle: ModelBundle,
    prefix_token_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    scale: float,
) -> tuple[float, torch.Tensor]:
    model = bundle.model
    embedding_layer = model.get_input_embeddings()
    prefix_embeds = embedding_layer(prefix_token_ids.unsqueeze(0)).detach().clone().requires_grad_(True)
    prompt_embeds = embedding_layer(prompt_ids.unsqueeze(0)).detach()
    target_embeds = embedding_layer(target_ids.unsqueeze(0)).detach()

    input_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)
    labels = torch.full((1, input_embeds.shape[1]), -100, dtype=torch.long, device=bundle.device)
    labels[:, prefix_token_ids.shape[0] + prompt_ids.shape[0] :] = target_ids.unsqueeze(0)

    with steering_hook(model, layer, steering_vector, scale):
        outputs = model(inputs_embeds=input_embeds, labels=labels, use_cache=False)
        loss = outputs.loss

    loss.backward()
    grad = prefix_embeds.grad[0].detach()
    return float(loss.detach().item()), grad


@torch.no_grad()
def _teacher_forced_loss(
    bundle: ModelBundle,
    prefix_token_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer: int,
    scale: float,
) -> float:
    model = bundle.model
    embedding_layer = model.get_input_embeddings()
    prefix_embeds = embedding_layer(prefix_token_ids.unsqueeze(0))
    prompt_embeds = embedding_layer(prompt_ids.unsqueeze(0))
    target_embeds = embedding_layer(target_ids.unsqueeze(0))

    input_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)
    labels = torch.full((1, input_embeds.shape[1]), -100, dtype=torch.long, device=bundle.device)
    labels[:, prefix_token_ids.shape[0] + prompt_ids.shape[0] :] = target_ids.unsqueeze(0)

    with steering_hook(model, layer, steering_vector, scale):
        outputs = model(inputs_embeds=input_embeds, labels=labels, use_cache=False)
    return float(outputs.loss.detach().item())


def _default_fill_token_id(tokenizer) -> int:
    for token in [" the", "quiet", ".", ","]:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
    return 0
