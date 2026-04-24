from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import load_bundle, read_jsonl
    from adv_steering.text_backend import encode_chat_prompt, split_chat_prompt_for_user_suffix
else:
    from .backend import load_bundle, read_jsonl
    from .text_backend import encode_chat_prompt, split_chat_prompt_for_user_suffix


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize token-level conceptness using a steering vector as a linear probe.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="causal_lm", choices=["auto", "causal_lm"], help="Only causal_lm is supported.")
    parser.add_argument("--residual-file", required=True, help="Path to poscon_negcon_residuals.pt.")
    parser.add_argument("--layer", type=int, default=None, help="Layer/probe to use. Defaults to best_layer in the residual file.")
    parser.add_argument("--dataset", default="data/llama_happy_sad_corpus.jsonl", help="JSONL corpus with positive/negative prompts and responses.")
    parser.add_argument("--max-examples", type=int, default=6, help="Maximum corpus rows to visualize.")
    parser.add_argument("--story-side", default="both", choices=["positive", "negative", "both"], help="Which responses from the corpus to visualize.")
    parser.add_argument("--prompt", default="", help="Optional single prompt for custom visualization.")
    parser.add_argument("--story", default="", help="Optional single story/response for custom visualization.")
    parser.add_argument("--suffix", default="", help="Optional suffix artifact JSON. If provided, baseline and suffix-applied views are shown.")
    parser.add_argument("--soft-prompt", default="", help="Optional continuous soft suffix artifact from rep_suffix_attack_soft_prompt.py.")
    parser.add_argument("--soft-prefix", "--prefix", default="", help="Optional continuous soft prefix artifact from rep_prefix_attack_soft_prompt.py.")
    parser.add_argument("--step", type=int, default=-1, help="Optimization step to load from a soft prompt/prefix trace. Defaults to final.")
    parser.add_argument("--exact-suffix-ids", action="store_true", help="Use suffix_token_ids from the suffix artifact instead of re-tokenizing suffix_text.")
    parser.add_argument("--score-metric", default="dot", choices=["cosine", "dot"], help="Metric used for coloring and threshold hits.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Mark tokens whose selected score passes this threshold.")
    parser.add_argument("--threshold-mode", default="absolute", choices=["absolute", "positive"], help="Whether thresholding uses abs(score) or score >= threshold.")
    parser.add_argument("--color-scale", type=float, default=1.0, help="Score magnitude mapped to full red/green saturation.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <residual-run>/token_conceptness.")
    parser.add_argument("--output-html", default="", help="Optional explicit HTML output path.")
    parser.add_argument("--output-json", default="", help="Optional explicit JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    residual_payload = torch.load(args.residual_file, map_location="cpu")
    layer = args.layer if args.layer is not None else int(residual_payload["best_layer"])
    steering_vector = load_steering_vector(args.residual_file, layer)
    vector_norm = float(steering_vector.norm().item())

    bundle, resolved_backend = load_bundle(args.model, args.backend, args.dtype, args.device_map)
    if resolved_backend != "causal_lm":
        raise ValueError("visualize_token_conceptness.py currently supports only causal_lm.")
    attack_mode_count = int(bool(args.suffix)) + int(bool(args.soft_prompt)) + int(bool(args.soft_prefix))
    if attack_mode_count > 1:
        raise ValueError("Use at most one of --suffix, --soft-prompt, or --soft-prefix.")

    suffix_info = load_suffix_artifact(args.suffix, args.exact_suffix_ids) if args.suffix else None
    soft_prompt_matrix = load_soft_matrix_artifact(args.soft_prompt, args.step, "soft_prompt") if args.soft_prompt else None
    soft_prefix_matrix = load_soft_matrix_artifact(args.soft_prefix, args.step, "soft_prefix") if args.soft_prefix else None
    examples = load_examples(args)
    if not examples:
        raise ValueError("No examples to visualize.")

    analyzed_examples = []
    for example_index, example in enumerate(examples):
        variants = [
            analyze_variant(
                bundle=bundle,
                prompt=example["prompt"],
                story=example["story"],
                layer=layer,
                steering_vector=steering_vector.to(bundle.device),
                variant_name="baseline",
                suffix_text="",
                suffix_token_ids=None,
                soft_prompt_matrix=None,
                soft_prefix_matrix=None,
            )
        ]
        if suffix_info is not None:
            variants.append(
                analyze_variant(
                    bundle=bundle,
                    prompt=example["prompt"],
                    story=example["story"],
                    layer=layer,
                    steering_vector=steering_vector.to(bundle.device),
                    variant_name="suffix",
                    suffix_text=suffix_info["suffix_text"],
                    suffix_token_ids=suffix_info["suffix_token_ids"],
                    soft_prompt_matrix=None,
                    soft_prefix_matrix=None,
                )
            )
        if soft_prompt_matrix is not None:
            variants.append(
                analyze_variant(
                    bundle=bundle,
                    prompt=example["prompt"],
                    story=example["story"],
                    layer=layer,
                    steering_vector=steering_vector.to(bundle.device),
                    variant_name="soft_prompt_suffix",
                    suffix_text="",
                    suffix_token_ids=None,
                    soft_prompt_matrix=soft_prompt_matrix.to(bundle.device),
                    soft_prefix_matrix=None,
                )
            )
        if soft_prefix_matrix is not None:
            variants.append(
                analyze_variant(
                    bundle=bundle,
                    prompt=example["prompt"],
                    story=example["story"],
                    layer=layer,
                    steering_vector=steering_vector.to(bundle.device),
                    variant_name="soft_prefix",
                    suffix_text="",
                    suffix_token_ids=None,
                    soft_prompt_matrix=None,
                    soft_prefix_matrix=soft_prefix_matrix.to(bundle.device),
                )
            )
        analyzed_examples.append(
            {
                "index": example_index,
                "concept": example.get("concept", ""),
                "label": example.get("label", ""),
                "prompt": example["prompt"],
                "story": example["story"],
                "variants": variants,
            }
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.residual_file).resolve().parent / "token_conceptness"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = Path(args.output_html) if args.output_html else output_dir / f"token_conceptness_layer_{layer}_{timestamp}.html"
    json_path = Path(args.output_json) if args.output_json else output_dir / f"token_conceptness_layer_{layer}_{timestamp}.json"

    payload = {
        "model": args.model,
        "backend": resolved_backend,
        "residual_file": args.residual_file,
        "layer": layer,
        "vector_norm": vector_norm,
        "score_metric": args.score_metric,
        "threshold": args.threshold,
        "threshold_mode": args.threshold_mode,
        "color_scale": args.color_scale,
        "dataset": args.dataset,
        "suffix": args.suffix,
        "soft_prompt": args.soft_prompt,
        "soft_prefix": args.soft_prefix,
        "step": args.step,
        "exact_suffix_ids": args.exact_suffix_ids,
        "examples": analyzed_examples,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "layer": layer,
                "vector_norm": vector_norm,
                "examples": len(analyzed_examples),
                "html_path": str(html_path),
                "json_path": str(json_path),
            },
            indent=2,
        )
    )


def load_steering_vector(path: str, layer: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if "steering_vectors" in payload:
        return payload["steering_vectors"][layer].float()
    if "candidates" in payload:
        for row in payload["candidates"]:
            if row["layer"] == layer:
                return torch.tensor(row["vector"], dtype=torch.float32)
    raise ValueError(f"Unsupported steering file format or missing layer {layer}: {path}")


def load_examples(args) -> list[dict]:
    if args.prompt or args.story:
        if not (args.prompt and args.story):
            raise ValueError("Provide both --prompt and --story for custom visualization.")
        return [{"concept": "custom", "label": "custom", "prompt": args.prompt, "story": args.story}]

    rows = read_jsonl(args.dataset)
    examples = []
    for row in rows[: args.max_examples]:
        concept = row.get("concept", "")
        if args.story_side in {"positive", "both"}:
            prompt = row.get("poscon_prompt") or row.get("positive_prompt") or row.get("truth_prompt")
            story = row.get("poscon_response") or row.get("positive_response") or row.get("truth_response")
            if prompt and story:
                examples.append({"concept": concept, "label": "positive", "prompt": prompt, "story": story})
        if args.story_side in {"negative", "both"}:
            prompt = row.get("negcon_prompt") or row.get("negative_prompt") or row.get("lie_prompt")
            story = row.get("negcon_response") or row.get("negative_response") or row.get("lie_response")
            if prompt and story:
                examples.append({"concept": concept, "label": "negative", "prompt": prompt, "story": story})
    return examples


def load_suffix_artifact(path: str, exact_suffix_ids: bool) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    suffix_text = payload.get("suffix_text", "")
    suffix_token_ids = payload.get("suffix_token_ids") if exact_suffix_ids else None
    if exact_suffix_ids and suffix_token_ids is None:
        raise ValueError(f"--exact-suffix-ids was set, but artifact has no suffix_token_ids: {path}")
    return {"suffix_text": suffix_text, "suffix_token_ids": suffix_token_ids}


def load_soft_matrix_artifact(path: str, step: int, matrix_key: str) -> torch.Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trace = payload.get("trace", []) if isinstance(payload, dict) else []
    if trace and step >= 0:
        if step >= len(trace):
            raise ValueError(f"step={step} is out of range for trace length {len(trace)} in {path}")
        matrix = trace[step].get(matrix_key)
        if matrix is None:
            raise ValueError(f"Trace step {step} in {path} does not contain {matrix_key}. Was --save-all-steps used?")
        return torch.tensor(matrix, dtype=torch.float32)
    matrix = payload.get(matrix_key) if isinstance(payload, dict) else None
    if matrix is None:
        raise ValueError(f"{path} does not contain top-level {matrix_key}.")
    return torch.tensor(matrix, dtype=torch.float32)


@torch.no_grad()
def analyze_variant(
    bundle,
    prompt: str,
    story: str,
    layer: int,
    steering_vector: torch.Tensor,
    variant_name: str,
    suffix_text: str,
    suffix_token_ids: list[int] | None,
    soft_prompt_matrix: torch.Tensor | None,
    soft_prefix_matrix: torch.Tensor | None,
) -> dict:
    input_ids, inputs_embeds, attention_mask, position_rows, prompt_structure = build_prompt_story_inputs(
        bundle=bundle,
        prompt=prompt,
        story=story,
        suffix_text=suffix_text,
        suffix_token_ids=suffix_token_ids,
        soft_prompt_matrix=soft_prompt_matrix,
        soft_prefix_matrix=soft_prefix_matrix,
    )
    model_kwargs = {"attention_mask": attention_mask, "output_hidden_states": True, "use_cache": False}
    if inputs_embeds is not None:
        model_kwargs["inputs_embeds"] = inputs_embeds
    else:
        model_kwargs["input_ids"] = input_ids
    outputs = bundle.model(**model_kwargs)
    sequence_states = outputs.hidden_states[layer + 1][0, : len(position_rows), :].float()
    vector = steering_vector.float()
    vector_norm = vector.norm().clamp_min(1e-8)
    dots = torch.matmul(sequence_states, vector)
    cosines = dots / (sequence_states.norm(dim=-1).clamp_min(1e-8) * vector_norm)

    tokens = []
    for offset, row in enumerate(position_rows):
        tokens.append(
            {
                "token_index": offset,
                "token_id": row["token_id"],
                "token_text": row["token_text"],
                "segment": row["segment"],
                "segment_index": row["segment_index"],
                "cosine": float(cosines[offset].item()),
                "dot": float(dots[offset].item()),
            }
        )
    return {
        "name": variant_name,
        "suffix_text": suffix_text,
        "used_exact_suffix_ids": suffix_token_ids is not None,
        "soft_prompt_shape": None if soft_prompt_matrix is None else list(soft_prompt_matrix.shape),
        "soft_prefix_shape": None if soft_prefix_matrix is None else list(soft_prefix_matrix.shape),
        "prompt_structure": prompt_structure,
        "tokens": tokens,
        "summary": summarize_tokens(tokens),
    }


def build_prompt_story_inputs(
    bundle,
    prompt: str,
    story: str,
    suffix_text: str,
    suffix_token_ids: list[int] | None,
    soft_prompt_matrix: torch.Tensor | None,
    soft_prefix_matrix: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, list[dict], str]:
    if soft_prompt_matrix is not None and soft_prefix_matrix is not None:
        raise ValueError("Use at most one continuous matrix variant at a time.")
    response_ids = torch.tensor(
        bundle.tokenizer.encode(story, add_special_tokens=False),
        dtype=torch.long,
        device=bundle.device,
    )
    if response_ids.numel() == 0:
        raise ValueError("Story tokenized to zero tokens.")

    if soft_prompt_matrix is not None:
        segments = split_chat_prompt_for_user_suffix(bundle, prompt)
        embedding_layer = bundle.model.get_input_embeddings()
        user_embeds = embedding_layer(segments["user_input_ids"]).detach()
        assistant_prefix_embeds = embedding_layer(segments["assistant_prefix_ids"]).detach()
        response_embeds = embedding_layer(response_ids.unsqueeze(0)).detach()
        soft_prompt = soft_prompt_matrix.to(bundle.device).to(embedding_layer.weight.dtype)
        inputs_embeds = torch.cat([user_embeds, soft_prompt, assistant_prefix_embeds, response_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=bundle.device)
        position_rows = []
        position_rows.extend(build_token_rows(bundle, segments["user_input_ids"][0], "user_prompt_and_format"))
        position_rows.extend(build_soft_rows(soft_prompt.shape[1], "soft_prompt_suffix"))
        position_rows.extend(build_token_rows(bundle, segments["assistant_prefix_ids"][0], "assistant_generation_prefix"))
        position_rows.extend(build_token_rows(bundle, response_ids, "story"))
        return None, inputs_embeds, attention_mask, position_rows, "user_prompt + soft_prompt_suffix + assistant_generation_prefix + story"

    if soft_prefix_matrix is not None:
        prompt_inputs = encode_chat_prompt(bundle, prompt, add_generation_prompt=True)
        prompt_ids = prompt_inputs["input_ids"]
        embedding_layer = bundle.model.get_input_embeddings()
        prompt_embeds = embedding_layer(prompt_ids).detach()
        response_embeds = embedding_layer(response_ids.unsqueeze(0)).detach()
        soft_prefix = soft_prefix_matrix.to(bundle.device).to(embedding_layer.weight.dtype)
        inputs_embeds = torch.cat([prompt_embeds, soft_prefix, response_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=bundle.device)
        position_rows = []
        position_rows.extend(build_token_rows(bundle, prompt_ids[0], "prompt_with_generation_prefix"))
        position_rows.extend(build_soft_rows(soft_prefix.shape[1], "soft_prefix"))
        position_rows.extend(build_token_rows(bundle, response_ids, "story"))
        return None, inputs_embeds, attention_mask, position_rows, "chat_formatted_prompt_with_assistant_generation_prefix + soft_prefix + story"

    if suffix_token_ids is not None:
        segments = split_chat_prompt_for_user_suffix(bundle, prompt)
        suffix_tensor = torch.tensor(suffix_token_ids, dtype=torch.long, device=bundle.device)
        prompt_ids = torch.cat(
            [
                segments["user_input_ids"][0],
                suffix_tensor,
                segments["assistant_prefix_ids"][0],
            ],
            dim=0,
        )
        position_rows = []
        position_rows.extend(build_token_rows(bundle, segments["user_input_ids"][0], "user_prompt_and_format"))
        position_rows.extend(build_token_rows(bundle, suffix_tensor, "exact_suffix"))
        position_rows.extend(build_token_rows(bundle, segments["assistant_prefix_ids"][0], "assistant_generation_prefix"))
    else:
        prompt_for_encoding = prompt + suffix_text if suffix_text else prompt
        prompt_inputs = encode_chat_prompt(bundle, prompt_for_encoding, add_generation_prompt=True)
        prompt_ids = prompt_inputs["input_ids"][0]
        if suffix_text:
            segments = split_chat_prompt_for_user_suffix(bundle, prompt_for_encoding)
            position_rows = []
            position_rows.extend(build_token_rows(bundle, segments["user_input_ids"][0], "user_prompt_and_format"))
            position_rows.extend(build_token_rows(bundle, segments["assistant_prefix_ids"][0], "assistant_generation_prefix"))
        else:
            position_rows = build_token_rows(bundle, prompt_ids, "prompt_with_generation_prefix")

    input_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=bundle.device)
    position_rows.extend(build_token_rows(bundle, response_ids, "story"))
    if suffix_token_ids is not None:
        prompt_structure = "user_prompt + exact_discrete_suffix_ids + assistant_generation_prefix + story"
    elif suffix_text:
        prompt_structure = "chat_formatted_user_prompt_with_text_suffix + assistant_generation_prefix + story"
    else:
        prompt_structure = "chat_formatted_prompt_with_assistant_generation_prefix + story"
    return input_ids, None, attention_mask, position_rows, prompt_structure


def build_token_rows(bundle, token_ids: torch.Tensor, segment: str) -> list[dict]:
    rows = []
    for segment_index, token_id in enumerate(token_ids.tolist()):
        rows.append(
            {
                "segment": segment,
                "segment_index": segment_index,
                "token_id": int(token_id),
                "token_text": bundle.tokenizer.decode([int(token_id)], skip_special_tokens=False),
            }
        )
    return rows


def build_soft_rows(length: int, segment: str) -> list[dict]:
    return [
        {
            "segment": segment,
            "segment_index": segment_index,
            "token_id": None,
            "token_text": f"<{segment}_{segment_index}>",
        }
        for segment_index in range(length)
    ]


def summarize_tokens(tokens: list[dict]) -> dict:
    if not tokens:
        return {"num_tokens": 0}
    cosines = [row["cosine"] for row in tokens]
    dots = [row["dot"] for row in tokens]
    segment_counts = {}
    for row in tokens:
        segment = row.get("segment", "unknown")
        segment_counts[segment] = segment_counts.get(segment, 0) + 1
    return {
        "num_tokens": len(tokens),
        "mean_cosine": sum(cosines) / len(cosines),
        "max_cosine": max(cosines),
        "min_cosine": min(cosines),
        "mean_dot": sum(dots) / len(dots),
        "max_dot": max(dots),
        "min_dot": min(dots),
        "segment_counts": segment_counts,
    }


def render_html(payload: dict) -> str:
    body = []
    for example in payload["examples"]:
        body.append(
            f"<section class='example'>"
            f"<h2>{escape(example['label'])}: {escape(example['concept'])}</h2>"
            f"<p class='prompt'>{escape(example['prompt'])}</p>"
        )
        for variant in example["variants"]:
            body.append(render_variant(variant, payload))
        body.append("</section>")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Token Conceptness Layer {payload['layer']}</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f4ee;
      color: #1f2933;
    }}
    .header, .example {{
      max-width: 1180px;
      margin: 0 auto 24px auto;
      background: rgba(255,255,255,0.86);
      border: 1px solid #e5dfd3;
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(60, 45, 24, 0.08);
    }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    .meta, .prompt, .summary {{ color: #5f6b76; }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 8px 0 12px 0;
    }}
    .chip {{
      padding: 4px 8px;
      border-radius: 999px;
      background: #efe8dc;
      color: #4b5563;
      font-size: 12px;
      border: 1px solid #ddd2bf;
    }}
    .variant {{
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid #ebe5d9;
    }}
    .tokens {{
      line-height: 2.1;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 14px;
    }}
    .tok {{
      white-space: pre-wrap;
      border-radius: 5px;
      padding: 2px 3px;
      margin: 1px;
      border: 1px solid rgba(0,0,0,0.04);
    }}
    .hit {{
      outline: 2px solid #111827;
      outline-offset: 1px;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 10px;
    }}
    .bar {{
      width: 220px;
      height: 14px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgb(190, 57, 43), white, rgb(34, 139, 84));
      border: 1px solid #d8d1c4;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Token Conceptness Probe</h1>
    <p class="meta">Layer {payload['layer']} | metric {escape(payload['score_metric'])} | threshold {payload['threshold']} ({escape(payload['threshold_mode'])}) | vector norm {payload['vector_norm']:.6f}</p>
    <div class="legend"><span>-1 / negative</span><div class="bar"></div><span>+1 / positive</span></div>
  </div>
  {''.join(body)}
</body>
</html>
"""


def render_variant(variant: dict, payload: dict) -> str:
    tokens_html = []
    for token in variant["tokens"]:
        score = token[payload["score_metric"]]
        color_score = max(-1.0, min(1.0, score / max(float(payload["color_scale"]), 1e-8)))
        hit = is_hit(score, payload["threshold"], payload["threshold_mode"])
        tokens_html.append(
            "<span "
            f"class='tok {'hit' if hit else ''}' "
            f"style='background:{score_to_color(color_score)}' "
            f"title='idx={token['token_index']} segment={escape(token.get('segment', ''))} "
            f"segment_idx={token.get('segment_index', 0)} id={token['token_id']} cosine={token['cosine']:.4f} dot={token['dot']:.4f}'>"
            f"{escape(token['token_text'])}</span>"
        )
    summary = variant["summary"]
    segment_counts = summary.get("segment_counts", {})
    chips_html = "".join(
        f"<span class='chip'>{escape(segment)}: {count}</span>"
        for segment, count in segment_counts.items()
    )
    return (
        f"<div class='variant'>"
        f"<h3>{escape(variant['name'])}</h3>"
        f"<p class='summary'>structure: {escape(variant.get('prompt_structure', ''))}</p>"
        f"<div class='chips'>{chips_html}</div>"
        f"<p class='summary'>tokens={summary.get('num_tokens', 0)} "
        f"mean_cos={summary.get('mean_cosine', 0.0):.4f} "
        f"max_cos={summary.get('max_cosine', 0.0):.4f} "
        f"min_cos={summary.get('min_cosine', 0.0):.4f} "
        f"mean_dot={summary.get('mean_dot', 0.0):.4f}</p>"
        f"<div class='tokens'>{''.join(tokens_html)}</div>"
        f"</div>"
    )


def is_hit(score: float, threshold: float, mode: str) -> bool:
    if mode == "positive":
        return score >= threshold
    return abs(score) >= threshold


def score_to_color(score: float) -> str:
    score = max(-1.0, min(1.0, score))
    if score >= 0:
        alpha = abs(score)
        red = round((1 - alpha) * 255 + alpha * 34)
        green = round((1 - alpha) * 255 + alpha * 139)
        blue = round((1 - alpha) * 255 + alpha * 84)
    else:
        alpha = abs(score)
        red = round((1 - alpha) * 255 + alpha * 190)
        green = round((1 - alpha) * 255 + alpha * 57)
        blue = round((1 - alpha) * 255 + alpha * 43)
    return f"rgb({red}, {green}, {blue})"


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
