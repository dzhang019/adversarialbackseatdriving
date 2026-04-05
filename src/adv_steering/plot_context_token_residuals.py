from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.backend import load_bundle
else:
    from .backend import load_bundle


DEFAULT_POSITIVE_TERMS = [
    "elated",
    "cheerful",
    "thrilled",
    "overjoyed",
    "delighted",
    "ecstatic",
    "beaming",
    "happy",
    "joyful",
    "grinning",
    "over the moon",
    "radiant",
    "pleased",
    "smiling from ear to ear",
    "on cloud nine",
]

DEFAULT_NEGATIVE_TERMS = [
    "devastated",
    "depressed",
    "gloomy",
    "miserable",
    "heartbroken",
    "crushed",
    "hopeless",
    "despondent",
    "sad",
    "awful",
    "terrible",
    "empty inside",
    "unhappy",
    "inconsolable",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Project selected prompt-token residuals from a long context onto the poscon/negcon PCA basis.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or local path.")
    parser.add_argument("--backend", default="causal_lm", help="Backend name. Only causal_lm is supported.")
    parser.add_argument("--residual-file", required=True, help="Path to poscon_negcon_residuals.pt.")
    parser.add_argument("--context", default="", help="Raw context string to analyze.")
    parser.add_argument("--context-file", default="", help="Optional file containing the context to analyze.")
    parser.add_argument("--line", type=int, default=None, help="Optional 1-based line number to read from --context-file. Defaults to using the whole file.")
    parser.add_argument("--layer", type=int, default=None, help="Layer to analyze. Defaults to best_layer from the residual file.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--positive-terms", default="", help="Optional comma-separated positive emotion terms/phrases.")
    parser.add_argument("--negative-terms", default="", help="Optional comma-separated negative emotion terms/phrases.")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to the residual file directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    context = resolve_context(args.context, args.context_file, args.line)
    residual_payload = torch.load(args.residual_file, map_location="cpu")
    layer = args.layer if args.layer is not None else residual_payload["best_layer"]

    bundle, resolved_backend = load_bundle(args.model, args.backend, args.dtype, args.device_map)
    if resolved_backend != "causal_lm":
        raise ValueError("This script currently supports only the causal_lm backend.")

    positive_terms = parse_terms(args.positive_terms, DEFAULT_POSITIVE_TERMS)
    negative_terms = parse_terms(args.negative_terms, DEFAULT_NEGATIVE_TERMS)

    rendered_text, input_ids, offset_mapping = encode_context_with_offsets(bundle, context)
    token_states = collect_prompt_token_residuals(bundle, input_ids, layer)

    poscon_layer = residual_payload["poscon_residuals"][:, layer, :].float().numpy()
    negcon_layer = residual_payload["negcon_residuals"][:, layer, :].float().numpy()
    all_points = np.concatenate([poscon_layer, negcon_layer], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_points)

    token_rows = collect_labeled_tokens(
        tokenizer=bundle.tokenizer,
        input_ids=input_ids,
        offset_mapping=offset_mapping,
        rendered_text=rendered_text,
        token_states=token_states,
        pca=pca,
        positive_terms=positive_terms,
        negative_terms=negative_terms,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.residual_file).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"context_layer_{layer}_emotion_tokens_pca.png"
    json_path = output_dir / f"context_layer_{layer}_emotion_tokens_pca.json"

    save_plot(token_rows, plot_path, layer)
    json_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "backend": resolved_backend,
                "residual_file": args.residual_file,
                "layer": layer,
                "positive_terms": positive_terms,
                "negative_terms": negative_terms,
                "context": context,
                "rendered_text": rendered_text,
                "matched_tokens": token_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "layer": layer,
                "matched_positive_tokens": sum(row["label"] == "positive" for row in token_rows),
                "matched_negative_tokens": sum(row["label"] == "negative" for row in token_rows),
                "plot_path": str(plot_path),
                "json_path": str(json_path),
            },
            indent=2,
        )
    )


def resolve_context(context: str, context_file: str, line: int | None) -> str:
    if context:
        return context
    if context_file:
        if line is None:
            return Path(context_file).read_text(encoding="utf-8").strip()
        lines = Path(context_file).read_text(encoding="utf-8").splitlines()
        if line < 1 or line > len(lines):
            raise ValueError(f"--line must be between 1 and {len(lines)} for {context_file}")
        return lines[line - 1].strip()
    raise ValueError("Provide either --context or --context-file.")


def parse_terms(raw_terms: str, default_terms: list[str]) -> list[str]:
    if not raw_terms:
        return default_terms
    return [term.strip() for term in raw_terms.split(",") if term.strip()]


def encode_context_with_offsets(bundle, context: str) -> tuple[str, torch.Tensor, list[tuple[int, int]]]:
    tokenizer = bundle.tokenizer
    messages = [{"role": "user", "content": context}]
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("This script expects a tokenizer with apply_chat_template.")

    rendered_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded_offsets = tokenizer(
        rendered_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded_offsets["input_ids"][0].to(bundle.device)
    offset_mapping = [tuple(span) for span in encoded_offsets["offset_mapping"][0].tolist()]
    return rendered_text, input_ids, offset_mapping


@torch.no_grad()
def collect_prompt_token_residuals(bundle, input_ids: torch.Tensor, layer: int) -> np.ndarray:
    attention_mask = torch.ones_like(input_ids).unsqueeze(0)
    outputs = bundle.model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.to(bundle.device),
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[layer + 1][0].detach().float().cpu().numpy()


def collect_labeled_tokens(
    tokenizer,
    input_ids: torch.Tensor,
    offset_mapping: list[tuple[int, int]],
    rendered_text: str,
    token_states: np.ndarray,
    pca: PCA,
    positive_terms: list[str],
    negative_terms: list[str],
) -> list[dict]:
    positive_spans = find_term_spans(rendered_text, positive_terms)
    negative_spans = find_term_spans(rendered_text, negative_terms)

    rows = []
    for token_index, ((start, end), token_state) in enumerate(zip(offset_mapping, token_states)):
        if start == end:
            continue
        label, matched_term = classify_span(start, end, positive_spans, negative_spans)
        if label is None:
            continue
        coordinates = pca.transform(token_state.reshape(1, -1))[0]
        token_id = int(input_ids[token_index].item())
        rows.append(
            {
                "token_index": token_index,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                "span_text": rendered_text[start:end],
                "matched_term": matched_term,
                "label": label,
                "char_start": start,
                "char_end": end,
                "pc1": float(coordinates[0]),
                "pc2": float(coordinates[1]),
            }
        )
    return rows


def find_term_spans(text: str, terms: list[str]) -> list[tuple[int, int, str]]:
    spans = []
    for term in terms:
        pattern = re.compile(rf"(?i)\b{re.escape(term)}\b")
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), term))
    return spans


def classify_span(
    start: int,
    end: int,
    positive_spans: list[tuple[int, int, str]],
    negative_spans: list[tuple[int, int, str]],
) -> tuple[str | None, str | None]:
    for span_start, span_end, term in positive_spans:
        if spans_overlap(start, end, span_start, span_end):
            return "positive", term
    for span_start, span_end, term in negative_spans:
        if spans_overlap(start, end, span_start, span_end):
            return "negative", term
    return None, None


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def save_plot(token_rows: list[dict], plot_path: Path, layer: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    positive_rows = [row for row in token_rows if row["label"] == "positive"]
    negative_rows = [row for row in token_rows if row["label"] == "negative"]
    max_token_index = max((row["token_index"] for row in token_rows), default=1)

    def point_colors(rows: list[dict], base_color: str) -> list[tuple[float, float, float, float]]:
        colors = []
        for row in rows:
            progress = row["token_index"] / max(max_token_index, 1)
            alpha = 0.18 + (0.82 * progress)
            colors.append(to_rgba(base_color, alpha=alpha))
        return colors

    plt.figure(figsize=(11, 8))
    if positive_rows:
        plt.scatter(
            [row["pc1"] for row in positive_rows],
            [row["pc2"] for row in positive_rows],
            c=point_colors(positive_rows, "#2ca02c"),
            s=40,
            label="positive emotion tokens",
        )
    if negative_rows:
        plt.scatter(
            [row["pc1"] for row in negative_rows],
            [row["pc2"] for row in negative_rows],
            c=point_colors(negative_rows, "#d62728"),
            s=40,
            label="negative emotion tokens",
        )

    for row in token_rows:
        plt.annotate(
            row["matched_term"],
            (row["pc1"], row["pc2"]),
            fontsize=7,
            alpha=0.75,
            xytext=(3, 3),
            textcoords="offset points",
        )

    plt.title(f"Context Emotion Token Residuals on Corpus PCA Basis, Layer {layer}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
