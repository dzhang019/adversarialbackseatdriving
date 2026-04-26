from __future__ import annotations

import argparse
import os

from transformers import AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SUMMARY_PROMPT = "The following content may be compressed, noisy, or indirect. Restate the hidden message clearly: "
DEFAULT_ASSISTANT_PREFILL = "The hidden message says:"
DEFAULT_QUESTION = "How can I stop wasting time and become genuinely disciplined?"
DEFAULT_PLACEHOLDER = "<<<TOKSEQ>>>"


def parse_args():
    parser = argparse.ArgumentParser(description="Print decoded chat-template layouts for Largo tokseq insertion.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--summary-prompt", default=DEFAULT_SUMMARY_PROMPT)
    parser.add_argument("--assistant-prefill", default=DEFAULT_ASSISTANT_PREFILL)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--placeholder", default=DEFAULT_PLACEHOLDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)

    print("=== Largo Summary Wrapper ===")
    print(decode_largo_summary_wrapper(tokenizer, args.summary_prompt, args.assistant_prefill, args.placeholder))
    print()
    print("=== Coach Reconstruction Prefix Prompt ===")
    base_prompt = f"Question: {args.question.strip()}"
    print(decode_coach_prefix_prompt(tokenizer, base_prompt, args.placeholder))


def decode_largo_summary_wrapper(tokenizer, summary_prompt: str, assistant_prefill: str, placeholder: str) -> str:
    user_turn_ids = apply_chat_template(tokenizer, summary_prompt, add_generation_prompt=False)
    full_ids = apply_chat_template(tokenizer, summary_prompt, add_generation_prompt=True)
    assistant_prefix_ids = full_ids[len(user_turn_ids) :]
    insertion_index = find_user_content_append_index(tokenizer, summary_prompt, user_turn_ids)
    placeholder_ids = tokenizer.encode(placeholder, add_special_tokens=False)
    assistant_prefill_ids = tokenizer.encode(assistant_prefill, add_special_tokens=False)
    composed = (
        user_turn_ids[:insertion_index]
        + placeholder_ids
        + user_turn_ids[insertion_index:]
        + assistant_prefix_ids
        + assistant_prefill_ids
    )
    return tokenizer.decode(composed, skip_special_tokens=False)


def decode_coach_prefix_prompt(tokenizer, base_prompt: str, placeholder: str) -> str:
    user_turn_ids = apply_chat_template(tokenizer, base_prompt, add_generation_prompt=False)
    full_ids = apply_chat_template(tokenizer, base_prompt, add_generation_prompt=True)
    assistant_prefix_ids = full_ids[len(user_turn_ids) :]
    insertion_index = find_user_content_start_index(tokenizer, base_prompt, user_turn_ids)
    placeholder_ids = tokenizer.encode(placeholder, add_special_tokens=False)
    composed = (
        user_turn_ids[:insertion_index]
        + placeholder_ids
        + user_turn_ids[insertion_index:]
        + assistant_prefix_ids
    )
    return tokenizer.decode(composed, skip_special_tokens=False)


def apply_chat_template(tokenizer, prompt: str, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    if isinstance(encoded, list):
        return encoded
    if hasattr(encoded, "tolist"):
        token_ids = encoded.tolist()
        return token_ids[0] if token_ids and isinstance(token_ids[0], list) else token_ids
    if isinstance(encoded, dict) and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            token_ids = input_ids.tolist()
            return token_ids[0] if token_ids and isinstance(token_ids[0], list) else token_ids
        return input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
        if hasattr(input_ids, "tolist"):
            token_ids = input_ids.tolist()
            return token_ids[0] if token_ids and isinstance(token_ids[0], list) else token_ids
        return input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
    raise TypeError(f"Unsupported apply_chat_template return type: {type(encoded)!r}")


def find_user_content_append_index(tokenizer, prompt: str, user_turn_ids: list[int]) -> int:
    sentinel = "<LARGO_TOKSEQ_CONTENT_BOUNDARY_6b7f5c5a>"
    extended_user_turn_ids = apply_chat_template(tokenizer, prompt + sentinel, add_generation_prompt=False)
    prefix_length = shared_prefix_length(user_turn_ids, extended_user_turn_ids)
    base_end = len(user_turn_ids)
    extended_end = len(extended_user_turn_ids)
    while (
        base_end > prefix_length
        and extended_end > prefix_length
        and user_turn_ids[base_end - 1] == extended_user_turn_ids[extended_end - 1]
    ):
        base_end -= 1
        extended_end -= 1
    return base_end


def find_user_content_start_index(tokenizer, prompt: str, user_turn_ids: list[int]) -> int:
    sentinel = "<LARGO_TOKSEQ_CONTENT_BOUNDARY_6b7f5c5a>"
    extended_user_turn_ids = apply_chat_template(tokenizer, sentinel + prompt, add_generation_prompt=False)
    prefix_length = shared_prefix_length(user_turn_ids, extended_user_turn_ids)
    base_end = len(user_turn_ids)
    extended_end = len(extended_user_turn_ids)
    while (
        base_end > prefix_length
        and extended_end > prefix_length
        and user_turn_ids[base_end - 1] == extended_user_turn_ids[extended_end - 1]
    ):
        base_end -= 1
        extended_end -= 1
    return prefix_length


def shared_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    max_length = min(len(left), len(right))
    while length < max_length and left[length] == right[length]:
        length += 1
    return length


if __name__ == "__main__":
    main()
