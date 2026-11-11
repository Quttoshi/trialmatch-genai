from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trialmatch.data import extract_numbered_sentences
from trialmatch.prompts import build_messages, build_target

MAX_TARGET_TOKENS = 128


def build_prompt_ids(
    tokenizer: Any,
    record: Mapping[str, Any],
    max_prompt_length: int,
) -> dict[str, Any]:
    if max_prompt_length < 64:
        raise ValueError("max_prompt_length must be at least 64 tokens")

    original_note = record["note"]
    note = original_note
    original_note_tokens = len(tokenizer.encode(original_note, add_special_tokens=False))

    while True:
        messages = build_messages(record, note=note)
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(prompt_ids) <= max_prompt_length:
            visible_sentence_ids = sorted(extract_numbered_sentences(note))
            return {
                "input_ids": prompt_ids,
                "note": note,
                "visible_sentence_ids": visible_sentence_ids,
                "truncated": note != original_note,
                "original_note_tokens": original_note_tokens,
                "retained_note_tokens": len(tokenizer.encode(note, add_special_tokens=False)),
            }

        numbered_sentences = extract_numbered_sentences(note)
        if len(numbered_sentences) > 1:
            retained = list(numbered_sentences.items())[:-1]
            note = " ".join(f"{sentence_id}. {text}" for sentence_id, text in retained)
            continue

        note_ids = tokenizer.encode(note, add_special_tokens=False)
        excess = len(prompt_ids) - max_prompt_length
        retained_token_count = max(1, len(note_ids) - excess - 8)
        if retained_token_count >= len(note_ids):
            retained_token_count = max(1, len(note_ids) // 2)
        shortened_note = tokenizer.decode(
            note_ids[:retained_token_count],
            skip_special_tokens=True,
        )
        if not shortened_note or shortened_note == note:
            raise ValueError(
                "Prompt exceeds max_prompt_length="
                f"{max_prompt_length} even after note truncation"
            )
        note = shortened_note


def encode_supervised_record(
    tokenizer: Any,
    record: Mapping[str, Any],
    max_length: int,
) -> dict[str, Any]:
    if max_length < MAX_TARGET_TOKENS + 64:
        raise ValueError(
            f"max_length must be at least {MAX_TARGET_TOKENS + 64} tokens"
        )

    prompt = build_prompt_ids(
        tokenizer,
        record,
        max_prompt_length=max_length - MAX_TARGET_TOKENS,
    )
    prompt_ids = prompt["input_ids"]
    messages = build_messages(record, note=prompt["note"])
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": build_target(record)}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the prompt prefix")
    target_ids = full_ids[len(prompt_ids) :]
    if len(target_ids) > MAX_TARGET_TOKENS:
        raise ValueError(
            f"Target needs {len(target_ids)} tokens; reserved budget is {MAX_TARGET_TOKENS}"
        )
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "truncated": prompt["truncated"],
        "original_note_tokens": prompt["original_note_tokens"],
        "retained_note_tokens": prompt["retained_note_tokens"],
        "visible_sentence_ids": prompt["visible_sentence_ids"],
    }
