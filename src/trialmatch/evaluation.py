from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from json import JSONDecoder, JSONDecodeError
from typing import Any

from trialmatch.data import extract_numbered_sentences
from trialmatch.metrics import classification_metrics, evidence_metrics

INVALID_LABEL = "__INVALID_OUTPUT__"
OUTPUT_KEYS = {"eligibility", "evidence_sentence_ids"}


def parse_model_output(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("Generated output does not contain a JSON object")
    try:
        payload, end = JSONDecoder().raw_decode(text[start:])
    except JSONDecodeError as error:
        raise ValueError("Generated output contains invalid JSON") from error
    if text[start + end :].strip():
        raise ValueError("Generated output contains text outside the JSON object")
    if not isinstance(payload, dict) or set(payload) != OUTPUT_KEYS:
        raise ValueError(f"Generated JSON must contain exactly {sorted(OUTPUT_KEYS)}")
    label = payload["eligibility"]
    sentence_ids = payload["evidence_sentence_ids"]
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Generated eligibility category must be a non-empty string")
    if not isinstance(sentence_ids, list) or any(
        type(sentence_id) is not int or sentence_id < 0 for sentence_id in sentence_ids
    ):
        raise ValueError("Generated evidence_sentence_ids must be non-negative integers")
    if len(set(sentence_ids)) != len(sentence_ids):
        raise ValueError("Generated evidence_sentence_ids must not contain duplicates")
    return {
        "eligibility": label,
        "evidence_sentence_ids": sentence_ids,
    }


def build_explanation(
    predicted_label: str,
    sentence_ids: Sequence[int],
    sentence_text: Mapping[int, str],
) -> tuple[list[dict[str, Any]], list[int], str]:
    evidence = [
        {"sentence_id": sentence_id, "text": sentence_text[sentence_id]}
        for sentence_id in sentence_ids
        if sentence_id in sentence_text
    ]
    invalid_ids = [sentence_id for sentence_id in sentence_ids if sentence_id not in sentence_text]
    if evidence:
        cited_text = " ".join(
            f"[{item['sentence_id']}] {item['text']}" for item in evidence
        )
        explanation = f"Predicted category: {predicted_label}. Cited patient-record evidence: {cited_text}"
    else:
        explanation = (
            f"Predicted category: {predicted_label}. No valid supporting sentence was cited; "
            "missing evidence is not proof of eligibility or ineligibility."
        )
    if invalid_ids:
        explanation += f" Unrecognized sentence IDs: {invalid_ids}."
    return evidence, invalid_ids, explanation


def score_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction set")
    gold_labels = [str(item["gold_eligibility"]) for item in predictions]
    predicted_labels = [str(item["predicted_eligibility"]) for item in predictions]
    criteria_types = [str(item["criterion_type"]) for item in predictions]

    by_criterion_type: dict[str, Any] = {}
    for criterion_type in ("inclusion", "exclusion"):
        indices = [
            index
            for index, value in enumerate(criteria_types)
            if value.lower() == criterion_type
        ]
        by_criterion_type[criterion_type] = (
            classification_metrics(
                [gold_labels[index] for index in indices],
                [predicted_labels[index] for index in indices],
            )
            if indices
            else None
        )

    return {
        "classification_overall": classification_metrics(gold_labels, predicted_labels),
        "classification_by_criterion_type": by_criterion_type,
        "evidence_retrieval": evidence_metrics(
            [item["gold_evidence_sentence_ids"] for item in predictions],
            [item["predicted_evidence_sentence_ids"] for item in predictions],
        ),
        "invalid_output_count": sum(not item["output_valid"] for item in predictions),
        "invalid_evidence_id_count": sum(
            len(item["invalid_evidence_ids"]) for item in predictions
        ),
        "out_of_context_citation_count": sum(
            len(item.get("out_of_context_evidence_ids", [])) for item in predictions
        ),
    }


def make_prediction_record(
    record: Mapping[str, Any],
    raw_output: str,
) -> dict[str, Any]:
    output_valid = True
    output_error: str | None = None
    try:
        parsed = parse_model_output(raw_output)
        predicted_label = parsed["eligibility"]
        predicted_ids = parsed["evidence_sentence_ids"]
    except ValueError as error:
        output_valid = False
        output_error = str(error)
        predicted_label = INVALID_LABEL
        predicted_ids = []

    sentence_text = extract_numbered_sentences(record["note"])
    evidence, invalid_ids, explanation = build_explanation(
        predicted_label,
        predicted_ids,
        sentence_text,
    )
    return {
        "annotation_id": record["annotation_id"],
        "patient_id": record["patient_id"],
        "criterion_type": record["criterion_type"],
        "criterion_text": record["criterion_text"],
        "gold_eligibility": record["expert_eligibility"],
        "predicted_eligibility": predicted_label,
        "gold_evidence_sentence_ids": record["expert_sentence_ids"],
        "predicted_evidence_sentence_ids": predicted_ids,
        "evidence_text": evidence,
        "invalid_evidence_ids": invalid_ids,
        "explanation": explanation,
        "output_valid": output_valid,
        "output_error": output_error,
        "raw_output": raw_output,
    }
