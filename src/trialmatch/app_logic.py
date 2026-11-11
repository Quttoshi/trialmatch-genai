from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from trialmatch.data import extract_numbered_sentences
from trialmatch.evaluation import build_explanation


_NUMBERED_LINE_PREFIX = re.compile(r"^\d+\.\s+")


def normalize_patient_record(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty patient record")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [_NUMBERED_LINE_PREFIX.sub("", line, count=1) for line in lines]
    if any(not sentence for sentence in sentences):
        raise ValueError("Patient record contains an empty sentence")
    return " ".join(f"{sentence_id}. {sentence}" for sentence_id, sentence in enumerate(sentences))


def parse_criteria(text: str) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter at least one criterion")
    criteria = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        criterion_type, separator, criterion_text = line.partition(":")
        criterion_type = criterion_type.strip().lower()
        criterion_text = criterion_text.strip()
        if not separator:
            raise ValueError(
                f"Criterion line {line_number} must use 'inclusion: text' or 'exclusion: text'"
            )
        if criterion_type not in {"inclusion", "exclusion"}:
            raise ValueError(
                f"Criterion line {line_number} must be inclusion or exclusion"
            )
        if not criterion_text:
            raise ValueError(f"Criterion line {line_number} has no criterion text")
        criteria.append(
            {
                "criterion_type": criterion_type,
                "criterion_text": criterion_text,
            }
        )
    if not criteria:
        raise ValueError("Enter at least one non-empty criterion")
    return criteria


def application_result(
    criterion: Mapping[str, str],
    note: str,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    sentence_map = extract_numbered_sentences(note)
    evidence, invalid_ids, explanation = build_explanation(
        str(prediction["eligibility"]),
        prediction["evidence_sentence_ids"],
        sentence_map,
    )
    return {
        "criterion_type": criterion["criterion_type"],
        "criterion_text": criterion["criterion_text"],
        "predicted_eligibility": prediction["eligibility"],
        "cited_sentence_ids": prediction["evidence_sentence_ids"],
        "evidence_text": evidence,
        "invalid_sentence_ids": invalid_ids,
        "explanation": explanation,
    }


def render_screening_report(results: Sequence[Mapping[str, Any]]) -> str:
    if not results:
        raise ValueError("Cannot render a report without criterion results")
    sections = [
        "# Clinical Trial Eligibility Screening Report",
        "",
        "> Educational screening prototype only; not a clinical enrollment decision.",
        "",
    ]
    for index, result in enumerate(results, start=1):
        sections.extend(
            [
                f"## Criterion {index}: {result['criterion_type']}",
                "",
                f"**Criterion:** {result['criterion_text']}",
                "",
                f"**Predicted category:** {result['predicted_eligibility']}",
                "",
                "**Supporting patient-record evidence:**",
            ]
        )
        evidence = result["evidence_text"]
        if evidence:
            sections.extend(
                f"- Sentence {item['sentence_id']}: {item['text']}" for item in evidence
            )
        else:
            sections.append("- No valid supporting sentence was cited.")
        if result["invalid_sentence_ids"]:
            sections.append(
                f"- Unrecognized sentence IDs: {result['invalid_sentence_ids']}"
            )
        sections.extend([ "", f"**Explanation:** {result['explanation']}", ""])
    return "\n".join(sections)
