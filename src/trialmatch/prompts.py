from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = (
    "You screen one patient against one clinical-trial criterion. Use only the "
    "provided patient record and criterion. Do not treat missing information as "
    "evidence for or against eligibility. Return exactly one JSON object with "
    'string field "eligibility" and integer-array field "evidence_sentence_ids". '
    "Use the original eligibility category, including insufficient-information "
    "or not-applicable categories when appropriate. Cite only sentence IDs shown "
    "in the patient record. Do not include any other text."
)


def build_messages(record: Mapping[str, Any], note: str | None = None) -> list[dict[str, str]]:
    patient_note = record["note"] if note is None else note
    criterion_text = record["criterion_text"]
    if not criterion_text:
        criterion_text = "[MISSING IN SOURCE]"
    user_content = (
        "Patient record with numbered sentences:\n"
        f"{patient_note}\n\n"
        f"Criterion type:\n{record['criterion_type']}\n\n"
        f"Criterion text:\n{criterion_text}\n\n"
        "Return the required JSON object."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_target(record: Mapping[str, Any]) -> str:
    target = {
        "eligibility": record["expert_eligibility"],
        "evidence_sentence_ids": record["expert_sentence_ids"],
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))
