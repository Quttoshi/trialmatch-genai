from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
RANDOM_SEED = 42
SPLIT_SIZES = {"train": 37, "validation": 8, "test": 8}
REQUIRED_FIELDS = (
    "annotation_id",
    "patient_id",
    "note",
    "criterion_type",
    "criterion_text",
    "expert_eligibility",
    "expert_sentences",
)


def parse_sentence_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", r"\[\]"}:
            return []
        if text.startswith(r"\[") and text.endswith(r"\]"):
            text = text[2:-2].strip()
            if not text:
                return []
            text = f"[{text}]"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Invalid expert sentence ID list: {value!r}") from error
    else:
        raise TypeError(f"Expected sentence IDs as a list or string, got {type(value).__name__}")

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expert sentence IDs must be a list, got {type(parsed).__name__}")
    if any(type(sentence_id) is not int or sentence_id < 0 for sentence_id in parsed):
        raise ValueError(f"Sentence IDs must be non-negative integers: {parsed!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Sentence IDs must not contain duplicates: {parsed!r}")
    return list(parsed)


def extract_numbered_sentences(note: str) -> dict[int, str]:
    markers = list(re.finditer(r"(?:^|\s)(\d+)\.\s+", note))
    selected = []
    expected_id = 0
    for marker in markers:
        if int(marker.group(1)) == expected_id:
            selected.append(marker)
            expected_id += 1

    sentences: dict[int, str] = {}
    for index, marker in enumerate(selected):
        end = selected[index + 1].start() if index + 1 < len(selected) else len(note)
        sentence = note[marker.end() : end].strip()
        if sentence:
            sentences[index] = sentence
    return sentences


def split_patient_ids(patient_ids: Iterable[str]) -> dict[str, list[str]]:
    ordered_ids = np.asarray(sorted(set(patient_ids)), dtype=str)
    if len(ordered_ids) != sum(SPLIT_SIZES.values()):
        raise ValueError(
            "Expected exactly 53 unique patient IDs for the prescribed 37/8/8 split; "
            f"found {len(ordered_ids)}."
        )

    np.random.default_rng(RANDOM_SEED).shuffle(ordered_ids)
    shuffled_ids = ordered_ids.tolist()
    train_end = SPLIT_SIZES["train"]
    validation_end = train_end + SPLIT_SIZES["validation"]
    return {
        "train": shuffled_ids[:train_end],
        "validation": shuffled_ids[train_end:validation_end],
        "test": shuffled_ids[validation_end:],
    }


def _text_field(row: Mapping[str, Any], name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise ValueError(f"Expected {name!r} to be text, got {type(value).__name__}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _training_record(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Dataset row is missing required fields: {missing}")
    try:
        annotation_id = int(row["annotation_id"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid annotation_id: {row['annotation_id']!r}") from error

    patient_id = _text_field(row, "patient_id")
    if not patient_id:
        raise ValueError("patient_id must not be empty")
    criterion_value = row["criterion_text"]
    missing_fields: list[str] = []
    if criterion_value is None or criterion_value == "":
        criterion_text = ""
        missing_fields.append("criterion_text")
    elif isinstance(criterion_value, str):
        criterion_text = criterion_value
    else:
        raise ValueError(
            "Expected 'criterion_text' to be text or missing, "
            f"got {type(criterion_value).__name__}"
        )

    return {
        "annotation_id": annotation_id,
        "patient_id": patient_id,
        "note": _text_field(row, "note"),
        "criterion_type": _text_field(row, "criterion_type"),
        "criterion_text": criterion_text,
        "expert_eligibility": _text_field(row, "expert_eligibility"),
        "expert_sentence_ids": parse_sentence_ids(row["expert_sentences"]),
        "source_missing_fields": missing_fields,
    }


def prepare_dataset(output_dir: Path) -> dict[str, Any]:
    from datasets import load_dataset

    source_rows = load_dataset(DATASET_ID, split="train").to_list()
    records = [_training_record(row) for row in source_rows]
    split_ids = split_patient_ids(record["patient_id"] for record in records)

    patient_to_split = {
        patient_id: split_name
        for split_name, patient_ids in split_ids.items()
        for patient_id in patient_ids
    }
    if len(patient_to_split) != sum(SPLIT_SIZES.values()):
        raise ValueError("Patient split contains duplicate IDs")
    patient_sets = [set(patient_ids) for patient_ids in split_ids.values()]
    for left_index, left_ids in enumerate(patient_sets):
        if any(left_ids & right_ids for right_ids in patient_sets[left_index + 1 :]):
            raise ValueError("Patient leakage detected between splits")
    if set(patient_to_split) != {record["patient_id"] for record in records}:
        raise ValueError("Patient split does not cover every annotation")

    split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        split_records[patient_to_split[record["patient_id"]]].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_SIZES:
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in split_records[split_name]:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_id": DATASET_ID,
        "seed": RANDOM_SEED,
        "patient_counts": {name: len(ids) for name, ids in split_ids.items()},
        "annotation_counts": {name: len(split_records[name]) for name in SPLIT_SIZES},
        "patient_ids": split_ids,
        "input_fields": ["note", "criterion_type", "criterion_text"],
        "target_fields": ["expert_eligibility", "expert_sentence_ids"],
        "excluded_dataset_fields": [
            "gpt4_explanation",
            "explanation_correctness",
            "gpt4_sentences",
            "gpt4_eligibility",
        ],
        "source_missing_fields": {
            field: [
                record["annotation_id"]
                for record in records
                if field in record["source_missing_fields"]
            ]
            for field in ("criterion_text",)
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
