from __future__ import annotations

import unittest

from trialmatch.data import (
    _training_record,
    extract_numbered_sentences,
    parse_sentence_ids,
    split_patient_ids,
)


class SentenceIdTests(unittest.TestCase):
    def test_parse_json_and_python_list_strings(self) -> None:
        self.assertEqual(parse_sentence_ids("[0, 3]"), [0, 3])
        self.assertEqual(parse_sentence_ids("(1, 2)"), [1, 2])
        self.assertEqual(parse_sentence_ids(r"\[0, 1\]"), [0, 1])

    def test_empty_and_invalid_sentence_ids(self) -> None:
        self.assertEqual(parse_sentence_ids("[]"), [])
        self.assertEqual(parse_sentence_ids(None), [])
        with self.assertRaises(ValueError):
            parse_sentence_ids("[1, -1]")
        with self.assertRaises(ValueError):
            parse_sentence_ids("[0, 0]")

    def test_extract_numbered_sentences(self) -> None:
        note = "0. First patient sentence. 1. Second sentence with 2.5 mg."
        self.assertEqual(
            extract_numbered_sentences(note),
            {
                0: "First patient sentence.",
                1: "Second sentence with 2.5 mg.",
            },
        )


class PatientSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        patient_ids = [f"patient-{index:02d}" for index in range(53)]
        first = split_patient_ids(patient_ids)
        second = split_patient_ids(reversed(patient_ids))

        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(ids) for name, ids in first.items()},
            {"train": 37, "validation": 8, "test": 8},
        )
        self.assertEqual(len(set(sum(first.values(), []))), 53)

    def test_rejects_wrong_patient_count(self) -> None:
        with self.assertRaises(ValueError):
            split_patient_ids(["only-one"])


class TrainingRecordTests(unittest.TestCase):
    def test_record_excludes_generated_predictions(self) -> None:
        row = {
            "annotation_id": 7,
            "patient_id": "patient-01",
            "note": "0. A patient sentence.",
            "criterion_type": "inclusion",
            "criterion_text": "Age 18 or older",
            "expert_eligibility": "included",
            "expert_sentences": "[0]",
            "gpt4_explanation": "must not enter model features",
            "gpt4_eligibility": "not enough information",
        }

        record = _training_record(row)
        self.assertEqual(record["expert_sentence_ids"], [0])
        self.assertNotIn("gpt4_explanation", record)
        self.assertNotIn("gpt4_eligibility", record)

    def test_missing_criterion_is_explicit_not_fabricated(self) -> None:
        row = {
            "annotation_id": 8,
            "patient_id": "patient-02",
            "note": "0. A patient sentence.",
            "criterion_type": "exclusion",
            "criterion_text": None,
            "expert_eligibility": "not excluded",
            "expert_sentences": "[]",
        }

        record = _training_record(row)
        self.assertEqual(record["criterion_text"], "")
        self.assertEqual(record["source_missing_fields"], ["criterion_text"])


if __name__ == "__main__":
    unittest.main()
