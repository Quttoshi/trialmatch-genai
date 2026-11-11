from __future__ import annotations

import unittest

from trialmatch.app_logic import (
    application_result,
    normalize_patient_record,
    parse_criteria,
    render_screening_report,
)


class ApplicationLogicTests(unittest.TestCase):
    def test_normalizes_user_sentences_to_stable_ids(self) -> None:
        self.assertEqual(
            normalize_patient_record("58-year-old patient\n0. Reports chest pain"),
            "0. 58-year-old patient 1. Reports chest pain",
        )

    def test_rejects_empty_patient_record_and_bad_criterion(self) -> None:
        with self.assertRaises(ValueError):
            normalize_patient_record(" \n ")
        with self.assertRaises(ValueError):
            parse_criteria("other: criterion")
        with self.assertRaises(ValueError):
            parse_criteria("inclusion:")

    def test_parses_multiple_criterion_types(self) -> None:
        self.assertEqual(
            parse_criteria(
                "inclusion: Age 18 or older\nexclusion: No active malignancy"
            ),
            [
                {"criterion_type": "inclusion", "criterion_text": "Age 18 or older"},
                {"criterion_type": "exclusion", "criterion_text": "No active malignancy"},
            ],
        )

    def test_builds_exportable_result_from_exact_evidence(self) -> None:
        note = normalize_patient_record("Patient is 58 years old.")
        result = application_result(
            {"criterion_type": "inclusion", "criterion_text": "Age 18 or older"},
            note,
            {"eligibility": "included", "evidence_sentence_ids": [0]},
        )
        report = render_screening_report([result])
        self.assertIn("not a clinical enrollment decision", report)
        self.assertIn("Patient is 58 years old.", report)
        self.assertEqual(result["evidence_text"][0]["sentence_id"], 0)


if __name__ == "__main__":
    unittest.main()
