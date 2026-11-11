from __future__ import annotations

import unittest

from trialmatch.evaluation import (
    INVALID_LABEL,
    build_explanation,
    make_prediction_record,
    parse_model_output,
    score_predictions,
)


class OutputParserTests(unittest.TestCase):
    def test_parses_raw_category_and_evidence_ids(self) -> None:
        parsed = parse_model_output(
            '{"eligibility":"not enough information","evidence_sentence_ids":[]}'
        )
        self.assertEqual(parsed["eligibility"], "not enough information")
        self.assertEqual(parsed["evidence_sentence_ids"], [])

    def test_rejects_malformed_and_extra_fields(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_output("not json")
        with self.assertRaises(ValueError):
            parse_model_output(
                '{"eligibility":"included","evidence_sentence_ids":[],"explanation":"x"}'
            )
        with self.assertRaises(ValueError):
            parse_model_output(
                '{"eligibility":"included","evidence_sentence_ids":[-1]}'
            )

    def test_invalid_prediction_is_visible_not_silently_defaulted(self) -> None:
        record = {
            "annotation_id": 1,
            "patient_id": "p1",
            "criterion_type": "inclusion",
            "criterion_text": "Criterion",
            "expert_eligibility": "included",
            "expert_sentence_ids": [0],
            "note": "0. Evidence sentence.",
        }
        prediction = make_prediction_record(record, "bad output")
        self.assertEqual(prediction["predicted_eligibility"], INVALID_LABEL)
        self.assertFalse(prediction["output_valid"])
        self.assertTrue(prediction["output_error"])

    def test_explanation_quotes_only_available_source_sentences(self) -> None:
        evidence, invalid_ids, explanation = build_explanation(
            "included",
            [0, 9],
            {0: "Patient is 58 years old."},
        )
        self.assertEqual(evidence, [{"sentence_id": 0, "text": "Patient is 58 years old."}])
        self.assertEqual(invalid_ids, [9])
        self.assertIn("Unrecognized sentence IDs: [9]", explanation)


class PredictionScoringTests(unittest.TestCase):
    def test_reports_classification_by_criterion_type_and_evidence(self) -> None:
        predictions = [
            {
                "gold_eligibility": "included",
                "predicted_eligibility": "included",
                "criterion_type": "inclusion",
                "gold_evidence_sentence_ids": [0],
                "predicted_evidence_sentence_ids": [0],
                "output_valid": True,
                "invalid_evidence_ids": [],
            },
            {
                "gold_eligibility": "not excluded",
                "predicted_eligibility": INVALID_LABEL,
                "criterion_type": "exclusion",
                "gold_evidence_sentence_ids": [],
                "predicted_evidence_sentence_ids": [],
                "output_valid": False,
                "invalid_evidence_ids": [],
            },
        ]
        metrics = score_predictions(predictions)
        self.assertIn("inclusion", metrics["classification_by_criterion_type"])
        self.assertIn("exclusion", metrics["classification_by_criterion_type"])
        self.assertEqual(metrics["invalid_output_count"], 1)
        self.assertEqual(metrics["evidence_retrieval"]["micro"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
