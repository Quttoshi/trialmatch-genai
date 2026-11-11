from __future__ import annotations

import unittest

from trialmatch.metrics import classification_metrics, evidence_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_preserves_labels_and_reports_confusion(self) -> None:
        result = classification_metrics(
            ["included", "not enough information", "included"],
            ["included", "included", "not applicable"],
        )
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(
            set(result["labels"]),
            {"included", "not enough information", "not applicable"},
        )
        self.assertEqual(
            result["confusion_matrix"]["not enough information"]["included"],
            1,
        )

    def test_rejects_mismatched_sequences(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(["included"], [])


class EvidenceMetricsTests(unittest.TestCase):
    def test_empty_gold_records_are_explicit_and_penalize_unsupported_predictions(self) -> None:
        result = evidence_metrics(
            [[], [1, 2], []],
            [[], [1, 3], [4]],
        )
        self.assertEqual(result["empty_gold_records"], 2)
        self.assertEqual(result["empty_gold_exact_matches"], 1)
        self.assertEqual(result["empty_gold_with_predictions"], 1)
        self.assertAlmostEqual(result["micro"]["precision"], 1 / 3)
        self.assertEqual(result["micro"]["recall"], 0.5)
        self.assertEqual(result["micro"]["f1"], 0.4)

    def test_all_empty_and_empty_prediction_cases(self) -> None:
        self.assertEqual(
            evidence_metrics([[]], [[]])["micro"],
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )
        result = evidence_metrics([[1]], [[]])
        self.assertEqual(result["micro"]["recall"], 0.0)
        self.assertEqual(result["micro"]["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
