from __future__ import annotations

import unittest

from trialmatch.prompts import build_messages, build_target


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "note": "0. Patient is 58 years old.",
            "criterion_type": "inclusion",
            "criterion_text": "Age 18 or older",
            "expert_eligibility": "included",
            "expert_sentence_ids": [0],
        }

    def test_prompt_uses_only_patient_and_criterion_fields(self) -> None:
        record = {
            **self.record,
            "gpt4_explanation": "must not appear",
            "gpt4_eligibility": "incorrect leaked label",
        }
        rendered = "\n".join(message["content"] for message in build_messages(record))
        self.assertIn("Patient is 58 years old", rendered)
        self.assertIn("Age 18 or older", rendered)
        self.assertNotIn("must not appear", rendered)
        self.assertNotIn("incorrect leaked label", rendered)

    def test_target_preserves_label_and_evidence_ids(self) -> None:
        self.assertEqual(
            build_target(self.record),
            '{"eligibility":"included","evidence_sentence_ids":[0]}',
        )

    def test_missing_criterion_is_explicit(self) -> None:
        record = {**self.record, "criterion_text": ""}
        rendered = build_messages(record)[1]["content"]
        self.assertIn("[MISSING IN SOURCE]", rendered)


if __name__ == "__main__":
    unittest.main()
