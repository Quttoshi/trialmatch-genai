from __future__ import annotations

import unittest
from typing import Any

from trialmatch.prompts import build_messages
from trialmatch.training import encode_supervised_record


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered)


class TrainingEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.record: dict[str, Any] = {
            "note": "0. Patient is 58 years old.",
            "criterion_type": "inclusion",
            "criterion_text": "Age 18 or older",
            "expert_eligibility": "included",
            "expert_sentence_ids": [0],
        }

    def test_loss_mask_covers_prompt_only(self) -> None:
        encoded = encode_supervised_record(self.tokenizer, self.record, max_length=1024)
        prompt_ids = self.tokenizer.apply_chat_template(
            build_messages(self.record),
            tokenize=True,
            add_generation_prompt=True,
        )
        self.assertFalse(encoded["truncated"])
        self.assertTrue(all(label == -100 for label in encoded["labels"][: len(prompt_ids)]))
        self.assertTrue(all(label != -100 for label in encoded["labels"][len(prompt_ids) :]))
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))
        self.assertEqual(len(encoded["attention_mask"]), len(encoded["input_ids"]))

    def test_long_note_is_truncated_to_context_budget(self) -> None:
        record = {
            **self.record,
            "note": f"0. {'A' * 900} 1. {'B' * 900}",
        }
        encoded = encode_supervised_record(self.tokenizer, record, max_length=900)
        self.assertTrue(encoded["truncated"])
        self.assertLessEqual(len(encoded["input_ids"]), 900)
        self.assertGreater(encoded["original_note_tokens"], encoded["retained_note_tokens"])

    def test_too_small_context_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_supervised_record(self.tokenizer, self.record, max_length=32)


if __name__ == "__main__":
    unittest.main()
