from __future__ import annotations

import unittest

import numpy as np

from trialmatch.decoding import cache_sequence_length, crop_cache
from trialmatch.speculative import assert_tokenizer_compatibility


class FakeDynamicCache:
    def __init__(self, tensor: np.ndarray) -> None:
        self.tensor = tensor

    def get_seq_length(self) -> int:
        return int(self.tensor.shape[-2])

    def crop(self, sequence_length: int) -> None:
        self.tensor = self.tensor[..., :sequence_length, :]


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        key = np.zeros((1, 2, 9, 4), dtype=np.float32)
        value = np.ones((1, 2, 9, 4), dtype=np.float32)
        self.legacy_cache = ((key, value),)

    def test_reads_legacy_cache_length(self) -> None:
        self.assertEqual(cache_sequence_length(self.legacy_cache), 9)

    def test_crops_legacy_cache_key_and_value(self) -> None:
        cropped = crop_cache(self.legacy_cache, 5)
        self.assertEqual(cache_sequence_length(cropped), 5)
        self.assertEqual(cropped[0][0].shape[-2], 5)
        self.assertEqual(cropped[0][1].shape[-2], 5)

    def test_crops_dynamic_cache(self) -> None:
        cache = FakeDynamicCache(np.zeros((1, 2, 9, 4), dtype=np.float32))
        cropped = crop_cache(cache, 3)
        self.assertIs(cropped, cache)
        self.assertEqual(cache_sequence_length(cropped), 3)

    def test_rejects_invalid_crop_length(self) -> None:
        with self.assertRaises(ValueError):
            crop_cache(self.legacy_cache, 10)


class TokenizerCompatibilityTests(unittest.TestCase):
    class Tokenizer:
        def __init__(self, vocabulary: dict[str, int]) -> None:
            self.vocabulary = vocabulary
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 0

        def get_vocab(self) -> dict[str, int]:
            return self.vocabulary

    def test_accepts_identical_tokenizers(self) -> None:
        vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "token": 3}
        assert_tokenizer_compatibility(
            self.Tokenizer(vocabulary),
            self.Tokenizer(dict(vocabulary)),
        )

    def test_rejects_different_vocabularies(self) -> None:
        with self.assertRaises(ValueError):
            assert_tokenizer_compatibility(
                self.Tokenizer({"<pad>": 0, "x": 1}),
                self.Tokenizer({"<pad>": 0, "y": 1}),
            )


if __name__ == "__main__":
    unittest.main()
