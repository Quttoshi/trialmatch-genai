from __future__ import annotations

import unittest

import numpy as np

from trialmatch.attention import (
    attention_error,
    scaled_dot_product_attention,
    tiled_online_softmax_attention,
)


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        random = np.random.default_rng(42)
        self.query = random.normal(size=(2, 3, 7, 5)).astype(np.float32)
        self.key = random.normal(size=(2, 3, 9, 5)).astype(np.float32)
        self.value = random.normal(size=(2, 3, 9, 4)).astype(np.float32)

    def assert_tiled_matches_reference(self, **kwargs: object) -> None:
        reference = scaled_dot_product_attention(
            self.query,
            self.key,
            self.value,
            **kwargs,
        )
        tiled = tiled_online_softmax_attention(
            self.query,
            self.key,
            self.value,
            query_block_size=3,
            key_block_size=4,
            **kwargs,
        )
        error = attention_error(reference, tiled)
        self.assertLess(error["max_absolute_error"], 2e-6)
        self.assertLess(error["mean_absolute_error"], 5e-7)

    def test_matches_reference_without_mask(self) -> None:
        self.assert_tiled_matches_reference()

    def test_matches_reference_with_causal_mask(self) -> None:
        self.assert_tiled_matches_reference(causal=True)

    def test_matches_reference_with_broadcast_padding_mask(self) -> None:
        mask = np.ones((2, 1, 1, 9), dtype=np.bool_)
        mask[:, :, :, -2:] = False
        self.assert_tiled_matches_reference(mask=mask)

    def test_all_masked_queries_return_zeros(self) -> None:
        mask = np.zeros((7, 9), dtype=np.bool_)
        reference = scaled_dot_product_attention(
            self.query,
            self.key,
            self.value,
            mask=mask,
        )
        tiled = tiled_online_softmax_attention(
            self.query,
            self.key,
            self.value,
            query_block_size=2,
            key_block_size=3,
            mask=mask,
        )
        self.assertTrue(np.all(reference == 0))
        self.assertTrue(np.all(tiled == 0))

    def test_rejects_non_boolean_mask(self) -> None:
        with self.assertRaises(TypeError):
            scaled_dot_product_attention(
                self.query,
                self.key,
                self.value,
                mask=np.ones((7, 9), dtype=np.int8),
            )


if __name__ == "__main__":
    unittest.main()
