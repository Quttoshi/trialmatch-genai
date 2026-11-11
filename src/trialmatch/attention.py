from __future__ import annotations

from typing import Any

import numpy as np


def _validate_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[tuple[int, int, int, int], np.dtype[Any], np.ndarray | None]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, length, dim]")
    batch_size, heads, query_length, key_dim = query.shape
    key_batch, key_heads, key_length, key_key_dim = key.shape
    value_batch, value_heads, value_length, value_dim = value.shape
    if (key_batch, key_heads, key_key_dim) != (batch_size, heads, key_dim):
        raise ValueError("key dimensions must match query batch, heads, and feature size")
    if (value_batch, value_heads, value_length) != (batch_size, heads, key_length):
        raise ValueError("value batch, heads, and sequence length must match key")
    if query_length < 1 or key_length < 1 or key_dim < 1 or value_dim < 1:
        raise ValueError("sequence and feature dimensions must be positive")
    if not all(
        np.issubdtype(array.dtype, np.floating) for array in (query, key, value)
    ):
        raise TypeError("query, key, and value must use floating-point dtypes")

    broadcast_mask = None
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.dtype != np.bool_:
            raise TypeError("mask must be boolean, with True indicating an allowed position")
        try:
            broadcast_mask = np.broadcast_to(
                mask_array,
                (batch_size, heads, query_length, key_length),
            )
        except ValueError as error:
            raise ValueError("mask is not broadcastable to the attention score shape") from error

    compute_dtype = np.result_type(query.dtype, key.dtype, value.dtype, np.float32)
    return (batch_size, heads, query_length, value_dim), compute_dtype, broadcast_mask


def _causal_tile_mask(
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    query_length: int,
    key_length: int,
) -> np.ndarray:
    query_positions = (
        np.arange(query_start, query_end) + max(key_length - query_length, 0)
    )
    key_positions = np.arange(key_start, key_end)
    return key_positions[None, :] <= query_positions[:, None]


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    causal: bool = False,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    (batch_size, heads, query_length, value_dim), compute_dtype, allowed_mask = (
        _validate_inputs(query, key, value, mask)
    )
    key_length = key.shape[2]
    query_work = query.astype(compute_dtype, copy=False)
    key_work = key.astype(compute_dtype, copy=False)
    value_work = value.astype(compute_dtype, copy=False)
    scores = (
        np.matmul(query_work, np.swapaxes(key_work, -1, -2))
        / np.sqrt(query.shape[-1])
    )
    allowed = np.ones(scores.shape, dtype=np.bool_)
    if allowed_mask is not None:
        allowed &= allowed_mask
    if causal:
        allowed &= _causal_tile_mask(
            0,
            query_length,
            0,
            key_length,
            query_length,
            key_length,
        )[None, None, :, :]

    masked_scores = np.where(allowed, scores, -np.inf)
    row_max = np.max(masked_scores, axis=-1, keepdims=True)
    safe_row_max = np.where(np.isfinite(row_max), row_max, 0)
    weights = np.where(allowed, np.exp(masked_scores - safe_row_max), 0)
    normalizer = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(
        weights,
        normalizer,
        out=np.zeros_like(weights),
        where=normalizer > 0,
    )
    result = np.matmul(weights, value_work)
    return result.astype(query.dtype, copy=False)


def tiled_online_softmax_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    query_block_size: int = 32,
    key_block_size: int = 32,
    causal: bool = False,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if query_block_size < 1 or key_block_size < 1:
        raise ValueError("attention tile sizes must be positive")
    (batch_size, heads, query_length, value_dim), compute_dtype, allowed_mask = (
        _validate_inputs(query, key, value, mask)
    )
    key_length = key.shape[2]
    key_dim = query.shape[-1]
    query_work = query.astype(compute_dtype, copy=False)
    key_work = key.astype(compute_dtype, copy=False)
    value_work = value.astype(compute_dtype, copy=False)
    result = np.zeros(
        (batch_size, heads, query_length, value_dim),
        dtype=compute_dtype,
    )
    scale = 1 / np.sqrt(key_dim)

    for query_start in range(0, query_length, query_block_size):
        query_end = min(query_start + query_block_size, query_length)
        query_tile = query_work[:, :, query_start:query_end, :]
        tile_query_length = query_end - query_start
        running_max = np.full(
            (batch_size, heads, tile_query_length, 1),
            -np.inf,
            dtype=compute_dtype,
        )
        running_sum = np.zeros_like(running_max)
        running_output = np.zeros(
            (batch_size, heads, tile_query_length, value_dim),
            dtype=compute_dtype,
        )

        for key_start in range(0, key_length, key_block_size):
            key_end = min(key_start + key_block_size, key_length)
            key_tile = key_work[:, :, key_start:key_end, :]
            value_tile = value_work[:, :, key_start:key_end, :]
            scores = np.matmul(query_tile, np.swapaxes(key_tile, -1, -2)) * scale
            allowed = np.ones(scores.shape, dtype=np.bool_)
            if allowed_mask is not None:
                allowed &= allowed_mask[:, :, query_start:query_end, key_start:key_end]
            if causal:
                allowed &= _causal_tile_mask(
                    query_start,
                    query_end,
                    key_start,
                    key_end,
                    query_length,
                    key_length,
                )[None, None, :, :]

            masked_scores = np.where(allowed, scores, -np.inf)
            block_max = np.max(masked_scores, axis=-1, keepdims=True)
            new_max = np.maximum(running_max, block_max)
            safe_new_max = np.where(np.isfinite(new_max), new_max, 0)
            old_scale = np.zeros_like(new_max)
            previous_is_valid = np.isfinite(running_max)
            old_scale[previous_is_valid] = np.exp(
                running_max[previous_is_valid] - safe_new_max[previous_is_valid]
            )

            shifted_scores = masked_scores - safe_new_max
            exponentials = np.where(allowed, np.exp(shifted_scores), 0)
            running_sum = old_scale * running_sum + exponentials.sum(
                axis=-1,
                keepdims=True,
            )
            running_output = (
                old_scale * running_output
                + np.matmul(exponentials, value_tile)
            )
            running_max = new_max

        result[:, :, query_start:query_end, :] = np.divide(
            running_output,
            running_sum,
            out=np.zeros_like(running_output),
            where=running_sum > 0,
        )
    return result.astype(query.dtype, copy=False)


def attention_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("attention outputs must have the same shape")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "max_absolute_error": float(difference.max(initial=0)),
        "mean_absolute_error": float(difference.mean()),
    }
