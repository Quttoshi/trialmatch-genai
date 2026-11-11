from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from trialmatch.attention import (
    attention_error,
    scaled_dot_product_attention,
    tiled_online_softmax_attention,
)


def _median_latency(function: Callable[[], Any], repeats: int) -> float:
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def run_benchmark(
    *,
    query_length: int,
    key_length: int,
    heads: int,
    head_dim: int,
    query_block_size: int,
    key_block_size: int,
    repeats: int,
) -> dict[str, Any]:
    if min(query_length, key_length, heads, head_dim, repeats) < 1:
        raise ValueError("sequence lengths, dimensions, and repeats must be positive")
    random = np.random.default_rng(42)
    query = random.normal(size=(1, heads, query_length, head_dim)).astype(np.float32)
    key = random.normal(size=(1, heads, key_length, head_dim)).astype(np.float32)
    value = random.normal(size=(1, heads, key_length, head_dim)).astype(np.float32)

    reference = scaled_dot_product_attention(query, key, value, causal=True)
    tiled = tiled_online_softmax_attention(
        query,
        key,
        value,
        causal=True,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
    )
    error = attention_error(reference, tiled)
    if error["max_absolute_error"] > 2e-5:
        raise RuntimeError(f"Tiled attention failed the numerical check: {error}")

    reference_latency = _median_latency(
        lambda: scaled_dot_product_attention(query, key, value, causal=True),
        repeats,
    )
    tiled_latency = _median_latency(
        lambda: tiled_online_softmax_attention(
            query,
            key,
            value,
            causal=True,
            query_block_size=query_block_size,
            key_block_size=key_block_size,
        ),
        repeats,
    )
    reference_score_bytes = (
        query.shape[0]
        * heads
        * query_length
        * key_length
        * np.dtype(np.float32).itemsize
    )
    tiled_score_tile_bytes = (
        query.shape[0]
        * heads
        * min(query_block_size, query_length)
        * min(key_block_size, key_length)
        * np.dtype(np.float32).itemsize
    )
    return {
        "device": "CPU",
        "shape": {
            "batch": 1,
            "heads": heads,
            "query_length": query_length,
            "key_length": key_length,
            "head_dim": head_dim,
        },
        "tiles": {
            "query_block_size": query_block_size,
            "key_block_size": key_block_size,
        },
        "repeats": repeats,
        "reference_median_seconds": reference_latency,
        "tiled_median_seconds": tiled_latency,
        "reference_over_tiled_speed_ratio": (
            reference_latency / tiled_latency if tiled_latency else None
        ),
        "numerical_error": error,
        "estimated_score_tensor_bytes": {
            "reference_full_score_matrix": reference_score_bytes,
            "tiled_score_tile": tiled_score_tile_bytes,
            "note": "Theoretical score-tensor sizes only, not process peak memory.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reference and tiled attention.")
    parser.add_argument("--query-length", type=int, default=256)
    parser.add_argument("--key-length", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--query-block-size", type=int, default=32)
    parser.add_argument("--key-block-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        query_length=args.query_length,
        key_length=args.key_length,
        heads=args.heads,
        head_dim=args.head_dim,
        query_block_size=args.query_block_size,
        key_block_size=args.key_block_size,
        repeats=args.repeats,
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
