from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trialmatch.decoding import (
    DecodeMeasurement,
    greedy_decode_cached_timed,
    greedy_decode_uncached_timed,
)
from trialmatch.train import DEFAULT_MODEL


def _parse_lengths(value: str) -> list[int]:
    try:
        lengths = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Lengths must be comma-separated integers") from error
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("Lengths must be positive integers")
    return lengths


def _make_prompt(tokenizer: Any, sequence_length: int, device: Any, torch: Any) -> Any:
    seed = tokenizer.encode(
        "Patient history and trial eligibility criterion. ",
        add_special_tokens=False,
    )
    if not seed:
        raise ValueError("Tokenizer produced an empty benchmark prompt")
    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(tokenizer.bos_token_id)
    while len(token_ids) < sequence_length:
        token_ids.extend(seed)
    return torch.tensor(
        [token_ids[:sequence_length]],
        dtype=torch.long,
        device=device,
    )


def _measure_path(
    decode: Callable[..., DecodeMeasurement],
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | None,
    repeats: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    decode(model, input_ids, min(max_new_tokens, 3), eos_token_id)
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    output_tokens: list[list[int]] = []
    peak_allocations: list[int] = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        measurement = decode(model, input_ids, max_new_tokens, eos_token_id)
        torch.cuda.synchronize(device)
        prefill_samples.append(measurement.prefill_seconds)
        decode_samples.append(measurement.decode_seconds)
        output_tokens.append(measurement.tokens.tolist()[0])
        peak_allocations.append(
            max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
        )

    median_prefill = statistics.median(prefill_samples)
    median_decode = statistics.median(decode_samples)
    median_total = median_prefill + median_decode
    token_count = statistics.median(len(tokens) for tokens in output_tokens)
    return {
        "prefill_seconds_median": median_prefill,
        "decode_seconds_median": median_decode,
        "total_seconds_median": median_total,
        "output_tokens_median": token_count,
        "tokens_per_second": token_count / median_total if median_total else None,
        "peak_incremental_memory_bytes_median": int(statistics.median(peak_allocations)),
        "outputs": output_tokens,
    }


def run_benchmarks(
    model: Any,
    tokenizer: Any,
    *,
    input_lengths: list[int],
    output_lengths: list[int],
    repeats: int,
    torch: Any,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    device = model.get_input_embeddings().weight.device
    if device.type != "cuda":
        raise ValueError("Decoding benchmark requires a CUDA model")
    model.eval()
    cases = []
    for input_length in input_lengths:
        input_ids = _make_prompt(tokenizer, input_length, device, torch)
        for output_length in output_lengths:
            cached = _measure_path(
                greedy_decode_cached_timed,
                model,
                input_ids,
                output_length,
                tokenizer.eos_token_id,
                repeats,
                torch,
                device,
            )
            uncached = _measure_path(
                greedy_decode_uncached_timed,
                model,
                input_ids,
                output_length,
                tokenizer.eos_token_id,
                repeats,
                torch,
                device,
            )
            cached_outputs = cached.pop("outputs")
            uncached_outputs = uncached.pop("outputs")
            first_mismatch = None
            if cached_outputs[0] != uncached_outputs[0]:
                first_mismatch = next(
                    (
                        index
                        for index, (cached_token, uncached_token) in enumerate(
                            zip(cached_outputs[0], uncached_outputs[0])
                        )
                        if cached_token != uncached_token
                    ),
                    min(len(cached_outputs[0]), len(uncached_outputs[0])),
                )
            cases.append(
                {
                    "input_tokens": input_length,
                    "requested_output_tokens": output_length,
                    "greedy_token_agreement": first_mismatch is None,
                    "first_mismatch": first_mismatch,
                    "cached": cached,
                    "uncached_reference": uncached,
                }
            )
    return {
        "model": getattr(model.config, "_name_or_path", None),
        "device": str(device),
        "repeats": repeats,
        "decoding": "deterministic greedy",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KV-cached and uncached decoding.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--input-lengths", type=_parse_lengths, default=[128, 512, 1024])
    parser.add_argument("--output-lengths", type=_parse_lengths, default=[16, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/decoding_benchmark.json"))
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            'Install the optional training dependencies with "pip install -e .[train]".'
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Run the decoding benchmark on a CUDA GPU runtime")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results = run_benchmarks(
        model,
        tokenizer,
        input_lengths=args.input_lengths,
        output_lengths=args.output_lengths,
        repeats=args.repeats,
        torch=torch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
