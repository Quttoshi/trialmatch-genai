from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

from trialmatch.data import extract_numbered_sentences
from trialmatch.decoding import greedy_decode_cached_timed
from trialmatch.speculative import (
    assert_tokenizer_compatibility,
    greedy_speculative_decode,
)
from trialmatch.data import read_jsonl
from trialmatch.train import DEFAULT_MODEL
from trialmatch.training import MAX_TARGET_TOKENS, build_prompt_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare target greedy decoding with speculative decoding."
    )
    parser.add_argument("--validation-file", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--target-model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-adapter", type=Path)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--draft-token-counts", default="2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/speculative_benchmark.json"))
    args = parser.parse_args()
    try:
        args.draft_token_counts = [int(value) for value in args.draft_token_counts.split(",")]
    except ValueError as error:
        parser.error("--draft-token-counts must be comma-separated integers")
        raise error
    if not args.draft_token_counts or any(value < 1 for value in args.draft_token_counts):
        parser.error("--draft-token-counts must be positive")
    if args.examples < 1 or args.repeats < 1 or args.max_new_tokens < 1:
        parser.error("--examples, --repeats, and --max-new-tokens must be positive")
    if args.max_seq_len < MAX_TARGET_TOKENS + 64:
        parser.error(f"--max-seq-len must be at least {MAX_TARGET_TOKENS + 64}")
    return args


def _sync(torch: Any, device: Any) -> None:
    torch.cuda.synchronize(device)


def _measure_speculative(
    target_model: Any,
    draft_model: Any,
    input_ids: Any,
    max_new_tokens: int,
    draft_tokens: int,
    eos_token_id: int | None,
    repeats: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    greedy_speculative_decode(
        target_model,
        draft_model,
        input_ids,
        min(max_new_tokens, 4),
        draft_tokens,
        eos_token_id,
    )
    _sync(torch, device)
    gc.collect()
    torch.cuda.empty_cache()
    elapsed_samples: list[float] = []
    output_sequences: list[list[int]] = []
    stats_samples: list[dict[str, int | float]] = []
    peak_increments: list[int] = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        _sync(torch, device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        output, stats = greedy_speculative_decode(
            target_model,
            draft_model,
            input_ids,
            max_new_tokens,
            draft_tokens,
            eos_token_id,
        )
        _sync(torch, device)
        elapsed_samples.append(time.perf_counter() - started)
        output_sequences.append(output.tolist()[0])
        stats_samples.append(stats)
        peak_increments.append(
            max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
        )
    return {
        "median_seconds": statistics.median(elapsed_samples),
        "output_tokens_median": statistics.median(
            len(sequence) for sequence in output_sequences
        ),
        "tokens_per_second": (
            statistics.median(len(sequence) for sequence in output_sequences)
            / statistics.median(elapsed_samples)
        ),
        "acceptance_rate_mean": statistics.mean(
            float(stats["acceptance_rate"]) for stats in stats_samples
        ),
        "proposed_tokens_mean": statistics.mean(
            int(stats["proposed_tokens"]) for stats in stats_samples
        ),
        "accepted_tokens_mean": statistics.mean(
            int(stats["accepted_tokens"]) for stats in stats_samples
        ),
        "rounds_mean": statistics.mean(int(stats["rounds"]) for stats in stats_samples),
        "peak_incremental_memory_bytes_median": int(
            statistics.median(peak_increments)
        ),
        "outputs": output_sequences,
        "stats": stats_samples,
    }


def _measure_target_greedy(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | None,
    repeats: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    greedy_decode_cached_timed(
        model,
        input_ids,
        min(max_new_tokens, 4),
        eos_token_id,
    )
    _sync(torch, device)
    gc.collect()
    torch.cuda.empty_cache()
    measurements = []
    outputs = []
    peak_increments = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        _sync(torch, device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        measurement = greedy_decode_cached_timed(
            model,
            input_ids,
            max_new_tokens,
            eos_token_id,
        )
        _sync(torch, device)
        measurements.append(measurement)
        outputs.append(measurement.tokens.tolist()[0])
        peak_increments.append(
            max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
        )
    median_total = statistics.median(item.total_seconds for item in measurements)
    median_tokens = statistics.median(len(output) for output in outputs)
    return {
        "seconds": median_total,
        "prefill_seconds_median": statistics.median(
            item.prefill_seconds for item in measurements
        ),
        "decode_seconds_median": statistics.median(
            item.decode_seconds for item in measurements
        ),
        "tokens_per_second": median_tokens / median_total if median_total else None,
        "peak_incremental_memory_bytes_median": int(
            statistics.median(peak_increments)
        ),
        "outputs": outputs,
    }


def run_benchmark(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_seq_len: int,
    max_new_tokens: int,
    draft_token_counts: list[int],
    repeats: int,
    torch: Any,
) -> dict[str, Any]:
    target_device = target_model.get_input_embeddings().weight.device
    draft_device = draft_model.get_input_embeddings().weight.device
    if target_device != draft_device or target_device.type != "cuda":
        raise ValueError("Target and draft must be loaded on the same CUDA device")
    target_model.eval()
    draft_model.eval()
    cases = []
    for record in records:
        prompt = build_prompt_ids(
            tokenizer,
            record,
            max_prompt_length=max_seq_len - MAX_TARGET_TOKENS,
        )
        input_ids = torch.tensor(
            [prompt["input_ids"]],
            dtype=torch.long,
            device=target_device,
        )
        reference = _measure_target_greedy(
            target_model,
            input_ids,
            max_new_tokens,
            tokenizer.eos_token_id,
            repeats,
            torch,
            target_device,
        )
        reference_outputs = reference.pop("outputs")
        case = {
            "annotation_id": record["annotation_id"],
            "criterion_type": record["criterion_type"],
            "input_tokens": len(prompt["input_ids"]),
            "prompt_truncated": prompt["truncated"],
            "target_greedy": {**reference, "output_tokens": reference_outputs[0]},
            "draft_token_counts": {},
        }
        for draft_tokens in draft_token_counts:
            result = _measure_speculative(
                target_model,
                draft_model,
                input_ids,
                max_new_tokens,
                draft_tokens,
                tokenizer.eos_token_id,
                repeats,
                torch,
                target_device,
            )
            sequences = result.pop("outputs")
            run_stats = result.pop("stats")
            reference_tokens = reference_outputs[0]
            speculative_tokens = sequences[0]
            first_mismatch = None
            if reference_tokens != speculative_tokens:
                first_mismatch = next(
                    (
                        index
                        for index, (expected, actual) in enumerate(
                            zip(reference_tokens, speculative_tokens)
                        )
                        if expected != actual
                    ),
                    min(len(reference_tokens), len(speculative_tokens)),
                )
            case["draft_token_counts"][str(draft_tokens)] = {
                **result,
                "greedy_token_agreement": first_mismatch is None,
                "first_mismatch": first_mismatch,
                "per_repeat_stats": run_stats,
            }
        cases.append(case)
    return {
        "target_model": getattr(target_model.config, "_name_or_path", None),
        "draft_model": getattr(draft_model.config, "_name_or_path", None),
        "decoding": "greedy speculative decoding",
        "max_new_tokens": max_new_tokens,
        "repeats": repeats,
        "cases": cases,
    }


def main() -> None:
    args = _parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            'Install the optional training dependencies with "pip install -e .[train]".'
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Run speculative decoding benchmarks on a CUDA GPU runtime")

    records = read_jsonl(args.validation_file)[: args.examples]
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    assert_tokenizer_compatibility(target_tokenizer, draft_tokenizer)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    if args.target_adapter is not None:
        target_model = PeftModel.from_pretrained(
            target_model,
            args.target_adapter,
            is_trainable=False,
        )
    results = run_benchmark(
        target_model,
        draft_model,
        target_tokenizer,
        records,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        draft_token_counts=args.draft_token_counts,
        repeats=args.repeats,
        torch=torch,
    )
    results["target_adapter"] = (
        str(args.target_adapter) if args.target_adapter is not None else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
