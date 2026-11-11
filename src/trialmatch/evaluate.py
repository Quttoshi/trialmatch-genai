from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trialmatch.data import extract_numbered_sentences, read_jsonl
from trialmatch.evaluation import make_prediction_record, score_predictions
from trialmatch.training import MAX_TARGET_TOKENS, build_prompt_ids
from trialmatch.train import DEFAULT_MODEL


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy baseline and optional fine-tuned model evaluation."
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required to evaluate the held-out test split after all tuning decisions are fixed.",
    )
    return parser.parse_args()


def _write_results(
    output_dir: Path,
    name: str,
    predictions: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    result_dir = output_dir / name
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    metrics = score_predictions(predictions)
    metrics["run"] = metadata
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{name}: accuracy={metrics['classification_overall']['accuracy']:.4f}, "
          f"macro_f1={metrics['classification_overall']['macro_f1']:.4f}")


def _generate(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_seq_len: int,
    torch: Any,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    input_device = model.get_input_embeddings().weight.device
    for record in records:
        prompt = build_prompt_ids(
            tokenizer,
            record,
            max_prompt_length=max_seq_len - MAX_TARGET_TOKENS,
        )
        input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=input_device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=MAX_TARGET_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        prediction = make_prediction_record(record, raw_output)
        prompt_sentence_ids = set(extract_numbered_sentences(prompt["note"]))
        prediction["out_of_context_evidence_ids"] = [
            sentence_id
            for sentence_id in prediction["predicted_evidence_sentence_ids"]
            if sentence_id not in prompt_sentence_ids
        ]
        if prediction["out_of_context_evidence_ids"]:
            prediction["explanation"] += (
                " These cited sentences were not visible after input truncation: "
                f"{prediction['out_of_context_evidence_ids']}."
            )
        prediction["prompt_truncated"] = prompt["truncated"]
        prediction["visible_sentence_ids"] = sorted(prompt_sentence_ids)
        predictions.append(prediction)
    return predictions


def main() -> None:
    args = _parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "The test split is locked. Pass --allow-test only for the final evaluation "
            "after all model, prompt, and threshold decisions are fixed."
        )
    if args.max_seq_len < MAX_TARGET_TOKENS + 64:
        raise ValueError(
            f"--max-seq-len must be at least {MAX_TARGET_TOKENS + 64}"
        )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            'Install the optional training dependencies with "pip install -e .[train]".'
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Evaluation currently requires a CUDA GPU for 4-bit model loading. "
            "Use the RTX 3050 for inference or a free Colab/Kaggle GPU."
        )
    records = read_jsonl(args.data_dir / f"{args.split}.jsonl")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = True
    model.eval()
    run_metadata = {
        "model_name": args.model_name,
        "split": args.split,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": MAX_TARGET_TOKENS,
        "decoding": "greedy",
        "adapter": None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_predictions = _generate(
        model,
        tokenizer,
        records,
        args.max_seq_len,
        torch,
    )
    _write_results(args.output_dir, "baseline", baseline_predictions, run_metadata)

    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model.eval()
        tuned_metadata = {**run_metadata, "adapter": str(args.adapter)}
        tuned_predictions = _generate(
            model,
            tokenizer,
            records,
            args.max_seq_len,
            torch,
        )
        _write_results(args.output_dir, "finetuned", tuned_predictions, tuned_metadata)


if __name__ == "__main__":
    main()
