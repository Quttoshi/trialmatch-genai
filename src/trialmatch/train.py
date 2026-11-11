from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trialmatch.data import read_jsonl
from trialmatch.training import encode_supervised_record

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for TrialGPT criterion labels.")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen25-1.5b-qlora"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from the latest checkpoint in the output directory when available.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            'Install the optional training dependencies with "pip install -e .[train]".'
        ) from error

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a CUDA GPU. Run this script on a free Colab/Kaggle "
            "GPU runtime or configure a supported local CUDA environment."
        )
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("--epochs and --learning-rate must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")

    train_records = read_jsonl(args.train_file)
    validation_records = read_jsonl(args.validation_file)
    set_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    class TokenizedRecords(torch.utils.data.Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.examples = [
                encode_supervised_record(tokenizer, record, args.max_seq_len)
                for record in records
            ]

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            example = self.examples[index]
            return {
                key: torch.tensor(example[key], dtype=torch.long)
                for key in ("input_ids", "attention_mask", "labels")
            }

    train_dataset = TokenizedRecords(train_records)
    validation_dataset = TokenizedRecords(validation_records)

    def truncation_details(
        records: list[dict[str, Any]],
        examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "annotation_id": record["annotation_id"],
                "original_note_tokens": example["original_note_tokens"],
                "retained_note_tokens": example["retained_note_tokens"],
                "gold_evidence_ids_omitted_from_prompt": sorted(
                    set(record["expert_sentence_ids"])
                    - set(example["visible_sentence_ids"])
                ),
            }
            for record, example in zip(records, examples)
            if example["truncated"]
        ]

    train_truncation = truncation_details(train_records, train_dataset.examples)
    validation_truncation = truncation_details(
        validation_records,
        validation_dataset.examples,
    )
    truncation_metadata = {
        "max_seq_len": args.max_seq_len,
        "train_examples": len(train_records),
        "train_truncated_examples": len(train_truncation),
        "validation_examples": len(validation_records),
        "validation_truncated_examples": len(validation_truncation),
        "train_truncation_details": train_truncation,
        "validation_truncation_details": validation_truncation,
        "train_label_counts": {
            label: sum(record["expert_eligibility"] == label for record in train_records)
            for label in sorted({record["expert_eligibility"] for record in train_records})
        },
        "validation_label_counts": {
            label: sum(record["expert_eligibility"] == label for record in validation_records)
            for label in sorted(
                {record["expert_eligibility"] for record in validation_records}
            )
        },
        "max_original_note_tokens": max(
            example["original_note_tokens"]
            for example in train_dataset.examples + validation_dataset.examples
        ),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(truncation_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        processing_class=tokenizer,
    )
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        from transformers.trainer_utils import get_last_checkpoint

        resume_checkpoint = get_last_checkpoint(str(args.output_dir))
        if resume_checkpoint:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print(f"No checkpoint found in {args.output_dir}; starting a new run.")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    (args.output_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved adapter and truncation metadata to {args.output_dir}")
    print(json.dumps(truncation_metadata, indent=2))


if __name__ == "__main__":
    main()
