# TrialMatch: Clinical Trial Eligibility Research Prototype

This repository is being built for Generative AI Assignment 2. It uses the required [TrialGPT Criterion Annotations dataset](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations), preserves the expert eligibility categories, and separates patient records before model training.

## Selected model pair

- Target: `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, Apache-2.0)
- Draft: `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters, Apache-2.0)

The pair shares the Qwen2.5 tokenizer, which is required for the planned speculative-decoding comparison. The target model is intended for QLoRA training on a free Colab or Kaggle GPU if the local RTX 3050's 4GB VRAM is insufficient. Model weights and adapters will not be committed to this repository.

## Architecture comparison

| Candidate target | Parameters and architecture | License | Memory / hardware considerations | Decision |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.49B; 24 decoder layers; GQA with 14 query and 2 KV heads; 32,768-token context | Apache-2.0 | At least about 0.23 GiB for 4-bit parameter payload alone; actual runtime/training memory is higher due to quantization metadata, activations, KV cache, and framework overhead. Most practical on the 4GB RTX 3050. | Retained as draft; using it as target would require sourcing another smaller model with a compatible tokenizer. |
| Qwen2.5-1.5B-Instruct | 1.54B; 28 decoder layers; GQA with 12 query and 2 KV heads; 32,768-token context | Apache-2.0 | At least about 0.72 GiB for 4-bit parameter payload alone; actual training footprint is substantially higher. QLoRA on the local 4GB GPU is not assumed to fit; use free Colab/Kaggle GPU for training and measure local inference separately. | Selected target; stronger capacity than 0.5B and paired with the tokenizer-compatible 0.5B draft. |

The 4-bit figures are lower-bound arithmetic (parameter count × 4 bits), not measured peak VRAM. The model cards document the architecture and context lengths; training and inference memory must be measured and reported rather than inferred from weight size.

## Data integrity

The data-preparation command will sort the unique patient IDs, shuffle with NumPy `default_rng(42)`, and assign the first 37 patients to training, the next 8 to validation, and the remaining 8 to test. It writes the patient-ID manifest and separate JSONL files. Model inputs contain only the patient note, criterion type, and criterion text. Dataset-generated GPT-4 predictions and explanations are excluded; only expert labels and expert evidence IDs are used as targets.

The source dataset has one annotation with a missing criterion text. It is retained with an empty `criterion_text` and an explicit `source_missing_fields` marker; no criterion is invented and the expert label is preserved.

Do not use the test examples to select prompts, hyperparameters, or thresholds. Run test evaluation only after the pipeline and decisions are fixed.

## Initial setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Colab/Kaggle, use the notebook's available Python/PyTorch environment, install this package with `pip install -e .`, and install the optional training dependencies only when needed.

## Fine-tune on Colab/Kaggle

For an interactive Colab workflow covering split preparation, training, validation, optional final testing, and benchmarks, use [`notebooks/colab_training_and_evaluation.ipynb`](./notebooks/colab_training_and_evaluation.ipynb). The command-line steps below are also available for manual runs.

After cloning or uploading this project into a GPU runtime, prepare the same split and run:

```bash
pip install -e ".[train,test]"
python -m trialmatch.prepare_data --output-dir data/processed
python -m trialmatch.train \
  --train-file data/processed/train.jsonl \
  --validation-file data/processed/validation.jsonl \
  --output-dir /content/drive/MyDrive/trialmatch/qwen25-1.5b-qlora \
  --max-seq-len 1024 \
  --epochs 3
```

The trainer uses 4-bit NF4 with double quantization, LoRA rank 8, microbatch 1, gradient accumulation, and gradient checkpointing. It selects the best checkpoint by validation loss and writes training logs, label counts, and per-example truncation metadata, including gold evidence IDs omitted from the prompt. To fit the context budget, it removes trailing complete numbered note sentences first; if needed, it token-truncates the remaining note from the end. Criterion text and target labels are retained. The script never loads the test split.

To continue after a Colab disconnect, remount the same Google Drive and pass `--resume-from-checkpoint` when rerunning the training command. The latest saved checkpoint is resumed when present; checkpoints are saved at epoch boundaries, so work since the last completed epoch must run again.

## Evaluate on validation

The evaluator runs the base model and, when `--adapter` is supplied, the fine-tuned adapter on the same validation examples with deterministic greedy decoding:

```bash
python -m trialmatch.evaluate \
  --split validation \
  --adapter /content/drive/MyDrive/trialmatch/qwen25-1.5b-qlora/adapter \
  --output-dir outputs/validation
```

It saves prediction-level JSONL, overall and inclusion/exclusion classification metrics, confusion matrices, and evidence retrieval metrics. Malformed model outputs become an explicit `__INVALID_OUTPUT__` prediction and are counted. Evidence metrics use set-based IDs; for a record with no annotated evidence, an empty prediction is scored as an exact empty match (1 for per-record P/R/F1), while any predicted ID is scored as 0 for that record. Evidence cited from a sentence removed by input truncation is flagged. Only after model/prompt decisions are fixed, run final test evaluation with `--split test --allow-test`.

## Browser application

The Streamlit interface accepts a patient record (one sentence per line), accepts inclusion/exclusion criteria, shows each prediction with exact cited record sentences, and exports a Markdown screening report. It also exposes KV-cache and speculative-decoding comparisons on the same input.

Install a CUDA-enabled PyTorch build suitable for the runtime, then install the app dependencies and launch:

```powershell
python -m pip install -e ".[app,train]"
$env:TRIALMATCH_ADAPTER = "D:\path\to\adapter"
python -m streamlit run src\trialmatch\app.py
```

Leave `TRIALMATCH_ADAPTER` unset to use the base model. The adapter directory is loaded locally and must not be added to Git. Streamlit serves the application in a browser at its local URL; restart the server after changing the adapter.

## Inference benchmarks

On a CUDA runtime with the target model available:

```bash
python -m trialmatch.benchmark_decoding \
  --input-lengths 128,512,1024 \
  --output-lengths 16,32 \
  --output outputs/decoding_benchmark.json
python -m trialmatch.benchmark_speculative \
  --target-adapter /path/to/adapter \
  --draft-token-counts 2,4,8 \
  --output outputs/speculative_benchmark.json
```

The decoding benchmark records prefill/decode latency, output throughput, incremental peak CUDA allocation, and greedy token agreement. Speculative benchmarking uses validation examples and reports acceptance rate and any token mismatch. The NumPy tiled-attention benchmark checks output agreement and reports latency plus theoretical score-tensor sizes; it deliberately does not claim those estimates are measured process peak memory.

## Build the report

The IEEE-style source is in `report/main.tex`. From the `report` directory, build it with a LaTeX installation that includes IEEEtran:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Replace the author/course placeholders and populate the results, failure analysis, screenshots, and demo details only after running the experiments. Do not present placeholder or unmeasured values as results.

## Prepare the prescribed dataset split

```powershell
python -m trialmatch.prepare_data --output-dir data/processed
```

The command downloads the public dataset through Hugging Face Datasets, writes `split_manifest.json` and one JSONL file per split, and checks for patient leakage. Do not upload the JSONL records or model weights to GitHub.

## Current status

Implemented so far: the prescribed split/preprocessing pipeline, QLoRA training script, baseline/fine-tuned evaluator, criterion/evidence metrics, reference and tiled attention, explicit KV-cache and speculative decoding paths, benchmark commands, the Streamlit browser application, and an IEEE-style report draft. The public dataset produced the prescribed 37/8/8 patient split with no patient overlap (603/206/206 annotation rows). The report still needs measured results, failure analysis, screenshots, a demo link, and final formatting.

Verification so far: 37 unit tests pass; the split was generated from the actual public dataset and checked for patient overlap and GPT-generated field leakage. This Windows Python environment has no PyTorch/CUDA package or `pdflatex` installed, so QLoRA training, CUDA cache/speculative benchmarks, the live Streamlit app, and PDF compilation have not yet been executed. Run GPU-dependent work on free Colab/Kaggle or after configuring a supported local CUDA environment; do not report unmeasured results.

## Sources

- TrialGPT dataset card: https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations
- Qwen2.5-1.5B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Qwen2.5-0.5B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- TrialGPT reference implementation: https://github.com/ncbi-nlp/TrialGPT (reference only; not run as the assignment solution)
- Scholarly references: see [`references.bib`](./references.bib)
