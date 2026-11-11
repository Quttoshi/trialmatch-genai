from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from trialmatch.app_logic import (
    application_result,
    normalize_patient_record,
    parse_criteria,
    render_screening_report,
)
from trialmatch.data import extract_numbered_sentences
from trialmatch.decoding import (
    greedy_decode_cached_timed,
    greedy_decode_uncached_timed,
)
from trialmatch.evaluation import parse_model_output
from trialmatch.prompts import build_messages
from trialmatch.speculative import (
    assert_tokenizer_compatibility,
    greedy_speculative_decode,
)
from trialmatch.train import DEFAULT_MODEL
from trialmatch.training import MAX_TARGET_TOKENS, build_prompt_ids

MAX_SEQUENCE_LENGTH = 1024
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@st.cache_resource
def load_target_model(model_name: str, adapter_path: str) -> tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            'Install the optional app and training dependencies with "pip install -e .[app,train]".'
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("The application requires a CUDA GPU for 4-bit target-model inference")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    if adapter_path:
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_draft_model(model_name: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            'Install the optional app and training dependencies with "pip install -e .[app,train]".'
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("Speculative decoding requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _predict_one(
    model: Any,
    tokenizer: Any,
    note: str,
    criterion: dict[str, str],
    torch: Any,
) -> dict[str, Any]:
    record = {
        "note": note,
        "criterion_type": criterion["criterion_type"],
        "criterion_text": criterion["criterion_text"],
    }
    prompt = build_prompt_ids(
        tokenizer,
        record,
        max_prompt_length=MAX_SEQUENCE_LENGTH - MAX_TARGET_TOKENS,
    )
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=MAX_TARGET_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw_output = tokenizer.decode(
        output[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    prediction = parse_model_output(raw_output)
    result = application_result(criterion, note, prediction)
    result["prompt_truncated"] = prompt["truncated"]
    result["visible_sentence_ids"] = sorted(
        extract_numbered_sentences(prompt["note"])
    )
    outside_context = [
        sentence_id
        for sentence_id in result["cited_sentence_ids"]
        if sentence_id not in result["visible_sentence_ids"]
    ]
    result["out_of_context_sentence_ids"] = outside_context
    if outside_context:
        result["explanation"] += (
            f" These cited sentences were not visible after input truncation: {outside_context}."
        )
    return result


def _show_benchmark(
    model_name: str,
    adapter_path: str,
    note: str,
    criterion: dict[str, str],
    mode: str,
    output_tokens: int,
) -> None:
    try:
        import torch
        target_model, tokenizer = load_target_model(model_name, adapter_path)
        prompt = build_prompt_ids(
            tokenizer,
            {
                "note": note,
                "criterion_type": criterion["criterion_type"],
                "criterion_text": criterion["criterion_text"],
            },
            max_prompt_length=MAX_SEQUENCE_LENGTH - MAX_TARGET_TOKENS,
        )
        device = target_model.get_input_embeddings().weight.device
        input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)

        if mode == "KV cache vs uncached":
            reference = greedy_decode_uncached_timed(
                target_model,
                input_ids,
                output_tokens,
                tokenizer.eos_token_id,
            )
            cached = greedy_decode_cached_timed(
                target_model,
                input_ids,
                output_tokens,
                tokenizer.eos_token_id,
            )
            st.write(
                {
                    "token_agreement": bool(torch.equal(reference.tokens, cached.tokens)),
                    "uncached_prefill_seconds": reference.prefill_seconds,
                    "uncached_decode_seconds": reference.decode_seconds,
                    "uncached_tokens_per_second": (
                        reference.tokens.shape[-1] / reference.total_seconds
                        if reference.total_seconds
                        else None
                    ),
                    "cached_prefill_seconds": cached.prefill_seconds,
                    "cached_decode_seconds": cached.decode_seconds,
                    "cached_tokens_per_second": (
                        cached.tokens.shape[-1] / cached.total_seconds
                        if cached.total_seconds
                        else None
                    ),
                    "prompt_truncated": prompt["truncated"],
                }
            )
        else:
            draft_model, draft_tokenizer = load_draft_model(DRAFT_MODEL)
            assert_tokenizer_compatibility(tokenizer, draft_tokenizer)
            reference = greedy_decode_cached_timed(
                target_model,
                input_ids,
                output_tokens,
                tokenizer.eos_token_id,
            )
            started = time.perf_counter()
            speculative, stats = greedy_speculative_decode(
                target_model,
                draft_model,
                input_ids,
                output_tokens,
                draft_tokens=4,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            st.write(
                {
                    "token_agreement": bool(torch.equal(reference.tokens, speculative)),
                    "target_greedy_seconds": reference.total_seconds,
                    "target_greedy_tokens_per_second": (
                        reference.tokens.shape[-1] / reference.total_seconds
                        if reference.total_seconds
                        else None
                    ),
                    "speculative_seconds": elapsed,
                    "speculative_tokens_per_second": (
                        speculative.shape[-1] / elapsed if elapsed else None
                    ),
                    **stats,
                }
            )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Benchmark failed: {error}")


def main() -> None:
    st.set_page_config(page_title="TrialMatch Screening", layout="wide")
    st.title("TrialMatch: Clinical Trial Eligibility Screening")
    st.warning(
        "Educational screening prototype only. This tool is not a clinical enrollment "
        "decision and must not be used as medical advice."
    )
    model_name = os.getenv("TRIALMATCH_MODEL", DEFAULT_MODEL)
    adapter_path = os.getenv("TRIALMATCH_ADAPTER", "")
    st.caption(f"Target model: {model_name}; QLoRA adapter: {adapter_path or 'none'}")

    with st.form("screening_form"):
        patient_text = st.text_area(
            "Patient record",
            height=220,
            help="Enter one patient sentence per line. Sentence IDs are assigned in order.",
        )
        criteria_text = st.text_area(
            "Trial eligibility criteria",
            height=180,
            help="Enter one criterion per line, for example 'inclusion: Age 18 or older'.",
        )
        submitted = st.form_submit_button("Screen criteria")

    if submitted:
        st.session_state.pop("screening_results", None)
        st.session_state.pop("screening_request", None)
        try:
            note = normalize_patient_record(patient_text)
            criteria = parse_criteria(criteria_text)
        except ValueError as error:
            st.error(str(error))
        else:
            try:
                import torch
                model, tokenizer = load_target_model(model_name, adapter_path)
                results = []
                with st.spinner("Screening criterion-level inputs..."):
                    for criterion in criteria:
                        results.append(_predict_one(model, tokenizer, note, criterion, torch))
                st.session_state["screening_results"] = results
                st.session_state["screening_request"] = {
                    "note": note,
                    "criteria": criteria,
                }
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"Screening failed: {error}")

    results = st.session_state.get("screening_results", [])
    request = st.session_state.get("screening_request")
    if results:
        st.subheader("Criterion-level results")
        for index, result in enumerate(results, start=1):
            with st.expander(
                f"{index}. {result['criterion_type']}: "
                f"{result['predicted_eligibility']}",
                expanded=True,
            ):
                st.write(result["criterion_text"])
                st.write(result["explanation"])
                if result["evidence_text"]:
                    st.write("Exact source evidence")
                    for item in result["evidence_text"]:
                        st.write(f"Sentence {item['sentence_id']}: {item['text']}")
                if result["prompt_truncated"]:
                    st.warning("The patient record was truncated to fit the model context.")
                if result["out_of_context_sentence_ids"]:
                    st.warning(
                        "The model cited sentence IDs not visible after truncation: "
                        f"{result['out_of_context_sentence_ids']}"
                    )
        report = render_screening_report(results)
        st.download_button(
            "Download screening report",
            data=report,
            file_name="trialmatch_screening_report.md",
            mime="text/markdown",
        )

    st.subheader("Inference comparison")
    benchmark_mode = st.selectbox(
        "Benchmark mechanism",
        ["KV cache vs uncached", "Speculative vs target greedy"],
    )
    output_tokens = st.select_slider(
        "Maximum generated tokens",
        options=[8, 16, 32, 64],
        value=16,
    )
    if not request:
        st.info("Run a screening first to use the inference comparison on the same input.")
    elif st.button("Run inference comparison"):
        _show_benchmark(
            model_name,
            adapter_path,
            request["note"],
            request["criteria"][0],
            benchmark_mode,
            output_tokens,
        )


if __name__ == "__main__":
    main()
