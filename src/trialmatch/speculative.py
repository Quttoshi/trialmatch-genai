from __future__ import annotations

from typing import Any

from trialmatch.decoding import (
    cache_sequence_length,
    crop_cache,
    greedy_decode_cached,
    validate_decode_inputs,
)


def assert_tokenizer_compatibility(target_tokenizer: Any, draft_tokenizer: Any) -> None:
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Target and draft tokenizers do not have identical vocabularies")
    for token_name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(target_tokenizer, token_name) != getattr(draft_tokenizer, token_name):
            raise ValueError(f"Target and draft tokenizers differ in {token_name}")


def _prefill(model: Any, input_ids: Any, torch: Any) -> tuple[Any, Any]:
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    cache = outputs.past_key_values
    if cache is None:
        raise RuntimeError("Model did not return a key/value cache")
    if cache_sequence_length(cache) != input_ids.shape[1]:
        raise RuntimeError("Prefill cache length does not match prompt length")
    return outputs.logits[:, -1, :], cache


def _forward_cached(
    model: Any,
    token_ids: Any,
    cache: Any,
    torch: Any,
) -> tuple[Any, Any]:
    cache_start = cache_sequence_length(cache)
    token_count = token_ids.shape[1]
    cache_position = torch.arange(
        cache_start,
        cache_start + token_count,
        dtype=torch.long,
        device=token_ids.device,
    )
    position_ids = cache_position.unsqueeze(0).expand(token_ids.shape[0], -1)
    attention_mask = torch.ones(
        (token_ids.shape[0], cache_start + token_count),
        dtype=token_ids.dtype,
        device=token_ids.device,
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    next_cache = outputs.past_key_values
    if next_cache is None:
        raise RuntimeError("Model stopped returning the key/value cache")
    expected_length = cache_start + token_count
    actual_length = cache_sequence_length(next_cache)
    if actual_length != expected_length:
        raise RuntimeError(
            f"Cache grew to {actual_length} tokens; expected {expected_length}"
        )
    return outputs.logits, next_cache


def greedy_speculative_decode(
    target_model: Any,
    draft_model: Any,
    input_ids: Any,
    max_new_tokens: int,
    draft_tokens: int = 4,
    eos_token_id: int | None = None,
) -> tuple[Any, dict[str, int | float]]:
    import torch

    validate_decode_inputs(input_ids, max_new_tokens)
    if draft_tokens < 1:
        raise ValueError("draft_tokens must be positive")
    target_device = target_model.get_input_embeddings().weight.device
    draft_device = draft_model.get_input_embeddings().weight.device
    if target_device != draft_device or input_ids.device != target_device:
        raise ValueError("Target model, draft model, and prompt must use the same device")
    target_model.eval()
    draft_model.eval()

    target_next_logits, target_cache = _prefill(target_model, input_ids, torch)
    draft_next_logits, draft_cache = _prefill(draft_model, input_ids, torch)
    generated: list[int] = []
    proposed_token_count = 0
    accepted_token_count = 0
    rounds = 0

    while len(generated) < max_new_tokens:
        rounds += 1
        prefix_length = cache_sequence_length(target_cache)
        if cache_sequence_length(draft_cache) != prefix_length:
            raise RuntimeError("Target and draft cache lengths diverged")
        proposal_limit = min(draft_tokens, max_new_tokens - len(generated))
        proposals: list[int] = []

        for _ in range(proposal_limit):
            draft_token = draft_next_logits.argmax(dim=-1, keepdim=True)
            token_id = int(draft_token.item())
            proposals.append(token_id)
            draft_logits, draft_cache = _forward_cached(
                draft_model,
                draft_token,
                draft_cache,
                torch,
            )
            draft_next_logits = draft_logits[:, -1, :]
            if eos_token_id is not None and token_id == eos_token_id:
                break

        proposed_token_count += len(proposals)
        proposal_tensor = torch.tensor(
            [proposals],
            dtype=input_ids.dtype,
            device=target_device,
        )
        target_logits, target_cache = _forward_cached(
            target_model,
            proposal_tensor,
            target_cache,
            torch,
        )
        if cache_sequence_length(target_cache) != prefix_length + len(proposals):
            raise RuntimeError("Target verification cache has an unexpected length")

        accepted = 0
        correction_token: Any | None = None
        accepted_eos = False
        verification_logits = target_next_logits
        for proposal_index, proposal in enumerate(proposals):
            target_token = verification_logits.argmax(dim=-1, keepdim=True)
            target_token_id = int(target_token.item())
            if target_token_id != proposal:
                correction_token = target_token
                break
            accepted += 1
            if eos_token_id is not None and proposal == eos_token_id:
                accepted_eos = True
                break
            verification_logits = target_logits[:, proposal_index, :]

        accepted_token_count += accepted
        generated.extend(proposals[:accepted])
        if accepted_eos:
            break

        if correction_token is not None:
            generated.append(int(correction_token.item()))
            target_cache = crop_cache(target_cache, prefix_length + accepted)
            draft_cache = crop_cache(draft_cache, prefix_length + accepted)
            if len(generated) >= max_new_tokens:
                break
            if eos_token_id is not None and int(correction_token.item()) == eos_token_id:
                break
            target_logits, target_cache = _forward_cached(
                target_model,
                correction_token,
                target_cache,
                torch,
            )
            draft_logits, draft_cache = _forward_cached(
                draft_model,
                correction_token.to(draft_device),
                draft_cache,
                torch,
            )
            target_next_logits = target_logits[:, -1, :]
            draft_next_logits = draft_logits[:, -1, :]
            continue

        if accepted != len(proposals):
            raise RuntimeError("Speculative verification stopped without a correction token")
        if len(generated) >= max_new_tokens:
            break

        bonus_token = target_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(bonus_token.item()))
        if eos_token_id is not None and int(bonus_token.item()) == eos_token_id:
            break
        if len(generated) >= max_new_tokens:
            break

        target_logits, target_cache = _forward_cached(
            target_model,
            bonus_token,
            target_cache,
            torch,
        )
        draft_logits, draft_cache = _forward_cached(
            draft_model,
            bonus_token.to(draft_device),
            draft_cache,
            torch,
        )
        target_next_logits = target_logits[:, -1, :]
        draft_next_logits = draft_logits[:, -1, :]

    output = torch.tensor([generated], dtype=input_ids.dtype, device=target_device)
    stats: dict[str, int | float] = {
        "draft_tokens": draft_tokens,
        "proposed_tokens": proposed_token_count,
        "accepted_tokens": accepted_token_count,
        "rounds": rounds,
        "acceptance_rate": (
            accepted_token_count / proposed_token_count if proposed_token_count else 0.0
        ),
    }
    return output, stats


def verify_speculative_agreement(
    target_model: Any,
    draft_model: Any,
    input_ids: Any,
    max_new_tokens: int,
    draft_tokens: int = 4,
    eos_token_id: int | None = None,
) -> dict[str, Any]:
    reference = greedy_decode_cached(
        target_model,
        input_ids,
        max_new_tokens,
        eos_token_id,
    )
    speculative, stats = greedy_speculative_decode(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens,
        draft_tokens,
        eos_token_id,
    )
    if reference.shape != speculative.shape:
        first_mismatch = min(reference.shape[-1], speculative.shape[-1])
        matches = False
    else:
        mismatch_positions = (reference != speculative).nonzero(as_tuple=False)
        first_mismatch = (
            int(mismatch_positions[0, -1].item())
            if mismatch_positions.numel()
            else None
        )
        matches = first_mismatch is None
    return {
        "matches": matches,
        "first_mismatch": first_mismatch,
        "reference_tokens": reference.tolist(),
        "speculative_tokens": speculative.tolist(),
        "stats": stats,
    }
