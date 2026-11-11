from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def cache_sequence_length(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        length = cache.get_seq_length()
        return int(length.item()) if hasattr(length, "item") else int(length)
    if not isinstance(cache, (tuple, list)) or not cache:
        raise TypeError("Unsupported or empty key/value cache")
    first_layer = cache[0]
    if not isinstance(first_layer, (tuple, list)) or not first_layer:
        raise TypeError("Legacy key/value cache has an invalid layer entry")
    key_tensor = first_layer[0]
    if not hasattr(key_tensor, "shape") or len(key_tensor.shape) < 3:
        raise TypeError("Key cache tensor must have a sequence dimension")
    return int(key_tensor.shape[-2])


def crop_cache(cache: Any, sequence_length: int) -> Any:
    current_length = cache_sequence_length(cache)
    if sequence_length < 0 or sequence_length > current_length:
        raise ValueError(
            f"Cannot crop cache of length {current_length} to {sequence_length}"
        )
    if hasattr(cache, "crop"):
        cropped = cache.crop(sequence_length)
        result = cache if cropped is None else cropped
    elif isinstance(cache, (tuple, list)):
        layers = []
        for layer in cache:
            if not isinstance(layer, (tuple, list)):
                raise TypeError("Legacy cache layer must be a tuple or list")
            layers.append(
                tuple(
                    tensor[..., :sequence_length, :]
                    if hasattr(tensor, "shape") and len(tensor.shape) >= 3
                    else tensor
                    for tensor in layer
                )
            )
        result = tuple(layers)
    else:
        raise TypeError("Cache cannot be cropped with this model implementation")
    if cache_sequence_length(result) != sequence_length:
        raise RuntimeError("Cache crop did not produce the requested sequence length")
    return result


def validate_decode_inputs(input_ids: Any, max_new_tokens: int) -> None:
    if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
        raise ValueError("Greedy decoding expects input_ids with shape [1, sequence_length]")
    if input_ids.shape[1] < 1:
        raise ValueError("Prompt must contain at least one token")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class DecodeMeasurement:
    tokens: Any
    prefill_seconds: float
    decode_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds


def _synchronize(device: Any, torch: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def greedy_decode_uncached_timed(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> DecodeMeasurement:
    import torch

    validate_decode_inputs(input_ids, max_new_tokens)
    model.eval()
    generated = input_ids
    prefill_seconds = 0.0
    decode_seconds = 0.0
    for step in range(max_new_tokens):
        _synchronize(input_ids.device, torch)
        started = time.perf_counter()
        attention_mask = torch.ones_like(generated)
        with torch.inference_mode():
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=-1)
        _synchronize(input_ids.device, torch)
        elapsed = time.perf_counter() - started
        if step == 0:
            prefill_seconds = elapsed
        else:
            decode_seconds += elapsed
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return DecodeMeasurement(
        tokens=generated[:, input_ids.shape[1] :],
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
    )


def greedy_decode_cached_timed(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> DecodeMeasurement:
    import torch

    validate_decode_inputs(input_ids, max_new_tokens)
    model.eval()
    _synchronize(input_ids.device, torch)
    started = time.perf_counter()
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    next_logits = outputs.logits[:, -1, :]
    next_token = next_logits.argmax(dim=-1, keepdim=True)
    generated_tokens = [next_token]
    _synchronize(input_ids.device, torch)
    prefill_seconds = time.perf_counter() - started

    cache = outputs.past_key_values
    if cache is None:
        raise RuntimeError("Model did not return a key/value cache")
    cache_length = cache_sequence_length(cache)
    if cache_length != input_ids.shape[1]:
        raise RuntimeError(
            f"Prefill cache length {cache_length} does not match prompt length "
            f"{input_ids.shape[1]}"
        )

    decode_seconds = 0.0
    for _ in range(1, max_new_tokens):
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

        position = cache_sequence_length(cache)
        cache_position = torch.arange(
            position,
            position + 1,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = cache_position.unsqueeze(0)
        attention_mask = torch.ones(
            (input_ids.shape[0], position + 1),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        _synchronize(input_ids.device, torch)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        next_logits = outputs.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        _synchronize(input_ids.device, torch)
        decode_seconds += time.perf_counter() - started

        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("Model stopped returning the key/value cache")
        expected_length = position + 1
        actual_length = cache_sequence_length(cache)
        if actual_length != expected_length:
            raise RuntimeError(
                f"Cache grew to {actual_length} tokens; expected {expected_length}"
            )
    return DecodeMeasurement(
        tokens=torch.cat(generated_tokens, dim=-1),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
    )


def greedy_decode_uncached(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Any:
    return greedy_decode_uncached_timed(
        model,
        input_ids,
        max_new_tokens,
        eos_token_id,
    ).tokens


def greedy_decode_cached(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Any:
    return greedy_decode_cached_timed(
        model,
        input_ids,
        max_new_tokens,
        eos_token_id,
    ).tokens


def verify_greedy_agreement(
    model: Any,
    input_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> dict[str, Any]:
    reference = greedy_decode_uncached(
        model,
        input_ids,
        max_new_tokens,
        eos_token_id,
    )
    cached = greedy_decode_cached(
        model,
        input_ids,
        max_new_tokens,
        eos_token_id,
    )
    if reference.shape != cached.shape:
        return {
            "matches": False,
            "reference_tokens": reference.tolist(),
            "cached_tokens": cached.tolist(),
            "first_mismatch": min(reference.shape[-1], cached.shape[-1]),
        }
    mismatches = (reference != cached).nonzero(as_tuple=False)
    first_mismatch = int(mismatches[0, -1].item()) if mismatches.numel() else None
    return {
        "matches": first_mismatch is None,
        "reference_tokens": reference.tolist(),
        "cached_tokens": cached.tolist(),
        "first_mismatch": first_mismatch,
    }
