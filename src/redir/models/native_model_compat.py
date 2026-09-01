"""Compatibility contracts for non-Qwen native-tool models.

The defaults in the latent-memory stack remain Qwen-compatible.  New model
families opt into this module explicitly so model loading, text-backbone
metadata and language-only LoRA targeting cannot silently fall back to Qwen
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

import torch


@dataclass(frozen=True, slots=True)
class NativeModelProfile:
    """Architecture-level choices that cannot be inferred from Qwen defaults."""

    family: str
    repo_id: str
    auto_model_kind: str
    tokenizer_kind: str
    language_model_prefix: str
    tool_protocol: str
    allow_empty_generation_suffix: bool
    default_lora_modules: tuple[str, ...]


NATIVE_MODEL_PROFILES: dict[str, NativeModelProfile] = {
    "ministral3": NativeModelProfile(
        family="ministral3",
        repo_id="mistralai/Ministral-3-8B-Instruct-2512-BF16",
        auto_model_kind="image_text_to_text",
        tokenizer_kind="mistral_common_or_fixed_regex_auto_tokenizer",
        language_model_prefix="model.language_model.layers",
        tool_protocol="mistral_tool_calls_args",
        allow_empty_generation_suffix=True,
        default_lora_modules=("q_proj", "v_proj", "o_proj"),
    ),
    "gemma4": NativeModelProfile(
        family="gemma4",
        repo_id="google/gemma-4-E4B-it",
        auto_model_kind="causal_lm",
        tokenizer_kind="auto_tokenizer_text_only",
        language_model_prefix="model.language_model.layers",
        tool_protocol="gemma4_tool_call_channel",
        # Gemma 4 emits ``<|turn>model\n`` after a user turn, but a history
        # ending in a tool response is already inside the model turn and has a
        # valid empty generation suffix.
        allow_empty_generation_suffix=True,
        # Gemma 4 E4B shares K/V projections above layer 23. Its last eight
        # language layers expose q_proj/o_proj but no per-layer v_proj.
        default_lora_modules=("q_proj", "o_proj"),
    ),
}


def native_model_profile(family: str | None) -> NativeModelProfile | None:
    """Return an explicitly selected native profile, preserving Qwen defaults."""

    normalized = str(family or "").strip().lower()
    if normalized in {"", "qwen", "qwen3", "qwen35", "qwen3.5"}:
        return None
    try:
        return NATIVE_MODEL_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported native model family: {family!r}") from exc


def text_config(config: Any) -> Any:
    """Return the text backbone config for text-only or multimodal wrappers."""

    nested = getattr(config, "text_config", None)
    return nested if nested is not None else config


def text_hidden_size(config: Any) -> int:
    value = getattr(text_config(config), "hidden_size", None)
    if value is None:
        raise ValueError("model config does not expose a text hidden_size")
    return int(value)


def text_num_hidden_layers(config: Any) -> int:
    value = getattr(text_config(config), "num_hidden_layers", None)
    if value is None:
        raise ValueError("model config does not expose text num_hidden_layers")
    return int(value)


def last_layer_indices(config: Any, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    total = text_num_hidden_layers(config)
    if count > total:
        raise ValueError(f"requested {count} layers from a {total}-layer text model")
    return tuple(range(total - count, total))


def language_lora_target_regex(
    profile: NativeModelProfile,
    *,
    layer_indices: Iterable[int],
    module_names: Iterable[str] | None = None,
) -> str:
    """Build a PEFT regex that excludes vision/audio towers by construction."""

    layers = tuple(sorted({int(index) for index in layer_indices}))
    modules = tuple(sorted({str(name) for name in (module_names or profile.default_lora_modules)}))
    if not layers:
        raise ValueError("at least one language layer is required")
    if not modules:
        raise ValueError("at least one LoRA target module is required")
    layer_alternation = "|".join(str(index) for index in layers)
    module_paths = []
    for name in modules:
        if "." in name:
            module_paths.append(name)
        elif name in {"gate_proj", "up_proj", "down_proj"}:
            module_paths.append(f"mlp.{name}")
        else:
            module_paths.append(f"self_attn.{name}")
    module_alternation = "|".join(re.escape(path) for path in module_paths)
    return (
        rf"^{re.escape(profile.language_model_prefix)}\."
        rf"(?:{layer_alternation})\."
        rf"(?:{module_alternation})$"
    )


@dataclass(frozen=True, slots=True)
class GenerationBoundary:
    history_text: str
    prompt_text: str
    generation_suffix: str
    insertion_character_index: int


def detect_generation_boundary(
    history_text: str,
    prompt_text: str,
    *,
    allow_empty_suffix: bool,
) -> GenerationBoundary:
    """Validate a template-derived generation boundary without tokenizing it."""

    if not prompt_text.startswith(history_text):
        raise ValueError("add_generation_prompt rendering is not prefixed by history")
    suffix = prompt_text[len(history_text) :]
    if not suffix and not allow_empty_suffix:
        raise ValueError("model-native generation suffix is empty")
    return GenerationBoundary(
        history_text=history_text,
        prompt_text=prompt_text,
        generation_suffix=suffix,
        insertion_character_index=len(history_text),
    )


def _gemma4_text_model(model: Any) -> Any | None:
    """Return Gemma 4's text model when ``model`` is base or PEFT-wrapped."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    model_type = str(getattr(getattr(base, "config", None), "model_type", ""))
    if model_type != "gemma4":
        return None
    backbone = getattr(base, "model", None)
    text_model = getattr(backbone, "language_model", None)
    if text_model is None:
        raise ValueError("Gemma 4 model does not expose model.language_model")
    return text_model


def native_per_layer_inputs_for_embeds(
    model: Any,
    inputs_embeds: torch.Tensor,
    *,
    token_input_ids: torch.Tensor | None = None,
    token_positions_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build Gemma 4 PLE inputs for a sequence containing soft latent tokens.

    Gemma 4 normally recovers token ids from ``inputs_embeds`` with a
    vocabulary-wide equality broadcast.  That is both prohibitively large and
    invalid for learned latent embeddings.  Real token positions keep the exact
    token-identity PLE component.  Soft positions are assigned the algebraic
    raw component which makes Gemma 4's subsequent projection reduce to its
    documented context-only PLE path.

    Other model families do not use PLE and return ``None``.
    """

    text_model = _gemma4_text_model(model)
    if text_model is None:
        return None

    context_only = text_model.project_per_layer_inputs(inputs_embeds, None)
    scale = text_model.per_layer_input_scale
    soft_raw = context_only / scale - context_only
    if token_input_ids is None:
        if token_positions_mask is not None:
            raise ValueError("token_positions_mask requires token_input_ids")
        return soft_raw

    token_input_ids = token_input_ids.to(
        device=inputs_embeds.device, dtype=torch.long
    )
    if token_input_ids.shape != inputs_embeds.shape[:2]:
        raise ValueError(
            "Gemma 4 aligned token ids must match inputs_embeds sequence shape: "
            f"{tuple(token_input_ids.shape)} != {tuple(inputs_embeds.shape[:2])}"
        )
    if token_positions_mask is None:
        token_positions_mask = torch.ones_like(token_input_ids, dtype=torch.bool)
    else:
        token_positions_mask = token_positions_mask.to(
            device=inputs_embeds.device, dtype=torch.bool
        )
    if token_positions_mask.shape != token_input_ids.shape:
        raise ValueError("Gemma 4 token position mask must match aligned token ids")

    token_identity = text_model.get_per_layer_inputs(token_input_ids, None)
    token_identity = token_identity.to(
        device=inputs_embeds.device, dtype=context_only.dtype
    )
    return torch.where(
        token_positions_mask[..., None, None], token_identity, soft_raw
    )
