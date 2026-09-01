"""Token-exact rendering for native assistant-generation boundary injection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


BOUNDARY_INJECTION_MODE = "before_assistant_generation_boundary"


def token_ids_sha256(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().to(device="cpu", dtype=torch.long).tolist()
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_generation_boundary_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    device: torch.device,
    chat_template: str | None = None,
    assistant_generation_prefix: str = "",
    allow_empty_generation_suffix: bool = False,
) -> dict[str, Any]:
    """Render history and native generation suffix from one template contract.

    The returned ``input_ids`` are token-identical to the ordinary no-latent
    native prompt.  ``latent_insertion_index`` is the boundary after history and
    before the dynamically rendered assistant-generation suffix.
    """

    common: dict[str, Any] = {
        "tools": tools,
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if chat_template is not None:
        common["chat_template"] = chat_template
    history = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        **common,
    )
    full = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        **common,
    )
    history_ids = history["input_ids"]
    full_ids = full["input_ids"]
    if history_ids.ndim != 2 or full_ids.ndim != 2 or history_ids.size(0) != 1:
        raise ValueError("boundary rendering currently requires batch size one")
    history_len = history_ids.size(1)
    if full_ids.size(1) < history_len or not torch.equal(
        full_ids[:, :history_len], history_ids
    ):
        raise ValueError(
            "native add_generation_prompt output is not prefixed by the rendered history"
        )

    if assistant_generation_prefix:
        prefix_ids = tokenizer(
            assistant_generation_prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        if prefix_ids.numel() > 0 and not (
            full_ids.size(1) >= prefix_ids.size(1)
            and torch.equal(full_ids[:, -prefix_ids.size(1) :], prefix_ids)
        ):
            full_ids = torch.cat([full_ids, prefix_ids], dim=1)
            full["attention_mask"] = torch.cat(
                [
                    full["attention_mask"],
                    torch.ones(
                        prefix_ids.shape,
                        dtype=full["attention_mask"].dtype,
                    ),
                ],
                dim=1,
            )

    suffix_ids = full_ids[:, history_len:]
    if suffix_ids.numel() == 0 and not allow_empty_generation_suffix:
        raise ValueError("native generation suffix is empty")
    full_attention = full["attention_mask"]
    if full_attention.shape != full_ids.shape:
        raise ValueError("native boundary input/attention shapes differ")
    return {
        "input_ids": full_ids.to(device),
        "attention_mask": full_attention.to(device),
        "latent_insertion_index": history_len,
        "history_input_ids_sha256": token_ids_sha256(history_ids),
        "generation_suffix_ids_sha256": token_ids_sha256(suffix_ids),
        "rendered_input_ids_sha256": token_ids_sha256(full_ids),
        "generation_suffix_token_count": int(suffix_ids.numel()),
    }
