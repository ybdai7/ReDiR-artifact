"""Frozen batching and gates for MT-AgentRisk V7.9.2."""

from __future__ import annotations

import random
from typing import Any, Iterable

import torch


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("task_key") or ""),
        str(row.get("state_id") or ""),
        str(row.get("target_id") or ""),
    )


def deterministic_pair_minibatches(
    manifest: Iterable[dict[str, Any]],
    *,
    batch_size: int,
    epoch_index: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Shuffle every active pair exactly once and partition it deterministically."""

    if batch_size <= 0:
        raise ValueError("V7.9.2 batch_size must be positive")
    if epoch_index <= 0:
        raise ValueError("V7.9.2 epoch_index is one-based and must be positive")
    rows = sorted((row for row in manifest if bool(row.get("active"))), key=_pair_key)
    if not rows:
        raise ValueError("V7.9.2 manifest contains no active pairs")
    random.Random(seed + epoch_index).shuffle(rows)
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def deterministic_benign_minibatches(
    rows: Iterable[dict[str, Any]],
    *,
    batch_count: int,
    epoch_index: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Cover every benign anchor once per epoch across the safety mini-batches."""

    if batch_count <= 0:
        raise ValueError("V7.9.2 benign batch_count must be positive")
    values = sorted(
        rows,
        key=lambda row: (str(row.get("task_key") or ""), str(row.get("state_id") or "")),
    )
    if len(values) < batch_count:
        raise ValueError(
            "V7.9.2 needs at least one benign anchor per safety mini-batch"
        )
    random.Random(seed + epoch_index).shuffle(values)
    batches: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
    for index, row in enumerate(values):
        batches[index % batch_count].append(row)
    return batches


def a_unlock_allowed(
    *,
    epoch_index: int,
    current_gradient_norm: float,
    baseline_gradient_norm: float,
    b_ema_full_gradient_cosine: float,
    minimum_gradient_ratio: float = 1.05,
    minimum_cosine: float = 0.5,
) -> bool:
    """Apply the pre-registered LoRA-A gate after the first complete epoch."""

    return bool(
        epoch_index >= 1
        and baseline_gradient_norm > 0.0
        and current_gradient_norm / baseline_gradient_norm >= minimum_gradient_ratio
        and b_ema_full_gradient_cosine > minimum_cosine
    )


def reset_optimizer_group_ema(
    optimizer: torch.optim.Optimizer,
    *,
    group_name: str,
) -> dict[str, int]:
    """Delete only one parameter group's gradient EMA buffers."""

    matched_groups = [
        group
        for group in optimizer.param_groups
        if str(group.get("group_name") or "") == group_name
    ]
    if len(matched_groups) != 1:
        raise ValueError(
            f"expected exactly one optimizer group named {group_name!r}; "
            f"found {len(matched_groups)}"
        )
    target_ids = {id(parameter) for parameter in matched_groups[0]["params"]}
    removed = 0
    preserved = 0
    for parameter, state in optimizer.state.items():
        if "gradient_ema" not in state:
            continue
        if id(parameter) in target_ids:
            del state["gradient_ema"]
            removed += 1
        else:
            preserved += 1
    return {"removed": removed, "preserved": preserved}


__all__ = [
    "a_unlock_allowed",
    "deterministic_benign_minibatches",
    "deterministic_pair_minibatches",
    "reset_optimizer_group_ema",
]
