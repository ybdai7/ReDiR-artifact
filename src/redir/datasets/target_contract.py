"""Validation helpers for the frozen teacher-target contract used by training."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


FORMAT_INSTRUCTION = (
    "Teacher-only response-format instruction: If you determine that the "
    "request must be refused, do not end with plain assistant prose. End the "
    "turn with the native finish tool call, put the complete refusal in its "
    "message argument, preserve the native tool-call syntax exactly, and do "
    "not call any other tool. This instruction controls response format only; "
    "make the safety decision yourself."
)
FORMAT_INSTRUCTION_SHA256 = hashlib.sha256(FORMAT_INSTRUCTION.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_target_contract(
    *,
    target_path: Path,
    availability_path: Path,
    expected_target_count: int,
    expected_supported_task_count: int,
) -> dict[str, Any]:
    """Validate targets and their per-task availability without resampling."""

    target_path = target_path.resolve()
    availability_path = availability_path.resolve()
    targets = _load_jsonl(target_path)
    availability_rows = _load_jsonl(availability_path)
    availability_by_task = {
        str(row.get("task_key") or ""): row for row in availability_rows
    }
    if (
        len(availability_rows) != 55
        or len(availability_by_task) != 55
        or "" in availability_by_task
    ):
        raise ValueError("target contract requires 55 unique availability rows")
    if len(targets) != expected_target_count:
        raise ValueError(
            f"frozen target count mismatch: {len(targets)} != {expected_target_count}"
        )

    targets_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_target_ids: set[str] = set()
    elicited_count = 0
    for target in targets:
        task_key = str(target.get("task_key") or "")
        target_id = str(target.get("target_id") or "")
        if (
            task_key not in availability_by_task
            or not target_id
            or target_id in seen_target_ids
            or target.get("available") is not True
            or target.get("target_protocol") != "native"
            or target.get("source_teacher_completion_exact") is not True
            or target.get("heldout15_used") is not False
        ):
            raise ValueError(f"invalid frozen native target: {task_key}/{target_id}")
        seen_target_ids.add(target_id)
        targets_by_task[task_key].append(target)
        elicited_count += int(target.get("elicited_with_format_instruction") is True)

    supported_tasks: list[str] = []
    for task_key, availability in availability_by_task.items():
        task_targets = targets_by_task.get(task_key, [])
        selected_ids = [str(value) for value in availability.get("selected_target_ids") or []]
        if (
            len(task_targets) > 3
            or selected_ids != [str(target["target_id"]) for target in task_targets]
            or int(availability.get("selected_unique_targets", -1)) != len(task_targets)
            or bool(availability.get("has_valid_target")) != bool(task_targets)
            or availability.get("heldout15_used") is not False
        ):
            raise ValueError(f"frozen availability/target mismatch: {task_key}")
        if task_targets:
            supported_tasks.append(task_key)
    if len(supported_tasks) != expected_supported_task_count:
        raise ValueError(
            "frozen supported-task count mismatch: "
            f"{len(supported_tasks)} != {expected_supported_task_count}"
        )

    return {
        "targets": targets,
        "targets_by_task": dict(targets_by_task),
        "availability_rows": availability_rows,
        "availability_by_task": availability_by_task,
        "supported_task_keys": sorted(supported_tasks),
        "target_count": len(targets),
        "elicited_target_count": elicited_count,
        "artifacts": {
            "targets": {"path": str(target_path), "sha256": _file_sha256(target_path)},
            "availability": {
                "path": str(availability_path),
                "sha256": _file_sha256(availability_path),
            },
        },
    }


__all__ = ["FORMAT_INSTRUCTION", "FORMAT_INSTRUCTION_SHA256", "load_frozen_target_contract"]
