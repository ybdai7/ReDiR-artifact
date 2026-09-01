"""Build the V7.8 filesystem55 teacher universe and Final-Turn state pool."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from redir.datasets.mtagentrisk_v75_privileged_teacher import FILESYSTEM_GUIDANCE
from redir.datasets.mtagentrisk_v77_canonical_teacher import (
    EXPECTED_SYSTEM_SHA256,
    EXPECTED_TOOL_SCHEMA_SHA256,
    file_sha256,
    stable_hash,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _manifest_keys(path: Path) -> list[str]:
    keys = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f"empty or duplicate V7.8 manifest: {path}")
    return keys


def _system_text(row: dict[str, Any]) -> str:
    messages = row.get("student_state_messages") or []
    values = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
    ]
    if len(values) != 1 or not values[0]:
        raise ValueError(f"state does not contain one system message: {row.get('state_id')}")
    return values[0]


def _canonical_messages(
    *,
    task_key: str,
    system_text: str,
    single_turn_root: Path,
) -> tuple[list[dict[str, str]], Path]:
    task_path = single_turn_root / f"safety-{task_key}" / "task.md"
    if not task_path.is_file():
        raise ValueError(f"missing filesystem55 canonical task: {task_path}")
    task_text = task_path.read_text(encoding="utf-8").strip()
    if not task_text:
        raise ValueError(f"empty filesystem55 canonical task: {task_path}")
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": f"{task_text}\n\n{FILESYSTEM_GUIDANCE}",
        },
    ], task_path


def _validate_state(
    row: dict[str, Any],
    *,
    expected_system_sha256: str,
    expected_tool_schema_sha256: str,
    expected_tool_count: int,
) -> None:
    system_text = _system_text(row)
    if (
        row.get("v485_route") != "protocol_safety"
        or row.get("protocol_source") != "native"
        or row.get("native_tool_calling") is not True
        or row.get("state_kind") != "final_state"
        or row.get("latest_user_is_actual_final_turn") is not True
        or row.get("source_system_prompt_sha256") != expected_system_sha256
        or hashlib.sha256(system_text.encode("utf-8")).hexdigest()
        != expected_system_sha256
        or row.get("source_tool_schema_sha256") != expected_tool_schema_sha256
        or len(row.get("available_tools") or []) != expected_tool_count
        or row.get("heldout15_used") is not False
    ):
        raise ValueError(f"invalid V7.8 Final-Turn state: {row.get('state_id')}")


def _annotate_safety(
    row: dict[str, Any],
    *,
    original_split: str,
    teacher_messages: list[dict[str, str]],
    task_path: Path,
    prefix_audit: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(row)
    student_messages = result.get("student_state_messages") or []
    teacher_hash = stable_hash(teacher_messages)
    judgment = prefix_audit.get("judgment") or {}
    result.update(
        {
            "split": "train",
            "source_split": "train",
            "v72_safety_split": "train",
            "v78_original_split": original_split,
            "v78_training_universe": "filesystem55_train45_plus_dev10",
            "v78_final_turn_only": True,
            "v78_target_state_coherence": {
                "same_underlying_task": True,
                "actual_final_turn": True,
                "future_turn_used": False,
                "harmful_goal_completed_before_state": False,
                "source_prefix_audit_cache_key": prefix_audit.get("cache_key"),
                "source_prefix_audit_confidence": judgment.get("confidence"),
            },
            "v72_safety_candidate": True,
            "v72_optimizer_candidate": True,
            "v485_training_candidate": True,
            "v5_precise_mask_candidate": True,
            "v48_teacher_exact_state_messages": deepcopy(teacher_messages),
            "v485_teacher_source": "v78_canonical_single_turn_native",
            "v75_teacher_context_mode": "privileged_collapsed_native",
            "v75_teacher_messages_sha256": teacher_hash,
            "v75_teacher_view": {
                "kind": "v78_canonical_single_turn_native",
                "task_key": str(result.get("task_key") or ""),
                "messages_sha256": teacher_hash,
                "student_messages_sha256": stable_hash(student_messages),
                "source_task_path": str(task_path.resolve()),
                "source_task_sha256": file_sha256(task_path.resolve()),
                "recoverable_privilege": True,
                "hidden_future_turns_used": False,
                "assistant_prose_used": False,
                "historical_refusal_used": False,
                "continuation_prompt_used": False,
            },
            "v485_protocol_replay": {
                "passed": True,
                "strategy": "v78_final_turn_student_canonical_teacher_native_v1",
                "source_prompt_sha256": result.get("source_prompt_sha256"),
                "teacher_messages_sha256": teacher_hash,
            },
            "heldout15_used": False,
            "dangerous_saved_suffix_gradient_used": False,
        }
    )
    for key in (
        "v6_native_refusal_transition_target",
        "v6_teacher_target_origin",
        "v72_teacher_target_seed",
        "v78_native_refusal_targets",
        "v78_target_availability",
    ):
        result.pop(key, None)
    return result


def build_v78_canonical_target_pool(
    *,
    train_safety_dataset: Path,
    dev_safety_dataset: Path,
    benign_dataset: Path,
    prefix_audit: Path,
    train_manifest: Path,
    dev_manifest: Path,
    heldout_manifest: Path,
    single_turn_root: Path,
    output_dir: Path,
    portable_contract: bool = False,
    expected_system_sha256: str = EXPECTED_SYSTEM_SHA256,
    expected_tool_schema_sha256: str = EXPECTED_TOOL_SCHEMA_SHA256,
    expected_tool_count: int = 20,
) -> dict[str, Any]:
    """Consume old dev10 into training and expose all filesystem55 teacher views."""

    paths = {
        "train_safety": train_safety_dataset.resolve(),
        "dev_safety": dev_safety_dataset.resolve(),
        "benign": benign_dataset.resolve(),
        "prefix_audit": prefix_audit.resolve(),
        "train_manifest": train_manifest.resolve(),
        "dev_manifest": dev_manifest.resolve(),
        "heldout_manifest": heldout_manifest.resolve(),
        "single_turn_root": single_turn_root.resolve(),
    }
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty V7.8 pool: {output_dir}")

    train_keys = _manifest_keys(paths["train_manifest"])
    dev_keys = _manifest_keys(paths["dev_manifest"])
    heldout_keys = _manifest_keys(paths["heldout_manifest"])
    split_sets = {
        "train": set(train_keys),
        "dev": set(dev_keys),
        "heldout": set(heldout_keys),
    }
    if (
        split_sets["train"] & split_sets["dev"]
        or split_sets["train"] & split_sets["heldout"]
        or split_sets["dev"] & split_sets["heldout"]
    ):
        raise ValueError("V7.8 train/dev/heldout manifests overlap")
    filesystem55 = [*train_keys, *dev_keys]
    if (len(train_keys), len(dev_keys), len(heldout_keys), len(set(filesystem55))) != (
        45,
        10,
        15,
        55,
    ):
        raise ValueError("V7.8 requires frozen filesystem train45/dev10/heldout15")

    prefix_rows = _load_jsonl(paths["prefix_audit"])
    prefix_by_state = {
        str(row.get("state_id") or ""): row for row in prefix_rows
    }
    if len(prefix_by_state) != len(prefix_rows):
        raise ValueError("V7.8 prefix audit contains duplicate/empty state ids")
    if any(row.get("heldout15_used") is not False for row in prefix_rows):
        raise ValueError("V7.8 prefix audit touched heldout15")

    safety_sources = {
        "train": _load_jsonl(paths["train_safety"]),
        "dev": _load_jsonl(paths["dev_safety"]),
    }
    raw_safety: list[tuple[str, dict[str, Any]]] = []
    for split, rows in safety_sources.items():
        for row in rows:
            if row.get("v485_route") == "protocol_safety":
                raw_safety.append((split, row))
    if not portable_contract and (
        len(raw_safety),
        len({str(row.get("task_key") or "") for _, row in raw_safety}),
    ) != (78, 43):
        raise ValueError("V7.8 requires the frozen V7.4 safety78/43 source")
    if portable_contract and not raw_safety:
        raise ValueError("portable V7.8 pool has no protocol-safety states")
    if set(prefix_by_state) != {
        str(row.get("state_id") or "") for _, row in raw_safety
    }:
        raise ValueError("V7.8 prefix audit does not exactly cover safety78")

    reference = raw_safety[0][1]
    system_text = _system_text(reference)
    if hashlib.sha256(system_text.encode("utf-8")).hexdigest() != expected_system_sha256:
        raise ValueError("V7.8 reference P3 hash mismatch")
    tools = deepcopy(reference.get("available_tools") or [])
    if (
        stable_hash(tools) != expected_tool_schema_sha256
        or len(tools) != expected_tool_count
    ):
        raise ValueError("V7.8 reference native tool schema hash mismatch")

    teacher_views: list[dict[str, Any]] = []
    teacher_by_task: dict[str, tuple[list[dict[str, str]], Path]] = {}
    for task_key in filesystem55:
        messages, task_path = _canonical_messages(
            task_key=task_key,
            system_text=system_text,
            single_turn_root=paths["single_turn_root"],
        )
        teacher_by_task[task_key] = (messages, task_path)
        teacher_views.append(
            {
                "task_key": task_key,
                "state_id": f"v78_teacher_view:{task_key}",
                "source_split": "train" if task_key in split_sets["train"] else "dev",
                "v48_teacher_exact_state_messages": deepcopy(messages),
                "v75_teacher_context_mode": "privileged_collapsed_native",
                "v75_teacher_messages_sha256": stable_hash(messages),
                "v75_teacher_view": {
                    "kind": "v78_canonical_single_turn_native",
                    "task_key": task_key,
                    "messages_sha256": stable_hash(messages),
                    "source_task_path": str(task_path.resolve()),
                    "source_task_sha256": file_sha256(task_path.resolve()),
                    "recoverable_privilege": True,
                    "hidden_future_turns_used": False,
                    "assistant_prose_used": False,
                    "historical_refusal_used": False,
                    "continuation_prompt_used": False,
                },
                "available_tools": deepcopy(tools),
                "available_tool_names": [
                    str((tool.get("function") or {}).get("name") or "")
                    for tool in tools
                ],
                "protocol_source": "native",
                "native_tool_calling": True,
                "source_system_prompt_sha256": expected_system_sha256,
                "source_tool_schema_sha256": expected_tool_schema_sha256,
                "heldout15_used": False,
            }
        )

    train_safety: list[dict[str, Any]] = []
    post_harm_excluded: list[dict[str, Any]] = []
    for original_split, row in raw_safety:
        _validate_state(
            row,
            expected_system_sha256=expected_system_sha256,
            expected_tool_schema_sha256=expected_tool_schema_sha256,
            expected_tool_count=expected_tool_count,
        )
        task_key = str(row.get("task_key") or "")
        expected_split = (
            "train" if task_key in split_sets["train"] else
            "dev" if task_key in split_sets["dev"] else ""
        )
        if expected_split != original_split:
            raise ValueError(f"V7.8 source split mismatch: {task_key}/{original_split}")
        audit = prefix_by_state[str(row.get("state_id") or "")]
        judgment = audit.get("judgment") or {}
        if judgment.get("harmful_goal_completed_before_state") is True:
            post_harm_excluded.append(
                {
                    "task_key": task_key,
                    "state_id": str(row.get("state_id") or ""),
                    "prefix_audit_cache_key": audit.get("cache_key"),
                    "reason": "harmful_goal_completed_before_state",
                }
            )
            continue
        messages, task_path = teacher_by_task[task_key]
        train_safety.append(
            _annotate_safety(
                row,
                original_split=original_split,
                teacher_messages=messages,
                task_path=task_path,
                prefix_audit=audit,
            )
        )
    state_ids = [str(row.get("state_id") or "") for row in train_safety]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("V7.8 training states are duplicated")
    safety_tasks = {str(row.get("task_key") or "") for row in train_safety}
    if safety_tasks & split_sets["heldout"]:
        raise ValueError("V7.8 safety states overlap heldout15")

    benign_rows: list[dict[str, Any]] = []
    for row in _load_jsonl(paths["benign"]):
        if row.get("v485_route") != "benign_retention":
            continue
        if (
            row.get("protocol_source") != "native"
            or row.get("native_tool_calling") is not True
            or row.get("source_system_prompt_sha256") != expected_system_sha256
            or row.get("source_tool_schema_sha256") != expected_tool_schema_sha256
            or len(row.get("available_tools") or []) != expected_tool_count
        ):
            raise ValueError(f"invalid V7.8 benign retention: {row.get('state_id')}")
        value = deepcopy(row)
        value.update(
            {
                "split": "train",
                "source_split": "train",
                "v72_optimizer_candidate": False,
                "v485_training_candidate": True,
                "v5_precise_mask_candidate": True,
                "v78_benign_retention": True,
                "heldout15_used": False,
            }
        )
        benign_rows.append(value)
    if len({str(row.get("task_key") or "") for row in benign_rows}) < 20:
        raise ValueError("V7.8 benign source lacks 20-task coverage")

    output_dir.mkdir(parents=True)
    train_safety.sort(key=lambda row: (str(row["task_key"]), str(row["state_id"])))
    benign_rows.sort(key=lambda row: (str(row.get("task_key") or ""), str(row.get("state_id") or "")))
    teacher_views.sort(key=lambda row: str(row["task_key"]))
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    teacher_path = output_dir / "teacher_views.jsonl"
    excluded_path = output_dir / "post_harm_excluded.jsonl"
    _write_jsonl(train_path, [*train_safety, *benign_rows])
    _write_jsonl(dev_path, [])
    _write_jsonl(teacher_path, teacher_views)
    _write_jsonl(excluded_path, post_harm_excluded)

    per_task_states = Counter(str(row["task_key"]) for row in train_safety)
    missing_state_tasks = sorted(set(filesystem55) - safety_tasks)
    stats = {
        "strategy": (
            "v78_filesystem55_canonical_target_final_turn_pool_portable_v1"
            if portable_contract
            else "v78_filesystem55_canonical_target_final_turn_pool_v1"
        ),
        "portable_contract": bool(portable_contract),
        "native_protocol_contract": {
            "system_prompt_sha256": expected_system_sha256,
            "tool_schema_sha256": expected_tool_schema_sha256,
            "tool_count": expected_tool_count,
        },
        "training_universe": {
            "tasks": 55,
            "former_train_tasks": 45,
            "former_dev_tasks": 10,
            "former_dev_consumed_for_training": True,
        },
        "teacher_views": {
            "tasks": len(teacher_views),
            "views": len(teacher_views),
            "messages_per_view": 2,
            "historical_teacher_used": False,
            "extra_protocol_instruction_used": False,
            "forced_tool_choice_used": False,
        },
        "student_states": {
            "states": len(train_safety),
            "tasks": len(safety_tasks),
            "states_per_task": dict(sorted(per_task_states.items())),
            "missing_final_turn_state_tasks": missing_state_tasks,
            "missing_final_turn_state_task_count": len(missing_state_tasks),
            "post_harm_excluded": len(post_harm_excluded),
        },
        "benign_retention": {
            "states": len(benign_rows),
            "tasks": len({str(row.get("task_key") or "") for row in benign_rows}),
        },
        "dev": {"states": 0, "tasks": 0},
        "heldout_overlap": sorted(safety_tasks & split_sets["heldout"]),
        "heldout15_used": False,
        "sources": {key: str(value) for key, value in paths.items()},
        "source_sha256": {
            key: file_sha256(value)
            for key, value in paths.items()
            if value.is_file()
        },
        "artifacts": {
            "train_sha256": file_sha256(train_path),
            "dev_sha256": file_sha256(dev_path),
            "teacher_views_sha256": file_sha256(teacher_path),
            "post_harm_excluded_sha256": file_sha256(excluded_path),
        },
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


__all__ = ["build_v78_canonical_target_pool"]
