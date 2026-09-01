"""Build V7.5 safety records with an independent privileged teacher view."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FILESYSTEM_GUIDANCE = """**CRITICAL INFRASTRUCTURE NOTICE: FILESYSTEM PROTOCOL**
You have access to the following MCP Filesystem tools: read_file, read_text_file, read_media_file, read_multiple_files, write_file, edit_file, create_directory, list_directory, list_directory_with_sizes, directory_tree, move_file, search_files, get_file_info, list_allowed_directories.

**MANDATORY RULES:**
1. **MCP FIRST:** You MUST attempt to use the MCP tools listed above for ALL file reading, writing, editing, searching, and listing operations. It it encouraged to rewrite instead of edit based on previous version.
2. **FALLBACK PROTOCOL:** Standard shell commands are generally discouraged in this environment. Use them **ONLY** if the MCP tools fail persistently (more than twice) and you cannot resolve the error."""


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


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_route_map(summary: dict[str, Any]) -> dict[str, str]:
    routes: dict[str, str] = {}
    for row in summary.get("state_rows") or []:
        state_id = str(row.get("state_id") or "")
        route = str((row.get("decision") or {}).get("route") or "unusable")
        if state_id:
            routes[state_id] = route
    return routes


def _historical_teacher_by_task(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    route_rank = {"strong": 2, "weak": 1, "unusable": 0}
    routes = _historical_route_map(summary)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("v485_route") != "protocol_safety":
            continue
        task_key = str(row.get("task_key") or "")
        messages = row.get("v48_teacher_exact_state_messages")
        if task_key and isinstance(messages, list) and messages:
            candidates.setdefault(task_key, []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for task_key, task_rows in candidates.items():
        task_rows.sort(
            key=lambda row: (
                route_rank.get(routes.get(str(row.get("state_id") or ""), "unusable"), 0),
                int(
                    str(row.get("v485_teacher_source") or "")
                    == "native_clean_finish_action"
                ),
                -len(row.get("v48_teacher_exact_state_messages") or []),
                str(row.get("state_id") or ""),
            ),
            reverse=True,
        )
        selected[task_key] = task_rows[0]
    return selected


def _system_message(record: dict[str, Any]) -> dict[str, str]:
    messages = record.get("student_state_messages") or []
    for message in messages:
        if message.get("role") == "system" and str(message.get("content") or ""):
            return {"role": "system", "content": str(message["content"])}
    raise ValueError(f"missing system message: {record.get('state_id')}")


def _collapsed_teacher_messages(
    record: dict[str, Any],
    *,
    single_turn_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    task_key = str(record.get("task_key") or "")
    task_path = single_turn_root / f"safety-{task_key}" / "task.md"
    if not task_path.is_file():
        raise ValueError(f"missing collapsed single-turn task for {task_key}: {task_path}")
    task_text = task_path.read_text(encoding="utf-8").strip()
    if not task_text:
        raise ValueError(f"empty collapsed single-turn task: {task_path}")
    messages = [
        _system_message(record),
        {
            "role": "user",
            "content": f"{task_text}\n\n{FILESYSTEM_GUIDANCE}",
        },
    ]
    return messages, {
        "kind": "collapsed_single_turn_native",
        "source_task_path": str(task_path.resolve()),
        "source_task_sha256": file_sha256(task_path),
        "recoverable_privilege": True,
        "hidden_future_turns_used": False,
    }


def _annotate_safety_record(
    row: dict[str, Any],
    *,
    historical: dict[str, dict[str, Any]],
    single_turn_root: Path,
) -> dict[str, Any]:
    result = deepcopy(row)
    task_key = str(result.get("task_key") or "")
    historical_row = historical.get(task_key)
    if historical_row is not None:
        teacher_messages = deepcopy(
            historical_row["v48_teacher_exact_state_messages"]
        )
        teacher_view = {
            "kind": "historical_privileged_native",
            "historical_state_id": str(historical_row.get("state_id") or ""),
            "historical_teacher_source": str(
                historical_row.get("v485_teacher_source") or ""
            ),
            "recoverable_privilege": True,
            "hidden_future_turns_used": False,
        }
    else:
        teacher_messages, teacher_view = _collapsed_teacher_messages(
            result,
            single_turn_root=single_turn_root,
        )
    student_messages = result.get("student_state_messages") or []
    if teacher_messages == student_messages:
        raise ValueError(
            f"V7.5 privileged teacher unexpectedly equals student state: {result.get('state_id')}"
        )
    teacher_sha = stable_hash(teacher_messages)
    result.update(
        {
            "v48_teacher_exact_state_messages": teacher_messages,
            "v485_teacher_source": "v75_privileged_collapsed_native",
            "v75_teacher_context_mode": "privileged_collapsed_native",
            "v75_teacher_messages_sha256": teacher_sha,
            "v75_teacher_view": {
                **teacher_view,
                "messages_sha256": teacher_sha,
                "student_messages_sha256": stable_hash(student_messages),
            },
            "v485_protocol_replay": {
                "passed": True,
                "strategy": "v75_student_native_privileged_teacher_v1",
                "source_prompt_sha256": result.get("source_prompt_sha256"),
                "teacher_messages_sha256": teacher_sha,
            },
            "heldout15_used": False,
        }
    )
    result.pop("v6_native_refusal_transition_target", None)
    result.pop("v6_teacher_target_origin", None)
    result.pop("v72_teacher_target_seed", None)
    return result


def build_v75_privileged_teacher_pool(
    *,
    train_dataset: Path,
    dev_dataset: Path,
    historical_dataset: Path,
    historical_teacher_summary: Path,
    single_turn_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty V7.5 pool: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    historical_rows = _load_jsonl(historical_dataset.resolve())
    historical_summary = json.loads(
        historical_teacher_summary.resolve().read_text(encoding="utf-8")
    )
    historical = _historical_teacher_by_task(
        historical_rows,
        historical_summary,
    )

    outputs: dict[str, list[dict[str, Any]]] = {}
    view_counts: Counter[str] = Counter()
    teacher_hashes: set[str] = set()
    for split, source in (
        ("train", train_dataset.resolve()),
        ("dev", dev_dataset.resolve()),
    ):
        rows: list[dict[str, Any]] = []
        for row in _load_jsonl(source):
            if row.get("v485_route") == "protocol_safety":
                row = _annotate_safety_record(
                    row,
                    historical=historical,
                    single_turn_root=single_turn_root.resolve(),
                )
                view = row["v75_teacher_view"]
                view_counts[str(view["kind"])] += 1
                teacher_hashes.add(str(view["messages_sha256"]))
            rows.append(row)
        outputs[split] = rows
        _write_jsonl(output_dir / f"{split}.jsonl", rows)

    safety_by_split = {
        split: [row for row in rows if row.get("v485_route") == "protocol_safety"]
        for split, rows in outputs.items()
    }
    train_tasks = {str(row.get("task_key") or "") for row in safety_by_split["train"]}
    dev_tasks = {str(row.get("task_key") or "") for row in safety_by_split["dev"]}
    if train_tasks & dev_tasks:
        raise ValueError("V7.5 train/dev task overlap")
    stats = {
        "stage": "v75_privileged_teacher_pool",
        "train": {
            "states": len(safety_by_split["train"]),
            "tasks": len(train_tasks),
        },
        "dev": {
            "states": len(safety_by_split["dev"]),
            "tasks": len(dev_tasks),
        },
        "teacher_view_counts": dict(sorted(view_counts.items())),
        "unique_teacher_views": len(teacher_hashes),
        "historical_teacher_tasks_available": len(historical),
        "student_teacher_context_equal": False,
        "teacher_context_mode": "privileged_collapsed_native",
        "historical_regression": {
            "summary_sha256": file_sha256(historical_teacher_summary.resolve()),
            "dataset_sha256": file_sha256(historical_dataset.resolve()),
            "states": int(historical_summary.get("states", 0)),
            "tasks": int(historical_summary.get("tasks", 0)),
            "rollouts": int(historical_summary.get("rollouts", 0)),
            "clean_native_refusals": int(
                historical_summary.get("clean_native_refusals", 0)
            ),
            "dangerous_nonfinish_actions": int(
                historical_summary.get("dangerous_nonfinish_actions", 0)
            ),
            "routes": historical_summary.get("routes") or {},
        },
        "heldout15_used": False,
    }
    stats_path = output_dir / "stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


__all__ = ["build_v75_privileged_teacher_pool", "stable_hash"]
