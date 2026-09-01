"""Build task-disjoint MAGE datasets from MT-AgentRisk OpenHands traces."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .openhands_adapter import is_guarded_action, normalize_pending_tool_calls


FINAL_TURN_RE = re.compile(r"(?im)^\s*Final\s+Turn\s*:")


def stable_partition(key: str, test_fraction: float) -> str:
    """Assign a task (not a trajectory) to a deterministic train/test split."""

    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be within [0, 1]")
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
    return "test" if value < test_fraction else "train"


def normalize_task_key(value: str) -> str:
    key = value.strip().replace("_", "-")
    key = re.sub(r"^traj-", "", key)
    key = re.sub(r"^multi-turn-", "", key)
    return key


def load_task_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = normalize_task_key(str(row["task_key"]))
        rows[key] = row
    return rows


def trajectory_task_key(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith(("task.", "benign_task.")):
        return f"benign/{parent.replace('benign_task.', 'task.')}"
    return normalize_task_key(parent or path.stem.removeprefix("traj_"))


def parse_seed(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def iter_trajectory_paths(roots: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if root.is_file():
            found.add(root.resolve())
        elif root.is_dir():
            found.update(path.resolve() for path in root.rglob("traj_*.json"))
        else:
            raise FileNotFoundError(root)
    return sorted(found)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(1, (limit - 64) // 2)
    return text[:keep] + f"\n...[truncated {len(text) - 2 * keep} chars]...\n" + text[-keep:]


def _trim_context(context: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    if max_chars <= 0:
        return deepcopy(context)
    kept: list[dict[str, Any]] = []
    total = 0
    for message in reversed(context):
        size = len(str(message.get("content", "")))
        if kept and total + size > max_chars:
            break
        kept.append(deepcopy(message))
        total += size
    kept.reverse()

    # Preserve the root policy and original task when the recent-window trim
    # would otherwise remove the intent needed by the guard.
    for role in ("system", "user"):
        first = next((item for item in context if item.get("role") == role), None)
        if first is not None and first not in kept:
            kept.insert(0, deepcopy(first))
    return kept


def event_to_context_message(event: dict[str, Any], max_message_chars: int) -> dict[str, Any] | None:
    source = event.get("source")
    action = event.get("action")
    if action == "recall":
        return None

    message = str(event.get("message") or event.get("content") or "")
    if source == "environment" or (event.get("cause") is not None and event.get("observation")):
        role = "tool"
        content = str(event.get("content") or message or event.get("observation") or "")
    elif source == "user":
        role = "user"
        content = message
    elif source == "agent" and event.get("id") == 0 and action in (None, "system"):
        role = "system"
        content = message
    elif source == "agent":
        role = "assistant"
        prefix = f"[OpenHands action={action}]\n" if action else ""
        content = prefix + message
    else:
        return None

    return {"role": role, "content": _truncate(content, max_message_chars)}


def build_trajectory_sample(
    traj_path: Path,
    *,
    model_family: str,
    split: str,
    is_attack: bool,
    max_context_chars: int = 24000,
    max_message_chars: int = 6000,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build one MAGE trajectory sample and an auditable extraction record."""

    events = json.loads(traj_path.read_text())
    if not isinstance(events, list):
        raise ValueError(f"Expected list trajectory: {traj_path}")

    task_key = trajectory_task_key(traj_path)
    seed = parse_seed(traj_path)
    context: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    labels: list[int] = []
    phase = "pre_final"
    final_turn_event_id: int | None = None
    bypass_counts: Counter[str] = Counter()

    for event in events:
        if event.get("source") == "user" and FINAL_TURN_RE.search(str(event.get("message") or "")):
            phase = "post_final"
            if final_turn_event_id is None:
                final_turn_event_id = event.get("id")

        if is_guarded_action(event):
            calls = normalize_pending_tool_calls(event)
            if not calls:
                raise ValueError(f"Guarded action failed normalization: {traj_path} event={event.get('id')}")
            label = int(is_attack and phase == "post_final")
            segments.append(
                {
                    "context": _trim_context(context, max_context_chars),
                    "tool_calls": calls,
                    "metadata": {
                        "event_id": event.get("id"),
                        "action": event.get("action"),
                        "phase": phase,
                        "label_reason": (
                            "unsafe_final_turn_execution"
                            if label
                            else ("benign_trajectory" if not is_attack else "pre_final_setup")
                        ),
                    },
                }
            )
            labels.append(label)
        elif event.get("source") == "agent" and event.get("action"):
            bypass_counts[str(event.get("action"))] += 1

        context_message = event_to_context_message(event, max_message_chars)
        if context_message is not None:
            context.append(context_message)

    audit = {
        "source_traj_path": str(traj_path.resolve()),
        "task_key": task_key,
        "model_family": model_family,
        "split": split,
        "is_attack": is_attack,
        "seed": seed,
        "num_events": len(events),
        "num_segments": len(segments),
        "num_approve": labels.count(0),
        "num_deny": labels.count(1),
        "final_turn_event_id": final_turn_event_id,
        "bypass_counts": dict(sorted(bypass_counts.items())),
    }
    if not segments:
        audit["skip_reason"] = "no_guarded_actions"
        return None, audit

    sample = {
        "segments": segments,
        "labels": labels,
        "file_path": f"{model_family}/{split}/{task_key}/seed_{seed if seed is not None else 'unknown'}",
        "task_key": task_key,
        "split": split,
        "model_family": model_family,
        "seed": seed,
        "source_traj_path": str(traj_path.resolve()),
        "is_attack": is_attack,
        "attack_type": "mtagentrisk_final_turn" if is_attack else "benign",
        "label_contract": "deny_all_external_actions_after_exact_final_turn_v1",
    }
    return sample, audit


def validate_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Dataset is empty")
    task_splits: dict[str, set[str]] = {}
    segments = approve = deny = 0
    action_counts: Counter[str] = Counter()
    for row in rows:
        split = row.get("split")
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}")
        task_key = str(row["task_key"])
        task_splits.setdefault(task_key, set()).add(split)
        row_segments = row.get("segments")
        labels = row.get("labels")
        if not isinstance(row_segments, list) or not isinstance(labels, list) or len(row_segments) != len(labels):
            raise ValueError(f"Segment/label mismatch: {row.get('file_path')}")
        for segment, label in zip(row_segments, labels):
            if label not in (0, 1):
                raise ValueError(f"Invalid label {label}: {row.get('file_path')}")
            calls = segment.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError(f"Empty tool calls: {row.get('file_path')}")
            if not isinstance(segment.get("context"), list):
                raise ValueError(f"Invalid context: {row.get('file_path')}")
            segments += 1
            approve += int(label == 0)
            deny += int(label == 1)
            action_counts.update(str(call.get("openhands_action", "unknown")) for call in calls)

    leakage = sorted(task for task, splits in task_splits.items() if len(splits) > 1)
    if leakage:
        raise ValueError(f"Task leakage across train/test: {leakage}")
    if approve == 0 or deny == 0:
        raise ValueError(f"Both labels are required: approve={approve}, deny={deny}")

    return {
        "num_rows": len(rows),
        "num_tasks": len(task_splits),
        "num_segments": segments,
        "num_approve": approve,
        "num_deny": deny,
        "action_counts": dict(sorted(action_counts.items())),
        "task_leakage": leakage,
    }


def stable_row_identity(row: dict[str, Any]) -> str:
    """Return a portable, deterministic identity for one trajectory row."""

    payload = {
        "model_family": row.get("model_family"),
        "split": row.get("split"),
        "task_key": row.get("task_key"),
        "seed": row.get("seed"),
        # ``file_path`` is the builder's portable unique key. Avoid the raw
        # source path because existing family datasets store checkout-specific
        # absolute paths there.
        "file_path": row.get("file_path"),
        "is_attack": bool(row.get("is_attack")),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def task_balanced_sample(
    rows: list[dict[str, Any]],
    count: int,
    *,
    salt: str,
) -> list[dict[str, Any]]:
    """Select rows deterministically while maximizing task coverage first.

    Selection proceeds in rounds: one row per task is selected before any task
    contributes a second row. Stable salted hashes prevent source file ordering
    from changing the frozen subset.
    """

    if count < 0 or count > len(rows):
        raise ValueError(f"Cannot select {count} rows from a pool of {len(rows)}")

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_key"]), []).append(row)

    def rank(value: str) -> str:
        return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()

    task_order = sorted(by_task, key=lambda task: (rank(f"task:{task}"), task))
    for task, task_rows in by_task.items():
        task_rows.sort(key=lambda row: (rank(f"row:{stable_row_identity(row)}"), stable_row_identity(row)))

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        made_progress = False
        for task in task_order:
            task_rows = by_task[task]
            if round_index < len(task_rows):
                selected.append(task_rows[round_index])
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            raise RuntimeError("Task-balanced selection exhausted the pool unexpectedly")
        round_index += 1
    return selected
