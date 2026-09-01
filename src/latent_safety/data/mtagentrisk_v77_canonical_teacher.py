"""Build the minimal V7.7 canonical-single-turn teacher pool."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SYSTEM_SHA256 = (
    "45252ef5950f6effc59acbf105f0ab7f0d3712517cd71d5c66563f60ee30baaf"
)
EXPECTED_TOOL_SCHEMA_SHA256 = (
    "f0f15a5be6c792c06af94fad7098aa323b242954ae5e31296f43134fdf575135"
)


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


def _manifest_keys(path: Path) -> set[str]:
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError(f"empty split manifest: {path}")
    return values


def canonical_single_turn_index(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Return one byte-identical two-message teacher view per task."""

    indexed: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        task_key = str(row.get("task_key") or "")
        raw_messages = row.get("canonical_single_messages")
        if not task_key or not isinstance(raw_messages, list):
            continue
        messages = [dict(message) for message in raw_messages]
        if (
            len(messages) != 2
            or [message.get("role") for message in messages] != ["system", "user"]
            or not all(str(message.get("content") or "").strip() for message in messages)
        ):
            raise ValueError(f"invalid canonical single-turn view for {task_key}")
        previous = indexed.get(task_key)
        if previous is not None and stable_hash(previous) != stable_hash(messages):
            raise ValueError(f"conflicting canonical single-turn views for {task_key}")
        indexed[task_key] = messages
    return indexed


def _system_content(messages: Any, *, label: str) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"missing messages for {label}")
    system = [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
    ]
    if len(system) != 1 or not system[0]:
        raise ValueError(f"expected one system message for {label}")
    return system[0]


def _annotate_safety(
    row: dict[str, Any],
    *,
    split: str,
    original_split: str,
    teacher_messages: list[dict[str, str]],
    gap_source: Path,
) -> dict[str, Any]:
    result = deepcopy(row)
    state_id = str(result.get("state_id") or "")
    task_key = str(result.get("task_key") or "")
    student_messages = result.get("student_state_messages")
    student_system = _system_content(student_messages, label=state_id)
    teacher_system = _system_content(teacher_messages, label=f"teacher:{task_key}")
    if hashlib.sha256(student_system.encode("utf-8")).hexdigest() != EXPECTED_SYSTEM_SHA256:
        raise ValueError(f"student P3 system hash mismatch: {state_id}")
    if teacher_system != student_system:
        raise ValueError(f"teacher/student system mismatch: {state_id}")
    if teacher_messages == student_messages:
        raise ValueError(f"teacher unexpectedly equals student history: {state_id}")
    teacher_user = str(teacher_messages[-1].get("content") or "")
    if "Please continue working on the task" in teacher_user:
        raise ValueError(f"continuation prompt leaked into teacher view: {state_id}")
    if any(message.get("role") in {"assistant", "tool"} for message in teacher_messages):
        raise ValueError(f"assistant/tool history leaked into teacher view: {state_id}")

    teacher_messages = deepcopy(teacher_messages)
    teacher_sha = stable_hash(teacher_messages)
    result.update(
        {
            "split": split,
            "source_split": split,
            "v72_safety_split": split,
            "v77_original_split": original_split,
            "v77_mechanism_smoke_only": True,
            "v72_safety_origin": "v77_frozen_safety17_anchor",
            "v72_safety_candidate": True,
            "v72_optimizer_candidate": True,
            "v485_training_candidate": True,
            "v5_precise_mask_candidate": True,
            "v48_teacher_exact_state_messages": teacher_messages,
            "v485_teacher_source": "v77_canonical_single_turn_native",
            "v75_teacher_context_mode": "privileged_collapsed_native",
            "v75_teacher_messages_sha256": teacher_sha,
            "v75_teacher_view": {
                "kind": "canonical_single_turn_native",
                "task_key": task_key,
                "messages_sha256": teacher_sha,
                "student_messages_sha256": stable_hash(student_messages),
                "source_gap_dataset": str(gap_source.resolve()),
                "source_gap_dataset_sha256": file_sha256(gap_source.resolve()),
                "recoverable_privilege": True,
                "hidden_future_turns_used": False,
                "assistant_prose_used": False,
                "historical_refusal_used": False,
                "continuation_prompt_used": False,
            },
            "source_system_prompt_sha256": EXPECTED_SYSTEM_SHA256,
            "v485_protocol_replay": {
                "passed": True,
                "strategy": "v77_canonical_single_turn_teacher_native_v1",
                "source_prompt_sha256": result.get("source_prompt_sha256"),
                "teacher_messages_sha256": teacher_sha,
            },
            "heldout15_used": False,
            "dangerous_saved_suffix_gradient_used": False,
        }
    )
    for key in (
        "v48_teacher_raw_completion",
        "v48_teacher_seed",
        "v48_teacher_state_kind",
        "v6_native_refusal_transition_target",
        "v6_teacher_target_origin",
        "v6_teacher_canonicalization",
        "v72_teacher_target_seed",
        "native_teacher_trajectory_path",
        "native_teacher_judge_cache_key",
    ):
        result.pop(key, None)
    return result


def _validate_native_safety(row: dict[str, Any]) -> None:
    if (
        row.get("v485_route") != "protocol_safety"
        or row.get("protocol_source") != "native"
        or row.get("native_tool_calling") is not True
        or row.get("state_kind") != "final_state"
        or row.get("latest_user_is_actual_final_turn") is not True
        or row.get("source_tool_schema_sha256") != EXPECTED_TOOL_SCHEMA_SHA256
        or len(row.get("available_tools") or []) != 20
        or not str(row.get("source_prompt_sha256") or "")
        or row.get("observed_contains_refusal") is not False
    ):
        raise ValueError(f"invalid frozen native safety state: {row.get('state_id')}")


def build_v77_canonical_teacher_pool(
    *,
    anchor_dataset: Path,
    gap_source: Path,
    train_manifest: Path,
    dev_manifest: Path,
    heldout_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Train on all safety17 states and attach one canonical teacher per task."""

    paths = {
        "anchor": anchor_dataset.resolve(),
        "gap": gap_source.resolve(),
        "train_manifest": train_manifest.resolve(),
        "dev_manifest": dev_manifest.resolve(),
        "heldout_manifest": heldout_manifest.resolve(),
    }
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty V7.7 pool: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_keys = {
        "train": _manifest_keys(paths["train_manifest"]),
        "dev": _manifest_keys(paths["dev_manifest"]),
        "heldout": _manifest_keys(paths["heldout_manifest"]),
    }
    if (
        split_keys["train"] & split_keys["dev"]
        or split_keys["train"] & split_keys["heldout"]
        or split_keys["dev"] & split_keys["heldout"]
    ):
        raise ValueError("train/dev/heldout manifests overlap")

    anchor_rows = _load_jsonl(paths["anchor"])
    safety = [row for row in anchor_rows if row.get("v485_route") == "protocol_safety"]
    benign = [row for row in anchor_rows if row.get("v485_route") == "benign_retention"]
    if (len(safety), len({str(row.get("task_key") or "") for row in safety})) != (17, 14):
        raise ValueError("V7.7 requires frozen safety17/14")
    if (len(benign), len({str(row.get("task_key") or "") for row in benign})) != (20, 20):
        raise ValueError("V7.7 requires frozen benign20/20")

    teacher_by_task = canonical_single_turn_index(_load_jsonl(paths["gap"]))
    safety_tasks = {str(row.get("task_key") or "") for row in safety}
    missing = sorted(safety_tasks - set(teacher_by_task))
    if missing:
        raise ValueError(f"canonical teacher view missing for tasks: {missing}")

    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    prompt_hashes: set[str] = set()
    for row in safety:
        _validate_native_safety(row)
        task_key = str(row["task_key"])
        if task_key in split_keys["heldout"]:
            raise ValueError(f"safety17 overlaps heldout: {task_key}")
        original_split = (
            "train"
            if task_key in split_keys["train"]
            else "dev"
            if task_key in split_keys["dev"]
            else ""
        )
        if not original_split:
            raise ValueError(f"safety17 task is outside frozen manifests: {task_key}")
        prompt_hash = str(row["source_prompt_sha256"])
        if prompt_hash in prompt_hashes:
            raise ValueError(f"duplicate safety prompt hash: {prompt_hash}")
        prompt_hashes.add(prompt_hash)
        outputs["train"].append(
            _annotate_safety(
                row,
                split="train",
                original_split=original_split,
                teacher_messages=teacher_by_task[task_key],
                gap_source=paths["gap"],
            )
        )

    for row in benign:
        if (
            row.get("protocol_source") != "native"
            or row.get("native_tool_calling") is not True
            or len(row.get("available_tools") or []) != 20
        ):
            raise ValueError(f"invalid native benign anchor: {row.get('state_id')}")
        row = deepcopy(row)
        row.update(
            {
                "split": "train",
                "source_split": "train",
                "v72_optimizer_candidate": False,
                "v485_training_candidate": True,
                "v5_precise_mask_candidate": True,
                "heldout15_used": False,
                "v77_benign20_retention": True,
            }
        )
        outputs["train"].append(row)

    train_safety = [row for row in outputs["train"] if row.get("v485_route") == "protocol_safety"]
    dev_safety = outputs["dev"]
    if (len(train_safety), len({row["task_key"] for row in train_safety})) != (17, 14):
        raise ValueError("V7.7 train split must contain all safety17/14")
    if dev_safety:
        raise ValueError("V7.7 mechanism smoke requires an empty internal dev split")
    original_split_counts = {
        split: sum(row.get("v77_original_split") == split for row in train_safety)
        for split in ("train", "dev")
    }
    if original_split_counts != {"train": 13, "dev": 4}:
        raise ValueError(
            f"V7.7 original split provenance mismatch: {original_split_counts}"
        )

    outputs["train"].sort(
        key=lambda row: (str(row.get("v485_route") or ""), str(row.get("task_key") or ""), str(row.get("state_id") or ""))
    )
    outputs["dev"].sort(key=lambda row: (str(row.get("task_key") or ""), str(row.get("state_id") or "")))
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    _write_jsonl(train_path, outputs["train"])
    _write_jsonl(dev_path, outputs["dev"])

    teacher_hashes = {
        str((row.get("v75_teacher_view") or {}).get("messages_sha256") or "")
        for row in [*train_safety, *dev_safety]
    }
    stats = {
        "strategy": "v77_safety17_all_train_canonical_single_turn_native_hybrid_v1",
        "experiment_scope": "mechanism_smoke_only",
        "train": {"states": 17, "tasks": 14},
        "dev": {"states": 0, "tasks": 0},
        "original_split_provenance": original_split_counts,
        "former_dev_consumed_for_training": True,
        "dev_probe_enabled": False,
        "safety_total": {"states": 17, "tasks": 14},
        "benign_retention": {"states": 20, "tasks": 20},
        "teacher": {
            "context_mode": "canonical_single_turn_native",
            "messages_per_view": 2,
            "unique_views": len(teacher_hashes),
            "historical_teacher_used": False,
            "assistant_or_tool_history_used": False,
            "hidden_future_turns_used": False,
        },
        "sources": {key: str(path) for key, path in paths.items()},
        "source_sha256": {key: file_sha256(path) for key, path in paths.items()},
        "train_jsonl_sha256": file_sha256(train_path),
        "dev_jsonl_sha256": file_sha256(dev_path),
        "heldout_overlap": [],
        "heldout15_used": False,
        "benign20_used_for_retention": True,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


__all__ = [
    "EXPECTED_SYSTEM_SHA256",
    "EXPECTED_TOOL_SCHEMA_SHA256",
    "build_v77_canonical_teacher_pool",
    "canonical_single_turn_index",
    "stable_hash",
]
