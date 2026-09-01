"""Build split-safe expanded genuine-native decision-state pools for V7.2."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SafetyStateSource:
    split: str
    seed: int
    path: Path


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_keys(path: Path) -> list[str]:
    keys = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"split manifest contains duplicate tasks: {path}")
    return keys


def _validate_state(
    row: dict[str, Any],
    *,
    label: str,
    expected_system_sha256: str,
    expected_tool_schema_sha256: str,
    expected_tool_count: int,
) -> None:
    if (
        row.get("protocol_source") != "native"
        or not row.get("native_tool_calling")
        or row.get("state_kind") != "final_state"
        or not row.get("latest_user_is_actual_final_turn")
        or row.get("source_system_prompt_sha256") != expected_system_sha256
        or row.get("source_tool_schema_sha256") != expected_tool_schema_sha256
        or len(row.get("available_tools") or []) != expected_tool_count
        or not str(row.get("source_prompt_sha256") or "")
    ):
        raise ValueError(f"invalid genuine-native decision state ({label}): {row.get('state_id')}")


def _state_score(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(row.get("native_history_assistant_tool_calls") or 0),
        int(row.get("native_history_tool_messages") or 0),
        len(row.get("student_state_messages") or []),
        str(row.get("state_id") or ""),
    )


def _annotate(
    row: dict[str, Any],
    *,
    split: str,
    origin: str,
    optimizer_candidate: bool,
) -> dict[str, Any]:
    result = deepcopy(row)
    result.update(
        {
            "split": split,
            "source_split": split,
            "v72_safety_split": split,
            "v72_safety_origin": origin,
            "v72_safety_candidate": True,
            "v72_optimizer_candidate": optimizer_candidate,
            # V7.2 scores the no-latent teacher on the byte-identical native
            # state.  Do not inherit the historical single-task teacher state.
            "v48_teacher_exact_state_messages": deepcopy(
                result.get("student_state_messages") or []
            ),
            "v485_teacher_source": "v72_exact_native_same_context_no_latent",
            "v485_route": "protocol_safety",
            "v485_training_candidate": optimizer_candidate,
            "v5_precise_mask_candidate": True,
            "v485_protocol_replay": {
                "passed": True,
                "strategy": "v72_genuine_native_exact_state_audit_v1",
                "source_prompt_sha256": result.get("source_prompt_sha256"),
            },
            "heldout15_used": False,
        }
    )
    return result


def _deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_hash = str(row["source_prompt_sha256"])
        previous = best.get(prompt_hash)
        if previous is None or _state_score(row) > _state_score(previous):
            best[prompt_hash] = row
    return sorted(
        best.values(),
        key=lambda row: (str(row.get("task_key") or ""), str(row.get("state_id") or "")),
    )


def _is_registered_benign_retention(row: dict[str, Any]) -> bool:
    if row.get("v485_route") == "benign_retention":
        return True
    return (
        row.get("v73_category") == "productive_benign"
        and row.get("productive_benign") is True
        and row.get("training_eligible") is True
        and row.get("harmful_tool_call_executed") is False
        and row.get("dangerous_nonfinish_before_refusal") is False
        and row.get("safety_action_conflict") is False
        and row.get("attack_success_record") is False
        and row.get("dangerous_completion_gradient_used") is False
        and row.get("heldout15_used") is False
    )


def build_v72_safety_pools(
    *,
    anchor_dataset: Path | None,
    sources: Iterable[SafetyStateSource],
    train_manifest: Path,
    dev_manifest: Path,
    heldout_manifest: Path,
    output_dir: Path,
    target_train_tasks: int = 40,
    min_train_tasks: int = 30,
    max_train_states: int = 60,
    max_states_per_task: int = 2,
    benign_dataset: Path | None = None,
    expected_system_sha256: str = EXPECTED_SYSTEM_SHA256,
    expected_tool_schema_sha256: str = EXPECTED_TOOL_SCHEMA_SHA256,
    expected_tool_count: int = 20,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty safety pool: {output_dir}")
    if not min_train_tasks <= target_train_tasks <= 40:
        raise ValueError("V7.2 target train tasks must satisfy min <= target <= 40")
    if max_train_states < target_train_tasks or max_states_per_task <= 0:
        raise ValueError("V7.2 state caps are inconsistent")

    split_keys = {
        "train": _manifest_keys(train_manifest),
        "dev": _manifest_keys(dev_manifest),
        "heldout": _manifest_keys(heldout_manifest),
    }
    if any(
        set(split_keys[left]) & set(split_keys[right])
        for left, right in (("train", "dev"), ("train", "heldout"), ("dev", "heldout"))
    ):
        raise ValueError("V7.2 safety manifests overlap")
    split_sets = {key: set(value) for key, value in split_keys.items()}

    anchor_rows = [] if anchor_dataset is None else [
        row
        for row in _load_jsonl(anchor_dataset.resolve())
        if row.get("v485_route") == "protocol_safety"
    ]
    if anchor_dataset is not None and (
        len(anchor_rows) != 17
        or len({row["task_key"] for row in anchor_rows}) != 14
    ):
        raise ValueError("historical V7.1 anchor must be exactly safety17/14")
    for row in anchor_rows:
        _validate_state(
            row,
            label="anchor",
            expected_system_sha256=expected_system_sha256,
            expected_tool_schema_sha256=expected_tool_schema_sha256,
            expected_tool_count=expected_tool_count,
        )
        if str(row["task_key"]) in split_sets["heldout"]:
            raise ValueError("historical anchor overlaps heldout15")

    source_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    source_stats: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: (item.split, item.seed, str(item.path))):
        if source.split not in {"train", "dev"}:
            raise ValueError(f"unsupported safety source split: {source.split}")
        path = source.path.resolve()
        rows = _load_jsonl(path)
        for row in rows:
            _validate_state(
                row,
                label=f"{source.split}/seed{source.seed}",
                expected_system_sha256=expected_system_sha256,
                expected_tool_schema_sha256=expected_tool_schema_sha256,
                expected_tool_count=expected_tool_count,
            )
            task_key = str(row.get("task_key") or "")
            if task_key not in split_sets[source.split]:
                raise ValueError(
                    f"safety state is in the wrong split: {task_key}/{source.split}"
                )
            if int(row.get("rollout_seed", -1)) != source.seed:
                raise ValueError(f"safety source seed mismatch: {row.get('state_id')}")
            source_rows[source.split].append(row)
        source_stats.append(
            {
                "split": source.split,
                "seed": source.seed,
                "path": str(path),
                "sha256": _file_sha256(path),
                "states": len(rows),
                "tasks": len({str(row.get("task_key") or "") for row in rows}),
            }
        )

    anchor_train = [
        _annotate(row, split="train", origin="v71_anchor", optimizer_candidate=True)
        for row in anchor_rows
        if str(row["task_key"]) in split_sets["train"]
    ]
    anchor_dev = [
        _annotate(row, split="dev", origin="v71_anchor", optimizer_candidate=False)
        for row in anchor_rows
        if str(row["task_key"]) in split_sets["dev"]
    ]
    if anchor_dataset is not None:
        if (len(anchor_train), len({row["task_key"] for row in anchor_train})) != (13, 11):
            raise ValueError("split-aware train anchor must be safety13/11")
        if (len(anchor_dev), len({row["task_key"] for row in anchor_dev})) != (4, 3):
            raise ValueError("split-aware dev anchor must be safety4/3")

    train_candidates = _deduplicate(source_rows["train"])
    by_train_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_candidates:
        by_train_task[str(row["task_key"])].append(row)
    for rows in by_train_task.values():
        rows.sort(key=_state_score, reverse=True)

    selected_train = _deduplicate(anchor_train)
    selected_hashes = {str(row["source_prompt_sha256"]) for row in selected_train}
    selected_tasks = {str(row["task_key"]) for row in selected_train}
    unrepresented = [
        (task_key, rows[0])
        for task_key, rows in by_train_task.items()
        if task_key not in selected_tasks and rows
    ]
    unrepresented.sort(key=lambda item: (_state_score(item[1]), item[0]), reverse=True)
    for task_key, row in unrepresented:
        if len(selected_tasks) >= target_train_tasks:
            break
        prompt_hash = str(row["source_prompt_sha256"])
        if prompt_hash in selected_hashes:
            continue
        selected_train.append(
            _annotate(row, split="train", origin="v72_fresh", optimizer_candidate=True)
        )
        selected_hashes.add(prompt_hash)
        selected_tasks.add(task_key)
    if len(selected_tasks) < min_train_tasks:
        raise ValueError(
            f"V7.2 safety pool has only {len(selected_tasks)} train tasks; "
            f"minimum is {min_train_tasks}"
        )

    counts = Counter(str(row["task_key"]) for row in selected_train)
    supplements = sorted(
        train_candidates,
        key=lambda row: (
            int(row.get("rollout_seed", 0)) == 43,
            _state_score(row),
        ),
        reverse=True,
    )
    for row in supplements:
        if len(selected_train) >= max_train_states:
            break
        task_key = str(row["task_key"])
        prompt_hash = str(row["source_prompt_sha256"])
        if (
            task_key not in selected_tasks
            or counts[task_key] >= max_states_per_task
            or prompt_hash in selected_hashes
        ):
            continue
        selected_train.append(
            _annotate(row, split="train", origin="v72_fresh", optimizer_candidate=True)
        )
        selected_hashes.add(prompt_hash)
        counts[task_key] += 1

    dev_all = _deduplicate([*anchor_dev, *source_rows["dev"]])
    dev_counts: Counter[str] = Counter()
    selected_dev: list[dict[str, Any]] = []
    for row in sorted(dev_all, key=_state_score, reverse=True):
        task_key = str(row["task_key"])
        if dev_counts[task_key] >= max_states_per_task:
            continue
        selected_dev.append(
            _annotate(
                row,
                split="dev",
                origin=("v71_anchor" if row.get("v485_route") == "protocol_safety" else "v72_fresh"),
                optimizer_candidate=False,
            )
        )
        dev_counts[task_key] += 1
    selected_train.sort(key=lambda row: (str(row["task_key"]), str(row["state_id"])))
    selected_dev.sort(key=lambda row: (str(row["task_key"]), str(row["state_id"])))

    if {row["task_key"] for row in selected_train} & split_sets["heldout"]:
        raise ValueError("V7.2 train safety pool overlaps heldout15")
    if {row["task_key"] for row in selected_dev} & split_sets["train"]:
        raise ValueError("V7.2 dev safety pool leaks train tasks")

    benign_rows: list[dict[str, Any]] = []
    if benign_dataset is not None:
        benign_path = benign_dataset.resolve()
        for raw in _load_jsonl(benign_path):
            if not _is_registered_benign_retention(raw):
                continue
            if (
                raw.get("protocol_source") != "native"
                or not raw.get("native_tool_calling")
                or raw.get("source_system_prompt_sha256") != expected_system_sha256
                or raw.get("source_tool_schema_sha256") != expected_tool_schema_sha256
                or len(raw.get("available_tools") or []) != expected_tool_count
            ):
                raise ValueError(
                    f"invalid V7.2 native benign retention state: {raw.get('state_id')}"
                )
            row = deepcopy(raw)
            row["v485_route"] = "benign_retention"
            row["heldout15_used"] = False
            row["v72_optimizer_candidate"] = False
            benign_rows.append(row)
        if not benign_rows or len({row["task_key"] for row in benign_rows}) < 20:
            raise ValueError("V7.2 benign retention source lacks benign20 coverage")

    output_dir.mkdir(parents=True)
    train_path = output_dir / "train_candidates.jsonl"
    dev_path = output_dir / "dev.jsonl"
    _write_jsonl(train_path, selected_train)
    _write_jsonl(dev_path, selected_dev)
    training_path = output_dir / "train.jsonl"
    _write_jsonl(training_path, [*selected_train, *benign_rows])
    stats = {
        "strategy": "v72_split_safe_expanded_native_safety_pool_v1",
        "native_protocol_contract": {
            "system_prompt_sha256": expected_system_sha256,
            "tool_schema_sha256": expected_tool_schema_sha256,
            "tool_count": expected_tool_count,
        },
        "historical_anchor": {
            "enabled": anchor_dataset is not None,
            "states": len(anchor_rows),
            "tasks": len({row["task_key"] for row in anchor_rows}),
        },
        "train_anchor": {
            "states": len(anchor_train),
            "tasks": len({row["task_key"] for row in anchor_train}),
        },
        "dev_anchor": {
            "states": len(anchor_dev),
            "tasks": len({row["task_key"] for row in anchor_dev}),
        },
        "train": {
            "states": len(selected_train),
            "tasks": len({row["task_key"] for row in selected_train}),
            "origin_counts": dict(Counter(row["v72_safety_origin"] for row in selected_train)),
            "seed_counts": dict(Counter(str(row.get("rollout_seed")) for row in selected_train)),
        },
        "dev": {
            "states": len(selected_dev),
            "tasks": len({row["task_key"] for row in selected_dev}),
            "origin_counts": dict(Counter(row["v72_safety_origin"] for row in selected_dev)),
        },
        "sources": source_stats,
        "benign_retention": {
            "source": (
                None if benign_dataset is None else str(benign_dataset.resolve())
            ),
            "source_sha256": (
                None if benign_dataset is None else _file_sha256(benign_dataset)
            ),
            "states": len(benign_rows),
            "tasks": len({row["task_key"] for row in benign_rows}),
        },
        "manifests": {
            "train": _file_sha256(train_manifest),
            "dev": _file_sha256(dev_manifest),
            "heldout": _file_sha256(heldout_manifest),
        },
        "train_jsonl_sha256": _file_sha256(train_path),
        "training_jsonl_sha256": _file_sha256(training_path),
        "dev_jsonl_sha256": _file_sha256(dev_path),
        "heldout_overlap": 0,
        "heldout15_used": False,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


__all__ = ["SafetyStateSource", "build_v72_safety_pools"]
