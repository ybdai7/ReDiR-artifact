"""Build official V2 live final-state datasets for MT-AgentRisk OPD."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from latent_safety.data.mtagentrisk_opd_pairs import normalize_task_id
from latent_safety.data.mtagentrisk_opd_pairs import event_kind, load_trajectory, relative_or_abs
from latent_safety.data.mtagentrisk_final_state_rounds import (
    final_user_index,
    first_agent_message_after,
)


DEFAULT_SOURCE_DATASET_ROOT = (
    "outputs/latent_training/mtagentrisk_qwen35_filesystem_p3_masked_opd"
)
DEFAULT_LIVE_OUTPUT_ROOT = (
    "outputs/latent_training/"
    "mtagentrisk_qwen35_filesystem_p3_v2_iterative_finalstate_schema_mask_opd/"
    "live_round_0"
)


@dataclass(frozen=True)
class SourceRecord:
    row: dict[str, Any]
    source_split: str

    @property
    def task_key(self) -> str:
        return str(self.row["task_key"])

    @property
    def single_outcome(self) -> str:
        return str(self.row.get("single_outcome", "")).upper()

    @property
    def multi_outcome(self) -> str:
        return str(self.row.get("multi_outcome", "")).upper()

    @property
    def multi_task_path(self) -> str:
        return str(self.row.get("multi_task_path") or "")


def repo_relative_or_abs(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_records(source_dataset_root: Path, splits: Iterable[str]) -> dict[str, SourceRecord]:
    records: dict[str, SourceRecord] = {}
    for split in splits:
        for row in load_jsonl(source_dataset_root / f"{split}.jsonl"):
            key = normalize_task_id(str(row["task_key"]))
            record = dict(row)
            record["task_key"] = key
            records[key] = SourceRecord(record, split)
    return records


def load_live_snapshots(snapshot_path: Path) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(snapshot_path):
        key = normalize_task_id(str(row.get("task_key") or row.get("multi_task_id") or ""))
        if not key:
            continue
        # Keep the last snapshot if a task is retried in the same run.
        snapshots[key] = row
    return snapshots


def load_post_eval_outcomes(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("results", {})
    if not isinstance(raw, dict):
        return {}
    outcomes: dict[str, str] = {}
    for task_id, item in raw.items():
        outcome = item.get("outcome") if isinstance(item, dict) else item
        if isinstance(outcome, str):
            outcomes[normalize_task_id(task_id)] = outcome.upper()
    return outcomes


def trajectory_key_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("traj_"):
        stem = stem[len("traj_") :]
    return normalize_task_id(stem)


def infer_trajectory_root(snapshot_path: Path) -> Path:
    return snapshot_path.parent / "filesystem_multiturn"


def index_trajectories(trajectory_root: Path | None) -> dict[str, Path]:
    if trajectory_root is None or not trajectory_root.exists():
        return {}
    index: dict[str, Path] = {}
    for path in trajectory_root.glob("**/traj_*.json"):
        index[trajectory_key_from_path(path)] = path
    return index


def baseline_reference_after_final_user(traj_path: Path | None) -> tuple[str, str]:
    if traj_path is None or not traj_path.is_file():
        return "", ""
    traj = load_trajectory(traj_path)
    final_idx = final_user_index(traj)
    if final_idx is None:
        return "", ""
    reference = first_agent_message_after(traj, final_idx)
    return reference, event_kind(reference) if reference else ""


def materialize_task_root(
    *,
    repo_root: Path,
    source_dataset_root: Path,
    output_task_root: Path,
    source_splits: tuple[str, ...] = ("train", "dev"),
    single_outcome_filter: str = "REJECT",
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_source_records(source_dataset_root, source_splits)
    selected = [
        record
        for record in records.values()
        if record.single_outcome == single_outcome_filter.upper()
    ]
    selected.sort(key=lambda record: (record.source_split, record.task_key))
    if limit is not None:
        selected = selected[:limit]

    if output_task_root.exists():
        shutil.rmtree(output_task_root)
    output_task_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in selected:
        task_path = Path(record.multi_task_path)
        if not task_path.is_absolute():
            task_path = (repo_root / task_path).resolve()
        if not task_path.exists():
            raise FileNotFoundError(f"task path does not exist for {record.task_key}: {task_path}")
        link = output_task_root / task_path.name
        # run_mtagentrisk_batch.sh discovers tasks with `find -type d`, so the
        # task entries themselves must be real directories rather than symlinks.
        shutil.copytree(task_path, link, symlinks=True)
        rows.append(
            {
                "task_key": record.task_key,
                "source_split": record.source_split,
                "single_outcome": record.single_outcome,
                "multi_outcome": record.multi_outcome,
                "task_path": repo_relative_or_abs(task_path, repo_root),
                "task_link": repo_relative_or_abs(link, repo_root),
            }
        )

    write_jsonl(output_task_root / "task_manifest.jsonl", rows)
    (output_task_root / "task_keys.txt").write_text(
        "\n".join(row["task_key"] for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return {
        "source_dataset_root": str(source_dataset_root),
        "output_task_root": str(output_task_root),
        "source_splits": list(source_splits),
        "single_outcome_filter": single_outcome_filter.upper(),
        "selected_tasks": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def ensure_system_message(
    messages: list[dict[str, Any]],
    canonical_single_messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return messages
    if canonical_single_messages and canonical_single_messages[0].get("role") == "system":
        return [canonical_single_messages[0], *messages]
    return messages


def build_live_dataset(
    *,
    repo_root: Path,
    source_dataset_root: Path,
    snapshot_path: Path,
    output_root: Path,
    source_splits: tuple[str, ...] = ("train", "dev"),
    split_name: str = "train",
    live_post_eval_path: Path | None = None,
    trajectory_root: Path | None = None,
) -> dict[str, Any]:
    source_records = load_source_records(source_dataset_root, source_splits)
    snapshots = load_live_snapshots(snapshot_path)
    live_outcomes = load_post_eval_outcomes(live_post_eval_path)
    trajectory_index = index_trajectories(trajectory_root or infer_trajectory_root(snapshot_path))
    rows: list[dict[str, Any]] = []

    for key, source in sorted(source_records.items()):
        snapshot = snapshots.get(key)
        state_status = "missing_final_state"
        student_state_messages: list[dict[str, Any]] = []
        available_tools: list[dict[str, Any]] = []
        available_tool_names: list[str] = []
        state_source = "openhands_live_missing"
        rollout_policy = ""
        rollout_checkpoint = None
        trajectory_path = ""
        baseline_reference_message = ""
        baseline_reference_kind = ""
        created_at = datetime.now(timezone.utc).isoformat()
        if snapshot:
            state_status = str(snapshot.get("state_status") or "missing_final_state")
            state_source = str(snapshot.get("state_source") or "openhands_live_final_state")
            rollout_policy = str(snapshot.get("rollout_policy") or "")
            rollout_checkpoint = snapshot.get("rollout_checkpoint")
            student_state_messages = list(snapshot.get("student_state_messages") or [])
            available_tools = list(snapshot.get("available_tools") or [])
            available_tool_names = list(snapshot.get("available_tool_names") or [])
            trajectory_path = str(snapshot.get("trajectory_path") or "")
            baseline_reference_message = str(snapshot.get("baseline_reference_message") or "")
            baseline_reference_kind = str(snapshot.get("baseline_reference_kind") or "")
            created_at = str(snapshot.get("created_at") or created_at)

        indexed_traj_path = trajectory_index.get(key)
        if indexed_traj_path is not None:
            trajectory_path = repo_relative_or_abs(indexed_traj_path, repo_root)
            if state_status == "collected_final_state" and not baseline_reference_message:
                baseline_reference_message, baseline_reference_kind = baseline_reference_after_final_user(indexed_traj_path)

        canonical_single_messages = list(source.row["canonical_single_messages"])
        student_state_messages = ensure_system_message(student_state_messages, canonical_single_messages)
        no_latent_multiturn_outcome = live_outcomes.get(key, source.multi_outcome)
        is_gap_candidate = (
            source.single_outcome == "REJECT"
            and no_latent_multiturn_outcome == "COMPLETE"
            and state_status == "collected_final_state"
        )
        is_training_candidate = (
            source.single_outcome == "REJECT"
            and state_status == "collected_final_state"
            and bool(student_state_messages)
        )

        row = {
            "task_key": key,
            "split": split_name,
            "source_split": source.source_split,
            "single_task_id": source.row.get("single_task_id", ""),
            "multi_task_id": source.row.get("multi_task_id", ""),
            "single_outcome": source.single_outcome,
            "multi_outcome": no_latent_multiturn_outcome,
            "no_latent_multiturn_outcome": no_latent_multiturn_outcome,
            "outcome_source": "live_post_eval" if key in live_outcomes else "source_dataset",
            "canonical_single_messages": canonical_single_messages,
            "student_state_messages": student_state_messages,
            "available_tools": available_tools,
            "available_tool_names": available_tool_names,
            "state_source": state_source,
            "stop_point": "after_final_user_message_before_next_agent_action",
            "state_status": state_status,
            "rollout_policy": rollout_policy,
            "rollout_checkpoint": rollout_checkpoint,
            "trajectory_path": trajectory_path,
            "post_eval_outcome": no_latent_multiturn_outcome,
            "baseline_reference_message": baseline_reference_message,
            "baseline_reference_kind": baseline_reference_kind,
            "student_action_prefix_source": baseline_reference_message,
            "single_traj_path": source.row.get("single_traj_path", ""),
            "multi_traj_path": source.row.get("multi_traj_path", ""),
            "single_task_path": source.row.get("single_task_path", ""),
            "multi_task_path": source.row.get("multi_task_path", ""),
            "is_training_candidate": is_training_candidate,
            "is_gap_candidate": is_gap_candidate,
            "created_at": created_at,
        }
        rows.append(row)

    priority = {True: 0, False: 1}
    rows.sort(
        key=lambda row: (
            priority[row["is_gap_candidate"]],
            0 if row["is_training_candidate"] else 1,
            row["source_split"],
            row["task_key"],
        )
    )

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / f"{split_name}.jsonl", rows)
    manifest = live_dataset_manifest(
        rows,
        source_dataset_root=source_dataset_root,
        snapshot_path=snapshot_path,
        output_root=output_root,
        source_splits=source_splits,
        split_name=split_name,
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_stats(output_root / "stats.tsv", manifest)
    return manifest


def live_dataset_manifest(
    rows: list[dict[str, Any]],
    *,
    source_dataset_root: Path,
    snapshot_path: Path,
    output_root: Path,
    source_splits: tuple[str, ...],
    split_name: str,
) -> dict[str, Any]:
    single_counts = Counter(row["single_outcome"] for row in rows)
    status_counts = Counter(row["state_status"] for row in rows)
    outcome_counts = Counter((row["single_outcome"], row["no_latent_multiturn_outcome"]) for row in rows)
    return {
        "dataset": "mtagentrisk_qwen35_filesystem_p3_v2_live_finalstate_schema_mask_opd",
        "source_dataset_root": str(source_dataset_root),
        "snapshot_path": str(snapshot_path),
        "output_root": str(output_root),
        "source_splits": list(source_splits),
        "split_name": split_name,
        "train_pool_total": len(rows),
        "train_pool_single_reject": single_counts.get("REJECT", 0),
        "train_pool_filtered_complete": single_counts.get("COMPLETE", 0),
        "train_pool_filtered_failed": single_counts.get("FAILED", 0),
        "collected_final_state": status_counts.get("collected_final_state", 0),
        "training_candidates": sum(bool(row["is_training_candidate"]) for row in rows),
        "gap_candidates": sum(bool(row["is_gap_candidate"]) for row in rows),
        "single_outcome_counts": dict(single_counts),
        "state_status_counts": dict(status_counts),
        "outcome_counts": {f"{a}->{b}": count for (a, b), count in outcome_counts.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_stats(path: Path, manifest: dict[str, Any]) -> None:
    rows = ["metric\tvalue"]
    for key in (
        "train_pool_total",
        "train_pool_single_reject",
        "train_pool_filtered_complete",
        "train_pool_filtered_failed",
        "collected_final_state",
        "training_candidates",
        "gap_candidates",
    ):
        rows.append(f"{key}\t{manifest.get(key, 0)}")
    for key, value in sorted(manifest.get("single_outcome_counts", {}).items()):
        rows.append(f"single_outcome.{key}\t{value}")
    for key, value in sorted(manifest.get("state_status_counts", {}).items()):
        rows.append(f"state_status.{key}\t{value}")
    for key, value in sorted(manifest.get("outcome_counts", {}).items()):
        rows.append(f"outcome.{key}\t{value}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
