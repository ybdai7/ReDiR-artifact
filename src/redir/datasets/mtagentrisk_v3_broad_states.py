"""Build V3 broad/final decision-state datasets for MT-AgentRisk OPD."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from redir.datasets.mtagentrisk_final_state_rounds import (
    final_user_index,
    messages_until_index,
    read_dependencies,
    tool_names_for_dependencies,
    tool_schemas_for_names,
)
from redir.datasets.mtagentrisk_opd_pairs import (
    event_kind,
    event_message,
    is_continue_message,
    is_empty_assistant_wait,
    is_retrieval_message,
    load_trajectory,
    looks_like_observation,
    normalize_task_id,
)
from redir.datasets.mtagentrisk_v2_live_dataset import (
    SourceRecord,
    load_post_eval_outcomes,
    load_source_records,
    repo_relative_or_abs,
    write_jsonl,
)
from redir.engine.targets import (
    CanonicalTargetError,
    canonical_assistant_message_from_event,
)
from redir.engine.masks import (
    DecisionParseResult,
    parse_decision_spans,
    parse_event_lifecycle_spans,
)


DEFAULT_SOURCE_DATASET_ROOT = (
    "outputs/latent_training/mtagentrisk_qwen35_filesystem_p3_masked_opd"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/latent_training/"
    "mtagentrisk_qwen35_filesystem_p3_v3_state_weighted_iterative_broad_final_opd/"
    "round_0"
)

STATE_WEIGHTS = {
    "final_state": 1.0,
    "unsafe_or_gap_tool_action": 1.0,
    "safety_analysis": 1.0,
    "finish": 1.0,
    "ordinary_tool_action": 1.0,
    "shell_action": 1.0,
    "ordinary_message": 1.0,
}

MALFORMED_COMPLETION_MARKERS = (
    "Missing required parameters for function",
    "Parameter 'command' is expected to be one of",
    "Failed to parse function arguments",
    "Failed to parse tool call",
    "Invalid tool call",
    "malformed tool call",
)


def split_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def parse_outcome_filter(value: str | None) -> set[str]:
    if value is None or value.strip() in {"", "all", "*"}:
        return set()
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def is_malformed_completion(text: str) -> bool:
    return any(marker in text for marker in MALFORMED_COMPLETION_MARKERS)


def task_path_for(record: SourceRecord, repo_root: Path) -> Path:
    path = Path(record.multi_task_path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def materialize_task_root(
    *,
    repo_root: Path,
    source_dataset_root: Path,
    output_task_root: Path,
    source_splits: tuple[str, ...] = ("train", "dev"),
    single_outcome_filter: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Create a real-directory task root for live rollout collection."""
    records = load_source_records(source_dataset_root, source_splits)
    allowed_outcomes = parse_outcome_filter(single_outcome_filter)
    selected = [
        record
        for record in records.values()
        if not allowed_outcomes or record.single_outcome in allowed_outcomes
    ]
    selected.sort(key=lambda record: (record.source_split, record.task_key))
    if limit is not None:
        selected = selected[:limit]

    if output_task_root.exists():
        shutil.rmtree(output_task_root)
    output_task_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for record in selected:
        task_path = task_path_for(record, repo_root)
        if not task_path.exists():
            raise FileNotFoundError(f"task path does not exist for {record.task_key}: {task_path}")
        link = output_task_root / task_path.name
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
    manifest = {
        "dataset": "mtagentrisk_qwen35_filesystem_p3_v3_task_root",
        "source_dataset_root": str(source_dataset_root),
        "output_task_root": str(output_task_root),
        "source_splits": list(source_splits),
        "single_outcome_filter": sorted(allowed_outcomes) if allowed_outcomes else "all",
        "selected_tasks": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_task_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def trajectory_key_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("traj_"):
        stem = stem[len("traj_") :]
    return normalize_task_id(stem)


def index_trajectories(trajectory_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not trajectory_root.exists():
        return index
    for path in trajectory_root.glob("**/traj_*.json"):
        index[trajectory_key_from_path(path)] = path
    return index


def is_agent_decision_event(event: dict[str, Any], message: str) -> bool:
    if event.get("source") != "agent" or event.get("id") == 0:
        return False
    if not message or is_empty_assistant_wait(message) or looks_like_observation(message):
        return False
    return True


def user_turn_count_before(traj: list[dict[str, Any]], end_index: int) -> int:
    count = 0
    for event in traj[: end_index + 1]:
        if event.get("source") != "user":
            continue
        message = event_message(event)
        if not message or is_retrieval_message(message) or is_continue_message(message):
            continue
        count += 1
    return count


def classify_action_kind(message: str, parse: DecisionParseResult) -> str:
    if parse.parsed:
        if parse.mask_source == "shell_action":
            return "shell_action"
        if parse.mask_kind == "finish_head":
            return "finish"
        if parse.mask_source in {"finish", "refusal"} or parse.mask_kind == "refusal_sentence":
            return "safety_analysis"
        if parse.mask_kind in {"tool_name", "action_head"}:
            return "tool_action"
    kind = event_kind(message)
    if kind == "mcp_action":
        return "tool_action"
    if kind == "refusal":
        return "safety_analysis"
    return "ordinary_message"


def state_kind_for(
    *,
    action_kind: str,
    is_final_state: bool,
    single_outcome: str,
    multi_outcome: str,
) -> str:
    if is_final_state:
        return "final_state"
    if action_kind in {"tool_action", "shell_action"}:
        if single_outcome == "REJECT" and multi_outcome == "COMPLETE":
            return "unsafe_or_gap_tool_action"
        return "ordinary_tool_action" if action_kind == "tool_action" else "shell_action"
    if action_kind in {"safety_analysis", "finish"}:
        return action_kind
    return "ordinary_message"


def reveal_status_for(
    *,
    final_idx: int | None,
    event_index: int,
    final_reference_used: bool,
) -> str:
    if final_idx is None:
        return "unknown"
    if event_index <= final_idx:
        return "pre_final"
    if not final_reference_used:
        return "final_decision"
    return "post_final_decision"


def sample_weight_for(state_kind: str) -> float:
    return float(STATE_WEIGHTS.get(state_kind, 0.25))


def capped_state_rows(
    rows: list[dict[str, Any]],
    *,
    max_tool_states_per_task: int,
    max_safety_states_per_task: int,
    max_final_states_per_task: int,
    max_ordinary_states_per_task: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        state_kind = str(row["state_kind"])
        if state_kind == "final_state":
            bucket = "final"
        elif row["action_kind"] in {"tool_action", "shell_action"}:
            bucket = "tool"
        elif row["action_kind"] in {"safety_analysis", "finish"}:
            bucket = "safety"
        else:
            bucket = "ordinary"
        buckets[bucket].append(row)

    for values in buckets.values():
        values.sort(
            key=lambda row: (
                -float(row.get("sample_weight", 1.0)),
                0 if row.get("is_gap_candidate") else 1,
                int(row.get("event_index", 0)),
            )
        )

    selected: list[dict[str, Any]] = []
    selected.extend(buckets["final"][:max_final_states_per_task])
    selected.extend(buckets["tool"][:max_tool_states_per_task])
    selected.extend(buckets["safety"][:max_safety_states_per_task])
    selected.extend(buckets["ordinary"][:max_ordinary_states_per_task])
    selected.sort(key=lambda row: (int(row["event_index"]), str(row["state_id"])))
    return selected


def extract_states_for_record(
    source: SourceRecord,
    *,
    traj_path: Path,
    repo_root: Path,
    round_id: int,
    rollout_policy: str,
    rollout_checkpoint: str | None,
    live_outcome: str | None,
    max_tool_states_per_task: int,
    max_safety_states_per_task: int,
    max_final_states_per_task: int,
    max_ordinary_states_per_task: int,
) -> list[dict[str, Any]]:
    traj = load_trajectory(traj_path)
    final_idx = final_user_index(traj)
    system_content = source.row["canonical_single_messages"][0]["content"]
    task_path = task_path_for(source, repo_root)
    dependencies = read_dependencies(task_path) if task_path.exists() else []
    tool_names = tool_names_for_dependencies(dependencies)
    available_tools = tool_schemas_for_names(tool_names)
    multi_outcome = (live_outcome or source.multi_outcome).upper()

    candidate_rows: list[dict[str, Any]] = []
    final_reference_used = False
    for idx, event in enumerate(traj):
        message = event_message(event)
        if not is_agent_decision_event(event, message):
            continue
        reveal_status = reveal_status_for(
            final_idx=final_idx,
            event_index=idx,
            final_reference_used=final_reference_used,
        )
        is_final_state = reveal_status == "final_decision"
        if is_final_state:
            final_reference_used = True
        student_state_messages = messages_until_index(
            traj,
            idx - 1,
            system_content=system_content,
        )
        parser_result = parse_decision_spans(message, available_tools or tool_names)
        event_lifecycle_result = parse_event_lifecycle_spans(message, available_tools or tool_names)
        action_kind = classify_action_kind(message, parser_result)
        state_kind = state_kind_for(
            action_kind=action_kind,
            is_final_state=is_final_state,
            single_outcome=source.single_outcome,
            multi_outcome=multi_outcome,
        )
        sample_weight = sample_weight_for(state_kind)
        parsed = parser_result.parsed
        single_outcome = source.single_outcome
        malformed_completion = is_malformed_completion(message)
        try:
            canonical_assistant_message, canonical_meta = canonical_assistant_message_from_event(event)
            canonical_completion_valid = True
            canonical_skip_reason = ""
        except (CanonicalTargetError, TypeError, ValueError, json.JSONDecodeError) as exc:
            canonical_assistant_message = None
            canonical_meta = {
                "canonical_action_channel_label": "",
                "canonical_completion_source": "",
                "raw_openhands_action": event.get("action"),
            }
            canonical_completion_valid = False
            canonical_skip_reason = str(exc)
        is_format_candidate = (
            parsed
            and single_outcome in {"COMPLETE", "REJECT"}
            and bool(student_state_messages)
            and bool(message)
            and not malformed_completion
        )
        is_safety_candidate = (
            parsed
            and single_outcome == "REJECT"
            and bool(student_state_messages)
            and bool(message)
            and not malformed_completion
        )
        is_broad_opd_candidate = (
            parsed
            and single_outcome in {"COMPLETE", "REJECT"}
            and bool(student_state_messages)
            and bool(message)
            and not malformed_completion
        )
        retention_candidate = is_broad_opd_candidate
        safety_revealed_candidate = (
            parsed
            and single_outcome == "REJECT"
            and reveal_status in {"final_decision", "post_final_decision"}
            and bool(student_state_messages)
            and bool(message)
            and not malformed_completion
        )
        is_v35_training_candidate = retention_candidate or safety_revealed_candidate
        lifecycle_warmup_candidate = (
            event_lifecycle_result.parsed
            and single_outcome in {"COMPLETE", "REJECT"}
            and bool(student_state_messages)
            and bool(message)
            and not malformed_completion
            and not (
                state_kind == "finish"
                and reveal_status not in {"final_decision", "post_final_decision"}
            )
        )
        safety_opd_candidate = (
            event_lifecycle_result.parsed
            and single_outcome == "REJECT"
            and reveal_status in {"final_decision", "post_final_decision"}
            and bool(student_state_messages)
            and bool(message)
            and not malformed_completion
        )
        candidate_rows.append(
            {
                "task_key": source.task_key,
                "state_id": f"{source.task_key}:{idx}:{state_kind}",
                "split": "train",
                "source_split": source.source_split,
                "round_id": round_id,
                "source_rollout_policy": rollout_policy,
                "rollout_policy": rollout_policy,
                "rollout_checkpoint": rollout_checkpoint,
                "single_task_id": source.row.get("single_task_id", ""),
                "multi_task_id": source.row.get("multi_task_id", ""),
                "single_outcome": single_outcome,
                "multi_outcome": multi_outcome,
                "no_latent_multiturn_outcome": multi_outcome,
                "canonical_single_messages": source.row["canonical_single_messages"],
                "student_state_messages": student_state_messages,
                "raw_multiturn_messages": student_state_messages,
                "action_completion": message,
                "student_action_prefix_source": message,
                "baseline_reference_message": message,
                "raw_openhands_action": event.get("action"),
                "canonical_assistant_message": canonical_assistant_message,
                "canonical_action_channel_label": canonical_meta.get("canonical_action_channel_label", ""),
                "canonical_completion_source": canonical_meta.get("canonical_completion_source", ""),
                "canonical_completion_valid": canonical_completion_valid,
                "canonical_skip_reason": canonical_skip_reason,
                "available_tools": available_tools,
                "available_tool_names": tool_names,
                "state_source": "openhands_full_trajectory",
                "state_kind": state_kind,
                "action_kind": action_kind,
                "reveal_status": reveal_status,
                "is_final_state": is_final_state,
                "turn_index": user_turn_count_before(traj, idx - 1),
                "turn_count": user_turn_count_before(traj, len(traj) - 1),
                "event_index": idx,
                "event_id": event.get("id"),
                "trajectory_path": repo_relative_or_abs(traj_path, repo_root),
                "post_eval_outcome": multi_outcome,
                "parser_result": parser_result.as_dict(),
                "event_lifecycle_parser_result": event_lifecycle_result.as_dict(),
                "parsed_decision": parsed,
                "parsed_event_lifecycle": event_lifecycle_result.parsed,
                "malformed_completion": malformed_completion,
                "sample_weight": sample_weight,
                "state_weight": sample_weight,
                "is_training_candidate": is_format_candidate,
                "is_format_candidate": is_format_candidate,
                "is_broad_opd_candidate": is_broad_opd_candidate,
                "is_safety_candidate": is_safety_candidate,
                "retention_candidate": retention_candidate,
                "safety_revealed_candidate": safety_revealed_candidate,
                "is_v35_training_candidate": is_v35_training_candidate,
                "lifecycle_warmup_candidate": lifecycle_warmup_candidate,
                "safety_opd_candidate": safety_opd_candidate,
                "is_v37_training_candidate": lifecycle_warmup_candidate or safety_opd_candidate,
                "is_gap_candidate": (
                    single_outcome == "REJECT"
                    and multi_outcome == "COMPLETE"
                    and state_kind in {"final_state", "unsafe_or_gap_tool_action"}
                ),
                "single_traj_path": source.row.get("single_traj_path", ""),
                "multi_traj_path": source.row.get("multi_traj_path", ""),
                "single_task_path": source.row.get("single_task_path", ""),
                "multi_task_path": source.row.get("multi_task_path", ""),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return capped_state_rows(
        candidate_rows,
        max_tool_states_per_task=max_tool_states_per_task,
        max_safety_states_per_task=max_safety_states_per_task,
        max_final_states_per_task=max_final_states_per_task,
        max_ordinary_states_per_task=max_ordinary_states_per_task,
    )


def build_broad_state_dataset(
    *,
    repo_root: Path,
    source_dataset_root: Path,
    trajectory_root: Path,
    output_root: Path,
    source_splits: tuple[str, ...] = ("train", "dev"),
    split_name: str = "train",
    round_id: int = 0,
    rollout_policy: str = "qwen35_9b_p3_no_latent",
    rollout_checkpoint: str | None = None,
    live_post_eval_path: Path | None = None,
    max_tool_states_per_task: int = 3,
    max_safety_states_per_task: int = 2,
    max_final_states_per_task: int = 1,
    max_ordinary_states_per_task: int = 1,
) -> dict[str, Any]:
    source_records = load_source_records(source_dataset_root, source_splits)
    trajectory_index = index_trajectories(trajectory_root)
    live_outcomes = load_post_eval_outcomes(live_post_eval_path)
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for key, source in sorted(source_records.items()):
        traj_path = trajectory_index.get(key)
        if traj_path is None:
            missing.append(
                {
                    "task_key": key,
                    "source_split": source.source_split,
                    "single_outcome": source.single_outcome,
                    "multi_outcome": source.multi_outcome,
                    "reason": "missing_trajectory",
                }
            )
            continue
        rows.extend(
            extract_states_for_record(
                source,
                traj_path=traj_path,
                repo_root=repo_root,
                round_id=round_id,
                rollout_policy=rollout_policy,
                rollout_checkpoint=rollout_checkpoint,
                live_outcome=live_outcomes.get(key),
                max_tool_states_per_task=max_tool_states_per_task,
                max_safety_states_per_task=max_safety_states_per_task,
                max_final_states_per_task=max_final_states_per_task,
                max_ordinary_states_per_task=max_ordinary_states_per_task,
            )
        )

    rows.sort(
        key=lambda row: (
            0 if row.get("is_gap_candidate") else 1,
            row.get("source_split", ""),
            row.get("task_key", ""),
            int(row.get("event_index", 0)),
        )
    )
    for row in rows:
        row["split"] = split_name

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / f"{split_name}.jsonl", rows)
    write_jsonl(output_root / "debug_missing_trajectories.jsonl", missing)
    manifest = broad_state_manifest(
        rows,
        missing=missing,
        source_dataset_root=source_dataset_root,
        trajectory_root=trajectory_root,
        output_root=output_root,
        source_splits=source_splits,
        split_name=split_name,
        round_id=round_id,
        rollout_policy=rollout_policy,
        rollout_checkpoint=rollout_checkpoint,
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_stats(output_root / "stats.tsv", manifest)
    return manifest


def broad_state_manifest(
    rows: list[dict[str, Any]],
    *,
    missing: list[dict[str, Any]],
    source_dataset_root: Path,
    trajectory_root: Path,
    output_root: Path,
    source_splits: tuple[str, ...],
    split_name: str,
    round_id: int,
    rollout_policy: str,
    rollout_checkpoint: str | None,
) -> dict[str, Any]:
    single_counts = Counter(row["single_outcome"] for row in rows)
    multi_counts = Counter(row["multi_outcome"] for row in rows)
    state_counts = Counter(row["state_kind"] for row in rows)
    action_counts = Counter(row["action_kind"] for row in rows)
    reveal_counts = Counter(row.get("reveal_status", "unknown") for row in rows)
    outcome_counts = Counter((row["single_outcome"], row["multi_outcome"]) for row in rows)
    task_keys = {row["task_key"] for row in rows}
    parsed_count = sum(bool(row.get("parsed_decision")) for row in rows)
    parsed_event_lifecycle_count = sum(bool(row.get("parsed_event_lifecycle")) for row in rows)
    canonical_valid_count = sum(bool(row.get("canonical_completion_valid")) for row in rows)
    canonical_label_counts = Counter(row.get("canonical_action_channel_label", "") for row in rows)
    return {
        "dataset": "mtagentrisk_qwen35_filesystem_p3_v3_state_weighted_iterative_broad_final_opd",
        "source_dataset_root": str(source_dataset_root),
        "trajectory_root": str(trajectory_root),
        "output_root": str(output_root),
        "source_splits": list(source_splits),
        "split_name": split_name,
        "round_id": round_id,
        "rollout_policy": rollout_policy,
        "rollout_checkpoint": rollout_checkpoint,
        "task_count": len(task_keys),
        "state_count": len(rows),
        "missing_trajectory_count": len(missing),
        "training_candidates": sum(bool(row.get("is_training_candidate")) for row in rows),
        "format_candidates": sum(bool(row.get("is_format_candidate")) for row in rows),
        "broad_opd_candidates": sum(bool(row.get("is_broad_opd_candidate")) for row in rows),
        "safety_candidates": sum(bool(row.get("is_safety_candidate")) for row in rows),
        "retention_candidates": sum(bool(row.get("retention_candidate")) for row in rows),
        "safety_revealed_candidates": sum(bool(row.get("safety_revealed_candidate")) for row in rows),
        "v35_training_candidates": sum(bool(row.get("is_v35_training_candidate")) for row in rows),
        "lifecycle_warmup_candidates": sum(bool(row.get("lifecycle_warmup_candidate")) for row in rows),
        "safety_opd_candidates": sum(bool(row.get("safety_opd_candidate")) for row in rows),
        "v37_training_candidates": sum(bool(row.get("is_v37_training_candidate")) for row in rows),
        "gap_candidates": sum(bool(row.get("is_gap_candidate")) for row in rows),
        "parsed_count": parsed_count,
        "parsed_rate": parsed_count / max(len(rows), 1),
        "parsed_event_lifecycle_count": parsed_event_lifecycle_count,
        "parsed_event_lifecycle_rate": parsed_event_lifecycle_count / max(len(rows), 1),
        "canonical_completion_valid_count": canonical_valid_count,
        "canonical_completion_valid_rate": canonical_valid_count / max(len(rows), 1),
        "single_outcome_counts": dict(single_counts),
        "multi_outcome_counts": dict(multi_counts),
        "state_kind_counts": dict(state_counts),
        "action_kind_counts": dict(action_counts),
        "reveal_status_counts": dict(reveal_counts),
        "canonical_action_channel_label_counts": dict(canonical_label_counts),
        "outcome_counts": {f"{a}->{b}": count for (a, b), count in outcome_counts.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_stats(path: Path, manifest: dict[str, Any]) -> None:
    rows = ["metric\tvalue"]
    for key in (
        "task_count",
        "state_count",
        "missing_trajectory_count",
        "training_candidates",
        "format_candidates",
        "safety_candidates",
        "retention_candidates",
        "safety_revealed_candidates",
        "v35_training_candidates",
        "lifecycle_warmup_candidates",
        "safety_opd_candidates",
        "v37_training_candidates",
        "gap_candidates",
        "parsed_count",
        "parsed_rate",
        "parsed_event_lifecycle_count",
        "parsed_event_lifecycle_rate",
        "canonical_completion_valid_count",
        "canonical_completion_valid_rate",
    ):
        rows.append(f"{key}\t{manifest.get(key, 0)}")
    for group in (
        "single_outcome_counts",
        "multi_outcome_counts",
        "state_kind_counts",
        "action_kind_counts",
        "reveal_status_counts",
        "canonical_action_channel_label_counts",
        "outcome_counts",
    ):
        for key, value in sorted(manifest.get(group, {}).items()):
            rows.append(f"{group}.{key}\t{value}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
