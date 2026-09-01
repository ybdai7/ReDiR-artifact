"""Build V2 final-agent-state OPD rounds from MT-AgentRisk trajectories."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from redir.datasets.mtagentrisk_opd_pairs import (
    event_kind,
    event_message,
    is_empty_assistant_wait,
    is_retrieval_message,
    load_trajectory,
    looks_like_observation,
    relative_or_abs,
)


ALLOWED_STATE_STATUSES = {
    "collected_final_state",
    "early_refusal",
    "early_failed",
    "step_limit",
    "tool_error",
    "missing_final_state",
}

DEPENDENCY_TOOL_NAMES: dict[str, list[str]] = {
    "mcp-filesystem": [
        "read_file",
        "read_text_file",
        "read_media_file",
        "read_multiple_files",
        "write_file",
        "edit_file",
        "create_directory",
        "list_directory",
        "list_directory_with_sizes",
        "directory_tree",
        "move_file",
        "search_files",
        "get_file_info",
        "list_allowed_directories",
    ],
    "mcp-postgres": [
        "list_schemas",
        "list_objects",
        "get_object_details",
        "explain_query",
        "analyze_workload_indexes",
        "analyze_query_indexes",
        "analyze_db_health",
        "get_top_queries",
        "execute_sql",
    ],
    "mcp-notion": [
        "API-get-user",
        "API-get-users",
        "API-get-self",
        "API-post-search",
        "API-get-block-children",
        "API-patch-block-children",
        "API-retrieve-a-block",
        "API-update-a-block",
        "API-delete-a-block",
        "API-retrieve-a-page",
        "API-patch-page",
        "API-post-page",
        "API-retrieve-a-page-property",
        "API-retrieve-a-comment",
        "API-create-a-comment",
        "API-query-data-source",
        "API-retrieve-a-data-source",
        "API-update-a-data-source",
        "API-create-a-data-source",
        "API-list-data-source-templates",
        "API-move-page",
    ],
    "mcp-playwright": [
        "browser_close",
        "browser_resize",
        "browser_console_messages",
        "browser_handle_dialog",
        "browser_evaluate",
        "browser_file_upload",
        "browser_fill_form",
        "browser_install",
        "browser_press_key",
        "browser_type",
        "browser_navigate",
        "browser_navigate_back",
        "browser_network_requests",
        "browser_run_code",
        "browser_take_screenshot",
        "browser_snapshot",
        "browser_click",
        "browser_drag",
        "browser_hover",
        "browser_select_option",
        "browser_tabs",
        "browser_wait_for",
    ],
    "mcp-gitlab": [
        "create_issue",
        "create_merge_request",
        "get_project",
        "list_merge_requests",
        "create_branch",
        "create_or_update_file",
        "push_files",
        "get_file_contents",
        "search_repositories",
    ],
    "mcp-gmail": [
        "GMAIL_SEND_EMAIL",
        "GMAIL_FETCH_EMAILS",
        "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
        "GMAIL_FETCH_MESSAGE_BY_THREAD_ID",
        "GMAIL_LIST_THREADS",
        "GMAIL_CREATE_EMAIL_DRAFT",
        "GMAIL_UPDATE_DRAFT",
        "GMAIL_SEND_DRAFT",
        "GMAIL_LIST_DRAFTS",
        "GMAIL_GET_DRAFT",
        "GMAIL_DELETE_DRAFT",
        "GMAIL_FORWARD_MESSAGE",
        "GMAIL_REPLY_TO_THREAD",
        "GMAIL_ADD_LABEL_TO_EMAIL",
        "GMAIL_CREATE_LABEL",
        "GMAIL_DELETE_LABEL",
        "GMAIL_PATCH_LABEL",
        "GMAIL_LIST_LABELS",
        "GMAIL_MODIFY_THREAD_LABELS",
        "GMAIL_MOVE_TO_TRASH",
        "GMAIL_DELETE_MESSAGE",
        "GMAIL_BATCH_DELETE_MESSAGES",
        "GMAIL_BATCH_MODIFY_MESSAGES",
        "GMAIL_GET_ATTACHMENT",
        "GMAIL_GET_PROFILE",
        "GMAIL_GET_CONTACTS",
        "GMAIL_SEARCH_PEOPLE",
        "GMAIL_GET_PEOPLE",
        "GMAIL_LIST_FILTERS",
        "GMAIL_LIST_HISTORY",
        "GMAIL_GET_AUTO_FORWARDING",
        "GMAIL_GET_VACATION_SETTINGS",
        "GMAIL_GET_LANGUAGE_SETTINGS",
        "GMAIL_SETTINGS_GET_IMAP",
        "GMAIL_SETTINGS_GET_POP",
        "GMAIL_SETTINGS_SEND_AS_GET",
        "GMAIL_LIST_CSE_IDENTITIES",
        "GMAIL_LIST_CSE_KEYPAIRS",
        "GMAIL_LIST_SMIME_INFO",
    ],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_dependencies(task_path: Path) -> list[str]:
    dep_path = task_path / "utils" / "dependencies.yml"
    if not dep_path.exists():
        return []
    value = yaml.safe_load(dep_path.read_text(encoding="utf-8")) or []
    return [str(item) for item in value] if isinstance(value, list) else []


def tool_names_for_dependencies(dependencies: Iterable[str]) -> list[str]:
    names: list[str] = []
    for dep in dependencies:
        for name in DEPENDENCY_TOOL_NAMES.get(dep, []):
            if name not in names:
                names.append(name)
    return names


def tool_schemas_for_names(names: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def final_user_index(traj: list[dict[str, Any]]) -> int | None:
    for idx, event in reversed(list(enumerate(traj))):
        if event.get("source") != "user":
            continue
        message = event_message(event)
        if message.startswith("Final Turn") or "Final Turn:" in message:
            return idx
    return None


def first_agent_message_after(traj: list[dict[str, Any]], start_index: int) -> str:
    for event in traj[start_index + 1 :]:
        if event.get("source") != "agent" or event.get("id") == 0:
            continue
        message = event_message(event)
        if not message or is_empty_assistant_wait(message) or looks_like_observation(message):
            continue
        return message
    return ""


def state_status_for(record: dict[str, Any], traj: list[dict[str, Any]], final_idx: int | None) -> str:
    if final_idx is not None:
        return "collected_final_state"
    outcome = str(record.get("multi_outcome", "")).upper()
    if outcome == "REJECT":
        return "early_refusal"
    if outcome == "FAILED":
        return "early_failed"
    if any("step limit" in event_message(event).lower() for event in traj):
        return "step_limit"
    if any("error calling tool" in event_message(event).lower() for event in traj):
        return "tool_error"
    return "missing_final_state"


def messages_until_index(
    traj: list[dict[str, Any]],
    end_index: int,
    *,
    system_content: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for event in traj[1 : end_index + 1]:
        source = event.get("source")
        message = event_message(event)
        if not message or is_empty_assistant_wait(message):
            continue
        if is_retrieval_message(message):
            continue
        if source == "user":
            messages.append({"role": "user", "content": message})
        elif source == "environment":
            messages.append({"role": "tool", "content": message})
        elif source == "agent":
            if looks_like_observation(message):
                messages.append({"role": "tool", "content": message})
            else:
                messages.append({"role": "assistant", "content": message})
    return messages


def build_snapshot_record(
    record: dict[str, Any],
    *,
    repo_root: Path,
    round_id: int,
    rollout_policy: str,
    rollout_checkpoint: str | None,
) -> dict[str, Any]:
    raw_traj_path = str(record.get("multi_traj_path") or "")
    traj_path = Path(raw_traj_path) if raw_traj_path else Path()
    if raw_traj_path and not traj_path.is_absolute():
        traj_path = (repo_root / traj_path).resolve()
    traj = load_trajectory(traj_path) if raw_traj_path and traj_path.is_file() else []
    final_idx = final_user_index(traj) if traj else None
    status = state_status_for(record, traj, final_idx)

    task_path = Path(record.get("multi_task_path") or "")
    if not task_path.is_absolute():
        task_path = (repo_root / task_path).resolve()
    dependencies = read_dependencies(task_path) if task_path.exists() else []
    tool_names = tool_names_for_dependencies(dependencies)

    system_content = record["canonical_single_messages"][0]["content"]
    if final_idx is not None:
        student_state_messages = messages_until_index(traj, final_idx, system_content=system_content)
        baseline_reference = first_agent_message_after(traj, final_idx)
    else:
        student_state_messages = record.get("raw_multiturn_messages") or []
        baseline_reference = record.get("student_action_prefix_source") or ""

    turn_count = sum(
        1
        for event in traj
        if event.get("source") == "user"
        and event_message(event)
        and not is_retrieval_message(event_message(event))
        and not event_message(event).startswith("Please continue working")
        and not event_message(event).startswith("Thanks, that covers everything")
    )

    return {
        "task_key": record["task_key"],
        "split": record["split"],
        "round_id": round_id,
        "rollout_policy": rollout_policy,
        "rollout_checkpoint": rollout_checkpoint,
        "single_task_id": record["single_task_id"],
        "multi_task_id": record["multi_task_id"],
        "single_outcome": record["single_outcome"],
        "multi_outcome": record["multi_outcome"],
        "canonical_single_messages": record["canonical_single_messages"],
        "student_state_messages": student_state_messages,
        "available_tools": tool_schemas_for_names(tool_names),
        "available_tool_names": tool_names,
        "state_source": "openhands_trajectory_slice",
        "stop_point": "after_final_user_message_before_next_agent_action",
        "state_status": status,
        "turn_index": turn_count if status == "collected_final_state" else None,
        "turn_count": turn_count,
        "trajectory_path": relative_or_abs(traj_path, repo_root) if raw_traj_path and traj_path.exists() else "",
        "post_eval_outcome": record["multi_outcome"],
        "baseline_reference_message": baseline_reference,
        "baseline_reference_kind": event_kind(baseline_reference) if baseline_reference else "",
        "single_traj_path": record.get("single_traj_path", ""),
        "multi_traj_path": record.get("multi_traj_path", ""),
        "single_task_path": record.get("single_task_path", ""),
        "multi_task_path": record.get("multi_task_path", ""),
        "is_training_candidate": (
            record.get("single_outcome") == "REJECT"
            and status == "collected_final_state"
            and bool(student_state_messages)
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def materialize_round_dataset(
    *,
    repo_root: Path,
    source_dataset_root: Path,
    output_root: Path,
    round_id: int,
    rollout_policy: str,
    rollout_checkpoint: str | None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split in ("train", "dev", "test"):
        source_path = source_dataset_root / f"{split}.jsonl"
        records = [
            build_snapshot_record(
                record,
                repo_root=repo_root,
                round_id=round_id,
                rollout_policy=rollout_policy,
                rollout_checkpoint=rollout_checkpoint,
            )
            for record in load_jsonl(source_path)
        ]
        records.sort(key=lambda row: row["task_key"])
        write_jsonl(output_root / f"{split}.jsonl", records)
        by_split[split].extend(records)
        all_records.extend(records)

    status_counts = Counter(record["state_status"] for record in all_records)
    split_counts = Counter(record["split"] for record in all_records)
    training_candidates = {
        split: sum(record.get("is_training_candidate", False) for record in records)
        for split, records in by_split.items()
    }
    manifest = {
        "dataset": "mtagentrisk_qwen35_filesystem_p3_v2_iterative_finalstate_schema_mask_opd",
        "source_dataset_root": str(source_dataset_root),
        "output_root": str(output_root),
        "round_id": round_id,
        "rollout_policy": rollout_policy,
        "rollout_checkpoint": rollout_checkpoint,
        "record_count": len(all_records),
        "split_counts": dict(split_counts),
        "state_status_counts": dict(status_counts),
        "training_candidates": training_candidates,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stats_lines = ["metric\tvalue"]
    stats_lines.extend(f"split.{key}\t{value}" for key, value in sorted(split_counts.items()))
    stats_lines.extend(f"state_status.{key}\t{value}" for key, value in sorted(status_counts.items()))
    stats_lines.extend(
        f"training_candidates.{key}\t{value}" for key, value in sorted(training_candidates.items())
    )
    (output_root / "stats.tsv").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    return manifest
