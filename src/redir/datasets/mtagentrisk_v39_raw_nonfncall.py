"""Build V3.9 raw OpenHands non-fncall warmup datasets.

V3.9 trains against the text that OpenHands actually received in
``native_tool_calling=false`` mode.  Parsed trajectory events are used only for
alignment and filtering; they are not used to reconstruct the assistant target.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

from redir.datasets.mtagentrisk_final_state_rounds import (
    read_dependencies,
    tool_names_for_dependencies,
)
from redir.datasets.mtagentrisk_opd_pairs import (
    event_message,
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
from redir.datasets.mtagentrisk_v3_broad_states import (
    MALFORMED_COMPLETION_MARKERS,
    capped_state_rows,
    index_trajectories,
    state_kind_for,
    task_path_for,
    user_turn_count_before,
)
from redir.engine.masks import parse_event_lifecycle_spans


DEFAULT_SOURCE_DATASET_ROOT = (
    "outputs/latent_training/mtagentrisk_qwen35_filesystem_p3_masked_opd"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/latent_training/"
    "mtagentrisk_qwen35_filesystem_p3_v39_raw_nonfncall_warmup/round_0/dataset"
)

RAW_NONFNCALL_EXTRA_TOOL_NAMES = (
    "execute_bash",
    "str_replace_editor",
    "browser",
)

FUNCTION_CALL_RE = re.compile(r"<function\s*=\s*([A-Za-z0-9_.:-]+)\b", re.IGNORECASE)
FINISH_FUNCTION_RE = re.compile(r"<function\s*=\s*finish\b", re.IGNORECASE)
FINAL_TURN_RE = re.compile(r"(^|\n)\s*Final Turn\s*:", re.IGNORECASE)
EXIT_CODE_RE = re.compile(r"exit code\s+(-?\d+)", re.IGNORECASE)
EXPLICIT_FINISH_REQUEST_RE = re.compile(
    r"finish the action|wrap up your work|if you are done|please finish(?: the action)?|let'?s stop.*finish",
    re.IGNORECASE | re.DOTALL,
)


def latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def latest_user_kind(messages: list[dict[str, Any]]) -> str:
    latest = latest_user_message(messages).strip()
    if not latest:
        return "missing"
    if EXPLICIT_FINISH_REQUEST_RE.search(latest):
        return "explicit_finish_request"
    if latest.startswith("EXECUTION RESULT"):
        return "execution_result"
    if FINAL_TURN_RE.search(latest):
        return "final_turn_instruction"
    return "other"


def sample_weight_for_v311(action_kind: str, reveal_status: str, latest_kind: str) -> float:
    """V3.11 context-aware mild lifecycle weights.

    V3.10 tried a global finish weight of 2.0 and fixed finalization pressure
    too aggressively, causing early finish on smoke tasks. V3.11 only nudges
    explicit finish-request states and mildly reinforces pre-final messages.
    """
    if action_kind == "finish":
        return 1.5 if latest_kind == "explicit_finish_request" else 1.0
    if action_kind == "ordinary_message":
        if reveal_status == "pre_final":
            return 1.25
        if reveal_status in {"final_decision", "post_final_decision"}:
            return 0.75
    return 1.0


def split_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_response_id(log: dict[str, Any]) -> str:
    for key in ("fncall_response", "response"):
        response = log.get(key)
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
    return ""


def raw_response_content(log: dict[str, Any]) -> str:
    response = log.get("response")
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def normalize_completion_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def response_prompt_messages(log: dict[str, Any]) -> list[dict[str, str]]:
    messages = log.get("messages")
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str):
            normalized.append({"role": role, "content": content})
    return normalized


def raw_log_timestamp_seconds(log: dict[str, Any]) -> float | None:
    value = log.get("timestamp")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def event_timestamp_seconds(event: dict[str, Any]) -> float | None:
    value = event.get("timestamp")
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def index_raw_completion_logs(raw_completions_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not raw_completions_root.exists():
        return index
    for path in sorted(raw_completions_root.glob("*.json")):
        try:
            log = load_json(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(log, dict):
            continue
        response_id = raw_response_id(log)
        if not response_id:
            continue
        log["_raw_completion_log_path"] = str(path)
        index[response_id] = log
    return index


def index_raw_completion_logs_by_content(
    raw_logs: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for log in raw_logs:
        content_key = normalize_completion_for_match(raw_response_content(log))
        if content_key:
            index[content_key].append(log)
    for matches in index.values():
        matches.sort(key=lambda log: raw_log_timestamp_seconds(log) or 0.0)
    return index


def event_response_id(event: dict[str, Any]) -> str:
    metadata = event.get("tool_call_metadata")
    if not isinstance(metadata, dict):
        return ""
    response = metadata.get("model_response")
    if not isinstance(response, dict):
        return ""
    response_id = response.get("id")
    return response_id if isinstance(response_id, str) else ""


def find_raw_log_for_event_by_content(
    event: dict[str, Any],
    raw_content_index: dict[str, list[dict[str, Any]]],
    used_response_ids: set[str],
) -> dict[str, Any] | None:
    """Recover raw completion logs for events without response metadata.

    OpenHands ``MessageAction`` events often keep the exact assistant text but do
    not retain ``tool_call_metadata.model_response.id``.  V3.9 needs the raw
    prompt messages too, so fall back to matching the event text against logged
    raw completions.  If several logs share the same text, prefer an unused log
    closest in time to the trajectory event.
    """
    event_key = normalize_completion_for_match(event_text(event))
    if not event_key:
        return None
    matches = raw_content_index.get(event_key) or []
    if not matches:
        return None
    unused = [
        log
        for log in matches
        if raw_response_id(log) and raw_response_id(log) not in used_response_ids
    ]
    candidates = unused or matches
    event_ts = event_timestamp_seconds(event)
    if event_ts is None:
        return candidates[0]

    def score(log: dict[str, Any]) -> float:
        log_ts = raw_log_timestamp_seconds(log)
        return abs(log_ts - event_ts) if log_ts is not None else float("inf")

    return min(candidates, key=score)


def event_text(event: dict[str, Any]) -> str:
    message = event_message(event)
    if message:
        return message
    args = event.get("args")
    if isinstance(args, dict):
        content = args.get("content")
        if isinstance(content, str):
            return content
    content = event.get("content")
    return content if isinstance(content, str) else ""


def final_user_index_strict(traj: list[dict[str, Any]]) -> int | None:
    for idx, event in reversed(list(enumerate(traj))):
        if event.get("source") != "user":
            continue
        text = event_text(event)
        if FINAL_TURN_RE.search(text):
            return idx
    return None


def reveal_status_for_event(
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


def is_agent_action_event(event: dict[str, Any]) -> bool:
    if event.get("source") != "agent" or event.get("id") == 0:
        return False
    action = event.get("action")
    if action is None:
        return False
    text = event_text(event)
    if is_empty_assistant_wait(text) or looks_like_observation(text):
        return False
    return True


def raw_function_name(raw_completion: str) -> str:
    match = FUNCTION_CALL_RE.search(raw_completion or "")
    return match.group(1).lower() if match else ""


def action_kind_from_event(event: dict[str, Any], raw_completion: str) -> str:
    function_name = raw_function_name(raw_completion)
    if function_name:
        if function_name == "finish":
            return "finish"
        if function_name == "execute_bash":
            return "shell_action"
        return "tool_action"

    action = str(event.get("action") or "")
    if action == "finish" or FINISH_FUNCTION_RE.search(raw_completion):
        return "finish"
    if action == "message":
        return "ordinary_message"
    if action == "run":
        return "shell_action"
    if action in {
        "browse_interactive",
        "call_tool_mcp",
        "edit",
        "read",
        "run_ipython",
    }:
        return "tool_action"
    return "ordinary_message"


def is_malformed_completion(text: str) -> bool:
    return any(marker in text for marker in MALFORMED_COMPLETION_MARKERS)


def observation_error_after(traj: list[dict[str, Any]], event_index: int) -> tuple[bool, str]:
    for event in traj[event_index + 1 : event_index + 4]:
        if event.get("source") != "agent":
            continue
        text = event_text(event)
        if not text:
            continue
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                if payload.get("isError") is True:
                    return True, "mcp_is_error"
                if payload.get("isError") is False:
                    return False, ""
        if "Error calling tool" in text or "Traceback" in text:
            return True, "tool_error_text"
        match = EXIT_CODE_RE.search(text)
        if match:
            return match.group(1) != "0", f"exit_code_{match.group(1)}"
        # Stop at the first observation-like event for this action.
        if event.get("action") is None or "EXECUTION RESULT" in text:
            return False, ""
    return False, ""


def tool_names_for_source(source: SourceRecord, repo_root: Path) -> list[str]:
    task_path = task_path_for(source, repo_root)
    dependencies = read_dependencies(task_path) if task_path.exists() else []
    names = tool_names_for_dependencies(dependencies)
    for name in RAW_NONFNCALL_EXTRA_TOOL_NAMES:
        if name not in names:
            names.append(name)
    return names


def build_rows_for_record(
    source: SourceRecord,
    *,
    traj_path: Path,
    raw_log_index: dict[str, dict[str, Any]],
    raw_content_index: dict[str, list[dict[str, Any]]],
    repo_root: Path,
    round_id: int,
    rollout_policy: str,
    rollout_checkpoint: str | None,
    live_outcome: str | None,
    max_tool_states_per_task: int,
    max_safety_states_per_task: int,
    max_final_states_per_task: int,
    max_ordinary_states_per_task: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    traj = load_trajectory(traj_path)
    final_idx = final_user_index_strict(traj)
    multi_outcome = (live_outcome or source.multi_outcome).upper()
    tool_names = tool_names_for_source(source, repo_root)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    final_reference_used = False
    used_response_ids: set[str] = set()

    for idx, event in enumerate(traj):
        if not is_agent_action_event(event):
            continue
        response_id = event_response_id(event)
        log = raw_log_index.get(response_id)
        alignment_source = "model_response_id" if log is not None else ""
        if log is None:
            log = find_raw_log_for_event_by_content(event, raw_content_index, used_response_ids)
            if log is not None:
                recovered_response_id = raw_response_id(log)
                response_id = recovered_response_id or response_id
                alignment_source = "raw_content_match"
        if log is None:
            skipped.append(
                {
                    "task_key": source.task_key,
                    "event_index": idx,
                    "event_id": event.get("id"),
                    "response_id": response_id,
                    "raw_openhands_action": event.get("action"),
                    "reason": "missing_raw_completion_log",
                }
            )
            continue

        if response_id:
            used_response_ids.add(response_id)
        raw_completion = raw_response_content(log)
        student_state_messages = response_prompt_messages(log)
        reveal_status = reveal_status_for_event(
            final_idx=final_idx,
            event_index=idx,
            final_reference_used=final_reference_used,
        )
        is_final_state = reveal_status == "final_decision"
        if is_final_state:
            final_reference_used = True
        action_kind = action_kind_from_event(event, raw_completion)
        latest_kind = latest_user_kind(student_state_messages)
        state_kind = state_kind_for(
            action_kind=action_kind,
            is_final_state=is_final_state,
            single_outcome=source.single_outcome,
            multi_outcome=multi_outcome,
        )
        sample_weight = sample_weight_for_v311(action_kind, reveal_status, latest_kind)
        if action_kind == "finish" and latest_kind == "explicit_finish_request":
            sample_weight_reason = "v311_explicit_finish_request_finish"
        elif action_kind == "ordinary_message" and reveal_status == "pre_final":
            sample_weight_reason = "v311_pre_final_message_retention"
        elif action_kind == "ordinary_message" and reveal_status in {"final_decision", "post_final_decision"}:
            sample_weight_reason = "v311_final_post_final_message_downweight"
        else:
            sample_weight_reason = "v311_default"
        parser_result = parse_event_lifecycle_spans(raw_completion, tool_names)
        malformed = is_malformed_completion(raw_completion)
        action_error, action_error_reason = observation_error_after(traj, idx)
        pre_final_finish = (
            action_kind == "finish"
            or bool(FINISH_FUNCTION_RE.search(raw_completion))
        ) and reveal_status not in {"final_decision", "post_final_decision"}
        candidate = (
            source.single_outcome in {"COMPLETE", "REJECT"}
            and bool(student_state_messages)
            and bool(raw_completion.strip())
            and parser_result.parsed
            and not malformed
            and not action_error
            and not pre_final_finish
        )
        skip_reasons: list[str] = []
        if not student_state_messages:
            skip_reasons.append("missing_prompt_messages")
        if not raw_completion.strip():
            skip_reasons.append("empty_raw_completion")
        if not parser_result.parsed:
            skip_reasons.append("event_lifecycle_unparsed")
        if malformed:
            skip_reasons.append("malformed_completion")
        if action_error:
            skip_reasons.append(action_error_reason or "action_error")
        if pre_final_finish:
            skip_reasons.append("pre_final_finish")
        if source.single_outcome not in {"COMPLETE", "REJECT"}:
            skip_reasons.append("single_outcome_not_trainable")

        row = {
            "task_key": source.task_key,
            "state_id": f"{source.task_key}:{idx}:v39_raw_nonfncall:{state_kind}",
            "split": "train",
            "source_split": source.source_split,
            "round_id": round_id,
            "source_rollout_policy": rollout_policy,
            "rollout_policy": rollout_policy,
            "rollout_checkpoint": rollout_checkpoint,
            "single_task_id": source.row.get("single_task_id", ""),
            "multi_task_id": source.row.get("multi_task_id", ""),
            "single_outcome": source.single_outcome,
            "multi_outcome": multi_outcome,
            "no_latent_multiturn_outcome": multi_outcome,
            "canonical_single_messages": source.row["canonical_single_messages"],
            "student_state_messages": student_state_messages,
            "raw_multiturn_messages": student_state_messages,
            "raw_nonfncall_completion": raw_completion,
            "action_completion": raw_completion,
            "student_action_prefix_source": raw_completion,
            "baseline_reference_message": raw_completion,
            "raw_completion_log_path": repo_relative_or_abs(Path(log["_raw_completion_log_path"]), repo_root),
            "raw_completion_response_id": response_id,
            "raw_completion_alignment_source": alignment_source,
            # Keep prompt tools empty because the raw OpenHands request did not
            # pass OpenAI tools. Tool names below are for parsing/debug only.
            "available_tools": [],
            "available_tool_names": tool_names,
            "state_source": "openhands_raw_nonfncall_completion",
            "state_kind": state_kind,
            "action_kind": action_kind,
            "reveal_status": reveal_status,
            "latest_user_kind": latest_kind,
            "is_final_state": is_final_state,
            "turn_index": user_turn_count_before(traj, idx - 1),
            "turn_count": user_turn_count_before(traj, len(traj) - 1),
            "event_index": idx,
            "event_id": event.get("id"),
            "raw_openhands_action": event.get("action"),
            "trajectory_path": repo_relative_or_abs(traj_path, repo_root),
            "post_eval_outcome": multi_outcome,
            "parser_result": parser_result.as_dict(),
            "event_lifecycle_parser_result": parser_result.as_dict(),
            "parsed_event_lifecycle": parser_result.parsed,
            "malformed_completion": malformed,
            "action_error": action_error,
            "action_error_reason": action_error_reason,
            "pre_final_finish": pre_final_finish,
            "sample_weight": sample_weight,
            "state_weight": sample_weight,
            "sample_weight_profile": "v311_context_aware_mild",
            "sample_weight_reason": sample_weight_reason,
            "raw_nonfncall_warmup_candidate": candidate,
            "is_v39_training_candidate": candidate,
            "is_gap_candidate": (
                source.single_outcome == "REJECT"
                and multi_outcome == "COMPLETE"
                and state_kind in {"final_state", "unsafe_or_gap_tool_action"}
            ),
            "skip_reasons": skip_reasons,
            "single_traj_path": source.row.get("single_traj_path", ""),
            "multi_traj_path": source.row.get("multi_traj_path", ""),
            "single_task_path": source.row.get("single_task_path", ""),
            "multi_task_path": source.row.get("multi_task_path", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        if not candidate:
            skipped.append(
                {
                    "task_key": source.task_key,
                    "event_index": idx,
                    "event_id": event.get("id"),
                    "response_id": response_id,
                    "raw_completion_alignment_source": alignment_source,
                    "action_kind": action_kind,
                    "state_kind": state_kind,
                    "reveal_status": reveal_status,
                    "skip_reasons": skip_reasons,
                    "raw_completion_preview": raw_completion[:300],
                }
            )

    capped = capped_state_rows(
        rows,
        max_tool_states_per_task=max_tool_states_per_task,
        max_safety_states_per_task=max_safety_states_per_task,
        max_final_states_per_task=max_final_states_per_task,
        max_ordinary_states_per_task=max_ordinary_states_per_task,
    )
    return capped, skipped


def build_raw_nonfncall_dataset(
    *,
    repo_root: Path,
    source_dataset_root: Path,
    trajectory_root: Path,
    raw_completions_root: Path,
    output_root: Path,
    source_splits: tuple[str, ...] = ("train", "dev"),
    split_name: str = "train",
    round_id: int = 0,
    rollout_policy: str = "qwen35_9b_p3_no_latent_raw_nonfncall",
    rollout_checkpoint: str | None = None,
    live_post_eval_path: Path | None = None,
    max_tool_states_per_task: int = 3,
    max_safety_states_per_task: int = 2,
    max_final_states_per_task: int = 1,
    max_ordinary_states_per_task: int = 2,
) -> dict[str, Any]:
    source_records = load_source_records(source_dataset_root, source_splits)
    trajectory_index = index_trajectories(trajectory_root)
    raw_log_index = index_raw_completion_logs(raw_completions_root)
    raw_content_index = index_raw_completion_logs_by_content(raw_log_index.values())
    live_outcomes = load_post_eval_outcomes(live_post_eval_path)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
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
        task_rows, task_skipped = build_rows_for_record(
            source,
            traj_path=traj_path,
            raw_log_index=raw_log_index,
            raw_content_index=raw_content_index,
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
        rows.extend(task_rows)
        skipped.extend(task_skipped)

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
    write_jsonl(output_root / "debug_skipped_raw_nonfncall.jsonl", skipped)
    write_jsonl(output_root / "debug_missing_trajectories.jsonl", missing)
    manifest = raw_nonfncall_manifest(
        rows,
        skipped=skipped,
        missing=missing,
        source_dataset_root=source_dataset_root,
        trajectory_root=trajectory_root,
        raw_completions_root=raw_completions_root,
        output_root=output_root,
        source_splits=source_splits,
        split_name=split_name,
        round_id=round_id,
        rollout_policy=rollout_policy,
        rollout_checkpoint=rollout_checkpoint,
        raw_completion_log_count=len(raw_log_index),
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_stats(output_root / "stats.tsv", manifest)
    return manifest


def raw_nonfncall_manifest(
    rows: list[dict[str, Any]],
    *,
    skipped: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    source_dataset_root: Path,
    trajectory_root: Path,
    raw_completions_root: Path,
    output_root: Path,
    source_splits: tuple[str, ...],
    split_name: str,
    round_id: int,
    rollout_policy: str,
    rollout_checkpoint: str | None,
    raw_completion_log_count: int,
) -> dict[str, Any]:
    task_keys = {row["task_key"] for row in rows}
    single_counts = Counter(row["single_outcome"] for row in rows)
    multi_counts = Counter(row["multi_outcome"] for row in rows)
    state_counts = Counter(row["state_kind"] for row in rows)
    action_counts = Counter(row["action_kind"] for row in rows)
    reveal_counts = Counter(row.get("reveal_status", "unknown") for row in rows)
    latest_user_counts = Counter(row.get("latest_user_kind", "unknown") for row in rows)
    candidate_weight_counts = Counter(
        str(row.get("sample_weight", 1.0))
        for row in rows
        if row.get("raw_nonfncall_warmup_candidate")
    )
    candidate_weight_reason_counts = Counter(
        row.get("sample_weight_reason", "unknown")
        for row in rows
        if row.get("raw_nonfncall_warmup_candidate")
    )
    candidate_action_weight_counts = Counter(
        f"{row.get('action_kind', 'unknown')}@{row.get('sample_weight', 1.0)}"
        for row in rows
        if row.get("raw_nonfncall_warmup_candidate")
    )
    skip_counts: Counter[str] = Counter()
    for row in rows:
        for reason in row.get("skip_reasons") or []:
            skip_counts[str(reason)] += 1
    external_skip_counts: Counter[str] = Counter()
    for row in skipped:
        for reason in row.get("skip_reasons") or [row.get("reason", "unknown")]:
            external_skip_counts[str(reason)] += 1
    return {
        "dataset": "mtagentrisk_qwen35_filesystem_p3_v39_raw_nonfncall_warmup",
        "source_dataset_root": str(source_dataset_root),
        "trajectory_root": str(trajectory_root),
        "raw_completions_root": str(raw_completions_root),
        "output_root": str(output_root),
        "source_splits": list(source_splits),
        "split_name": split_name,
        "round_id": round_id,
        "rollout_policy": rollout_policy,
        "rollout_checkpoint": rollout_checkpoint,
        "raw_completion_log_count": raw_completion_log_count,
        "task_count": len(task_keys),
        "state_count": len(rows),
        "training_candidates": sum(bool(row.get("raw_nonfncall_warmup_candidate")) for row in rows),
        "candidate_rate": sum(bool(row.get("raw_nonfncall_warmup_candidate")) for row in rows)
        / max(len(rows), 1),
        "missing_trajectory_count": len(missing),
        "skipped_event_count": len(skipped),
        "parsed_event_lifecycle_count": sum(bool(row.get("parsed_event_lifecycle")) for row in rows),
        "pre_final_finish_count": sum(bool(row.get("pre_final_finish")) for row in rows),
        "action_error_count": sum(bool(row.get("action_error")) for row in rows),
        "malformed_completion_count": sum(bool(row.get("malformed_completion")) for row in rows),
        "single_outcome_counts": dict(single_counts),
        "multi_outcome_counts": dict(multi_counts),
        "state_kind_counts": dict(state_counts),
        "action_kind_counts": dict(action_counts),
        "reveal_status_counts": dict(reveal_counts),
        "latest_user_kind_counts": dict(latest_user_counts),
        "candidate_weight_counts": dict(candidate_weight_counts),
        "candidate_weight_reason_counts": dict(candidate_weight_reason_counts),
        "candidate_action_weight_counts": dict(candidate_action_weight_counts),
        "skip_reason_counts": dict(skip_counts),
        "external_skip_reason_counts": dict(external_skip_counts),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_stats(path: Path, manifest: dict[str, Any]) -> None:
    rows = ["metric\tvalue"]
    for key in (
        "raw_completion_log_count",
        "task_count",
        "state_count",
        "training_candidates",
        "candidate_rate",
        "missing_trajectory_count",
        "skipped_event_count",
        "parsed_event_lifecycle_count",
        "pre_final_finish_count",
        "action_error_count",
        "malformed_completion_count",
    ):
        rows.append(f"{key}\t{manifest.get(key, 0)}")
    for group in (
        "single_outcome_counts",
        "multi_outcome_counts",
        "state_kind_counts",
        "action_kind_counts",
        "reveal_status_counts",
        "latest_user_kind_counts",
        "candidate_weight_counts",
        "candidate_weight_reason_counts",
        "candidate_action_weight_counts",
        "skip_reason_counts",
        "external_skip_reason_counts",
    ):
        for key, value in sorted(manifest.get(group, {}).items()):
            rows.append(f"{group}.{key}\t{value}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
