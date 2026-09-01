"""Build protocol-audited native MT-AgentRisk states from LiteLLM logs."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from jsonschema import Draft7Validator

import yaml

from redir.datasets.native_messages import native_message_completion
from redir.server.native_messages import normalize_native_messages


_REFUSAL_RE = re.compile(
    r"\b(cannot|can't|will not|won't|unable to|refuse|decline|not able to)\b",
    re.IGNORECASE,
)


def _has_visible_refusal(text: str) -> bool:
    visible = text.rsplit("</think>", 1)[-1]
    return _REFUSAL_RE.search(visible) is not None


@dataclass(frozen=True, slots=True)
class NativeTaskSpec:
    task_key: str
    task_path: Path
    initial_instruction: str


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ValueError(f"unsupported message content type: {type(content).__name__}")
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("message content parts must be objects")
        value = part.get("text")
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _normalize_task_key(name: str) -> str:
    for prefix in ("multi-turn_", "multi-turn-", "safety-"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _task_instruction(task_path: Path) -> str:
    task_md = task_path / "task.md"
    if task_md.is_file():
        return task_md.read_text(encoding="utf-8").strip()
    turns_path = task_path / "turns.yml"
    if not turns_path.is_file():
        raise FileNotFoundError(f"task has neither task.md nor turns.yml: {task_path}")
    turns = yaml.safe_load(turns_path.read_text(encoding="utf-8")) or {}
    rows = turns.get("turns") or []
    if not rows or not isinstance(rows[0], dict):
        raise ValueError(f"task has no first turn: {turns_path}")
    instruction_path = task_path / str(rows[0].get("instruction_file") or "")
    if not instruction_path.is_file():
        raise FileNotFoundError(f"first turn instruction is missing: {instruction_path}")
    return instruction_path.read_text(encoding="utf-8").strip()


def load_task_specs(task_root: Path) -> list[NativeTaskSpec]:
    specs: list[NativeTaskSpec] = []
    for path in sorted(item for item in task_root.iterdir() if item.is_dir()):
        specs.append(
            NativeTaskSpec(
                task_key=_normalize_task_key(path.name),
                task_path=path.resolve(),
                initial_instruction=_task_instruction(path),
            )
        )
    if not specs:
        raise ValueError(f"task root has no task directories: {task_root}")
    keys = [spec.task_key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("normalized task keys are not unique")
    return specs


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return _flatten_content(message.get("content"))
    return ""


def identify_task(
    messages: list[dict[str, Any]],
    specs: Iterable[NativeTaskSpec],
) -> NativeTaskSpec:
    first_user = _first_user_text(messages).strip()
    matches = [
        spec
        for spec in specs
        # The multi-turn runner prepends an evaluation notice before turn 1.
        # Match the exact task instruction as a unique substring so that both
        # benign task.md prompts and safety turns.yml prompts map correctly.
        if spec.initial_instruction.strip() in first_user
    ]
    if len(matches) != 1:
        raise ValueError(
            f"native request maps to {len(matches)} tasks; first_user_sha256="
            f"{_json_sha256(first_user)}"
        )
    return matches[0]


def _choice_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = str((function or {}).get("name") or "").strip()
        if name:
            names.append(name)
    return names


def audit_native_messages(messages: list[dict[str, Any]]) -> dict[str, int]:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("native state must begin with a system message")
    known_call_ids: set[str] = set()
    assistant_calls = 0
    tool_messages = 0
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        content = _flatten_content(message.get("content"))
        if role == "assistant" and "<function=" in content:
            raise ValueError(f"messages[{index}] contains legacy function-call text")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError(f"messages[{index}].tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise ValueError(f"messages[{index}] has malformed structured tool call")
                call_id = str(call.get("id") or "").strip()
                name = str(call["function"].get("name") or "").strip()
                arguments = call["function"].get("arguments")
                if not call_id or not name:
                    raise ValueError(f"messages[{index}] tool call misses id or name")
                if isinstance(arguments, str):
                    json.loads(arguments or "{}")
                elif not isinstance(arguments, dict):
                    raise ValueError(f"messages[{index}] tool arguments are not JSON")
                known_call_ids.add(call_id)
                assistant_calls += 1
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if not tool_call_id or tool_call_id not in known_call_ids:
                raise ValueError(f"messages[{index}] has unmatched tool_call_id")
            tool_messages += 1
    return {
        "assistant_tool_calls": assistant_calls,
        "tool_messages": tool_messages,
    }


def _response_summary(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError("completion log misses response")
    message = _choice_message(response)
    if not message:
        raise ValueError("completion log response misses assistant message")
    completion = native_message_completion(message)
    calls = message.get("tool_calls") or []
    names = [
        str((call.get("function") or {}).get("name") or "")
        for call in calls
        if isinstance(call, dict)
    ]
    choices = response.get("choices") or []
    finish_reason = str((choices[0] if choices else {}).get("finish_reason") or "")
    if finish_reason == "tool_calls" and not names:
        raise ValueError("finish_reason=tool_calls without structured tool call")
    return {
        "completion": completion,
        "empty": not completion.strip(),
        "function_names": names,
        "first_function_name": names[0] if names else "",
        "finish_reason": finish_reason,
        "contains_refusal": _has_visible_refusal(completion),
    }


def _response_tool_argument_errors(
    payload: dict[str, Any],
    tools: list[dict[str, Any]],
) -> list[str]:
    message = _choice_message(payload.get("response") or {})
    calls = message.get("tool_calls") or []
    schemas = {
        str((tool.get("function") or {}).get("name") or ""): (
            (tool.get("function") or {}).get("parameters") or {}
        )
        for tool in tools
        if isinstance(tool, dict)
    }
    errors: list[str] = []
    for index, call in enumerate(calls):
        function = (call or {}).get("function") or {}
        name = str(function.get("name") or "")
        if name not in schemas:
            errors.append(f"tool_calls[{index}] function {name!r} is not requested")
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                errors.append(f"tool_calls[{index}] arguments are invalid JSON: {exc}")
                continue
        if not isinstance(arguments, dict):
            errors.append(f"tool_calls[{index}] arguments are not an object")
            continue
        validation_errors = sorted(
            Draft7Validator(schemas[name]).iter_errors(arguments),
            key=lambda error: list(error.absolute_path),
        )
        errors.extend(
            f"tool_calls[{index}] {name}: {error.message}"
            for error in validation_errors
        )
    return errors


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _flatten_content(message.get("content"))
    return ""


_FINAL_TURN_LINE = re.compile(r"(?m)^[ \t]*Final Turn[ \t]*:")


def _is_actual_final_turn_prompt(text: str) -> bool:
    """Distinguish the active final instruction from the runner preamble.

    Every multi-turn user prompt quotes the words ``"Final Turn"`` in a
    protocol notice.  Only the real final instruction contains an unquoted
    line beginning with ``Final Turn:``.
    """

    return _FINAL_TURN_LINE.search(text) is not None



def build_native_states(
    *,
    raw_completion_dir: Path,
    task_root: Path,
    domain: str,
    rollout_seed: int,
    min_tools: int = 1,
    expected_tools: int | None = None,
    required_system_prefix: str | None = None,
    required_system_sha256: str | None = None,
    expected_task_count: int | None = None,
    rollout_policy: str = "qwen35_9b_p3_base_native",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if domain not in {"safety", "benign", "identity"}:
        raise ValueError("domain must be safety, benign, or identity")
    specs = load_task_specs(task_root)
    paths = sorted(raw_completion_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"raw completion directory is empty: {raw_completion_dir}")
    loaded: list[tuple[float, Path, dict[str, Any], NativeTaskSpec]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload.get("messages")
        if not isinstance(messages, list) or not all(isinstance(row, dict) for row in messages):
            raise ValueError(f"completion log has invalid messages: {path}")
        system_text = _flatten_content(messages[0].get("content")) if messages else ""
        if required_system_prefix:
            if not system_text.startswith(required_system_prefix):
                raise ValueError(
                    f"completion log system prompt misses required prefix "
                    f"{required_system_prefix!r}: {path}"
                )
        if required_system_sha256:
            actual_system_sha256 = _text_sha256(system_text)
            if actual_system_sha256 != required_system_sha256:
                raise ValueError(
                    "completion log system prompt SHA-256 mismatch: "
                    f"{actual_system_sha256} != {required_system_sha256}: {path}"
                )
        spec = identify_task(messages, specs)
        loaded.append((float(payload.get("timestamp") or 0.0), path, payload, spec))
    loaded.sort(key=lambda item: (item[0], item[1].name))

    per_task_index: Counter[str] = Counter()
    safety_seen: set[str] = set()
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    raw_observed_refusals = 0
    raw_invalid_tool_arguments = 0
    invalid_tool_argument_examples: list[dict[str, Any]] = []
    task_log_counts: Counter[str] = Counter(spec.task_key for _, _, _, spec in loaded)
    for _, path, payload, spec in loaded:
        messages = deepcopy(payload["messages"])
        audit = audit_native_messages(messages)
        tools = deepcopy(((payload.get("kwargs") or {}).get("tools") or []))
        if len(tools) < min_tools:
            raise ValueError(f"request declares only {len(tools)} tools: {path}")
        if expected_tools is not None and len(tools) != expected_tools:
            raise ValueError(
                f"request declares {len(tools)} tools, expected {expected_tools}: {path}"
            )
        tool_names = _tool_names(tools)
        if len(tool_names) != len(tools) or len(tool_names) != len(set(tool_names)):
            raise ValueError(f"request tool names are missing or duplicated: {path}")
        response = _response_summary(payload)
        tool_argument_errors = _response_tool_argument_errors(payload, tools)
        if tool_argument_errors:
            raw_invalid_tool_arguments += 1
            if len(invalid_tool_argument_examples) < 20:
                invalid_tool_argument_examples.append(
                    {
                        "completion_log": str(path.resolve()),
                        "task_key": spec.task_key,
                        "errors": tool_argument_errors,
                    }
                )
            if domain == "identity":
                exclusions["identity_response_tool_arguments_invalid"] += 1
                continue
        raw_observed_refusals += int(bool(response["contains_refusal"]))
        latest_user = _latest_user_text(messages)
        is_actual_final_turn = _is_actual_final_turn_prompt(latest_user)

        if domain == "safety":
            if not (messages[-1].get("role") == "user" and is_actual_final_turn):
                exclusions["not_first_final_turn_prompt"] += 1
                continue
            if spec.task_key in safety_seen:
                exclusions["duplicate_final_turn_prompt"] += 1
                continue
            safety_seen.add(spec.task_key)
        elif domain in {"benign", "identity"} and response["empty"]:
            exclusions[f"{domain}_observed_empty"] += 1
            continue
        elif domain == "benign" and response["contains_refusal"]:
            # A successful trajectory can still contain a transient false
            # refusal.  Do not teach that response as benign retention data.
            exclusions["benign_observed_refusal"] += 1
            continue

        state_index = per_task_index[spec.task_key]
        per_task_index[spec.task_key] += 1
        normalized_messages = normalize_native_messages(messages)
        source_messages_sha256 = _json_sha256(normalized_messages)
        source_tool_schema_sha256 = _json_sha256(tools)
        function_name = str(response["first_function_name"])
        if function_name == "finish":
            state_kind = "finish_action"
        elif function_name:
            state_kind = "tool_action"
        else:
            state_kind = "assistant_message"
        record_state_kind = "final_state" if domain == "safety" else state_kind
        record = {
            "task_key": spec.task_key,
            "state_id": (
                f"{spec.task_key}:seed{rollout_seed}:native_real:{state_index}:"
                f"{record_state_kind}"
            ),
            "state_kind": record_state_kind,
            "reveal_status": (
                "final_decision"
                if domain == "safety" or is_actual_final_turn
                else "unknown"
            ),
            "is_final_state": domain == "safety",
            "student_state_messages": normalized_messages,
            "available_tools": tools,
            "available_tool_names": tool_names,
            "protocol_source": "native",
            "native_tool_calling": True,
            "native_real_state": True,
            "native_domain": domain,
            "native_state_candidate": True,
            "native_safety_candidate": domain == "safety",
            "native_benign_candidate": domain in {"benign", "identity"},
            "native_identity_candidate": domain == "identity",
            "rollout_seed": rollout_seed,
            "rollout_policy": str(rollout_policy),
            "source_task_path": str(spec.task_path),
            "source_completion_log": str(path.resolve()),
            "source_completion_log_sha256": _file_sha256(path),
            "source_messages_sha256": source_messages_sha256,
            "source_tool_schema_sha256": source_tool_schema_sha256,
            "source_system_prompt_sha256": _text_sha256(
                _flatten_content(messages[0].get("content"))
            ),
            "source_prompt_sha256": _json_sha256(
                {"messages": normalized_messages, "tools": tools}
            ),
            "observed_completion": response["completion"],
            "observed_finish_reason": response["finish_reason"],
            "observed_function_names": response["function_names"],
            "observed_first_function_name": function_name,
            "observed_contains_refusal": response["contains_refusal"],
            "native_history_assistant_tool_calls": audit["assistant_tool_calls"],
            "native_history_tool_messages": audit["tool_messages"],
            "latest_user_contains_final_turn": is_actual_final_turn,
            "latest_user_is_actual_final_turn": is_actual_final_turn,
        }
        records.append(record)

    tasks = {str(row["task_key"]) for row in records}
    duplicate_prompts = len(records) - len({str(row["source_prompt_sha256"]) for row in records})
    stats = {
        "strategy": "native_real_state_extraction",
        "domain": domain,
        "protocol_source": "native",
        "native_tool_calling": True,
        "required_system_prefix": required_system_prefix,
        "required_system_sha256": required_system_sha256,
        "expected_tools": expected_tools,
        "expected_task_count": expected_task_count,
        "rollout_seed": rollout_seed,
        "rollout_policy": str(rollout_policy),
        "raw_log_count": len(loaded),
        "source_raw_log_count": len(paths),
        "record_count": len(records),
        "task_count": len(tasks),
        "tasks": sorted(tasks),
        "task_log_counts": dict(task_log_counts),
        "state_kind_counts": dict(Counter(str(row["state_kind"]) for row in records)),
        "observed_function_counts": dict(
            Counter(str(row["observed_first_function_name"] or "plain") for row in records)
        ),
        "request_tool_count_counts": dict(
            Counter(str(len(row["available_tools"])) for row in records)
        ),
        "system_prompt_sha256_counts": dict(
            Counter(str(row["source_system_prompt_sha256"]) for row in records)
        ),
        "observed_refusal_count": sum(bool(row["observed_contains_refusal"]) for row in records),
        "raw_observed_refusal_count": raw_observed_refusals,
        "raw_invalid_tool_argument_count": raw_invalid_tool_arguments,
        "invalid_tool_argument_examples": invalid_tool_argument_examples,
        "native_history_tool_call_states": sum(
            int(row["native_history_assistant_tool_calls"]) > 0 for row in records
        ),
        "duplicate_prompt_count": duplicate_prompts,
        "exclusions": dict(exclusions),
    }
    if not records:
        raise ValueError("native extraction produced zero records")
    if expected_task_count is not None and len(tasks) != expected_task_count:
        raise ValueError(
            f"native extraction produced {len(tasks)} tasks, expected {expected_task_count}"
        )
    if duplicate_prompts:
        raise ValueError(f"native extraction produced {duplicate_prompts} duplicate prompts")
    return records, stats


__all__ = [
    "NativeTaskSpec",
    "audit_native_messages",
    "build_native_states",
    "identify_task",
    "load_task_specs",
]
