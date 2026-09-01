"""Canonical assistant targets for OpenHands action-channel training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class CanonicalTargetError(ValueError):
    """Raised when a trajectory event cannot be converted to a canonical target."""


def _json_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    if value is None:
        return {}
    raise CanonicalTargetError(f"tool arguments are not a mapping: {type(value).__name__}")


def _response_message(event: dict[str, Any]) -> dict[str, Any] | None:
    metadata = event.get("tool_call_metadata")
    if not isinstance(metadata, dict):
        return None
    response = metadata.get("model_response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = function.get("name")
        if not name:
            continue
        normalized.append(
            {
                "name": str(name),
                "arguments": _json_arguments(function.get("arguments")),
            }
        )
    return normalized


def _assistant_message_from_response(event: dict[str, Any]) -> dict[str, Any] | None:
    message = _response_message(event)
    if not message:
        return None
    tool_calls = _normalize_tool_calls(message.get("tool_calls"))
    if not tool_calls:
        return None
    return {
        "role": "assistant",
        "content": str(message.get("content") or ""),
        "tool_calls": tool_calls,
    }


def _filtered_editor_arguments(args: dict[str, Any]) -> dict[str, Any]:
    allowed = ("command", "path", "file_text", "old_str", "new_str", "insert_line", "view_range")
    return {key: args[key] for key in allowed if key in args and args[key] is not None}


def canonical_assistant_message_from_event(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a parsed OpenHands event back to Qwen/OpenAI assistant-message shape.

    The returned message is meant to be passed to ``tokenizer.apply_chat_template``.
    Message actions remain plain assistant content. Tool and finish actions become
    assistant ``tool_calls`` so the Qwen chat template emits native
    ``<tool_call><function=...>`` XML, not OpenHands display text.
    """

    action = event.get("action")
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    response_target = _assistant_message_from_response(event)
    if response_target is not None:
        return response_target, {
            "canonical_action_channel_label": "finish" if action == "finish" else "tool",
            "canonical_completion_source": "model_response_tool_calls",
            "raw_openhands_action": action,
        }

    if action == "message":
        content = str(args.get("content") or event.get("message") or "")
        if not content.strip():
            raise CanonicalTargetError("message action has empty content")
        return {
            "role": "assistant",
            "content": content,
        }, {
            "canonical_action_channel_label": "message",
            "canonical_completion_source": "event_message_content",
            "raw_openhands_action": action,
        }

    content = str(args.get("thought") or event.get("message") or "")
    if action == "call_tool_mcp":
        name = args.get("name")
        arguments = args.get("arguments")
        if not name or not isinstance(arguments, dict):
            raise CanonicalTargetError("call_tool_mcp event lacks name or mapping arguments")
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"name": str(name), "arguments": arguments}],
        }, {
            "canonical_action_channel_label": "tool",
            "canonical_completion_source": "event_args_mcp_tool",
            "raw_openhands_action": action,
        }

    if action == "finish":
        message = str(args.get("final_thought") or args.get("message") or event.get("message") or "")
        if not message.strip():
            raise CanonicalTargetError("finish event lacks final message")
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"name": "finish", "arguments": {"message": message}}],
        }, {
            "canonical_action_channel_label": "finish",
            "canonical_completion_source": "event_args_finish",
            "raw_openhands_action": action,
        }

    if action == "run":
        command = args.get("command")
        if not isinstance(command, str) or not command:
            raise CanonicalTargetError("run event lacks command")
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"name": "execute_bash", "arguments": {"command": command}}],
        }, {
            "canonical_action_channel_label": "tool",
            "canonical_completion_source": "event_args_run",
            "raw_openhands_action": action,
        }

    if action == "edit":
        arguments = _filtered_editor_arguments(args)
        if not arguments:
            raise CanonicalTargetError("edit event lacks str_replace_editor arguments")
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"name": "str_replace_editor", "arguments": arguments}],
        }, {
            "canonical_action_channel_label": "tool",
            "canonical_completion_source": "event_args_edit",
            "raw_openhands_action": action,
        }

    raise CanonicalTargetError(f"unsupported OpenHands action for canonical target: {action}")


def load_event_from_record(record: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    trajectory_path = record.get("trajectory_path")
    if not trajectory_path:
        raise CanonicalTargetError("record lacks trajectory_path")
    path = Path(str(trajectory_path))
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        raise CanonicalTargetError(f"trajectory_path does not exist: {path}")
    events = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise CanonicalTargetError("trajectory is not a list")
    event_index = int(record.get("event_index"))
    if event_index < 0 or event_index >= len(events):
        raise CanonicalTargetError(f"event_index out of range: {event_index}")
    event = events[event_index]
    if not isinstance(event, dict):
        raise CanonicalTargetError("selected event is not an object")
    return event


def canonical_assistant_message_from_record(
    record: dict[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    message = record.get("canonical_assistant_message")
    if isinstance(message, dict):
        return message, {
            "canonical_action_channel_label": record.get("canonical_action_channel_label", ""),
            "canonical_completion_source": record.get("canonical_completion_source", "record_canonical_assistant_message"),
            "raw_openhands_action": record.get("raw_openhands_action", ""),
        }
    event = load_event_from_record(record, repo_root=repo_root)
    return canonical_assistant_message_from_event(event)


def canonical_completion_ids(
    tokenizer: Any,
    *,
    state_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    device: torch.device,
    max_tokens: int,
) -> tuple[torch.Tensor, str, str]:
    """Return completion ids using the tokenizer's native chat template."""

    prompt_text_kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
    full_text_kwargs: dict[str, Any] = {"add_generation_prompt": False, "tokenize": False}
    if tools:
        prompt_text_kwargs["tools"] = tools
        full_text_kwargs["tools"] = tools

    prompt_text = tokenizer.apply_chat_template(state_messages, **prompt_text_kwargs)
    full_messages = list(state_messages) + [assistant_message]
    full_text = tokenizer.apply_chat_template(full_messages, **full_text_kwargs)
    if not isinstance(prompt_text, str) or not isinstance(full_text, str):
        raise CanonicalTargetError("chat template did not return text")
    if not full_text.startswith(prompt_text):
        raise CanonicalTargetError("canonical assistant target text does not align with generation prompt")
    completion_text = full_text[len(prompt_text) :]
    if not completion_text:
        raise CanonicalTargetError("canonical completion text is empty")
    encoded = tokenizer(completion_text, add_special_tokens=False, return_tensors="pt")
    completion = encoded["input_ids"]
    if max_tokens > 0:
        completion = completion[:, :max_tokens]
    completion = completion.to(device)
    decoded = tokenizer.decode(completion[0], skip_special_tokens=True)
    return completion, decoded, full_text
