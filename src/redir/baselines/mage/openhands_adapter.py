"""Normalize OpenHands native actions for MAGE action review.

MAGE protects a target agent at the action boundary.  OpenHands persists MCP
calls as well as built-in execution actions (``read``, ``edit``, ``run``, ...).
This adapter gives both the offline dataset builder and the future live
OpenHands hook one canonical representation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# These actions can touch the workspace, execute code, or access an external
# service and therefore pass through the guard.
GUARDED_ACTIONS = frozenset(
    {
        "call_tool_mcp",
        "read",
        "edit",
        "run",
        "run_ipython",
        "browse_interactive",
    }
)

# These are control-plane actions.  In particular, a native ``finish`` refusal
# must remain possible even when the guard denies all external execution.
BYPASS_ACTIONS = frozenset({"finish", "think", "message", "recall", "system"})

_DEFAULT_TOOL_NAMES = {
    "call_tool_mcp": "mcp.unknown",
    "read": "openhands.read",
    "edit": "openhands.edit",
    "run": "openhands.run",
    "run_ipython": "openhands.run_ipython",
    "browse_interactive": "openhands.browse_interactive",
}

_INTERNAL_ARGUMENT_KEYS = {
    "thought",
    "impl_source",
    "confirmation_state",
    "is_static",
    "blocking",
    "hidden",
    "wait_for_response",
    "file_urls",
    "image_urls",
}


def is_guarded_action(event: dict[str, Any]) -> bool:
    """Return whether an OpenHands event represents an executable action."""

    return event.get("source") == "agent" and event.get("action") in GUARDED_ACTIONS


def _clean_arguments(value: Any) -> Any:
    if not isinstance(value, dict):
        return deepcopy(value)
    return {
        str(key): deepcopy(item)
        for key, item in value.items()
        if key not in _INTERNAL_ARGUMENT_KEYS and item is not None
    }


def _metadata_function_name(event: dict[str, Any]) -> str | None:
    metadata = event.get("tool_call_metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("function_name")
    return str(value).strip() if value else None


def normalize_pending_tool_calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one persisted OpenHands action into MAGE ``tool_calls``.

    The persisted event describes the action OpenHands actually attempted.  We
    deliberately do not expand every tool call found inside the raw model
    response: some providers emit several candidate calls in one response,
    while OpenHands executes and records them as individual events.
    """

    if not is_guarded_action(event):
        return []

    action = str(event.get("action"))
    raw_args = event.get("args")
    args = raw_args if isinstance(raw_args, dict) else {}

    if action == "call_tool_mcp":
        name = str(args.get("name") or _metadata_function_name(event) or _DEFAULT_TOOL_NAMES[action])
        arguments = args.get("arguments", {})
    else:
        name = _metadata_function_name(event) or _DEFAULT_TOOL_NAMES[action]
        arguments = args

    return [
        {
            "name": name,
            "arguments": _clean_arguments(arguments),
            "openhands_action": action,
        }
    ]


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(1, limit // 2)
    return text[:keep] + f"\n...[truncated {len(text) - limit} chars]...\n" + text[-keep:]


def _context_message(
    event: dict[str, Any],
    *,
    max_message_chars: int,
    include_recall: bool,
    include_openhands_system: bool,
) -> dict[str, str] | None:
    source = event.get("source")
    action = event.get("action")
    action = getattr(action, "value", action)
    message = str(event.get("message") or event.get("content") or "")

    if action == "recall" and not include_recall:
        return None
    if source == "environment":
        return None
    if source == "agent" and action == "system" and not include_openhands_system:
        return None

    if source == "user":
        role = "user"
    elif source == "agent" and action is None:
        role = "tool"
    elif source == "agent" and action == "system":
        role = "system"
    elif source == "agent":
        role = "assistant"
    else:
        role = "user"

    prefix = f"[OpenHands action={action}]\n" if action else ""
    return {
        "role": role,
        "content": _truncate_text(prefix + message, max_message_chars),
    }


def normalize_context_events(
    events: list[dict[str, Any]],
    *,
    max_context_chars: int = 24_000,
    max_message_chars: int = 6_000,
    include_recall: bool = False,
    include_openhands_system: bool = False,
) -> list[dict[str, str]]:
    """Build the live MAGE context with the frozen dataset-builder contract."""

    context = [
        message
        for event in events
        if (
            message := _context_message(
                event,
                max_message_chars=max_message_chars,
                include_recall=include_recall,
                include_openhands_system=include_openhands_system,
            )
        )
        is not None
    ]
    if max_context_chars <= 0:
        return context

    kept_reversed: list[dict[str, str]] = []
    total = 0
    for message in reversed(context):
        size = len(message.get("content", ""))
        if kept_reversed and total + size > max_context_chars:
            break
        kept_reversed.append(message)
        total += size
    kept = list(reversed(kept_reversed))

    first_user = next((message for message in context if message.get("role") == "user"), None)
    if first_user is not None and first_user not in kept:
        kept.insert(0, first_user)
    return kept
