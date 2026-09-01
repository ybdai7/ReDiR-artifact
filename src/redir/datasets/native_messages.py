"""Reconstruct native Qwen completions from OpenAI-compatible responses."""

from __future__ import annotations

import json
from typing import Any


def _message_reasoning(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    provider = message.get("provider_specific_fields")
    if isinstance(provider, dict):
        for key in ("reasoning_content", "reasoning"):
            value = provider.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _has_reasoning_channel(message: dict[str, Any]) -> bool:
    if any(isinstance(message.get(key), str) for key in ("reasoning_content", "reasoning")):
        return True
    provider = message.get("provider_specific_fields")
    return isinstance(provider, dict) and any(
        isinstance(provider.get(key), str)
        for key in ("reasoning_content", "reasoning")
    )


def _parameter_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _structured_tool_calls(message: dict[str, Any]) -> str:
    blocks: list[str] = []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return ""
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"arguments": arguments}
        if not isinstance(arguments, dict):
            arguments = {"arguments": arguments}
        lines = ["<tool_call>", f"<function={name}>"]
        for parameter_name, value in arguments.items():
            lines.extend(
                [
                    f"<parameter={parameter_name}>",
                    _parameter_value(value),
                    "</parameter>",
                ]
            )
        lines.extend(["</function>", "</tool_call>"])
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def native_message_completion(message: dict[str, Any]) -> str:
    """Join reasoning, visible content, and structured calls in model space."""

    reasoning = _message_reasoning(message)
    content = message.get("content")
    content = content if isinstance(content, str) else ""
    tool_xml = _structured_tool_calls(message)
    pieces: list[str] = []
    if _has_reasoning_channel(message):
        pieces.append((reasoning.rstrip() + "\n" if reasoning else "") + "</think>")
    if content.strip():
        pieces.append(content.strip())
    if tool_xml:
        pieces.append(tool_xml)
    return "\n\n".join(pieces)


__all__ = ["native_message_completion"]
