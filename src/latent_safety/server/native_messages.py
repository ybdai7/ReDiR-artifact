"""Dependency-neutral OpenAI message normalization for Qwen native prompts."""

from __future__ import annotations

import json
from typing import Any


def flatten_native_content(content: Any, *, message_index: int) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(
            f"messages[{message_index}].content must be a string, list, or null"
        )

    texts: list[str] = []
    for part_index, part in enumerate(content):
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            raise ValueError(
                f"messages[{message_index}].content[{part_index}] must be an object"
            )
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            value = part.get("text")
        elif part_type == "refusal":
            value = part.get("refusal")
        elif part_type == "thinking":
            value = part.get("thinking")
        else:
            raise ValueError(
                f"messages[{message_index}].content[{part_index}] has unsupported "
                f"part type {part_type!r}; native MT-AgentRisk prompts are text-only"
            )
        if value is None and part_type in {"text", "refusal"}:
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"messages[{message_index}].content[{part_index}] text must be a string"
            )
        texts.append(value)
    return "\n".join(texts)


def normalize_native_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Mirror the vLLM preprocessing required by the Qwen native template."""

    normalized: list[dict[str, Any]] = []
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, dict):
            raise ValueError(f"messages[{message_index}] must be an object")
        message = dict(raw_message)
        message["content"] = flatten_native_content(
            message.get("content"),
            message_index=message_index,
        )
        if message.get("role") == "assistant":
            reasoning = message.get("reasoning")
            message.pop("reasoning_content", None)
            if isinstance(reasoning, str):
                message["reasoning_content"] = reasoning

            raw_calls = message.get("tool_calls")
            if raw_calls is not None:
                if not isinstance(raw_calls, list):
                    raise ValueError(
                        f"messages[{message_index}].tool_calls must be a list"
                    )
                if not raw_calls:
                    message.pop("tool_calls", None)
                    normalized.append(message)
                    continue
                calls: list[dict[str, Any]] = []
                for call_index, raw_call in enumerate(raw_calls):
                    if not isinstance(raw_call, dict):
                        raise ValueError(
                            f"messages[{message_index}].tool_calls[{call_index}] "
                            "must be an object"
                        )
                    call = dict(raw_call)
                    raw_function = call.get("function")
                    if not isinstance(raw_function, dict):
                        raise ValueError(
                            f"messages[{message_index}].tool_calls[{call_index}].function "
                            "must be an object"
                        )
                    function = dict(raw_function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments or "{}")
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"messages[{message_index}].tool_calls[{call_index}] "
                                "has invalid JSON arguments"
                            ) from exc
                    elif arguments is None:
                        function["arguments"] = {}
                    call["function"] = function
                    calls.append(call)
                message["tool_calls"] = calls
        normalized.append(message)
    return normalized
