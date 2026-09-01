"""Parser for Mistral/Ministral native ``[TOOL_CALLS]`` completions."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from latent_safety.server.qwen_native_protocol import (
    QwenNativeParseAnomaly,
    QwenNativeParseResult,
    _vllm_random_tool_call_id,
)


_CALL = "[TOOL_CALLS]"
_ARGS = "[ARGS]"


def _declared_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            names.add(name)
    return names


def parse_mistral_native_response(
    raw_completion: str,
    tools: list[dict[str, Any]] | None = None,
    *,
    tool_call_id_factory: Callable[[], str] | None = None,
) -> QwenNativeParseResult:
    """Convert a complete Mistral native generation to an OpenAI message.

    Ministral's model-visible protocol is
    ``[TOOL_CALLS]function_name[ARGS]{json}``.  We deliberately return the
    established result type so the OpenAI serving and audit code can remain
    protocol-neutral while the Qwen parser itself stays untouched.
    """

    if not isinstance(raw_completion, str):
        raise TypeError("raw_completion must be a string")
    if tools is not None and not isinstance(tools, list):
        raise TypeError("tools must be a list or None")

    # Generation includes the EOS token when special control tokens are kept
    # for native parsing. It is framing, not assistant content.
    text = raw_completion
    while text.endswith("</s>"):
        text = text[: -len("</s>")]

    if _CALL not in text:
        return QwenNativeParseResult(
            reasoning_content=None,
            content=text or None,
            tool_calls=[],
            finish_reason="stop",
            parse_status="plain_text",
            parse_error=None,
            anomalies=(),
        )

    prefix, marker, remainder = text.partition(_CALL)
    anomalies: list[QwenNativeParseAnomaly] = []
    calls: list[dict[str, Any]] = []
    declared = _declared_tool_names(tools)
    id_factory = tool_call_id_factory or _vllm_random_tool_call_id
    parse_error: str | None = None

    chunks = remainder.split(_CALL)
    for index, chunk in enumerate(chunks):
        name, args_marker, raw_args = chunk.partition(_ARGS)
        name = name.strip()
        if not args_marker or not name:
            parse_error = "Malformed Mistral tool call: function name or [ARGS] is missing."
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="malformed_tool_call",
                    message=parse_error,
                    tool_index=index,
                    function_name=name or None,
                )
            )
            calls = []
            break
        if declared and name not in declared:
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="unknown_tool",
                    message=f"Tool {name!r} is not declared in the request schema.",
                    tool_index=index,
                    function_name=name,
                )
            )
            parse_error = f"Unknown tool {name!r}."
            calls = []
            break
        try:
            arguments = json.loads(raw_args.strip())
        except json.JSONDecodeError as exc:
            parse_error = f"JSONDecodeError: {exc}"
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="tool_parse_exception",
                    message=parse_error,
                    tool_index=index,
                    function_name=name,
                )
            )
            calls = []
            break
        if not isinstance(arguments, dict):
            parse_error = "Mistral tool arguments must decode to a JSON object."
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="non_object_arguments",
                    message=parse_error,
                    tool_index=index,
                    function_name=name,
                )
            )
            calls = []
            break
        calls.append(
            {
                "id": id_factory(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    if calls:
        return QwenNativeParseResult(
            reasoning_content=None,
            content=prefix or None,
            tool_calls=calls,
            finish_reason="tool_calls",
            parse_status="parsed",
            parse_error=None,
            anomalies=tuple(anomalies),
        )
    return QwenNativeParseResult(
        reasoning_content=None,
        content=text or None,
        tool_calls=[],
        finish_reason="stop",
        parse_status="parse_failure",
        parse_error=parse_error or "No structurally valid Mistral tool call was extracted.",
        anomalies=tuple(anomalies),
    )


__all__ = ["parse_mistral_native_response"]
