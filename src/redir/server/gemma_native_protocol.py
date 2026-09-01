"""Parser for Gemma 4 native tool-call channel completions."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping

from redir.server.qwen_native_protocol import (
    QwenNativeParseAnomaly,
    QwenNativeParseResult,
    _vllm_random_tool_call_id,
)


_CALL_START = "<|tool_call>"
_CALL_END = "<tool_call|>"
_THOUGHT_START = "<|channel>thought\n"
_CHANNEL_END = "<channel|>"
_STRING_MARKER = '<|"|>'


class _GemmaArgumentsParser:
    def __init__(self, text: str):
        self.text = text
        self.position = 0

    def parse(self) -> dict[str, Any]:
        value = self._value()
        self._whitespace()
        if self.position != len(self.text) or not isinstance(value, dict):
            raise ValueError("Gemma tool arguments must be one object")
        return value

    def _whitespace(self) -> None:
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def _consume(self, expected: str) -> None:
        self._whitespace()
        if not self.text.startswith(expected, self.position):
            raise ValueError(f"expected {expected!r} at offset {self.position}")
        self.position += len(expected)

    def _value(self) -> Any:
        self._whitespace()
        if self.text.startswith(_STRING_MARKER, self.position):
            self.position += len(_STRING_MARKER)
            end = self.text.find(_STRING_MARKER, self.position)
            if end < 0:
                raise ValueError("unterminated Gemma string")
            value = self.text[self.position:end]
            self.position = end + len(_STRING_MARKER)
            return value
        raw_string = re.match(r'r(?P<hashes>#+)?"', self.text[self.position :])
        if raw_string is not None:
            hashes = raw_string.group("hashes") or ""
            self.position += raw_string.end()
            closing = '"' + hashes
            end = self.text.find(closing, self.position)
            if end < 0:
                raise ValueError("unterminated Gemma raw string")
            value = self.text[self.position:end]
            self.position = end + len(closing)
            return value
        if self.text.startswith('"', self.position):
            try:
                value, consumed = json.JSONDecoder().raw_decode(
                    self.text[self.position :]
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON string at offset {self.position}") from exc
            if not isinstance(value, str):
                raise ValueError("Gemma quoted value is not a string")
            self.position += consumed
            return value
        if self.position >= len(self.text):
            raise ValueError("unexpected end of Gemma arguments")
        char = self.text[self.position]
        if char == "{":
            return self._object()
        if char == "[":
            return self._array()
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in ",}]":
            self.position += 1
        atom = self.text[start:self.position].strip()
        if atom == "true":
            return True
        if atom == "false":
            return False
        if atom == "null":
            return None
        try:
            return float(atom) if any(marker in atom for marker in ".eE") else int(atom)
        except ValueError:
            return atom

    def _key(self) -> str:
        self._whitespace()
        if self.text.startswith(_STRING_MARKER, self.position):
            value = self._value()
            if not isinstance(value, str):
                raise ValueError("Gemma object key is not a string")
            return value
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in ":,{}[]":
            self.position += 1
        key = self.text[start:self.position].strip()
        if not key:
            raise ValueError("Gemma object key is empty")
        return key

    def _object(self) -> dict[str, Any]:
        self._consume("{")
        result: dict[str, Any] = {}
        self._whitespace()
        if self.position < len(self.text) and self.text[self.position] == "}":
            self.position += 1
            return result
        while True:
            key = self._key()
            self._consume(":")
            result[key] = self._value()
            self._whitespace()
            if self.position < len(self.text) and self.text[self.position] == "}":
                self.position += 1
                return result
            self._consume(",")

    def _array(self) -> list[Any]:
        self._consume("[")
        result: list[Any] = []
        self._whitespace()
        if self.position < len(self.text) and self.text[self.position] == "]":
            self.position += 1
            return result
        while True:
            result.append(self._value())
            self._whitespace()
            if self.position < len(self.text) and self.text[self.position] == "]":
                self.position += 1
                return result
            self._consume(",")


def _declared_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            names.add(name)
    return names


def _strip_framing(text: str) -> str:
    for marker in ("<eos>", "<turn|>"):
        while text.endswith(marker):
            text = text[: -len(marker)]
    return text


def _split_reasoning(text: str) -> tuple[str | None, str]:
    if not text.startswith(_THOUGHT_START):
        return None, text
    body = text[len(_THOUGHT_START) :]
    reasoning, marker, remainder = body.partition(_CHANNEL_END)
    if not marker:
        return reasoning or None, ""
    return reasoning or None, remainder


def parse_gemma_native_response(
    raw_completion: str,
    tools: list[dict[str, Any]] | None = None,
    *,
    tool_call_id_factory: Callable[[], str] | None = None,
) -> QwenNativeParseResult:
    """Convert Gemma 4's channel protocol to an OpenAI assistant message."""

    if not isinstance(raw_completion, str):
        raise TypeError("raw_completion must be a string")
    if tools is not None and not isinstance(tools, list):
        raise TypeError("tools must be a list or None")

    text = _strip_framing(raw_completion)
    reasoning, visible = _split_reasoning(text)
    if _CALL_START not in visible:
        return QwenNativeParseResult(
            reasoning_content=reasoning,
            content=visible or None,
            tool_calls=[],
            finish_reason="stop",
            parse_status="reasoning_only" if reasoning and not visible else "plain_text",
            parse_error=None,
            anomalies=(),
        )

    prefix, _, remainder = visible.partition(_CALL_START)
    declared = _declared_tool_names(tools)
    id_factory = tool_call_id_factory or _vllm_random_tool_call_id
    anomalies: list[QwenNativeParseAnomaly] = []
    calls: list[dict[str, Any]] = []
    parse_error: str | None = None
    chunks = remainder.split(_CALL_START)
    suffix = ""
    for index, chunk in enumerate(chunks):
        body, end_marker, trailing = chunk.partition(_CALL_END)
        hybrid_thought: str | None = None
        if not end_marker:
            body, channel_marker, trailing = chunk.partition(_CHANNEL_END)
            hybrid_prefix = f"call:think{{thought:{_STRING_MARKER}"
            if channel_marker and body.startswith(hybrid_prefix):
                # Gemma 4 sometimes starts a declared `think` tool call but
                # closes the payload with its native thought-channel marker.
                # This is unambiguous and recoverable; keep every other
                # unterminated tool call fail-closed.
                hybrid_thought = body[len(hybrid_prefix) :]
            else:
                parse_error = "Gemma tool call is missing <tool_call|>."
                anomalies.append(
                    QwenNativeParseAnomaly(
                        code="malformed_tool_call",
                        message=parse_error,
                        tool_index=index,
                    )
                )
                calls = []
                break
        if index == len(chunks) - 1:
            suffix = trailing
        call_prefix = "call:"
        if not body.startswith(call_prefix) or "{" not in body:
            parse_error = "Gemma tool call is missing call:name{arguments}."
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="malformed_tool_call",
                    message=parse_error,
                    tool_index=index,
                )
            )
            calls = []
            break
        name, raw_arguments = body[len(call_prefix) :].split("{", 1)
        name = name.strip()
        raw_arguments = "{" + raw_arguments
        if declared and name not in declared:
            parse_error = f"Unknown tool {name!r}."
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="unknown_tool",
                    message=parse_error,
                    tool_index=index,
                    function_name=name,
                )
            )
            calls = []
            break
        if hybrid_thought is not None:
            arguments = {"thought": hybrid_thought}
        else:
            try:
                arguments = _GemmaArgumentsParser(raw_arguments).parse()
            except ValueError as exc:
                parse_error = f"GemmaArgumentsError: {exc}"
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
        content = (prefix + suffix).strip() or None
        return QwenNativeParseResult(
            reasoning_content=reasoning,
            content=content,
            tool_calls=calls,
            finish_reason="tool_calls",
            parse_status="parsed",
            parse_error=None,
            anomalies=tuple(anomalies),
        )
    return QwenNativeParseResult(
        reasoning_content=reasoning,
        content=visible or None,
        tool_calls=[],
        finish_reason="stop",
        parse_status="parse_failure",
        parse_error=parse_error or "No structurally valid Gemma tool call was extracted.",
        anomalies=tuple(anomalies),
    )


__all__ = ["parse_gemma_native_response"]
