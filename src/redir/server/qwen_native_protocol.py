"""Pure-Python Qwen native response parsing for the local server.

The parser mirrors the non-streaming order used by the native baseline:

1. vLLM 0.19 ``qwen3`` reasoning extraction;
2. vLLM 0.19 ``qwen3_coder`` XML tool-call extraction for Qwen3.5;
3. Qwen3 JSON-in-``<tool_call>`` extraction.

It deliberately keeps vLLM's permissive behavior.  For example, an unknown
tool name and an unclosed final XML element can still produce a tool call.
Those conditions are exposed as audit anomalies instead of changing the
model-visible response.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any, Literal
import uuid


_THINK_START = "<think>"
_THINK_END = "</think>"
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_FUNCTION_PREFIX = "<function="
_FUNCTION_END = "</function>"
_PARAMETER_PREFIX = "<parameter="
_PARAMETER_END = "</parameter>"

_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$",
    re.DOTALL,
)
_FUNCTION_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)

ParseStatus = Literal["parsed", "plain_text", "reasoning_only", "parse_failure"]


@dataclass(frozen=True, slots=True)
class QwenNativeParseAnomaly:
    """A parser observation that does not itself alter upstream semantics."""

    code: str
    message: str
    tool_index: int | None = None
    function_name: str | None = None
    parameter_name: str | None = None


@dataclass(frozen=True, slots=True)
class QwenNativeParseResult:
    """JSON-ready fields produced from one complete model generation."""

    reasoning_content: str | None
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: Literal["stop", "tool_calls"]
    parse_status: ParseStatus
    parse_error: str | None
    anomalies: tuple[QwenNativeParseAnomaly, ...]

    @property
    def tools_called(self) -> bool:
        return bool(self.tool_calls)


def _vllm_random_tool_call_id() -> str:
    """Match vLLM 0.19's default random tool-call ID format."""

    low_64_bits = uuid.uuid4().int & ((1 << 64) - 1)
    return f"chatcmpl-tool-{low_64_bits:016x}"


def _split_reasoning(
    model_output: str,
    *,
    thinking_enabled: bool,
) -> tuple[str | None, str | None]:
    """Mirror Qwen3ReasoningParser.extract_reasoning from vLLM 0.19."""

    before_start, start, after_start = model_output.partition(_THINK_START)
    model_output = after_start if start else before_start

    if _THINK_END not in model_output:
        if not thinking_enabled:
            return None, model_output
        return model_output, None

    reasoning, _, content = model_output.partition(_THINK_END)
    return reasoning, content or None


def _function_schema(
    function_name: str,
    tools: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Return the first matching tool's property schema, like vLLM."""

    if tools is None:
        return {}
    for tool in tools:
        # ChatCompletionToolsParam defaults an omitted type to "function".
        if tool.get("type", "function") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping) or function.get("name") != function_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            return {}
        properties = parameters.get("properties")
        if isinstance(properties, Mapping):
            return dict(properties)
        return dict(parameters)
    return {}


def _declared_tool_names(
    tools: Sequence[Mapping[str, Any]] | None,
) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        if tool.get("type", "function") != "function":
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _convert_parameter_value(
    value: str,
    *,
    parameter_name: str,
    parameter_schema: dict[str, Any],
    function_name: str,
    tool_index: int,
    anomalies: list[QwenNativeParseAnomaly],
) -> Any:
    """Apply qwen3_coder's schema-guided conversion rules."""

    if value.lower() == "null":
        return None

    if parameter_name not in parameter_schema:
        if parameter_schema:
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="unknown_parameter",
                    message=(
                        f"Parameter {parameter_name!r} is not declared for "
                        f"tool {function_name!r}; preserved as a string."
                    ),
                    tool_index=tool_index,
                    function_name=function_name,
                    parameter_name=parameter_name,
                )
            )
        return value

    config = parameter_schema[parameter_name]
    if isinstance(config, Mapping) and "type" in config:
        parameter_type = str(config["type"]).strip().lower()
    elif isinstance(config, Mapping) and "anyOf" in config:
        # This is the compatibility behavior added in vLLM 0.19.
        parameter_type = "object"
    else:
        parameter_type = "string"

    if parameter_type in {"string", "str", "text", "varchar", "char", "enum"}:
        return value

    if parameter_type.startswith(
        ("int", "uint", "long", "short", "unsigned")
    ):
        try:
            return int(value)
        except (ValueError, TypeError):
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="schema_conversion_fallback",
                    message=(
                        f"Value for integer parameter {parameter_name!r} could "
                        "not be converted; preserved as a string."
                    ),
                    tool_index=tool_index,
                    function_name=function_name,
                    parameter_name=parameter_name,
                )
            )
            return value

    if parameter_type.startswith(("num", "float")):
        try:
            float_value = float(value)
            # Deliberately leave OverflowError uncaught to match vLLM 0.19.
            return float_value if float_value - int(float_value) != 0 else int(float_value)
        except (ValueError, TypeError):
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="schema_conversion_fallback",
                    message=(
                        f"Value for numeric parameter {parameter_name!r} could "
                        "not be converted; preserved as a string."
                    ),
                    tool_index=tool_index,
                    function_name=function_name,
                    parameter_name=parameter_name,
                )
            )
            return value

    if parameter_type in {"boolean", "bool", "binary"}:
        lowered = value.lower()
        if lowered not in {"true", "false"}:
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="invalid_boolean_coerced_false",
                    message=(
                        f"Value for boolean parameter {parameter_name!r} was "
                        "neither 'true' nor 'false'; coerced to false."
                    ),
                    tool_index=tool_index,
                    function_name=function_name,
                    parameter_name=parameter_name,
                )
            )
        return lowered == "true"

    parsed_value: Any = value
    if parameter_type in {"object", "array", "arr"} or parameter_type.startswith(
        ("dict", "list")
    ):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    try:
        parsed_value = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        anomalies.append(
            QwenNativeParseAnomaly(
                code="schema_conversion_fallback",
                message=(
                    f"Value for parameter {parameter_name!r} could not be "
                    "converted; preserved as a string."
                ),
                tool_index=tool_index,
                function_name=function_name,
                parameter_name=parameter_name,
            )
        )
    return parsed_value


def _extract_function_calls(model_output: str) -> list[str]:
    matched_tool_calls = _TOOL_CALL_RE.findall(model_output)
    raw_tool_calls = [closed if closed else open_ for closed, open_ in matched_tool_calls]
    if not raw_tool_calls:
        # qwen3_coder accepts a bare <function=...> block as a back-off.
        raw_tool_calls = [model_output]

    raw_function_calls: list[tuple[str, str]] = []
    for raw_tool_call in raw_tool_calls:
        raw_function_calls.extend(_FUNCTION_RE.findall(raw_tool_call))
    return [closed if closed else open_ for closed, open_ in raw_function_calls]


def _extract_json_tool_call_bodies(model_output: str) -> list[str]:
    matched_tool_calls = _TOOL_CALL_RE.findall(model_output)
    return [closed if closed else open_ for closed, open_ in matched_tool_calls]


def _parse_json_tool_calls(
    model_output: str,
    *,
    declared_tool_names: set[str],
    tool_call_id_factory: Callable[[], str],
    anomalies: list[QwenNativeParseAnomaly],
) -> list[dict[str, Any]]:
    """Parse Qwen3's JSON objects wrapped in ``<tool_call>`` tags."""

    bodies = _extract_json_tool_call_bodies(model_output)
    if not bodies:
        raise ValueError("no JSON tool-call body was extracted")
    parsed: list[dict[str, Any]] = []
    for tool_index, raw_body in enumerate(bodies):
        payload = json.loads(raw_body.strip())
        if not isinstance(payload, Mapping):
            raise ValueError("Qwen3 tool-call payload must be a JSON object")
        function_name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(function_name, str) or not function_name:
            raise ValueError("Qwen3 tool-call payload has no non-empty string name")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("Qwen3 string arguments are not valid JSON") from exc
        if not isinstance(arguments, Mapping):
            raise ValueError("Qwen3 tool-call arguments must be a JSON object")
        if function_name not in declared_tool_names:
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="unknown_tool",
                    message=(
                        f"Tool {function_name!r} is not present in the request tools; "
                        "the call was preserved."
                    ),
                    tool_index=tool_index,
                    function_name=function_name,
                )
            )
        parsed.append(
            {
                "id": tool_call_id_factory(),
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(dict(arguments), ensure_ascii=False),
                },
            }
        )
    return parsed


def _add_xml_anomalies(
    model_output: str,
    anomalies: list[QwenNativeParseAnomaly],
) -> None:
    """Audit permissively accepted XML without changing parser output."""

    function_count = model_output.count(_FUNCTION_PREFIX)
    if function_count and _TOOL_CALL_START not in model_output:
        anomalies.append(
            QwenNativeParseAnomaly(
                code="missing_tool_call_wrapper",
                message="A bare function block was accepted by the vLLM back-off parser.",
            )
        )

    missing_tool_ends = max(
        0,
        model_output.count(_TOOL_CALL_START) - model_output.count(_TOOL_CALL_END),
    )
    for _ in range(missing_tool_ends):
        anomalies.append(
            QwenNativeParseAnomaly(
                code="unclosed_tool_call",
                message="An unclosed <tool_call> block was accepted permissively.",
            )
        )

    missing_function_ends = max(
        0,
        function_count - model_output.count(_FUNCTION_END),
    )
    for _ in range(missing_function_ends):
        anomalies.append(
            QwenNativeParseAnomaly(
                code="unclosed_function",
                message="An unclosed <function> block was accepted permissively.",
            )
        )

    missing_parameter_ends = max(
        0,
        model_output.count(_PARAMETER_PREFIX) - model_output.count(_PARAMETER_END),
    )
    for _ in range(missing_parameter_ends):
        anomalies.append(
            QwenNativeParseAnomaly(
                code="unclosed_parameter",
                message="An unclosed <parameter> block was accepted permissively.",
            )
        )

    if _TOOL_CALL_START in model_output and _FUNCTION_PREFIX not in model_output:
        anomalies.append(
            QwenNativeParseAnomaly(
                code="tool_call_without_function",
                message="A <tool_call> block did not contain a function header.",
            )
        )


def _parse_function_call(
    function_call: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None,
    declared_tool_names: set[str],
    tool_index: int,
    tool_call_id_factory: Callable[[], str],
    anomalies: list[QwenNativeParseAnomaly],
) -> dict[str, Any] | None:
    header_end = function_call.find(">")
    if header_end == -1:
        anomalies.append(
            QwenNativeParseAnomaly(
                code="malformed_function_header",
                message="A function header was missing its closing '>'.",
                tool_index=tool_index,
            )
        )
        return None

    function_name = function_call[:header_end]
    if not function_name:
        anomalies.append(
            QwenNativeParseAnomaly(
                code="empty_function_name",
                message="The parser accepted an empty function name.",
                tool_index=tool_index,
                function_name=function_name,
            )
        )
    if function_name not in declared_tool_names:
        anomalies.append(
            QwenNativeParseAnomaly(
                code="unknown_tool",
                message=(
                    f"Tool {function_name!r} is not present in the request tools; "
                    "the call was preserved to match vLLM."
                ),
                tool_index=tool_index,
                function_name=function_name,
            )
        )

    parameter_schema = _function_schema(function_name, tools)
    parameters = function_call[header_end + 1 :]
    parsed_parameters: dict[str, Any] = {}
    for match_text in _PARAMETER_RE.findall(parameters):
        parameter_header_end = match_text.find(">")
        if parameter_header_end == -1:
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="malformed_parameter_header",
                    message="A parameter header was missing its closing '>'.",
                    tool_index=tool_index,
                    function_name=function_name,
                )
            )
            # vLLM calls str.index here; one bad parameter falls back to the
            # entire original content rather than returning partial calls.
            raise ValueError("parameter header is missing '>'")

        parameter_name = match_text[:parameter_header_end]
        parameter_value = str(match_text[parameter_header_end + 1 :])
        if parameter_value.startswith("\n"):
            parameter_value = parameter_value[1:]
        if parameter_value.endswith("\n"):
            parameter_value = parameter_value[:-1]

        if not parameter_name:
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="empty_parameter_name",
                    message="The parser accepted an empty parameter name.",
                    tool_index=tool_index,
                    function_name=function_name,
                    parameter_name=parameter_name,
                )
            )
        if parameter_name in parsed_parameters:
            anomalies.append(
                QwenNativeParseAnomaly(
                    code="duplicate_parameter",
                    message=(
                        f"Duplicate parameter {parameter_name!r} overwrote its "
                        "previous value, matching vLLM."
                    ),
                    tool_index=tool_index,
                    function_name=function_name,
                    parameter_name=parameter_name,
                )
            )

        parsed_parameters[parameter_name] = _convert_parameter_value(
            parameter_value,
            parameter_name=parameter_name,
            parameter_schema=parameter_schema,
            function_name=function_name,
            tool_index=tool_index,
            anomalies=anomalies,
        )

    return {
        "id": tool_call_id_factory(),
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(parsed_parameters, ensure_ascii=False),
        },
    }


def parse_qwen_native_response(
    raw_completion: str,
    tools: list[dict[str, Any]] | None = None,
    *,
    thinking_enabled: bool = True,
    tool_call_id_factory: Callable[[], str] | None = None,
) -> QwenNativeParseResult:
    """Parse a complete Qwen3/Qwen3.5 generation into an OpenAI-style message.

    ``tools`` uses the OpenAI request schema.  The optional ID factory exists
    for deterministic parity tests; production callers should leave it unset.
    """

    if not isinstance(raw_completion, str):
        raise TypeError("raw_completion must be a string")
    if tools is not None and not isinstance(tools, list):
        raise TypeError("tools must be a list or None")

    normalized_tools: list[Mapping[str, Any]] | None
    if tools is None:
        normalized_tools = None
    else:
        normalized_tools = [tool for tool in tools if isinstance(tool, Mapping)]

    reasoning_content, content = _split_reasoning(
        raw_completion,
        thinking_enabled=thinking_enabled,
    )
    anomalies: list[QwenNativeParseAnomaly] = []
    parse_error: str | None = None
    tool_calls: list[dict[str, Any]] = []
    parser_input = content if content is not None else ""

    if _FUNCTION_PREFIX in parser_input:
        _add_xml_anomalies(parser_input, anomalies)
        function_calls = _extract_function_calls(parser_input)
        if function_calls:
            declared_names = _declared_tool_names(normalized_tools)
            id_factory = tool_call_id_factory or _vllm_random_tool_call_id
            try:
                parsed_calls = [
                    _parse_function_call(
                        function_call,
                        tools=normalized_tools,
                        declared_tool_names=declared_names,
                        tool_index=index,
                        tool_call_id_factory=id_factory,
                        anomalies=anomalies,
                    )
                    for index, function_call in enumerate(function_calls)
                ]
                tool_calls = [call for call in parsed_calls if call is not None]

                # qwen3_coder always replaces content with the prefix once it
                # found at least one function match, even if all calls were
                # later rejected for a missing '>'.
                tool_start = parser_input.find(_TOOL_CALL_START)
                function_start = parser_input.find(_FUNCTION_PREFIX)
                content_index = tool_start if tool_start >= 0 else function_start
                content = parser_input[:content_index] or None
                if not tool_calls:
                    parse_error = "No structurally valid function call was extracted."
            except Exception as exc:
                # Exact qwen3_coder fallback: discard all partial calls and
                # expose the unmodified post-reasoning content.
                tool_calls = []
                content = parser_input
                parse_error = f"{type(exc).__name__}: {exc}"
                anomalies.append(
                    QwenNativeParseAnomaly(
                        code="tool_parse_exception",
                        message=parse_error,
                    )
                )
        else:
            parse_error = "No function block was extracted from the tool-call markup."
    elif _TOOL_CALL_START in parser_input:
        json_bodies = _extract_json_tool_call_bodies(parser_input)
        if json_bodies and not any(body.lstrip().startswith("{") for body in json_bodies):
            # Preserve the historical qwen3_coder audit for non-JSON markup.
            # Qwen3 JSON support must not relabel an arbitrary <tool_call>
            # body as a parser exception.
            _add_xml_anomalies(parser_input, anomalies)
            parse_error = "No function block was extracted from the tool-call markup."
        else:
            declared_names = _declared_tool_names(normalized_tools)
            id_factory = tool_call_id_factory or _vllm_random_tool_call_id
            try:
                tool_calls = _parse_json_tool_calls(
                    parser_input,
                    declared_tool_names=declared_names,
                    tool_call_id_factory=id_factory,
                    anomalies=anomalies,
                )
                tool_start = parser_input.find(_TOOL_CALL_START)
                content = parser_input[:tool_start] or None
                missing_tool_ends = max(
                    0,
                    parser_input.count(_TOOL_CALL_START)
                    - parser_input.count(_TOOL_CALL_END),
                )
                for _ in range(missing_tool_ends):
                    anomalies.append(
                        QwenNativeParseAnomaly(
                            code="unclosed_tool_call",
                            message="An unclosed Qwen3 JSON tool-call block was accepted.",
                        )
                    )
            except Exception as exc:
                tool_calls = []
                content = parser_input
                parse_error = f"{type(exc).__name__}: {exc}"
                anomalies.append(
                    QwenNativeParseAnomaly(
                        code="tool_parse_exception",
                        message=parse_error,
                    )
                )

    if tool_calls:
        parse_status: ParseStatus = "parsed"
        finish_reason: Literal["stop", "tool_calls"] = "tool_calls"
    elif parse_error is not None:
        parse_status = "parse_failure"
        finish_reason = "stop"
    elif content is None and reasoning_content is not None:
        parse_status = "reasoning_only"
        finish_reason = "stop"
    else:
        parse_status = "plain_text"
        finish_reason = "stop"

    return QwenNativeParseResult(
        reasoning_content=reasoning_content,
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        parse_status=parse_status,
        parse_error=parse_error,
        anomalies=tuple(anomalies),
    )


def parse_qwen_native_completion(
    raw_completion: str,
    tools: list[dict[str, Any]] | None = None,
    *,
    thinking_enabled: bool = True,
    tool_call_id_factory: Callable[[], str] | None = None,
) -> QwenNativeParseResult:
    """Compatibility name for callers that describe the input as a completion."""

    return parse_qwen_native_response(
        raw_completion,
        tools,
        thinking_enabled=thinking_enabled,
        tool_call_id_factory=tool_call_id_factory,
    )


__all__ = [
    "QwenNativeParseAnomaly",
    "QwenNativeParseResult",
    "parse_qwen_native_completion",
    "parse_qwen_native_response",
]
