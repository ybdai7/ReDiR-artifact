"""Schema-driven decision span parser for masked OPD training.

The parser returns character spans over a generated assistant completion.  It
tries to isolate the smallest safety-relevant decision unit: tool choice,
action head, finish head, or refusal sentence.  It deliberately avoids tool
arguments and long command parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable


ACTION_SIGNALS = (
    "arguments",
    "args",
    "parameters",
    "parameter",
    "tool",
    "call",
    "function",
    "use",
    "execute",
    "run",
)
NEGATION_SIGNALS = (
    "should not use",
    "shouldn't use",
    "cannot call",
    "can't call",
    "will not use",
    "won't use",
    "do not use",
    "don't use",
    "not call",
    "not use",
)
REFUSAL_SIGNALS = (
    "cannot",
    "can't",
    "will not",
    "won't",
    "unable to",
    "refuse",
    "not able to",
    "not safe",
    "unsafe",
    "harmful",
    "safety concern",
    "safety concerns",
    "should not proceed",
    "do not proceed",
)
SENSITIVE_DOMAIN_SIGNALS = (
    "policy",
    "malicious",
    "vulnerable",
    "vulnerability",
    "discriminatory",
    "hate",
    "hateful",
    "sensitive",
    "credentials",
    "secret",
    "private",
    "confidential",
)
READY_MESSAGE_SIGNALS = (
    "ready for the next instruction",
    "ready for next instruction",
    "ready for the next task",
    "ready for next task",
    "ready for the next step",
    "ready for next step",
    "what's next",
    "what is next",
    "next instruction",
    "next on the agenda",
    "wait for the next instruction",
    "awaiting the next instruction",
)
SAFETY_REASONING_SIGNALS = REFUSAL_SIGNALS + SENSITIVE_DOMAIN_SIGNALS + (
    "credential exposure",
    "credit card",
    "card number",
    "cvv",
    "private key",
    "exfiltration",
    "phishing",
    "backdoor",
    "destructive",
    "command execution",
    "remote code execution",
    "rce",
    "payroll",
    "medical",
    "employee",
    "personal data",
    "financial",
    "protected group",
    "should refuse",
    "should not proceed",
    "do not proceed",
)
ACTION_INTENT_SIGNALS = (
    "i should",
    "i need to",
    "i will",
    "i'll",
    "let me",
    "need to",
    "should use",
    "should call",
    "will use",
    "will call",
    "use the",
    "call the",
    "run the",
    "execute the",
)
DECISION_VOCAB_SIGNALS = SAFETY_REASONING_SIGNALS + ACTION_SIGNALS + (
    "finish",
    "refusal",
    "refuse",
    "comply",
    "proceed",
    "stop",
    "deny",
    "allow",
    "block",
    "read",
    "write",
    "edit",
    "delete",
    "send",
    "upload",
    "download",
    "create",
    "modify",
    "source",
    "append",
    "expose",
    "publish",
)


@dataclass(frozen=True)
class DecisionSpan:
    start: int
    end: int
    kind: str
    source: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionParseResult:
    spans: tuple[DecisionSpan, ...]
    mask_kind: str
    mask_source: str
    reason: str = ""

    @property
    def parsed(self) -> bool:
        return bool(self.spans)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spans": [span.as_dict() for span in self.spans],
            "mask_kind": self.mask_kind,
            "mask_source": self.mask_source,
            "reason": self.reason,
            "parsed": self.parsed,
        }


def available_tool_names(available_tools: Iterable[Any] | None) -> list[str]:
    """Extract tool names from OpenAI tool schemas, simple dicts, or strings."""
    names: list[str] = []
    if not available_tools:
        return names

    for tool in available_tools:
        name: str | None = None
        if isinstance(tool, str):
            name = tool
        elif isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                name = function["name"]
            elif isinstance(tool.get("name"), str):
                name = tool["name"]
            elif isinstance(tool.get("tool_name"), str):
                name = tool["tool_name"]
        if name and name not in names:
            names.append(name)
    return names


def parse_decision_spans(
    generated_text: str,
    available_tools: Iterable[Any] | None,
) -> DecisionParseResult:
    text = generated_text or ""
    if not text.strip():
        return DecisionParseResult((), "unknown", "empty", "empty completion")

    tools = available_tool_names(available_tools)

    result = _parse_structured_tool(text, tools)
    if result.parsed:
        return result

    result = _parse_fenced_tool(text, tools)
    if result.parsed:
        return result

    result = _parse_known_tool_near_signal(text, tools)
    if result.parsed:
        return result

    result = _parse_shell_action(text)
    if result.parsed:
        return result

    result = _parse_finish_or_refusal(text)
    if result.parsed:
        return result

    return DecisionParseResult((), "unknown", "unknown", "no decision span matched")


def parse_visible_decision_spans(
    generated_text: str,
    available_tools: Iterable[Any] | None,
) -> DecisionParseResult:
    """Parse only the visible assistant output after the final think block."""
    text = generated_text or ""
    visible_start = visible_text_start(text)
    visible = text[visible_start:]
    result = parse_decision_spans(visible, available_tools)
    if not result.parsed:
        return result
    spans = tuple(
        DecisionSpan(
            span.start + visible_start,
            span.end + visible_start,
            span.kind,
            span.source,
        )
        for span in result.spans
    )
    return DecisionParseResult(spans, result.mask_kind, result.mask_source, result.reason)


def parse_think_reasoning_spans(generated_text: str) -> DecisionParseResult:
    """Return low-weight safety/action reasoning spans inside think blocks.

    The spans are intentionally sentence/window level.  Downstream source-aware
    filtering decides which tokens within the spans should receive weight.
    """
    text = generated_text or ""
    spans: list[DecisionSpan] = []
    for think_start, think_end in think_content_ranges(text):
        for start, end in _reasoning_windows(text, think_start, think_end):
            spans.append(DecisionSpan(start, end, "think_reasoning", "think_safety_or_action"))
    if not spans:
        return DecisionParseResult((), "unknown", "think_reasoning", "no think reasoning span")
    return DecisionParseResult(tuple(spans), "think_reasoning", "think_safety_or_action")


def content_exclusion_spans(generated_text: str) -> tuple[DecisionSpan, ...]:
    """Find spans that should not be distilled as reasoning/content targets."""
    text = generated_text or ""
    spans: list[DecisionSpan] = []
    patterns = (
        ("code_fence", r"```.*?```"),
        ("url", r"https?://[^\s`\"')]+"),
        ("filesystem_path", r"(?<![\w:])/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+"),
        ("env_assignment", r"\b[A-Z][A-Z0-9_]{3,}\s*=\s*[^\s`\"']+"),
        ("private_key_block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"),
        ("long_hash_or_base64", r"\b[A-Za-z0-9+/=_-]{32,}\b"),
        ("cardish_number", r"\b(?:\d[ -]?){12,19}\b"),
    )
    for source, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            spans.append(DecisionSpan(match.start(), match.end(), "excluded_content", source))
    return tuple(_merge_spans(spans))


def retention_exclusion_spans(
    generated_text: str,
    available_tools: Iterable[Any] | None,
) -> tuple[DecisionSpan, ...]:
    """Find spans that should not be used for base-format retention.

    Unlike ``content_exclusion_spans``, this keeps simple tool-name code fences
    available so V3.4 can retain OpenHands wrapper syntax while still excluding
    argument/code/content fences.
    """
    text = generated_text or ""
    tools = set(available_tool_names(available_tools))
    spans: list[DecisionSpan] = []
    for span in content_exclusion_spans(text):
        if span.source != "code_fence":
            spans.append(span)
            continue
        body = _code_fence_body(text[span.start : span.end]).strip()
        if body not in tools:
            spans.append(span)
    return tuple(_merge_spans(spans))


def visible_text_start(text: str) -> int:
    """Return char offset after the final closing think tag, if present."""
    last_end = 0
    for match in re.finditer(r"</think>", text, re.IGNORECASE):
        last_end = match.end()
    return last_end


def think_content_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    open_re = re.compile(r"<think>", re.IGNORECASE)
    close_re = re.compile(r"</think>", re.IGNORECASE)
    pos = 0
    while True:
        open_match = open_re.search(text, pos)
        if not open_match:
            break
        close_match = close_re.search(text, open_match.end())
        if close_match:
            ranges.append((open_match.end(), close_match.start()))
            pos = close_match.end()
        else:
            ranges.append((open_match.end(), len(text)))
            break
    return ranges


def looks_like_decision_vocab(text: str, available_tools: Iterable[Any] | None = None) -> bool:
    lowered = (text or "").lower()
    if any(signal in lowered for signal in DECISION_VOCAB_SIGNALS):
        return True
    tools = available_tool_names(available_tools)
    return any(tool.lower() in lowered for tool in tools)


def parse_format_prefix_spans(
    generated_text: str,
    available_tools: Iterable[Any] | None,
) -> DecisionParseResult:
    """Return spans that preserve agent interface format without arguments.

    This is intentionally wider than ``parse_decision_spans`` for CE warmup:
    it can include OpenHands/MCP action scaffolding such as the action header,
    fenced tool name, and the ``and arguments:`` marker.  It must not include
    the argument code block/body, paths, secrets, or file contents.
    """
    text = generated_text or ""
    if not text.strip():
        return DecisionParseResult((), "unknown", "empty", "empty completion")

    tools = available_tool_names(available_tools)

    result = _parse_openhands_mcp_format_prefix(text, tools)
    if result.parsed:
        return result

    result = _parse_raw_function_call_format_prefix(text, tools)
    if result.parsed:
        return result

    result = _parse_structured_tool(text, tools)
    if result.parsed:
        return result

    result = _parse_known_tool_near_signal(text, tools)
    if result.parsed:
        return result

    result = _parse_shell_action(text)
    if result.parsed:
        return result

    result = _parse_finish_or_refusal(text)
    if result.parsed:
        return result

    return DecisionParseResult((), "unknown", "unknown", "no format prefix span matched")


def parse_event_lifecycle_spans(
    generated_text: str,
    available_tools: Iterable[Any] | None,
) -> DecisionParseResult:
    """Return V3.7 event-level lifecycle spans.

    This keeps the V1-style event target, but covers ordinary ready/status
    messages that are crucial for MT-AgentRisk's non-final-turn lifecycle.
    """
    text = generated_text or ""
    if not text.strip():
        return DecisionParseResult((), "unknown", "empty", "empty completion")

    tools = available_tool_names(available_tools)

    result = _parse_openhands_mcp_format_prefix(text, tools)
    if result.parsed:
        return DecisionParseResult(result.spans, "event_tool_action", result.mask_source, result.reason)

    result = _parse_raw_function_call_format_prefix(text, tools)
    if result.parsed:
        if result.mask_source == "raw_finish_function":
            return DecisionParseResult(result.spans, "finish_head", result.mask_source, result.reason)
        return DecisionParseResult(result.spans, "event_tool_action", result.mask_source, result.reason)

    result = _parse_structured_tool(text, tools)
    if result.parsed:
        return DecisionParseResult(result.spans, "event_tool_action", result.mask_source, result.reason)

    result = _parse_fenced_tool(text, tools)
    if result.parsed:
        return DecisionParseResult(result.spans, "event_tool_action", result.mask_source, result.reason)

    result = _parse_known_tool_near_signal(text, tools)
    if result.parsed:
        return DecisionParseResult(result.spans, "event_tool_action", result.mask_source, result.reason)

    result = _parse_shell_action(text)
    if result.parsed:
        return DecisionParseResult(result.spans, "event_shell_action", result.mask_source, result.reason)

    result = _parse_finish_or_refusal(text)
    if result.parsed:
        return result

    result = _parse_ordinary_message(text)
    if result.parsed:
        return result

    return DecisionParseResult((), "unknown", "ordinary_message", "no event lifecycle span matched")


def parse_safety_prefix_action_spans(
    generated_text: str,
    available_tools: Iterable[Any] | None,
) -> DecisionParseResult:
    """Return spans for V3.13 safety-only OPD.

    This profile is intentionally V1-like, but it also includes the generated
    reasoning prefix through ``</think>`` because the safety decision often
    appears there before the visible action. Tool/shell calls keep only action
    heads and scaffolding. Finish calls keep the whole finish block because the
    message itself is the safety/stop decision.
    """
    text = generated_text or ""
    if not text.strip():
        return DecisionParseResult((), "unknown", "empty", "empty completion")

    tools = available_tool_names(available_tools)
    prefix = _think_decision_prefix_span(text)

    finish = _parse_raw_finish_full_spans(text)
    if finish.parsed:
        return _combine_safety_prefix(text, prefix, finish, "finish_full_action", finish.mask_source, finish.reason)

    result = _parse_raw_function_call_format_prefix(text, tools)
    if result.parsed:
        return _combine_safety_prefix(text, prefix, result, "safety_tool_action", result.mask_source, result.reason)

    result = _parse_openhands_mcp_format_prefix(text, tools)
    if result.parsed:
        return _combine_safety_prefix(text, prefix, result, "safety_tool_action", result.mask_source, result.reason)

    result = _parse_structured_tool(text, tools)
    if result.parsed:
        return _combine_safety_prefix(text, prefix, result, "safety_tool_action", result.mask_source, result.reason)

    result = _parse_fenced_tool(text, tools)
    if result.parsed:
        return _combine_safety_prefix(text, prefix, result, "safety_tool_action", result.mask_source, result.reason)

    result = _parse_known_tool_near_signal(text, tools)
    if result.parsed:
        return _combine_safety_prefix(text, prefix, result, "safety_tool_action", result.mask_source, result.reason)

    result = _parse_shell_action(text)
    if result.parsed:
        return _combine_safety_prefix(text, prefix, result, "safety_shell_action", result.mask_source, result.reason)

    result = _parse_finish_or_refusal(text)
    if result.parsed:
        return _combine_safety_prefix(text, prefix, result, result.mask_kind, result.mask_source, result.reason)

    if prefix is not None:
        return DecisionParseResult((prefix,), "think_safety_prefix", "think_prefix", "reasoning prefix through </think>")

    return DecisionParseResult((), "unknown", "safety_prefix_action", "no safety prefix/action span matched")


def parse_retention_spans(
    generated_text: str,
    available_tools: Iterable[Any] | None,
) -> DecisionParseResult:
    """Return spans for preserving thinking continuation and tool-call format.

    The returned spans are candidates only. The trainer applies token-level
    filters to remove tool names, refusal/safety decision words, paths, secrets,
    and long arguments before applying base-teacher retention KL.
    """
    text = generated_text or ""
    if not text.strip():
        return DecisionParseResult((), "unknown", "empty", "empty completion")

    spans: list[DecisionSpan] = []
    spans.extend(_think_retention_spans(text))
    spans.extend(_openhands_mcp_retention_spans(text, available_tool_names(available_tools)))
    spans.extend(_structured_tool_retention_spans(text, available_tool_names(available_tools)))
    spans.extend(_tool_call_retention_spans(text))
    spans.extend(_markdown_wrapper_retention_spans(text))

    merged = tuple(_merge_spans(spans))
    if not merged:
        return DecisionParseResult((), "unknown", "retention", "no retention span matched")
    return DecisionParseResult(merged, "retention", "thinking_or_format_retention")


def _reasoning_windows(text: str, start: int, end: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    pos = start
    separators = re.compile(r"(?<=[.!?])\s+|\n+")
    for match in separators.finditer(text, start, end):
        piece_start, piece_end = pos, match.start()
        _maybe_add_reasoning_window(text, piece_start, piece_end, ranges)
        pos = match.end()
    _maybe_add_reasoning_window(text, pos, end, ranges)
    return ranges


def _maybe_add_reasoning_window(text: str, start: int, end: int, ranges: list[tuple[int, int]]) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return
    lowered = text[start:end].lower()
    if not (
        any(signal in lowered for signal in SAFETY_REASONING_SIGNALS)
        or any(signal in lowered for signal in ACTION_INTENT_SIGNALS)
    ):
        return
    ranges.append((start, min(end, start + 280)))


def _merge_spans(spans: list[DecisionSpan]) -> list[DecisionSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[DecisionSpan] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end:
            merged[-1] = DecisionSpan(
                last.start,
                max(last.end, span.end),
                last.kind,
                f"{last.source}+{span.source}" if span.source not in last.source else last.source,
            )
        else:
            merged.append(span)
    return merged


def _think_decision_prefix_span(text: str) -> DecisionSpan | None:
    close = re.search(r"</think>", text, re.IGNORECASE)
    if not close:
        return None
    return DecisionSpan(0, close.end(), "think_safety_prefix", "think_prefix")


def _combine_safety_prefix(
    text: str,
    prefix: DecisionSpan | None,
    result: DecisionParseResult,
    mask_kind: str,
    mask_source: str,
    reason: str,
) -> DecisionParseResult:
    spans = list(result.spans)
    if prefix is not None:
        spans.insert(0, prefix)
    elif spans and text[: spans[0].start].strip():
        spans.insert(0, DecisionSpan(0, spans[0].start, "safety_leading_prefix", "leading_prefix"))
    return DecisionParseResult(
        tuple(_merge_spans(spans)),
        mask_kind,
        mask_source,
        reason,
    )

def _parse_raw_finish_full_spans(text: str) -> DecisionParseResult:
    function_re = re.compile(r"<function\s*=\s*finish\s*>", re.IGNORECASE)
    match = function_re.search(text)
    if not match:
        return DecisionParseResult((), "unknown", "raw_finish_function_full", "no raw finish function")
    close_match = re.search(r"</function\s*>", text[match.end() :], re.IGNORECASE)
    block_end = match.end() + close_match.end() if close_match else len(text)
    return DecisionParseResult(
        (DecisionSpan(match.start(), block_end, "finish_full_action", "raw_finish_function_full"),),
        "finish_full_action",
        "raw_finish_function_full",
        "full raw finish function block",
    )


def _parse_openhands_mcp_format_prefix(text: str, tools: list[str]) -> DecisionParseResult:
    if not tools:
        return DecisionParseResult((), "unknown", "openhands_mcp_format", "no available tools")

    header = re.search(
        r"I am interacting with the MCP server with name:\s*",
        text,
        re.IGNORECASE,
    )
    if not header:
        return DecisionParseResult((), "unknown", "openhands_mcp_format", "no MCP action header")

    fence_re = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*\n?(.*?)\n?```", re.DOTALL)
    first_fence = fence_re.search(text, header.end())
    if not first_fence:
        return DecisionParseResult((), "unknown", "openhands_mcp_format", "no fenced tool name")

    tool = first_fence.group(1).strip()
    if tool not in tools:
        return DecisionParseResult((), "unknown", "openhands_mcp_format", "fenced block is not a tool name")

    spans = [
        DecisionSpan(header.start(), first_fence.end(), "format_prefix", "openhands_mcp_format"),
    ]

    arguments = re.search(r"\band arguments\s*:", text[first_fence.end() :], re.IGNORECASE)
    if arguments:
        marker_start = first_fence.end() + arguments.start()
        marker_end = first_fence.end() + arguments.end()
        spans.append(DecisionSpan(marker_start, marker_end, "format_prefix", "openhands_mcp_format"))

    return DecisionParseResult(
        tuple(spans),
        "format_prefix",
        "openhands_mcp_format",
    )


def _parse_raw_function_call_format_prefix(text: str, tools: list[str]) -> DecisionParseResult:
    """Parse OpenHands native_tool_calling=false function-call text.

    Example:
        <function=write_file>
        <parameter=path>/workspace/a</parameter>
        </function>

    The mask keeps the action/function scaffold and parameter tag names, but it
    deliberately excludes parameter values.
    """
    function_re = re.compile(r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>", re.IGNORECASE)
    for match in function_re.finditer(text):
        name = match.group(1)
        is_finish = name == "finish"
        if not is_finish and tools and name not in tools:
            continue
        if not is_finish and not tools:
            continue

        close_match = re.search(r"</function\s*>", text[match.end() :], re.IGNORECASE)
        block_end = match.end() + close_match.end() if close_match else min(len(text), match.end() + 320)
        spans: list[DecisionSpan] = [
            DecisionSpan(match.start(), match.end(), "raw_function_head", "raw_finish_function" if is_finish else "raw_function_call"),
        ]

        block = text[match.end() : block_end]
        for parameter_match in re.finditer(r"</?parameter(?:\s*=\s*[^>\s]+)?\s*>", block, re.IGNORECASE):
            spans.append(
                DecisionSpan(
                    match.end() + parameter_match.start(),
                    match.end() + parameter_match.end(),
                    "raw_function_scaffold",
                    "raw_finish_function" if is_finish else "raw_function_call",
                )
            )
        if close_match:
            close_start = match.end() + close_match.start()
            close_end = match.end() + close_match.end()
            spans.append(
                DecisionSpan(
                    close_start,
                    close_end,
                    "raw_function_scaffold",
                    "raw_finish_function" if is_finish else "raw_function_call",
                )
            )

        return DecisionParseResult(
            tuple(_merge_spans(spans)),
            "finish_head" if is_finish else "format_prefix",
            "raw_finish_function" if is_finish else "raw_function_call",
            "OpenHands raw function-call scaffold",
        )
    return DecisionParseResult((), "unknown", "raw_function_call", "no raw function call")


def _openhands_mcp_retention_spans(text: str, tools: list[str]) -> list[DecisionSpan]:
    spans: list[DecisionSpan] = []
    if not tools:
        return spans

    header = re.search(
        r"I am interacting with the MCP server with name:\s*",
        text,
        re.IGNORECASE,
    )
    if not header:
        return spans

    fence_re = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*\n?(.*?)\n?```", re.DOTALL)
    first_fence = fence_re.search(text, header.end())
    if not first_fence:
        spans.append(DecisionSpan(header.start(), header.end(), "retention_format", "openhands_mcp_scaffold"))
        return spans

    tool = first_fence.group(1).strip()
    if tool not in tools:
        return spans

    body_start = first_fence.start(1) + len(first_fence.group(1)) - len(first_fence.group(1).lstrip())
    body_end = body_start + len(tool)
    if header.start() < body_start:
        spans.append(DecisionSpan(header.start(), body_start, "retention_format", "openhands_mcp_scaffold"))
    if body_end < first_fence.end():
        spans.append(DecisionSpan(body_end, first_fence.end(), "retention_format", "openhands_mcp_scaffold"))

    arguments = re.search(r"\band arguments\s*:", text[first_fence.end() :], re.IGNORECASE)
    if arguments:
        marker_start = first_fence.end() + arguments.start()
        marker_end = first_fence.end() + arguments.end()
        spans.append(DecisionSpan(marker_start, marker_end, "retention_format", "openhands_mcp_scaffold"))
    return spans


def _structured_tool_retention_spans(text: str, tools: list[str]) -> list[DecisionSpan]:
    spans: list[DecisionSpan] = []
    if not tools:
        return spans
    for match in re.finditer(r"(['\"](?:name|tool_name)['\"]\s*:\s*['\"])([^'\"]+)(['\"])", text):
        if match.group(2) not in tools:
            continue
        spans.append(DecisionSpan(match.start(1), match.end(1), "retention_format", "structured_tool_scaffold"))
        spans.append(DecisionSpan(match.start(3), match.end(3), "retention_format", "structured_tool_scaffold"))
    for match in re.finditer(r"(['\"](?:arguments|args|parameters)['\"]\s*:)", text):
        spans.append(DecisionSpan(match.start(1), match.end(1), "retention_format", "structured_tool_scaffold"))
    return spans


def _tool_call_retention_spans(text: str) -> list[DecisionSpan]:
    spans: list[DecisionSpan] = []
    for match in re.finditer(r"(<tool_call>\s*(?:name|tool_name)\s*:)", text, re.IGNORECASE):
        spans.append(DecisionSpan(match.start(1), match.end(1), "retention_format", "tool_call_scaffold"))
    for match in re.finditer(r"\b(arguments|args|parameters)\s*:", text, re.IGNORECASE):
        spans.append(DecisionSpan(match.start(0), match.end(0), "retention_format", "tool_call_scaffold"))
    return spans


def _markdown_wrapper_retention_spans(text: str) -> list[DecisionSpan]:
    spans: list[DecisionSpan] = []
    for match in re.finditer(r"```(?:bash|python|json|sh)?\s*", text, re.IGNORECASE):
        spans.append(DecisionSpan(match.start(), match.end(), "retention_format", "markdown_fence"))
    return spans


def _think_retention_spans(text: str) -> list[DecisionSpan]:
    spans: list[DecisionSpan] = []
    open_re = re.compile(r"<think>", re.IGNORECASE)
    close_re = re.compile(r"</think>", re.IGNORECASE)
    pos = 0
    while True:
        open_match = open_re.search(text, pos)
        if not open_match:
            break
        close_match = close_re.search(text, open_match.end())
        if close_match:
            if open_match.end() < close_match.start():
                spans.append(
                    DecisionSpan(open_match.end(), close_match.start(), "retention_think", "think_continuation")
                )
            spans.append(DecisionSpan(close_match.start(), close_match.end(), "retention_think", "think_close"))
            pos = close_match.end()
        else:
            if open_match.end() < len(text):
                spans.append(DecisionSpan(open_match.end(), len(text), "retention_think", "think_continuation"))
            break
    return spans


def _code_fence_body(text: str) -> str:
    match = re.match(r"```(?:[A-Za-z0-9_-]+)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    return match.group(1) if match else ""


def _parse_structured_tool(text: str, tools: list[str]) -> DecisionParseResult:
    if not tools:
        return DecisionParseResult((), "unknown", "structured_tool", "no available tools")

    for match in re.finditer(r"['\"](?:name|tool_name)['\"]\s*:\s*['\"]([^'\"]+)['\"]", text):
        tool = match.group(1)
        if tool not in tools:
            continue
        window = _window(text, match.start(), match.end(), size=240).lower()
        if not _has_action_signal(window):
            continue
        return DecisionParseResult(
            (DecisionSpan(match.start(1), match.end(1), "tool_name", "structured_tool"),),
            "tool_name",
            "structured_tool",
        )

    for tool in tools:
        pattern = re.compile(rf"(?<![\w.-])({re.escape(tool)})\s*\(")
        match = pattern.search(text)
        if match:
            return DecisionParseResult(
                (DecisionSpan(match.start(1), match.end(1), "tool_name", "function_call"),),
                "tool_name",
                "function_call",
            )
    return DecisionParseResult((), "unknown", "structured_tool", "no structured tool call")


def _parse_fenced_tool(text: str, tools: list[str]) -> DecisionParseResult:
    if not tools:
        return DecisionParseResult((), "unknown", "fenced_tool", "no available tools")

    fence_re = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*\n?(.*?)\n?```", re.DOTALL)
    for match in fence_re.finditer(text):
        body = match.group(1).strip()
        if body not in tools:
            continue
        window = _window(text, match.start(), match.end(), size=220).lower()
        if not _has_action_signal(window):
            continue
        body_start = match.start(1) + len(match.group(1)) - len(match.group(1).lstrip())
        body_end = body_start + len(body)
        phrase_start = _minimal_phrase_start(text, match.start(), fallback=body_start)
        start = min(phrase_start, body_start) if phrase_start < body_start else body_start
        return DecisionParseResult(
            (DecisionSpan(start, body_end, "action_head", "fenced_tool_name"),),
            "action_head",
            "fenced_tool_name",
        )
    return DecisionParseResult((), "unknown", "fenced_tool", "no fenced tool name")


def _parse_known_tool_near_signal(text: str, tools: list[str]) -> DecisionParseResult:
    if not tools:
        return DecisionParseResult((), "unknown", "known_tool_near_signal", "no available tools")

    for tool in sorted(tools, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w.-])({re.escape(tool)})(?![\w.-])")
        for match in pattern.finditer(text):
            sentence = _sentence_containing(text, match.start(), match.end()).lower()
            if any(signal in sentence for signal in NEGATION_SIGNALS):
                refusal = _parse_finish_or_refusal(text)
                if refusal.parsed:
                    return refusal
            after = text[match.end() : match.end() + 2]
            window = _window(text, match.start(), match.end(), size=120).lower()
            if not (_has_action_signal(window) or after[:1] in {"(", "{", ":"}):
                continue
            start = _minimal_phrase_start(text, match.start(), fallback=match.start())
            return DecisionParseResult(
                (DecisionSpan(start, match.end(1), "action_head", "known_tool_near_signal"),),
                "action_head",
                "known_tool_near_signal",
            )
    return DecisionParseResult((), "unknown", "known_tool_near_signal", "no tool action signal")


def _parse_shell_action(text: str) -> DecisionParseResult:
    patterns = (
        r"\b(Running command:)",
        r"\b(Running Python code interactively:)",
        r"\b(Command to run:)",
        r"\b(Executing command:)",
        r"\b(I am interacting with the browser:)",
        r"\b(I edited the file)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return DecisionParseResult(
                (DecisionSpan(match.start(1), match.end(1), "shell_action_head", "shell_action"),),
                "shell_action_head",
                "shell_action",
            )
    return DecisionParseResult((), "unknown", "shell_action", "no shell action")


def _parse_finish_or_refusal(text: str) -> DecisionParseResult:
    finish = re.search(r"\b(AgentFinishAction|finish(?: the action)?|final answer)\b", text, re.IGNORECASE)
    if finish:
        return DecisionParseResult(
            (DecisionSpan(finish.start(1), finish.end(1), "finish_head", "finish"),),
            "finish_head",
            "finish",
        )

    lowered = text.lower()
    positions = [
        lowered.find(signal)
        for signal in REFUSAL_SIGNALS + NEGATION_SIGNALS
        if lowered.find(signal) != -1
    ]
    if not positions:
        return DecisionParseResult((), "unknown", "refusal", "no refusal signal")
    start_signal = min(positions)
    start = _sentence_start(text, start_signal)
    end = _sentence_end(text, start_signal)
    end = min(end, start + 220)
    return DecisionParseResult(
        (DecisionSpan(start, end, "refusal_or_safety_sentence", "refusal"),),
        "refusal_or_safety_sentence",
        "refusal",
    )


def _parse_ordinary_message(text: str) -> DecisionParseResult:
    visible_start = visible_text_start(text)
    visible = text[visible_start:].strip()
    offset = visible_start + len(text[visible_start:]) - len(text[visible_start:].lstrip())
    if not visible:
        visible = text.strip()
        offset = len(text) - len(text.lstrip())
    if not visible:
        return DecisionParseResult((), "unknown", "ordinary_message", "empty ordinary message")

    lowered = visible.lower()
    ready_positions = [lowered.find(signal) for signal in READY_MESSAGE_SIGNALS if lowered.find(signal) != -1]
    if ready_positions:
        signal_pos = min(pos for pos in ready_positions if pos >= 0)
        start = _sentence_start(visible, signal_pos)
        end = _sentence_end(visible, signal_pos)
        return DecisionParseResult(
            (DecisionSpan(offset + start, offset + min(end, start + 240), "ordinary_message", "ready_message"),),
            "ordinary_message",
            "ready_message",
        )

    if len(visible) <= 240:
        return DecisionParseResult(
            (DecisionSpan(offset, offset + len(visible), "ordinary_message", "ordinary_message"),),
            "ordinary_message",
            "ordinary_message",
        )

    end = min(_sentence_end(visible, 0), 240)
    return DecisionParseResult(
        (DecisionSpan(offset, offset + end, "ordinary_message", "ordinary_message"),),
        "ordinary_message",
        "ordinary_message",
    )


def _window(text: str, start: int, end: int, *, size: int) -> str:
    return text[max(0, start - size) : min(len(text), end + size)]


def _has_action_signal(text: str) -> bool:
    return any(signal in text for signal in ACTION_SIGNALS)


def _sentence_containing(text: str, start: int, end: int) -> str:
    return text[_sentence_start(text, start) : _sentence_end(text, end)]


def _sentence_start(text: str, pos: int) -> int:
    candidates = [text.rfind(sep, 0, pos) for sep in (".", "\n", "!", "?")]
    start = max(candidates)
    return 0 if start == -1 else start + 1


def _sentence_end(text: str, pos: int) -> int:
    candidates = [text.find(sep, pos) for sep in (".", "\n", "!", "?")]
    positives = [idx for idx in candidates if idx != -1]
    return len(text) if not positives else min(positives) + 1


def _minimal_phrase_start(text: str, tool_start: int, *, fallback: int) -> int:
    window_start = max(0, tool_start - 120)
    prefix = text[window_start:tool_start].lower()
    best: int | None = None
    for signal in ACTION_SIGNALS:
        idx = prefix.rfind(signal)
        if idx != -1:
            best = window_start + idx if best is None else max(best, window_start + idx)
    if best is None:
        return fallback
    return max(_sentence_start(text, best), best)
