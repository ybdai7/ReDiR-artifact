"""Task-agnostic masking helpers for V5 precise-mask OPD."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from redir.engine.behavior import (
    EXPLICIT_SAFETY_DECISION_MARKERS,
    SAFETY_ENTRY_MARKERS,
    action_consistent_safety,
    contains_explicit_safety_decision,
    native_finish_message_text,
    protocol_sequence_safety,
)


PrefixStratum = Literal["c", "b", "a", "unknown"]
FinishPreference = Literal["refusal", "success", "tie", "unavailable"]

_COMPLETED_CLAIM_RE = re.compile(
    r"\b(?:benign|safe request|completed successfully|task (?:is|has been) complete|"
    r"successfully completed|all done|done with|ready for (?:the )?next instruction|"
    r"ready for what(?:'s| is) next)\b",
    re.IGNORECASE,
)
_NEUTRAL_EXECUTION_RE = re.compile(
    r"(?:<function\s*=|\b(?:call|calling|use|using|execute|executing|run|running|"
    r"create|creating|write|writing|read|reading|inspect|inspecting|check|checking|"
    r"modify|modifying|edit|editing|search|searching|download|downloading|"
    r"upload|uploading|list|listing|open|opening)\b|"
    r"\b(?:i (?:will|should|need to|can)|let me|the user (?:asks|is asking))\b)",
    re.IGNORECASE,
)
_FINISH_MESSAGE_RE = re.compile(
    r"<function\s*=\s*finish\s*>.*?<parameter\s*=\s*message\s*>(.*?)</parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_REQUEST_VIOLATION_RE = re.compile(
    r"\b(?:this|the)\s+request\b.{0,600}\bviolat(?:e|es|ed|ing)\b",
    re.IGNORECASE | re.DOTALL,
)
_FINISH_STRUCTURE_PATTERNS = (
    re.compile(r"<function\s*=\s*finish\s*>", re.IGNORECASE),
    re.compile(r"<parameter\s*=\s*message\s*>", re.IGNORECASE),
    re.compile(r"</parameter\s*>", re.IGNORECASE),
    re.compile(r"</function\s*>", re.IGNORECASE),
)
_FINISH_OPENING_PATTERNS = _FINISH_STRUCTURE_PATTERNS[:2]

# ``assistant_generation_prefix`` already opens the thinking block.  The
# canonical target closes that block on the real next assistant turn before
# teaching only the task-independent OpenHands finish opening scaffold.
V52_FINISH_OPENING_SUFFIX = "\n\n<function=finish>\n<parameter=message>"
V52_CANONICAL_FINISH_OPENING = "\n</think>" + V52_FINISH_OPENING_SUFFIX
V53_FINISH_OPENING = "\n\n<function=finish>\n<parameter=message>\n"
V53_FINISH_CLOSING = "\n</parameter>\n</function>"
V6_NATIVE_FINISH_OPENING = (
    "\n</think>\n\n<tool_call>\n<function=finish>\n<parameter=message>\n"
)
V6_NATIVE_FINISH_CLOSING = (
    "\n</parameter>\n</function>\n</tool_call><|im_end|>"
)
_V53_REFUSAL_START_MARKERS = EXPLICIT_SAFETY_DECISION_MARKERS + (
    "i won't",
    "i refuse",
    "i am unable",
    "unable to comply",
    "unable to continue",
)

_REFUSAL_COMMITMENT_PATTERN = (
    r"i\s+refuse\b|"
    r"i\s+(?:must|should|need\s+to)\s+refuse|"
    r"i\s+(?:must|should|need\s+to)\s+(?:stand\s+by|maintain)\s+"
    r"(?:my|the)\s+refusal|"
    r"i\s+(?:cannot|can't)\s+and\s+(?:will\s+not|won't)\s+"
    r"(?:help|assist|comply|continue|proceed|do\s+(?:this|that))|"
    r"i\s+(?:will\s+not|won't)\s+(?:help|assist|comply|continue|proceed|do\s+(?:this|that))|"
    r"i\s+(?:cannot|can't)\s+(?:help|assist|comply|continue|proceed|fulfill|complete|do\s+(?:this|that))|"
    r"i(?:'m|\s+am)\s+unable\s+to\s+(?:help|assist|comply|continue|proceed)|"
    r"cannot\s+safely\s+(?:continue|proceed|help|assist|comply)|"
    r"(?:must|should|need\s+to)\s+(?:stop|refuse)|"
    r"requires?\s+immediate\s+refusal"
)
_REFUSAL_COMMITMENT_RE = re.compile(
    rf"\b(?:{_REFUSAL_COMMITMENT_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)
_TEACHER_REFUSAL_COMMITMENT_RE = re.compile(
    rf"\b(?:{_REFUSAL_COMMITMENT_PATTERN}|"
    r"i\s+(?:cannot|can't|will\s+not|won't)\s+(?:in\s+good\s+conscience\s+)?"
    r"(?:create|write|modify|change|execute|run|read|expose|continue|proceed|"
    r"implement|generate|perform|complete|fulfill|assist|help)\b)",
    re.IGNORECASE | re.DOTALL,
)
_COMMITTED_SAFETY_DECISION_RE = re.compile(
    rf"\b(?:{_REFUSAL_COMMITMENT_PATTERN}|"
    r"(?:this|the)\s+request\b.{0,600}\bviolat(?:e|es|ed|ing)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_TEACHER_CONCRETE_RISK_MARKERS = (
    "unauthorized",
    "credential",
    "private key",
    "secret",
    "confidential",
    "sensitive",
    "privacy",
    "deceptive",
    "malicious",
    "security risk",
    "security-sensitive",
    "weaken system security",
    "discriminatory",
    "discrimination",
    "hateful",
    "hate speech",
    "racist",
    "racial slur",
    "xenophobic",
    "protected group",
    "medical decision",
    "medical treatment",
    "medical decision-making",
    "gender",
    "gender-based",
    "harmful outcomes",
)


def _teacher_refusal_commitment_match(text: str) -> re.Match[str] | None:
    matches = list(_TEACHER_REFUSAL_COMMITMENT_RE.finditer(text))
    if not matches:
        return None
    strong = [
        match
        for match in matches
        if "stand by" not in match.group(0).lower()
        and "maintain" not in match.group(0).lower()
    ]
    return (strong or matches)[0]


@dataclass(frozen=True)
class Level5EntryTarget:
    available: bool
    teacher_prefix_token_ids: tuple[int, ...]
    decision_token_ids: tuple[int, ...]
    decision_span_start: int
    decision_span_end: int
    teacher_prefix_text: str
    decision_text: str
    check_text: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "teacher_prefix_token_ids": list(self.teacher_prefix_token_ids),
            "decision_token_ids": list(self.decision_token_ids),
            "decision_span_start": self.decision_span_start,
            "decision_span_end": self.decision_span_end,
            "span_length": self.decision_span_end - self.decision_span_start,
            "teacher_prefix_text": self.teacher_prefix_text,
            "decision_text": self.decision_text,
            "check_text": self.check_text,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FinishScaffoldTarget:
    available: bool
    completion_token_ids: tuple[int, ...]
    structural_token_indices: tuple[int, ...]
    completion_text: str
    structural_text: str
    reason: str
    prefix_source: str = "canonical_empty_reasoning"
    prefix_token_count: int = 0
    prefix_text: str = ""
    target_mode: str = "opening_only"
    message_token_count: int = 0
    message_text: str = ""
    message_token_indices: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "completion_token_ids": list(self.completion_token_ids),
            "structural_token_indices": list(self.structural_token_indices),
            "completion_text": self.completion_text,
            "structural_text": self.structural_text,
            "reason": self.reason,
            "prefix_source": self.prefix_source,
            "prefix_token_count": self.prefix_token_count,
            "prefix_text": self.prefix_text,
            "target_mode": self.target_mode,
            "message_token_count": self.message_token_count,
            "message_text": self.message_text,
            "message_token_indices": list(self.message_token_indices),
        }


@dataclass(frozen=True)
class NativeRefusalTransitionTarget:
    available: bool
    completion_token_ids: tuple[int, ...]
    supervised_token_indices: tuple[int, ...]
    supervised_token_weights: tuple[float, ...]
    completion_text: str
    reasoning_text: str
    message_text: str
    reasoning_token_count: int
    action_token_count: int
    message_token_count: int
    reason: str
    target_mode: str = "native_clean_teacher_full_refusal_transition"
    weighting_profile: str = "full_uniform_v1"
    source_reasoning_text: str = ""
    reasoning_selection: str = "full_clean_teacher_reasoning"
    early_reasoning_token_indices: tuple[int, ...] = ()
    decision_token_indices: tuple[int, ...] = ()
    action_onset_token_indices: tuple[int, ...] = ()
    message_token_indices: tuple[int, ...] = ()
    finish_function_name_token_indices: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "completion_token_ids": list(self.completion_token_ids),
            "supervised_token_indices": list(self.supervised_token_indices),
            "supervised_token_weights": list(self.supervised_token_weights),
            "completion_text": self.completion_text,
            "reasoning_text": self.reasoning_text,
            "message_text": self.message_text,
            "reasoning_token_count": self.reasoning_token_count,
            "action_token_count": self.action_token_count,
            "message_token_count": self.message_token_count,
            "reason": self.reason,
            "target_mode": self.target_mode,
            "weighting_profile": self.weighting_profile,
            "source_reasoning_text": self.source_reasoning_text,
            "reasoning_selection": self.reasoning_selection,
            "early_reasoning_token_indices": list(
                self.early_reasoning_token_indices
            ),
            "decision_token_indices": list(self.decision_token_indices),
            "action_onset_token_indices": list(
                self.action_onset_token_indices
            ),
            "message_token_indices": list(self.message_token_indices),
            "finish_function_name_token_indices": list(
                self.finish_function_name_token_indices
            ),
        }


def benign_completed_claim(text: str) -> bool:
    return bool(_COMPLETED_CLAIM_RE.search(text))


def is_neutral_execution(text: str) -> bool:
    return bool(_NEUTRAL_EXECUTION_RE.search(text))


def classify_prefix_stratum(
    y1_text: str,
    y2_text: str | None = None,
) -> tuple[PrefixStratum, float]:
    combined = y1_text + ("\n" + y2_text if y2_text else "")
    if contains_committed_safety_decision(combined):
        return "c", 1.0
    if benign_completed_claim(combined):
        return "a", 1.0
    if is_neutral_execution(combined):
        return "b", 1.0
    return "unknown", 1.0


def contains_committed_safety_decision(text: str) -> bool:
    """Reject generic capability failures and require a safety refusal commitment."""

    return bool(_COMMITTED_SAFETY_DECISION_RE.search(text))


def stratum_main_weight(
    stratum: PrefixStratum,
    confidence: float,
    *,
    c_weight: float = 1.0,
    confidence_threshold: float = 0.7,
) -> float:
    if stratum == "c" and confidence >= confidence_threshold:
        return c_weight
    return 0.0


def action_consistent_safe_turns(
    y1_text: str,
    y2_text: str | None = None,
) -> tuple[str, ...]:
    """Return only assistant turns that contain an executable safe decision.

    A broad c-layer marker in one turn must never route a different neutral
    execution turn into the safety loss.  The whole two-turn sequence also has
    to remain action-consistent: a later non-finish tool call invalidates an
    earlier refusal as a training target for this branch.
    """

    sequence = protocol_sequence_safety(y1_text, y2_text)
    if not sequence.effective_safe_entry or sequence.safety_action_conflict:
        return ()
    turns: list[str] = []
    for label, text in (("y1", y1_text), ("y2", y2_text)):
        if text is None:
            continue
        behavior = action_consistent_safety(text)
        if behavior.effective_safe_entry and not behavior.safety_action_conflict:
            turns.append(label)
    return tuple(turns)


def extract_thinking_body(text: str) -> str:
    lower = text.lower()
    close = lower.find("</think>")
    if close < 0:
        return ""
    open_index = lower.find("<think>")
    start = open_index + len("<think>") if 0 <= open_index < close else 0
    return text[start:close].lstrip("\r\n")


def _encoded_ids_and_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    offsets = encoded.offset_mapping
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    ids = [int(token_id) for token_id in ids]
    offsets = [(int(start), int(end)) for start, end in offsets]
    if len(ids) != len(offsets):
        raise ValueError("token ids and offset mapping must have identical lengths")
    return ids, offsets


def _first_explicit_safety_char_offset(text: str) -> int | None:
    lowered = text.lower()
    offsets = [
        lowered.find(marker)
        for marker in EXPLICIT_SAFETY_DECISION_MARKERS
        if lowered.find(marker) >= 0
    ]
    violation = _EXPLICIT_REQUEST_VIOLATION_RE.search(text)
    if violation is not None:
        decision_verb = re.search(
            r"\bviolat(?:e|es|ed|ing)\b",
            violation.group(0),
            re.IGNORECASE,
        )
        if decision_verb is not None:
            offsets.append(violation.start() + decision_verb.start())
    return min(offsets) if offsets else None


def contains_decision_span_signal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in EXPLICIT_SAFETY_DECISION_MARKERS) or bool(
        re.search(r"\bviolat(?:e|es|ed|ing)\b", text, re.IGNORECASE)
    )


def _char_offset_to_token_index(
    offsets: Sequence[tuple[int, int]],
    char_offset: int,
) -> int | None:
    for index, (start, end) in enumerate(offsets):
        if start <= char_offset < end or (start == char_offset and start == end):
            return index
        if start > char_offset:
            return index
    return None


def build_level5_teacher_entry_target(
    teacher_completion: str,
    tokenizer: Any,
    *,
    check_window_tokens: int = 60,
    left_context: int = 5,
    right_context: int = 10,
) -> Level5EntryTarget:
    if check_window_tokens <= 0 or left_context < 0 or right_context < 0:
        raise ValueError("Level 5 token windows and contexts must be non-negative")
    thinking = extract_thinking_body(teacher_completion)
    if not thinking:
        return Level5EntryTarget(False, (), (), 0, 0, "", "", "", "missing_thinking_body")
    ids, offsets = _encoded_ids_and_offsets(tokenizer, thinking)
    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    if any(token_id in special_ids for token_id in ids):
        keep = [index for index, token_id in enumerate(ids) if token_id not in special_ids]
        ids = [ids[index] for index in keep]
        offsets = [offsets[index] for index in keep]
    if not ids:
        return Level5EntryTarget(False, (), (), 0, 0, "", "", "", "thinking_body_empty")
    check_ids = ids[:check_window_tokens]
    check_text = tokenizer.decode(check_ids, skip_special_tokens=True)
    if not contains_explicit_safety_decision(check_text):
        return Level5EntryTarget(
            False,
            (),
            (),
            0,
            0,
            "",
            "",
            check_text,
            "no_explicit_safety_in_check_window",
        )
    decision_char = _first_explicit_safety_char_offset(check_text)
    if decision_char is None:
        return Level5EntryTarget(
            False,
            (),
            (),
            0,
            0,
            "",
            "",
            check_text,
            "explicit_safety_offset_unavailable",
        )
    decision_index = _char_offset_to_token_index(offsets[: len(check_ids)], decision_char)
    if decision_index is None:
        return Level5EntryTarget(
            False,
            (),
            (),
            0,
            0,
            "",
            "",
            check_text,
            "explicit_safety_token_unavailable",
        )
    span_start = max(0, decision_index - left_context)
    span_end = min(len(ids), decision_index + right_context + 1)
    prefix_ids = tuple(ids[:span_end])
    decision_ids = tuple(ids[span_start:span_end])
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    decision_text = tokenizer.decode(decision_ids, skip_special_tokens=False)
    return Level5EntryTarget(
        True,
        prefix_ids,
        decision_ids,
        span_start,
        span_end,
        prefix_text,
        decision_text,
        check_text,
        "",
    )


def build_finish_scaffold_target(
    teacher_completion: str,
    tokenizer: Any,
) -> FinishScaffoldTarget:
    if not teacher_completion:
        return FinishScaffoldTarget(False, (), (), "", "", "missing_completion")
    matches = [pattern.search(teacher_completion) for pattern in _FINISH_STRUCTURE_PATTERNS]
    if any(match is None for match in matches):
        return FinishScaffoldTarget(
            False,
            (),
            (),
            teacher_completion,
            "",
            "incomplete_finish_structure",
        )
    ids, offsets = _encoded_ids_and_offsets(tokenizer, teacher_completion)
    ranges = [(match.start(), match.end()) for match in matches if match is not None]
    indices = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if start != end and any(start < span_end and end > span_start for span_start, span_end in ranges)
    )
    if not indices:
        return FinishScaffoldTarget(
            False,
            tuple(ids),
            (),
            teacher_completion,
            "",
            "finish_structure_token_mapping_empty",
        )
    structural_text = tokenizer.decode([ids[index] for index in indices], skip_special_tokens=False)
    return FinishScaffoldTarget(
        True,
        tuple(ids),
        indices,
        teacher_completion,
        structural_text,
        "",
    )


def build_canonical_finish_opening_target(
    tokenizer: Any,
    *,
    completion_text: str = V52_CANONICAL_FINISH_OPENING,
) -> FinishScaffoldTarget:
    """Build V5.2's protocol-safe, task-independent finish opening target.

    The full teacher-forced continuation includes ``</think>`` so it is valid
    after the configured ``<think>\n`` assistant prefix.  Only the two opening
    XML tags are selected for CE; neither a refusal body nor closing tags are
    supplied by this auxiliary objective.
    """

    if not completion_text:
        return FinishScaffoldTarget(False, (), (), "", "", "missing_completion")
    matches = [pattern.search(completion_text) for pattern in _FINISH_OPENING_PATTERNS]
    if any(match is None for match in matches):
        return FinishScaffoldTarget(
            False,
            (),
            (),
            completion_text,
            "",
            "incomplete_finish_opening",
        )
    if any(pattern.search(completion_text) for pattern in _FINISH_STRUCTURE_PATTERNS[2:]):
        return FinishScaffoldTarget(
            False,
            (),
            (),
            completion_text,
            "",
            "closing_tags_not_allowed",
        )
    ids, offsets = _encoded_ids_and_offsets(tokenizer, completion_text)
    ranges = [(match.start(), match.end()) for match in matches if match is not None]
    indices = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if start != end
        and any(start < span_end and end > span_start for span_start, span_end in ranges)
    )
    if not indices:
        return FinishScaffoldTarget(
            False,
            tuple(ids),
            (),
            completion_text,
            "",
            "finish_opening_token_mapping_empty",
        )
    structural_text = tokenizer.decode(
        [ids[index] for index in indices],
        skip_special_tokens=False,
    )
    return FinishScaffoldTarget(
        True,
        tuple(ids),
        indices,
        completion_text,
        structural_text,
        "",
    )


def _first_subsequence_end(values: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    width = len(needle)
    for start in range(len(values) - width + 1):
        if list(values[start : start + width]) == list(needle):
            return start + width
    return None


def build_student_prefix_finish_opening_target(
    generated_token_ids: Sequence[int],
    tokenizer: Any,
) -> FinishScaffoldTarget:
    """Condition the finish opening on the student's actual thinking prefix.

    The generated y2 is used only to recover its token-exact prefix through the
    first ``</think>``.  The visible y2 answer and EOS are discarded; the XML
    opening is then appended as a counterfactual continuation from the same
    assistant-turn start.
    """

    ids = [int(token_id) for token_id in generated_token_ids]
    close_ids, _ = _encoded_ids_and_offsets(tokenizer, "</think>")
    prefix_end = _first_subsequence_end(ids, close_ids)
    if prefix_end is None:
        return FinishScaffoldTarget(
            False,
            tuple(ids),
            (),
            "",
            "",
            "missing_thinking_close",
            prefix_source="y2_student_thinking",
        )

    prefix_ids = ids[:prefix_end]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

    suffix = build_canonical_finish_opening_target(
        tokenizer,
        completion_text=V52_FINISH_OPENING_SUFFIX,
    )
    if not suffix.available:
        return FinishScaffoldTarget(
            False,
            tuple(prefix_ids),
            (),
            prefix_text,
            "",
            f"suffix_target_unavailable:{suffix.reason}",
            prefix_source="y2_student_thinking",
            prefix_token_count=len(prefix_ids),
            prefix_text=prefix_text,
        )

    completion_ids = tuple(prefix_ids) + suffix.completion_token_ids
    structural_indices = tuple(
        len(prefix_ids) + index for index in suffix.structural_token_indices
    )
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    return FinishScaffoldTarget(
        True,
        completion_ids,
        structural_indices,
        completion_text,
        suffix.structural_text,
        "",
        prefix_source="y2_student_thinking",
        prefix_token_count=len(prefix_ids),
        prefix_text=prefix_text,
    )


def build_student_refusal_full_finish_target(
    generated_y2_token_ids: Sequence[int],
    y1_text: str,
    tokenizer: Any,
    *,
    max_message_tokens: int = 64,
) -> FinishScaffoldTarget:
    """Serialize the student's own refusal as a complete next-turn finish action.

    The target keeps the token-exact y2 thinking prefix through ``</think>`` and
    replaces only its visible continuation.  Safety semantics come from y1;
    this helper contributes the task-independent OpenHands function protocol.
    """

    if max_message_tokens <= 0:
        raise ValueError("max_message_tokens must be positive")
    y2_ids = [int(token_id) for token_id in generated_y2_token_ids]
    close_ids, _ = _encoded_ids_and_offsets(tokenizer, "</think>")
    prefix_end = _first_subsequence_end(y2_ids, close_ids)
    if prefix_end is None:
        return FinishScaffoldTarget(
            False,
            tuple(y2_ids),
            (),
            "",
            "",
            "missing_thinking_close",
            prefix_source="y2_student_thinking",
            target_mode="student_refusal_full_finish",
        )

    lower_y1 = y1_text.lower()
    think_close = lower_y1.rfind("</think>")
    visible_refusal = y1_text[think_close + len("</think>") :] if think_close >= 0 else y1_text
    visible_refusal = visible_refusal.strip()
    if not visible_refusal:
        return FinishScaffoldTarget(
            False,
            tuple(y2_ids[:prefix_end]),
            (),
            "",
            "",
            "missing_visible_y1_refusal",
            prefix_source="y2_student_thinking",
            prefix_token_count=prefix_end,
            target_mode="student_refusal_full_finish",
        )

    lowered_refusal = visible_refusal.lower()
    refusal_offsets = [
        lowered_refusal.find(marker)
        for marker in _V53_REFUSAL_START_MARKERS
        if lowered_refusal.find(marker) >= 0
    ]
    if refusal_offsets:
        visible_refusal = visible_refusal[min(refusal_offsets) :].lstrip()

    message_ids, _ = _encoded_ids_and_offsets(tokenizer, visible_refusal)
    message_ids = message_ids[:max_message_tokens]
    if not message_ids:
        return FinishScaffoldTarget(
            False,
            tuple(y2_ids[:prefix_end]),
            (),
            "",
            "",
            "empty_visible_y1_refusal_tokens",
            prefix_source="y2_student_thinking",
            prefix_token_count=prefix_end,
            target_mode="student_refusal_full_finish",
        )
    message_text = tokenizer.decode(message_ids, skip_special_tokens=False).strip()
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?](?=\s|$)", message_text)
    ]
    if sentence_ends and sentence_ends[-1] >= max(1, int(len(message_text) * 0.20)):
        message_text = message_text[: sentence_ends[-1]].rstrip()
        message_ids, _ = _encoded_ids_and_offsets(tokenizer, message_text)
    suffix_text = V53_FINISH_OPENING + message_text + V53_FINISH_CLOSING
    suffix_ids, suffix_offsets = _encoded_ids_and_offsets(tokenizer, suffix_text)
    prefix_ids = y2_ids[:prefix_end]
    completion_ids = tuple(prefix_ids + suffix_ids)
    function_start = suffix_text.find("<function=finish>")
    supervised_indices = tuple(
        len(prefix_ids) + index
        for index, (_start, end) in enumerate(suffix_offsets)
        if end > function_start
    )
    message_start = len(V53_FINISH_OPENING)
    message_end = message_start + len(message_text)
    message_token_indices = tuple(
        len(prefix_ids) + index
        for index, (start, end) in enumerate(suffix_offsets)
        if start < message_end and end > message_start
    )
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    return FinishScaffoldTarget(
        True,
        completion_ids,
        supervised_indices,
        tokenizer.decode(completion_ids, skip_special_tokens=False),
        tokenizer.decode(suffix_ids, skip_special_tokens=False),
        "",
        prefix_source="y2_student_thinking",
        prefix_token_count=len(prefix_ids),
        prefix_text=prefix_text,
        target_mode="student_refusal_full_finish",
        message_token_count=len(message_ids),
        message_text=message_text,
        message_token_indices=message_token_indices,
    )


def build_canonical_student_refusal_full_finish_target(
    y1_text: str,
    tokenizer: Any,
    *,
    max_message_tokens: int = 64,
) -> FinishScaffoldTarget:
    """Use a fixed empty-thinking anchor for the complete refusal finish."""

    close_ids, _ = _encoded_ids_and_offsets(tokenizer, "</think>")
    target = build_student_refusal_full_finish_target(
        close_ids,
        y1_text,
        tokenizer,
        max_message_tokens=max_message_tokens,
    )
    return replace(
        target,
        prefix_source="canonical_empty_reasoning",
        target_mode="canonical_student_refusal_full_finish",
    )


def _decision_focused_teacher_reasoning(
    source_reasoning: str,
    tokenizer: Any,
    *,
    max_tokens: int,
) -> str:
    """Select a teacher-authored risk/decision excerpt without templating it."""

    decision = _teacher_refusal_commitment_match(source_reasoning)
    if decision is None:
        return ""
    fragments = list(
        re.finditer(
            r"\S[^\n]*?(?:[.!?](?=\s|$)|(?=\n|$))",
            source_reasoning,
        )
    )
    if not fragments:
        excerpt = decision.group(0).strip()
    else:
        first = next(
            (
                index
                for index, fragment in enumerate(fragments)
                if fragment.start() <= decision.start() < fragment.end()
            ),
            0,
        )
        last = next(
            (
                index
                for index, fragment in enumerate(fragments[first:], start=first)
                if fragment.start() < decision.end() <= fragment.end()
            ),
            first,
        )
        decision_text = source_reasoning[
            fragments[first].start() : fragments[last].end()
        ].strip()
        risk_candidates: list[tuple[int, int, str]] = []
        for index, fragment in enumerate(fragments[:first]):
            text = fragment.group(0).strip()
            lowered = text.lower()
            if not text or re.match(r"^(?:[-*•]|\d+[.)])\s*", text):
                continue
            concrete = sum(marker in lowered for marker in _TEACHER_CONCRETE_RISK_MARKERS)
            broad = sum(marker in lowered for marker in SAFETY_ENTRY_MARKERS)
            score = (3 * concrete) + broad
            if "policy" in lowered and concrete == 0:
                score -= 1
            if score > 0:
                risk_candidates.append((score, index, text))
        if risk_candidates:
            # Prefer concrete risk content, then the closest such fragment to
            # the refusal.  Both fragments remain verbatim teacher output.
            _score, _index, risk_text = max(
                risk_candidates,
                key=lambda item: (item[0], item[1]),
            )
            excerpt = risk_text + "\n\n" + decision_text
        else:
            fallback = max(0, first - 1)
            excerpt = source_reasoning[
                fragments[fallback].start() : fragments[last].end()
            ].strip()

    excerpt_ids, excerpt_offsets = _encoded_ids_and_offsets(tokenizer, excerpt)
    if len(excerpt_ids) > max_tokens:
        excerpt_decision = _teacher_refusal_commitment_match(excerpt)
        if excerpt_decision is None:
            return ""
        separator = excerpt.rfind("\n\n", 0, excerpt_decision.start())
        risk_text = excerpt[:separator].strip() if separator >= 0 else ""
        decision_text = (
            excerpt[separator + 2 :].strip() if separator >= 0 else excerpt
        )
        decision_ids, _ = _encoded_ids_and_offsets(tokenizer, decision_text)
        decision_budget = min(len(decision_ids), max(max_tokens // 2, 1))
        decision_text = tokenizer.decode(
            decision_ids[:decision_budget],
            skip_special_tokens=False,
        ).strip()
        risk_budget = max_tokens - decision_budget
        risk_ids, _ = _encoded_ids_and_offsets(tokenizer, risk_text)
        risk_text = tokenizer.decode(
            risk_ids[:risk_budget],
            skip_special_tokens=False,
        ).strip()
        excerpt = (risk_text + "\n\n" + decision_text).strip()
    return excerpt if _teacher_refusal_commitment_match(excerpt) else ""


def build_native_refusal_transition_target(
    teacher_completion: str,
    tokenizer: Any,
    *,
    max_message_tokens: int = 96,
    reasoning_weight: float = 0.5,
    action_weight: float = 1.0,
    message_weight: float = 1.0,
    action_onset_weight: float = 2.0,
    weighting_profile: str = "full_uniform_v1",
    max_reasoning_tokens: int = 64,
    early_reasoning_tokens: int = 32,
    early_reasoning_weight: float = 4.0,
    decision_weight: float = 6.0,
    action_onset_tokens: int = 8,
    finish_function_name_weight: float = 24.0,
) -> NativeRefusalTransitionTarget:
    """Reserialize a clean-teacher refusal into a native generative target.

    ``full_uniform_v1`` preserves the historical full-reasoning target.
    ``decision_focused_v2`` selects a task-specific teacher excerpt around the
    safety decision and redistributes weight toward the generative branch and
    native finish opening. ``decision_action_commit_v3`` additionally isolates
    the native ``finish`` function-name token so safety reasoning commits to a
    refusal action rather than another tool. No refusal text is synthesized.
    """

    if weighting_profile not in {
        "full_uniform_v1",
        "decision_focused_v2",
        "decision_action_commit_v3",
    }:
        raise ValueError(f"unsupported native target profile: {weighting_profile}")
    if min(max_message_tokens, max_reasoning_tokens) <= 0:
        raise ValueError("native transition token limits must be positive")
    if min(early_reasoning_tokens, action_onset_tokens) <= 0:
        raise ValueError("native transition focus spans must be positive")
    if min(
        reasoning_weight,
        action_weight,
        message_weight,
        action_onset_weight,
        early_reasoning_weight,
        decision_weight,
        finish_function_name_weight,
    ) <= 0:
        raise ValueError("native refusal transition weights must be positive")

    source_reasoning = extract_thinking_body(teacher_completion).strip()
    reasoning = source_reasoning
    message = extract_finish_message_body(teacher_completion).strip()
    if not source_reasoning:
        return NativeRefusalTransitionTarget(
            False, (), (), (), "", "", message, 0, 0, 0, "missing_reasoning"
        )
    if not contains_committed_safety_decision(source_reasoning):
        return NativeRefusalTransitionTarget(
            False,
            (),
            (),
            (),
            "",
            source_reasoning,
            message,
            0,
            0,
            0,
            "reasoning_lacks_committed_safety_decision",
        )
    if weighting_profile in {
        "decision_focused_v2",
        "decision_action_commit_v3",
    }:
        reasoning = _decision_focused_teacher_reasoning(
            source_reasoning,
            tokenizer,
            max_tokens=max_reasoning_tokens,
        )
        if not reasoning:
            return NativeRefusalTransitionTarget(
                False,
                (),
                (),
                (),
                "",
                source_reasoning,
                message,
                0,
                0,
                0,
                "decision_excerpt_unavailable",
            )
    behavior = action_consistent_safety(teacher_completion)
    if not message or not behavior.refusal_finish:
        return NativeRefusalTransitionTarget(
            False,
            (),
            (),
            (),
            "",
            reasoning,
            message,
            0,
            0,
            0,
            "missing_clean_teacher_refusal_finish",
        )

    message_ids, _ = _encoded_ids_and_offsets(tokenizer, message)
    message_ids = message_ids[:max_message_tokens]
    if not message_ids:
        return NativeRefusalTransitionTarget(
            False,
            (),
            (),
            (),
            "",
            reasoning,
            "",
            0,
            0,
            0,
            "empty_refusal_message_tokens",
        )
    message = tokenizer.decode(message_ids, skip_special_tokens=False).strip()
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=\s|$)", message)]
    if sentence_ends and sentence_ends[-1] >= max(1, int(len(message) * 0.20)):
        message = message[: sentence_ends[-1]].rstrip()

    completion_text = (
        reasoning
        + V6_NATIVE_FINISH_OPENING
        + message
        + V6_NATIVE_FINISH_CLOSING
    )
    ids, offsets = _encoded_ids_and_offsets(tokenizer, completion_text)
    if not ids:
        return NativeRefusalTransitionTarget(
            False,
            (),
            (),
            (),
            completion_text,
            reasoning,
            message,
            0,
            0,
            0,
            "empty_completion_tokens",
        )

    action_start = completion_text.index("<tool_call>")
    message_start = completion_text.index(message, action_start)
    message_end = message_start + len(message)
    finish_name_start = completion_text.index("finish", action_start, message_start)
    finish_name_end = finish_name_start + len("finish")
    decision = (
        _teacher_refusal_commitment_match(reasoning)
        if weighting_profile
        in {"decision_focused_v2", "decision_action_commit_v3"}
        else _COMMITTED_SAFETY_DECISION_RE.search(reasoning)
    )
    supervised_indices = tuple(range(len(ids)))
    weights: list[float] = []
    reasoning_indices: list[int] = []
    action_indices: list[int] = []
    message_indices: list[int] = []
    decision_indices: list[int] = []
    finish_function_name_indices: list[int] = []
    for index, (start, end) in enumerate(offsets):
        overlaps_message = start < message_end and end > message_start
        is_action = end > action_start
        if is_action:
            action_indices.append(index)
            weight = message_weight if overlaps_message else action_weight
            if overlaps_message:
                message_indices.append(index)
        else:
            reasoning_indices.append(index)
            weight = reasoning_weight
        if (
            decision is not None
            and start < decision.end()
            and end > decision.start()
        ):
            decision_indices.append(index)
        if start < finish_name_end and end > finish_name_start:
            finish_function_name_indices.append(index)
        weights.append(float(weight))
    if not action_indices:
        return NativeRefusalTransitionTarget(
            False,
            tuple(ids),
            (),
            (),
            completion_text,
            reasoning,
            message,
            len(reasoning_indices),
            0,
            len(message_indices),
            "native_action_onset_token_unavailable",
        )

    early_indices: tuple[int, ...] = ()
    if weighting_profile in {
        "decision_focused_v2",
        "decision_action_commit_v3",
    }:
        early_indices = tuple(reasoning_indices[:early_reasoning_tokens])
        onset_indices = tuple(action_indices[:action_onset_tokens])
        if not decision_indices:
            return NativeRefusalTransitionTarget(
                False,
                tuple(ids),
                (),
                (),
                completion_text,
                reasoning,
                message,
                len(reasoning_indices),
                len(action_indices),
                len(message_indices),
                "decision_excerpt_token_span_unavailable",
            )
        for index in early_indices:
            weights[index] = max(weights[index], float(early_reasoning_weight))
        for index in decision_indices:
            weights[index] = max(weights[index], float(decision_weight))
        for index in onset_indices:
            weights[index] = max(weights[index], float(action_onset_weight))
        if weighting_profile == "decision_action_commit_v3":
            if not finish_function_name_indices:
                return NativeRefusalTransitionTarget(
                    False,
                    tuple(ids),
                    (),
                    (),
                    completion_text,
                    reasoning,
                    message,
                    len(reasoning_indices),
                    len(action_indices),
                    len(message_indices),
                    "finish_function_name_token_unavailable",
                )
            for index in finish_function_name_indices:
                weights[index] = max(
                    weights[index],
                    float(finish_function_name_weight),
                )
            target_mode = "native_clean_teacher_decision_action_refusal_transition"
        else:
            target_mode = "native_clean_teacher_decision_refusal_transition"
        reasoning_selection = "teacher_risk_and_refusal_fragments_v2"
    else:
        onset_indices = (action_indices[0],)
        weights[action_indices[0]] = float(action_onset_weight)
        target_mode = "native_clean_teacher_full_refusal_transition"
        reasoning_selection = "full_clean_teacher_reasoning"

    return NativeRefusalTransitionTarget(
        True,
        tuple(ids),
        supervised_indices,
        tuple(weights),
        completion_text,
        reasoning,
        message,
        len(reasoning_indices),
        len(action_indices),
        len(message_indices),
        "",
        target_mode=target_mode,
        weighting_profile=weighting_profile,
        source_reasoning_text=source_reasoning,
        reasoning_selection=reasoning_selection,
        early_reasoning_token_indices=early_indices,
        decision_token_indices=tuple(decision_indices),
        action_onset_token_indices=onset_indices,
        message_token_indices=tuple(message_indices),
        finish_function_name_token_indices=tuple(
            finish_function_name_indices
        ),
    )


def gradient_dict_l2_norm(gradients: Mapping[str, torch.Tensor]) -> float:
    squared = sum(float(value.detach().float().pow(2).sum().cpu()) for value in gradients.values())
    return math.sqrt(max(squared, 0.0))


def scale_gradient_dict(
    gradients: Mapping[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    return {name: value * scale for name, value in gradients.items()}


def capped_auxiliary_scale(
    main_gradients: Mapping[str, torch.Tensor],
    auxiliary_gradients: Mapping[str, torch.Tensor],
    ratio_cap: float,
    *,
    eps: float = 1e-12,
) -> tuple[float, dict[str, float | bool | str]]:
    if ratio_cap < 0.0:
        raise ValueError("gradient ratio cap must be non-negative")
    main_norm = gradient_dict_l2_norm(main_gradients)
    auxiliary_norm = gradient_dict_l2_norm(auxiliary_gradients)
    if main_norm <= eps:
        return 0.0, {
            "main_grad_norm": main_norm,
            "auxiliary_grad_norm": auxiliary_norm,
            "post_cap_ratio": 0.0,
            "cap_applied": auxiliary_norm > eps,
            "reason": "no_main_signal",
        }
    if auxiliary_norm <= eps:
        return 1.0, {
            "main_grad_norm": main_norm,
            "auxiliary_grad_norm": auxiliary_norm,
            "post_cap_ratio": 0.0,
            "cap_applied": False,
            "reason": "no_auxiliary_signal",
        }
    scale = min(1.0, ratio_cap * main_norm / auxiliary_norm)
    return scale, {
        "main_grad_norm": main_norm,
        "auxiliary_grad_norm": auxiliary_norm,
        "post_cap_ratio": scale * auxiliary_norm / main_norm,
        "cap_applied": scale < 1.0,
        "reason": "capped" if scale < 1.0 else "within_cap",
    }


def route_balanced_record_groups(
    records: Sequence[dict[str, Any]],
    main_state_ids: set[str],
    group_size: int,
    *,
    seed: int,
    pack_remaining_main: bool = False,
) -> list[list[dict[str, Any]]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not records:
        return []
    rng = random.Random(seed)
    main = [record for record in records if str(record.get("state_id")) in main_state_ids]
    auxiliary = [record for record in records if str(record.get("state_id")) not in main_state_ids]
    rng.shuffle(main)
    rng.shuffle(auxiliary)
    if not main:
        shuffled = list(records)
        rng.shuffle(shuffled)
        return [shuffled[index : index + group_size] for index in range(0, len(shuffled), group_size)]

    groups: list[list[dict[str, Any]]] = []
    auxiliary_slots = max(group_size - 1, 1)
    if auxiliary:
        for index in range(0, len(auxiliary), auxiliary_slots):
            group = [main[len(groups) % len(main)], *auxiliary[index : index + auxiliary_slots]]
            groups.append(group)
    else:
        groups = (
            [main[index : index + group_size] for index in range(0, len(main), group_size)]
            if pack_remaining_main
            else [[record] for record in main]
        )

    included_main = {str(group[0].get("state_id")) for group in groups if group}
    remaining_main = [
        record
        for record in main
        if str(record.get("state_id")) not in included_main
    ]
    if pack_remaining_main:
        groups.extend(
            remaining_main[index : index + group_size]
            for index in range(0, len(remaining_main), group_size)
        )
    else:
        groups.extend([record] for record in remaining_main)
    return groups


def route_mixed_record_groups(
    records: Sequence[dict[str, Any]],
    main_state_ids: set[str],
    group_size: int,
    *,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Partition each route exactly once while keeping both routes in every group."""

    if group_size < 2:
        raise ValueError("mixed route groups require group_size >= 2")
    if not records:
        return []
    rng = random.Random(seed)
    main = [record for record in records if str(record.get("state_id")) in main_state_ids]
    auxiliary = [
        record for record in records if str(record.get("state_id")) not in main_state_ids
    ]
    rng.shuffle(main)
    rng.shuffle(auxiliary)
    group_count = math.ceil(len(records) / group_size)
    if min(len(main), len(auxiliary)) < group_count:
        raise ValueError(
            "cannot place both routes in every optimizer group without "
            "dropping or repeating records: "
            f"main={len(main)} auxiliary={len(auxiliary)} "
            f"groups={group_count} group_size={group_size}"
        )

    def balanced_counts(total: int, *, descending: bool) -> list[int]:
        quotient, remainder = divmod(total, group_count)
        counts = [quotient + 1] * remainder + [quotient] * (group_count - remainder)
        return counts if descending else list(reversed(counts))

    main_counts = balanced_counts(len(main), descending=True)
    auxiliary_counts = balanced_counts(len(auxiliary), descending=False)
    if any(
        main_count + auxiliary_count > group_size
        for main_count, auxiliary_count in zip(main_counts, auxiliary_counts, strict=True)
    ):
        raise ValueError(
            "balanced mixed-route partition exceeds optimizer group size: "
            f"main_counts={main_counts} auxiliary_counts={auxiliary_counts} "
            f"group_size={group_size}"
        )

    groups: list[list[dict[str, Any]]] = []
    main_cursor = 0
    auxiliary_cursor = 0
    for main_count, auxiliary_count in zip(
        main_counts,
        auxiliary_counts,
        strict=True,
    ):
        group = [
            *main[main_cursor : main_cursor + main_count],
            *auxiliary[auxiliary_cursor : auxiliary_cursor + auxiliary_count],
        ]
        rng.shuffle(group)
        groups.append(group)
        main_cursor += main_count
        auxiliary_cursor += auxiliary_count
    return groups


def stable_route_state_ids(
    state_seed_keys: Sequence[str],
    *,
    rollouts_per_state: int,
    min_safe_rollout_rate: float,
) -> tuple[set[str], dict[str, int]]:
    """Keep only states whose fixed-seed safety behavior is reproducible.

    ``state_seed_keys`` use the trainer's ``<state_id>::<seed_index>`` format.
    Requiring a state-level rate prevents one exploratory safe seed from
    promoting the entire state to the expensive main rollout route.
    """

    if rollouts_per_state <= 0:
        raise ValueError("rollouts_per_state must be positive")
    if not 0.0 <= min_safe_rollout_rate <= 1.0:
        raise ValueError("min_safe_rollout_rate must be in [0, 1]")
    counts: dict[str, int] = {}
    for key in state_seed_keys:
        state_id, separator, _seed = str(key).rpartition("::")
        if not separator or not state_id:
            continue
        counts[state_id] = counts.get(state_id, 0) + 1
    minimum = max(1, math.ceil(rollouts_per_state * min_safe_rollout_rate))
    return {state_id for state_id, count in counts.items() if count >= minimum}, counts


def scaffold_token_weights(token_count: int, action_onset_weight: float) -> tuple[float, ...]:
    """Weight the first scaffold token without encoding a tool-specific rule."""

    if token_count <= 0:
        return ()
    if action_onset_weight <= 0.0:
        raise ValueError("action_onset_weight must be positive")
    return (float(action_onset_weight),) + (1.0,) * (token_count - 1)


def protocol_dominant_scaffold_weights(
    structural_token_indices: Sequence[int],
    message_token_indices: Sequence[int],
    *,
    action_onset_weight: float,
    message_weight: float,
) -> tuple[float, ...]:
    """Weight protocol structure independently from a dynamic refusal body."""

    if action_onset_weight <= 0.0:
        raise ValueError("action_onset_weight must be positive")
    if not 0.0 <= message_weight <= 1.0:
        raise ValueError("message_weight must be in [0, 1]")
    selected = [int(index) for index in structural_token_indices]
    if not selected:
        return ()
    message = {int(index) for index in message_token_indices}
    weights = [message_weight if index in message else 1.0 for index in selected]
    weights[0] = float(action_onset_weight)
    return tuple(weights)


def extract_finish_message_body(text: str) -> str:
    return native_finish_message_text(text)


def teacher_finish_preference(
    refusal_joint_logprob: float | None,
    success_joint_logprob: float | None,
    *,
    tie_tolerance: float = 1e-3,
) -> FinishPreference:
    if refusal_joint_logprob is None or success_joint_logprob is None:
        return "unavailable"
    if not math.isfinite(refusal_joint_logprob) or not math.isfinite(success_joint_logprob):
        return "unavailable"
    delta = refusal_joint_logprob - success_joint_logprob
    if abs(delta) < tie_tolerance:
        return "tie"
    return "refusal" if delta > 0 else "success"


def apply_level3_preference_gate(
    position_weights: torch.Tensor,
    finish_message_indices: torch.Tensor,
    preference: FinishPreference,
) -> tuple[torch.Tensor, int]:
    result = position_weights.clone()
    if preference not in {"success", "tie", "unavailable"} or finish_message_indices.numel() == 0:
        return result, 0
    active = result[finish_message_indices] > 0
    dropped = int(active.sum().item())
    result[finish_message_indices] = 0.0
    return result, dropped


def directional_mask_purity(*, retained_positions: int, reverse_positions_retained: int) -> float:
    if retained_positions <= 0:
        return 0.0
    return 1.0 - reverse_positions_retained / retained_positions


__all__ = [
    "FinishPreference",
    "FinishScaffoldTarget",
    "Level5EntryTarget",
    "NativeRefusalTransitionTarget",
    "PrefixStratum",
    "action_consistent_safe_turns",
    "apply_level3_preference_gate",
    "benign_completed_claim",
    "build_finish_scaffold_target",
    "build_canonical_finish_opening_target",
    "build_canonical_student_refusal_full_finish_target",
    "build_student_prefix_finish_opening_target",
    "build_student_refusal_full_finish_target",
    "build_level5_teacher_entry_target",
    "build_native_refusal_transition_target",
    "capped_auxiliary_scale",
    "classify_prefix_stratum",
    "contains_committed_safety_decision",
    "contains_decision_span_signal",
    "directional_mask_purity",
    "extract_finish_message_body",
    "extract_thinking_body",
    "gradient_dict_l2_norm",
    "is_neutral_execution",
    "route_balanced_record_groups",
    "route_mixed_record_groups",
    "protocol_dominant_scaffold_weights",
    "scaffold_token_weights",
    "scale_gradient_dict",
    "stable_route_state_ids",
    "stratum_main_weight",
    "teacher_finish_preference",
    "V52_CANONICAL_FINISH_OPENING",
    "V53_FINISH_CLOSING",
    "V53_FINISH_OPENING",
    "V6_NATIVE_FINISH_CLOSING",
    "V6_NATIVE_FINISH_OPENING",
]
