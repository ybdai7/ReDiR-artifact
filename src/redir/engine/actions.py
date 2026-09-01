"""Pure helpers for V4.6 entry/action bridge training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
import re
from typing import Any, Sequence


@dataclass(frozen=True)
class V46ActionSpans:
    scaffold: tuple[tuple[int, int], ...]
    message_body: tuple[int, int] | None


def task_balanced_records(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    max_states_per_task: int,
) -> list[dict[str, Any]]:
    """Sample an equal number of states per task and interleave tasks."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_key") or "unknown")].append(record)

    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    for task, task_records in grouped.items():
        shuffled = list(task_records)
        rng.shuffle(shuffled)
        selected[task] = shuffled[:max_states_per_task] if max_states_per_task > 0 else shuffled

    tasks = sorted(selected)
    rng.shuffle(tasks)
    ordered: list[dict[str, Any]] = []
    depth = max((len(selected[task]) for task in tasks), default=0)
    for index in range(depth):
        round_tasks = [task for task in tasks if index < len(selected[task])]
        rng.shuffle(round_tasks)
        ordered.extend(selected[task][index] for task in round_tasks)
    return ordered


def v6_task_balanced_benign_tool_records(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    state_count: int,
) -> list[dict[str, Any]]:
    """Select one audited productive native tool state per benign task.

    Legacy datasets do not carry ``v6_native_real_state`` and retain the old
    task-balanced behavior.  A native-real dataset must be homogeneous and
    uses only non-finish tool states from judge-COMPLETE trajectories.
    """

    native_flags = [bool(record.get("v6_native_real_state")) for record in records]
    if any(native_flags):
        groups = v6_ordered_benign_tool_candidate_groups(records, seed=seed)
        return [group[0] for group in groups[:state_count] if group]

    return task_balanced_records(
        records,
        seed=seed,
        max_states_per_task=1,
    )[:state_count]


def v6_ordered_benign_tool_candidate_groups(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Return a frozen candidate order for each audited native benign task.

    The first record in every group exactly reproduces the historical
    ``v6_task_balanced_benign_tool_records`` selection.  Remaining records are
    deterministic initialization-admission fallbacks; training is not allowed
    to reorder them from post-update behavior.
    """

    native_flags = [bool(record.get("v6_native_real_state")) for record in records]
    if not records or not any(native_flags):
        raise ValueError("V6 native benign candidate groups require native-real states")
    if not all(native_flags):
        raise ValueError("V6 benign pool mixes native-real and legacy states")

    candidates = [
        record
        for record in records
        if str(record.get("source_post_eval_outcome") or "") == "COMPLETE"
        and not bool(record.get("observed_contains_refusal"))
        and str(record.get("observed_first_function_name") or "")
        not in {"", "finish"}
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record.get("task_key") or "unknown")].append(record)

    rng = random.Random(seed)
    ordered_by_task: dict[str, list[dict[str, Any]]] = {}
    for task, task_records in grouped.items():
        shuffled = list(task_records)
        rng.shuffle(shuffled)
        ordered_by_task[task] = shuffled

    # Preserve the two task-order shuffles performed by task_balanced_records
    # at depth zero.  This keeps the R3e first-choice anchors byte-for-byte
    # stable while exposing the unused same-task candidates as fallbacks.
    tasks = sorted(ordered_by_task)
    rng.shuffle(tasks)
    rng.shuffle(tasks)
    return [ordered_by_task[task] for task in tasks]


def finish_action_spans(text: str) -> V46ActionSpans:
    """Locate a canonical finish scaffold and its message body."""
    function_open = re.search(r"<function\s*=\s*finish\s*>", text, re.IGNORECASE)
    if function_open is None:
        return V46ActionSpans((), None)

    function_close = re.search(r"</function\s*>", text[function_open.end() :], re.IGNORECASE)
    function_close_span = None
    if function_close is not None:
        function_close_span = (
            function_open.end() + function_close.start(),
            function_open.end() + function_close.end(),
        )

    parameter_open = re.search(r"<parameter\s*=\s*message\s*>", text[function_open.end() :], re.IGNORECASE)
    parameter_open_span = None
    parameter_close_span = None
    message_body = None
    if parameter_open is not None:
        parameter_open_span = (
            function_open.end() + parameter_open.start(),
            function_open.end() + parameter_open.end(),
        )
        parameter_close = re.search(r"</parameter\s*>", text[parameter_open_span[1] :], re.IGNORECASE)
        if parameter_close is not None:
            parameter_close_span = (
                parameter_open_span[1] + parameter_close.start(),
                parameter_open_span[1] + parameter_close.end(),
            )
            message_body = (parameter_open_span[1], parameter_close_span[0])

    scaffold = [(function_open.start(), function_open.end())]
    for span in (parameter_open_span, parameter_close_span, function_close_span):
        if span is not None:
            scaffold.append(span)
    return V46ActionSpans(tuple(scaffold), message_body)


_VISIBLE_REFUSAL_PATTERNS = (
    re.compile(r"\bi\s+(?:cannot|can't|will\s+not|won't)\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:must|need\s+to)\s+refuse\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m|\s+am)\s+unable\s+to\b", re.IGNORECASE),
    re.compile(r"\bi\s+refuse\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+safely\b", re.IGNORECASE),
    re.compile(r"\b(?:unable|not\s+able)\s+to\s+(?:help|assist|comply|continue)\b", re.IGNORECASE),
)


def visible_response_text(text: str) -> str:
    """Return the user-visible finish message or text after hidden reasoning."""
    spans = finish_action_spans(text)
    if spans.message_body is not None:
        return text[slice(*spans.message_body)].strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def has_visible_refusal(text: str) -> bool:
    """Require an explicit refusal in the user-visible response, not hidden thinking."""
    visible = visible_response_text(text)
    return any(pattern.search(visible) is not None for pattern in _VISIBLE_REFUSAL_PATTERNS)


__all__ = [
    "V46ActionSpans",
    "finish_action_spans",
    "has_visible_refusal",
    "task_balanced_records",
    "v6_ordered_benign_tool_candidate_groups",
    "v6_task_balanced_benign_tool_records",
    "visible_response_text",
]
