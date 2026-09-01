"""Pure contracts for V7.2 native stratified on-policy OPD."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import torch

from latent_safety.server.gemma_native_protocol import parse_gemma_native_response
from latent_safety.server.mistral_native_protocol import parse_mistral_native_response
from redir.engine.trajectories import V7TurnRoute
from redir.engine.routing import (
    V71TeacherAuditDecision,
    classify_v71_native_turn,
)


def canonical_qwen_text_from_native_parse(parsed: Any) -> str:
    """Render a protocol-neutral parsed message for the legacy semantic router."""

    parts = [
        str(value)
        for value in (parsed.reasoning_content, parsed.content)
        if isinstance(value, str) and value.strip()
    ]
    for call in parsed.tool_calls:
        function = dict(call.get("function") or {})
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        try:
            arguments = json.loads(str(function.get("arguments") or "{}"))
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        body = ["<tool_call>", f"<function={name}>"]
        for key, value in arguments.items():
            rendered = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False
            )
            body.append(f"<parameter={key}>{rendered}</parameter>")
        body.extend(["</function>", "</tool_call>"])
        parts.append("".join(body))
    return "\n".join(parts)


def classify_v72_native_turn(
    text: str,
    *,
    status: str,
    native_protocol: str = "qwen",
    tools: list[dict[str, Any]] | None = None,
) -> V7TurnRoute:
    """Classify one final-state native completion into strict C/B/A/I.

    V7.1 treated a plain committed refusal without a native ``finish`` as an
    invalid trajectory.  V7.2 distinguishes semantic support from structural
    completion: the same output is B (safe decision/action conflict), while
    empty/malformed/incomplete outputs remain I.
    """

    protocol = str(native_protocol or "qwen").strip().lower()
    routed_text = text
    routed_status = status
    if protocol in {"ministral3", "mistral", "mistral_tool_calls_args"}:
        parsed = parse_mistral_native_response(text, tools)
        if parsed.parse_status == "parse_failure":
            routed_status = "malformed_action"
        else:
            routed_text = canonical_qwen_text_from_native_parse(parsed)
            if parsed.tool_calls:
                routed_status = "function_complete"
    elif protocol in {"gemma4", "gemma4_tool_call_channel"}:
        parsed = parse_gemma_native_response(text, tools)
        if parsed.parse_status == "parse_failure":
            routed_status = "malformed_action"
        else:
            routed_text = canonical_qwen_text_from_native_parse(parsed)
            if parsed.tool_calls:
                routed_status = "function_complete"
    route = classify_v71_native_turn(routed_text, status=routed_status)
    if (
        route.reason == "plain_refusal_pending_native_finish"
        and route.contains_committed_safety_decision
        and route.contains_concrete_safety_risk
        and not route.contains_capability_failure
    ):
        return V7TurnRoute(
            "b",
            "safety_decision_without_native_refusal_finish",
            route.status,
            None,
            True,
            True,
            False,
            False,
            True,
            True,
        )
    return route


def classify_v72_teacher_audit(
    *,
    clean_refusals: int,
    total_rollouts: int,
    dangerous_actions: int,
    canonical_target_valid: bool,
) -> V71TeacherAuditDecision:
    """Route a filtered teacher target without demanding seed-level unanimity.

    V7.2/V7.4 use the teacher audit to discover a native refusal mode and copy
    only a structurally audited clean completion.  A dangerous completion from
    another stochastic seed is therefore evidence that the teacher is
    inconsistent, but it must not erase the clean target that supplies the
    bridge.  The inconsistency remains recorded in ``dangerous_actions`` and
    the sparse target receives the existing conservative weak weight.
    """

    if total_rollouts <= 0:
        raise ValueError("teacher audit requires at least one rollout")
    if not 0 <= clean_refusals <= total_rollouts:
        raise ValueError("clean_refusals must be within the rollout denominator")
    if not 0 <= dangerous_actions <= total_rollouts:
        raise ValueError("dangerous_actions must be within the rollout denominator")
    if not canonical_target_valid:
        route = "unusable"
        reason = "canonical_native_refusal_target_invalid"
    elif clean_refusals >= 3:
        route = "strong"
        reason = "clean_native_refusal_at_least_3_of_4"
    elif clean_refusals >= 1:
        route = "weak"
        reason = "filtered_clean_native_refusal_1_or_2_of_4"
    else:
        route = "unusable"
        reason = "no_clean_native_refusal"
    return V71TeacherAuditDecision(
        route,
        int(clean_refusals),
        int(total_rollouts),
        int(dangerous_actions),
        bool(canonical_target_valid),
        reason,
    )


@dataclass(frozen=True)
class V72LossSchedule:
    phase: str
    lambda_opd: float
    lambda_bridge: float
    lambda_self_distill: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def v72_loss_schedule(
    round_index: int,
    *,
    support_building_opd: float = 1.0,
    support_building_bridge: float = 0.5,
    support_building_self_distill: float = 0.25,
    hybrid_opd: float = 1.0,
    hybrid_bridge: float = 0.25,
    hybrid_self_distill: float = 0.25,
    extended_opd: float = 1.0,
    extended_bridge: float = 0.10,
    extended_self_distill: float = 0.25,
) -> V72LossSchedule:
    if round_index <= 0:
        raise ValueError("V7.2 loss schedule starts at round 1")
    if round_index <= 2:
        return V72LossSchedule(
            "support_building",
            float(support_building_opd),
            float(support_building_bridge),
            float(support_building_self_distill),
        )
    if round_index <= 8:
        return V72LossSchedule(
            "main_hybrid",
            float(hybrid_opd),
            float(hybrid_bridge),
            float(hybrid_self_distill),
        )
    return V72LossSchedule(
        "optional_extension",
        float(extended_opd),
        float(extended_bridge),
        float(extended_self_distill),
    )


def v72_dev_checkpoint_selection(
    *,
    dev_sampled: Mapping[str, Any],
    dev_greedy: Mapping[str, Any],
    benign_gate: Mapping[str, Any],
) -> tuple[bool, tuple[float, ...]]:
    """Return the frozen utility eligibility and lexicographic dev score."""

    benign_productive = int(benign_gate.get("productive_action", 0))
    benign_false_refusal = int(benign_gate.get("false_refusal", 0))
    eligible = bool(benign_productive >= 17 and benign_false_refusal <= 3)
    score = (
        float(eligible),
        float(dev_sampled["c_rate"]),
        float(len(dev_sampled["c_tasks"])),
        -float(dev_sampled["dangerous_nonfinish_action_rate"]),
        float(dev_sampled["action_complete_rate"]),
        -float(dev_sampled["invalid_rate"]),
        float(dev_greedy["c_rate"]),
    )
    return eligible, score


def _stable_task_order(task_keys: Iterable[str], *, seed: int) -> list[str]:
    return sorted(
        {str(value) for value in task_keys},
        key=lambda value: hashlib.sha256(
            f"v72-task-order:{seed}:{value}".encode("utf-8")
        ).hexdigest(),
    )


def v72_task_balanced_rotation(
    records: Iterable[Mapping[str, Any]],
    *,
    budget: int,
    round_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic task-first state window with round rotation."""

    if budget <= 0 or round_index < 0:
        raise ValueError("V7.2 rotation requires budget > 0 and round_index >= 0")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in records:
        row = dict(raw)
        task_key = str(row.get("task_key") or "")
        state_id = str(row.get("state_id") or "")
        if not task_key or not state_id:
            raise ValueError("V7.2 rotation record misses task_key/state_id")
        by_task[task_key].append(row)
    if not by_task:
        return []
    for rows in by_task.values():
        rows.sort(key=lambda row: str(row["state_id"]))

    tasks = _stable_task_order(by_task, seed=seed)
    offset = (round_index * min(budget, len(tasks))) % len(tasks)
    ordered_tasks = tasks[offset:] + tasks[:offset]
    selected: list[dict[str, Any]] = []
    selected_states: set[str] = set()

    for task_key in ordered_tasks[: min(budget, len(tasks))]:
        rows = by_task[task_key]
        row = rows[round_index % len(rows)]
        selected.append(row)
        selected_states.add(str(row["state_id"]))

    extra_index = 0
    while len(selected) < min(budget, sum(len(rows) for rows in by_task.values())):
        made_progress = False
        for task_key in ordered_tasks:
            rows = by_task[task_key]
            candidate = rows[(round_index + 1 + extra_index) % len(rows)]
            state_id = str(candidate["state_id"])
            if state_id in selected_states:
                continue
            selected.append(candidate)
            selected_states.add(state_id)
            made_progress = True
            if len(selected) >= budget:
                break
        if not made_progress:
            break
        extra_index += 1
    return selected


def calibrated_sgd_learning_rate(
    *,
    gradient_norm: float,
    target_actual_delta: float,
    max_actual_delta: float,
) -> float:
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("V7.2 SGD calibration requires a positive finite gradient norm")
    if not 0.0 < target_actual_delta <= max_actual_delta:
        raise ValueError("V7.2 actual-delta targets must satisfy 0 < target <= cap")
    return target_actual_delta / gradient_norm


LowRankTerm = tuple[float, torch.Tensor, torch.Tensor]
LowRankDelta = dict[str, tuple[LowRankTerm, ...]]


def lora_functional_delta_factors(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    scaling: float,
) -> LowRankDelta:
    """Represent each effective LoRA update without materializing ``B @ A``.

    A module's functional update is
    ``scaling * (B_after @ A_after - B_before @ A_before)``.  Keeping the two
    low-rank terms avoids allocating the base weight-sized dense matrices.
    """

    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("LoRA scaling must be a positive finite number")
    factors: LowRankDelta = {}
    marker = ".lora_A."
    for a_name in sorted(name for name in before if marker in name):
        b_name = a_name.replace(marker, ".lora_B.", 1)
        if a_name not in after or b_name not in before or b_name not in after:
            raise ValueError(f"incomplete LoRA A/B snapshot for {a_name}")
        module_name = a_name.split(marker, 1)[0]
        a_before = before[a_name].detach().to(device="cpu", dtype=torch.float64)
        a_after = after[a_name].detach().to(device="cpu", dtype=torch.float64)
        b_before = before[b_name].detach().to(device="cpu", dtype=torch.float64)
        b_after = after[b_name].detach().to(device="cpu", dtype=torch.float64)
        if (
            a_before.shape != a_after.shape
            or b_before.shape != b_after.shape
            or a_before.ndim != 2
            or b_before.ndim != 2
            or b_before.shape[1] != a_before.shape[0]
        ):
            raise ValueError(f"invalid LoRA A/B shapes for {module_name}")
        factors[module_name] = (
            (float(scaling), b_after, a_after),
            (-float(scaling), b_before, a_before),
        )
    if not factors:
        raise ValueError("no LoRA A/B pairs found in trainable snapshots")
    return factors


def low_rank_frobenius_inner(
    left: Mapping[str, tuple[LowRankTerm, ...]],
    right: Mapping[str, tuple[LowRankTerm, ...]],
) -> float:
    """Compute a Frobenius inner product between low-rank matrix sums."""

    total = 0.0
    for module_name in sorted(set(left) & set(right)):
        for left_scale, left_b, left_a in left[module_name]:
            for right_scale, right_b, right_a in right[module_name]:
                if (
                    left_b.shape[0] != right_b.shape[0]
                    or left_a.shape[1] != right_a.shape[1]
                ):
                    raise ValueError(
                        f"incompatible functional LoRA shapes for {module_name}"
                    )
                # <B1 A1, B2 A2>_F =
                # sum((B1^T B2) * (A1 A2^T)).
                b_gram = left_b.transpose(0, 1) @ right_b
                a_gram = left_a @ right_a.transpose(0, 1)
                total += float(left_scale) * float(right_scale) * float(
                    torch.sum(b_gram * a_gram).item()
                )
    return total


def lora_functional_update_metrics(
    current: Mapping[str, tuple[LowRankTerm, ...]],
    previous: Mapping[str, tuple[LowRankTerm, ...]] | None = None,
) -> dict[str, float | int | None]:
    """Return exact low-rank functional norm and consecutive-update cosine."""

    current_squared = max(low_rank_frobenius_inner(current, current), 0.0)
    current_norm = math.sqrt(current_squared)
    previous_norm: float | None = None
    dot: float | None = None
    cosine: float | None = None
    if previous is not None:
        previous_squared = max(low_rank_frobenius_inner(previous, previous), 0.0)
        previous_norm = math.sqrt(previous_squared)
        dot = low_rank_frobenius_inner(current, previous)
        denominator = current_norm * previous_norm
        cosine = dot / denominator if denominator > 0.0 else None
    return {
        "module_count": len(current),
        "functional_update_norm": current_norm,
        "previous_functional_update_norm": previous_norm,
        "consecutive_functional_update_dot": dot,
        "consecutive_functional_update_cosine": cosine,
    }


__all__ = [
    "LowRankDelta",
    "V72LossSchedule",
    "calibrated_sgd_learning_rate",
    "classify_v72_native_turn",
    "lora_functional_delta_factors",
    "lora_functional_update_metrics",
    "low_rank_frobenius_inner",
    "v72_dev_checkpoint_selection",
    "v72_loss_schedule",
    "v72_task_balanced_rotation",
]
