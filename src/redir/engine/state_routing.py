"""Pure contracts for V7.6 state-conditioned leak-free native OPD.

The model-facing trainer is intentionally kept separate from these routing
contracts.  V7.6 has three independent labels:

* the safety of the history *before* the sampled assistant turn;
* the reliability and format of the state-conditioned teacher target; and
* the semantic effect of the current student's proposed action.

Keeping these labels independent prevents a task-level teacher profile, a
late refusal, or a mechanically read-only tool name from silently granting a
training gradient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from redir.engine.trajectories import V7TurnRoute


V76PrefixRoute = Literal[
    "pre_state_safe",
    "pre_state_partial_progress",
    "pre_state_harm_completed",
    "pre_state_ambiguous",
]
V76TeacherRoute = Literal[
    "native_consistent",
    "native_sparse",
    "native_mixed",
    "semantic_only",
    "evidence_seeking",
    "unsafe",
    "invalid",
]
V76SemanticAction = Literal[
    "safe_evidence",
    "harmful_progress",
    "neutral",
    "ambiguous",
    "not_applicable",
]
V76StudentSubtype = Literal[
    "clean_native_refusal",
    "b_plain_refusal_without_finish",
    "b_safe_tool_after_decision",
    "b_harmful_action_after_decision",
    "b_success_finish_conflict",
    "a_safe_evidence",
    "a_harmful_progress",
    "a_generic_noncommittal",
    "i_model_empty_or_malformed",
    "i_runtime_parser_environment",
]


@dataclass(frozen=True)
class V76PrefixDecision:
    route: V76PrefixRoute
    gradient_weight: float
    unsafe_goal_advanced_before_state: bool
    harmful_goal_completed_before_state: bool
    danger_revealed: bool
    missing_future_evidence: bool
    evidence_event_ids: tuple[int, ...]
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evidence_event_ids"] = list(self.evidence_event_ids)
        return payload


def classify_v76_prefix_audit(
    judgment: Mapping[str, Any],
    *,
    minimum_confidence: float = 0.80,
) -> V76PrefixDecision:
    """Convert a semantic prefix judgment into a fail-closed train route."""

    confidence = float(judgment.get("confidence", 0.0))
    advanced = bool(judgment.get("unsafe_goal_advanced_before_state", False))
    completed = bool(judgment.get("harmful_goal_completed_before_state", False))
    danger_revealed = bool(judgment.get("danger_revealed", False))
    missing_future = bool(judgment.get("missing_future_evidence", True))
    raw_ids = judgment.get("evidence_event_ids") or []
    evidence_ids = tuple(sorted({int(value) for value in raw_ids}))

    if completed:
        route: V76PrefixRoute = "pre_state_harm_completed"
        weight = 0.0
        reason = "harmful_goal_completed_before_current_turn"
    elif (
        confidence < minimum_confidence
        or not danger_revealed
        or missing_future
    ):
        route = "pre_state_ambiguous"
        weight = 0.0
        reason = "insufficient_recoverable_evidence"
    elif advanced:
        route = "pre_state_partial_progress"
        weight = 0.25
        reason = "recoverable_state_after_partial_unsafe_progress"
    else:
        route = "pre_state_safe"
        weight = 1.0
        reason = "risk_revealed_without_prior_unsafe_progress"
    return V76PrefixDecision(
        route=route,
        gradient_weight=weight,
        unsafe_goal_advanced_before_state=advanced,
        harmful_goal_completed_before_state=completed,
        danger_revealed=danger_revealed,
        missing_future_evidence=missing_future,
        evidence_event_ids=evidence_ids,
        confidence=confidence,
        reason=reason,
    )


@dataclass(frozen=True)
class V76TeacherDecision:
    route: V76TeacherRoute
    native_clean: int
    semantic_clean: int
    safe_evidence: int
    harmful: int
    invalid: int
    total_rollouts: int
    canonical_native_target_valid: bool
    semantic_target_valid: bool
    route_weight: float
    allows_teacher_kl: bool
    allows_native_bridge: bool
    native_bridge_ce_only: bool
    allows_semantic_prefix: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_v76_teacher_audit(
    *,
    native_clean: int,
    semantic_clean: int,
    safe_evidence: int,
    harmful: int,
    invalid: int,
    total_rollouts: int,
    canonical_native_target_valid: bool,
    semantic_target_valid: bool,
) -> V76TeacherDecision:
    """Route four fresh state-conditioned teacher samples.

    A clean sample from a mixed teacher is still a useful filtered target, but
    the full-vocabulary teacher distribution is not trusted.  Consequently a
    mixed state receives CE-only support creation and never teacher KL.
    """

    counts = (native_clean, semantic_clean, safe_evidence, harmful, invalid)
    if total_rollouts <= 0 or any(value < 0 for value in counts):
        raise ValueError("V7.6 teacher counts must be non-negative")
    if sum(counts) != total_rollouts:
        raise ValueError("V7.6 teacher routes must partition every rollout")

    if native_clean > 0 and canonical_native_target_valid and harmful > 0:
        route: V76TeacherRoute = "native_mixed"
        weight = 0.25
        allows_kl = False
        allows_native = True
        ce_only = True
        allows_semantic = False
        reason = "filtered_native_target_from_behaviorally_mixed_teacher"
    elif native_clean >= 3 and canonical_native_target_valid and harmful == 0:
        route = "native_consistent"
        weight = 1.0
        allows_kl = True
        allows_native = True
        ce_only = False
        allows_semantic = False
        reason = "clean_native_refusal_at_least_3_of_4"
    elif native_clean >= 1 and canonical_native_target_valid and harmful == 0:
        route = "native_sparse"
        weight = 0.25
        allows_kl = True
        allows_native = True
        ce_only = False
        allows_semantic = False
        reason = "clean_native_refusal_1_or_2_of_4"
    elif semantic_clean > 0 and semantic_target_valid and harmful == 0:
        route = "semantic_only"
        weight = 0.25
        allows_kl = False
        allows_native = False
        ce_only = False
        allows_semantic = True
        reason = "explicit_refusal_without_native_finish"
    elif harmful > 0:
        route = "unsafe"
        weight = 0.0
        allows_kl = False
        allows_native = False
        ce_only = False
        allows_semantic = False
        reason = "teacher_advanced_harm_without_filtered_native_target"
    elif safe_evidence > 0:
        route = "evidence_seeking"
        weight = 0.0
        allows_kl = False
        allows_native = False
        ce_only = False
        allows_semantic = False
        reason = "teacher_requested_more_recoverable_evidence"
    else:
        route = "invalid"
        weight = 0.0
        allows_kl = False
        allows_native = False
        ce_only = False
        allows_semantic = False
        reason = "no_clean_state_conditioned_teacher_target"
    return V76TeacherDecision(
        route=route,
        native_clean=int(native_clean),
        semantic_clean=int(semantic_clean),
        safe_evidence=int(safe_evidence),
        harmful=int(harmful),
        invalid=int(invalid),
        total_rollouts=int(total_rollouts),
        canonical_native_target_valid=bool(canonical_native_target_valid),
        semantic_target_valid=bool(semantic_target_valid),
        route_weight=weight,
        allows_teacher_kl=allows_kl,
        allows_native_bridge=allows_native,
        native_bridge_ce_only=ce_only,
        allows_semantic_prefix=allows_semantic,
        reason=reason,
    )


@dataclass(frozen=True)
class V76SemanticActionDecision:
    label: V76SemanticAction
    confidence: float
    rationale: str
    judge_cache_key: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class V76StudentRoute:
    stratum: Literal["c", "b", "a", "i"]
    subtype: V76StudentSubtype
    semantic_action: V76SemanticAction
    harmful_current_action: bool
    safe_evidence_current_action: bool
    model_generated_invalid: bool
    runtime_invalid: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_RUNTIME_FAILURE_STATUSES = frozenset(
    {
        "parser_failure",
        "runtime_failure",
        "environment_failure",
        "mcp_failure",
    }
)


def classify_v76_student_route(
    structural: V7TurnRoute,
    *,
    semantic: V76SemanticActionDecision | None = None,
) -> V76StudentRoute:
    """Add task-conditioned action semantics to the structural C/B/A/I route."""

    semantic_label: V76SemanticAction = (
        "not_applicable" if semantic is None else semantic.label
    )
    harmful = semantic_label == "harmful_progress"
    evidence = semantic_label == "safe_evidence"

    if structural.stratum == "c":
        return V76StudentRoute(
            "c", "clean_native_refusal", semantic_label, False, False, False, False,
            structural.reason,
        )
    if structural.stratum == "i":
        runtime = structural.status in _RUNTIME_FAILURE_STATUSES
        return V76StudentRoute(
            "i",
            (
                "i_runtime_parser_environment"
                if runtime
                else "i_model_empty_or_malformed"
            ),
            semantic_label,
            False,
            False,
            not runtime,
            runtime,
            structural.reason,
        )
    if structural.stratum == "b":
        if structural.function_name is None:
            subtype: V76StudentSubtype = "b_plain_refusal_without_finish"
        elif structural.function_name == "finish":
            subtype = "b_success_finish_conflict"
        elif harmful:
            subtype = "b_harmful_action_after_decision"
        else:
            subtype = "b_safe_tool_after_decision"
        return V76StudentRoute(
            "b", subtype, semantic_label, harmful, evidence, False, False,
            structural.reason,
        )
    if harmful:
        subtype = "a_harmful_progress"
    elif evidence:
        subtype = "a_safe_evidence"
    else:
        subtype = "a_generic_noncommittal"
    return V76StudentRoute(
        "a", subtype, semantic_label, harmful, evidence, False, False,
        structural.reason,
    )


@dataclass(frozen=True)
class V76LossPermissions:
    prefix_weight: float
    allow_reverse_kl: bool
    allow_self_distill: bool
    allow_native_bridge: bool
    native_bridge_ce_only: bool
    allow_semantic_bridge: bool
    subtype_multiplier: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def v76_loss_permissions(
    *,
    prefix: V76PrefixDecision,
    teacher: V76TeacherDecision,
    student: V76StudentRoute,
) -> V76LossPermissions:
    """Return the only gradient branches permitted by the frozen V7.6 matrix."""

    if prefix.gradient_weight <= 0.0:
        return V76LossPermissions(
            0.0, False, False, False, False, False, 0.0,
            f"prefix_route_blocks_gradient:{prefix.route}",
        )
    subtype = student.subtype
    allow_self = subtype == "clean_native_refusal"
    allow_rkl = bool(
        teacher.allows_teacher_kl
        and subtype
        in {
            "clean_native_refusal",
            "b_plain_refusal_without_finish",
            "b_harmful_action_after_decision",
            "b_success_finish_conflict",
        }
    )
    native_bridge = False
    semantic_bridge = False
    multiplier = 1.0
    if subtype in {
        "b_plain_refusal_without_finish",
        "b_harmful_action_after_decision",
        "b_success_finish_conflict",
        "a_harmful_progress",
        "i_model_empty_or_malformed",
    }:
        native_bridge = teacher.allows_native_bridge
        semantic_bridge = teacher.allows_semantic_prefix and subtype != "i_model_empty_or_malformed"
    elif subtype in {"b_safe_tool_after_decision", "a_safe_evidence"}:
        # The label name describes the only trainable case.  A semantic judge
        # failure/low-confidence action can still land in this structural B
        # bucket, so require the explicit safe-evidence bit fail-closed.
        native_bridge = bool(
            student.safe_evidence_current_action
            and teacher.route == "native_consistent"
        )
        multiplier = 0.25
    # Generic/ambiguous actions and runtime failures are diagnostic only.
    return V76LossPermissions(
        prefix_weight=prefix.gradient_weight,
        allow_reverse_kl=allow_rkl,
        allow_self_distill=allow_self,
        allow_native_bridge=native_bridge,
        native_bridge_ce_only=bool(native_bridge and teacher.native_bridge_ce_only),
        allow_semantic_bridge=semantic_bridge,
        subtype_multiplier=multiplier,
        reason="route_matrix_applied",
    )


def classify_v76_task_trajectory(judgment: Mapping[str, Any]) -> str:
    """Map the existing V4.7 full-trajectory judge schema to V7.6 diagnostics."""

    decision = str(judgment.get("decision_label") or "")
    terminal = str(judgment.get("terminal_label") or "")
    advanced = bool(judgment.get("unsafe_goal_advanced_before_refusal", False))
    completed = bool(judgment.get("harmful_goal_completed", False))
    safe_terminal = terminal in {"refusal_finish", "plain_refusal", "safe_alternative"}
    if completed or terminal == "unsafe_complete":
        return "unsafe_complete"
    if safe_terminal and advanced:
        return "unsafe_progress_then_refusal"
    if safe_terminal and decision == "immediate_finish_refusal":
        return "direct_native_refusal"
    if safe_terminal and decision == "immediate_message_refusal":
        return "direct_plain_refusal"
    if safe_terminal and decision == "tool_before_refusal":
        return "safe_evidence_then_refusal"
    return "invalid_or_failed"


def v76_paired_transitions(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, object]:
    """Summarize matched fresh-rollout C/B/A/I transitions."""

    keys = sorted(set(before) & set(after))
    transitions: dict[str, int] = {}
    for key in keys:
        source = str(before[key]).lower()
        target = str(after[key]).lower()
        if source not in {"c", "b", "a", "i"} or target not in {
            "c",
            "b",
            "a",
            "i",
        }:
            raise ValueError(f"invalid V7.6 paired route: {source}->{target}")
        label = f"{source}->{target}"
        transitions[label] = transitions.get(label, 0) + 1
    return {
        "matched": len(keys),
        "before_only": len(set(before) - set(after)),
        "after_only": len(set(after) - set(before)),
        "transitions": dict(sorted(transitions.items())),
        "b_to_c": transitions.get("b->c", 0),
        "a_to_b_or_c": transitions.get("a->b", 0)
        + transitions.get("a->c", 0),
        "c_retained": transitions.get("c->c", 0),
    }


__all__ = [
    "V76LossPermissions",
    "V76PrefixDecision",
    "V76SemanticActionDecision",
    "V76StudentRoute",
    "V76TeacherDecision",
    "classify_v76_prefix_audit",
    "classify_v76_student_route",
    "classify_v76_task_trajectory",
    "classify_v76_teacher_audit",
    "v76_paired_transitions",
    "v76_loss_permissions",
]
