"""Pure V7.8 target-availability and action-support contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


V78Stratum = Literal["c", "b", "a", "i"]
V78BridgeKind = Literal["none", "closure_ce", "full_target_ce"]


@dataclass(frozen=True, slots=True)
class V78TargetAvailability:
    """Whether filtered teacher sampling found at least one valid target.

    The raw sampling yield is deliberately descriptive.  It never scales the
    target CE weight and does not assert that the unfiltered teacher
    distribution is stable.
    """

    has_valid_target: bool
    valid_native_target_count: int
    total_rollouts: int
    raw_target_yield: float
    target_ce_weight: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_v78_target_availability(
    *,
    valid_native_targets: int,
    total_rollouts: int,
) -> V78TargetAvailability:
    if total_rollouts <= 0:
        raise ValueError("V7.8 target discovery requires at least one rollout")
    if not 0 <= valid_native_targets <= total_rollouts:
        raise ValueError("valid_native_targets must be within the rollout denominator")
    available = valid_native_targets > 0
    return V78TargetAvailability(
        has_valid_target=available,
        valid_native_target_count=int(valid_native_targets),
        total_rollouts=int(total_rollouts),
        raw_target_yield=float(valid_native_targets) / float(total_rollouts),
        target_ce_weight=1.0 if available else 0.0,
        reason=(
            "at_least_one_filtered_native_refusal_target"
            if available
            else "no_filtered_native_refusal_target"
        ),
    )


@dataclass(frozen=True, slots=True)
class V78ActionSupport:
    """Teacher-only eligibility for action-token reverse KL."""

    supported: bool
    teacher_finish_probability: float
    teacher_finish_vs_observed_log_odds: float | None
    observed_function_name: str | None
    teacher_body_is_refusal: bool
    minimum_finish_probability: float
    minimum_finish_vs_observed_log_odds: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_v78_action_support(
    *,
    teacher_finish_probability: float,
    teacher_finish_vs_observed_log_odds: float | None,
    observed_function_name: str | None,
    teacher_body_is_refusal: bool,
    minimum_finish_probability: float = 0.05,
    minimum_finish_vs_observed_log_odds: float = math.log(2.0),
) -> V78ActionSupport:
    """Gate reverse KL using only the teacher's own action distribution.

    Student finish probability and teacher-vs-student advantage are
    intentionally absent.  They remain useful diagnostics but cannot decide
    whether the teacher itself supports ``finish``.
    """

    probability_finite = math.isfinite(teacher_finish_probability)
    probability_valid = probability_finite and 0.0 <= teacher_finish_probability <= 1.0
    observed = None if observed_function_name in (None, "") else str(observed_function_name)
    needs_preference = observed not in (None, "finish")
    odds_finite = (
        teacher_finish_vs_observed_log_odds is not None
        and math.isfinite(float(teacher_finish_vs_observed_log_odds))
    )
    if not 0.0 < minimum_finish_probability < 1.0:
        raise ValueError("minimum_finish_probability must be in (0, 1)")
    if minimum_finish_vs_observed_log_odds <= 0.0:
        raise ValueError("minimum finish-vs-observed log odds must be positive")

    if not probability_valid:
        supported = False
        reason = "invalid_teacher_finish_probability"
    elif not teacher_body_is_refusal:
        supported = False
        reason = "teacher_target_body_is_not_refusal"
    elif teacher_finish_probability < minimum_finish_probability:
        supported = False
        reason = "teacher_finish_probability_below_absolute_threshold"
    elif needs_preference and not odds_finite:
        supported = False
        reason = "teacher_finish_vs_observed_log_odds_unavailable"
    elif needs_preference and float(teacher_finish_vs_observed_log_odds) < (
        minimum_finish_vs_observed_log_odds
    ):
        supported = False
        reason = "teacher_does_not_prefer_finish_over_observed_function"
    else:
        supported = True
        reason = "teacher_has_absolute_action_support"
    return V78ActionSupport(
        supported=supported,
        teacher_finish_probability=float(teacher_finish_probability),
        teacher_finish_vs_observed_log_odds=(
            None
            if teacher_finish_vs_observed_log_odds is None
            else float(teacher_finish_vs_observed_log_odds)
        ),
        observed_function_name=observed,
        teacher_body_is_refusal=bool(teacher_body_is_refusal),
        minimum_finish_probability=float(minimum_finish_probability),
        minimum_finish_vs_observed_log_odds=float(
            minimum_finish_vs_observed_log_odds
        ),
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class V78LossPermissions:
    stratum: V78Stratum
    bridge_kind: V78BridgeKind
    allow_action_reverse_kl: bool
    allow_self_distillation: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def v78_loss_permissions(
    *,
    stratum: V78Stratum,
    has_valid_target: bool,
    teacher_action_support: bool,
    has_action_boundary: bool,
    model_generated_invalid: bool = True,
    action_reverse_kl_enabled: bool = True,
) -> V78LossPermissions:
    """Return the V7.8/V7.9 C/B/A/I objective matrix.

    V7.9 keeps every CE/self-CE permission unchanged but globally disables
    action reverse KL during mode creation.
    """

    allow_rkl = bool(action_reverse_kl_enabled)

    if stratum == "c":
        return V78LossPermissions(
            stratum,
            "none",
            bool(
                allow_rkl
                and has_valid_target
                and teacher_action_support
                and has_action_boundary
            ),
            True,
            "strict_c_self_distillation_with_optional_action_opd",
        )
    if stratum == "b":
        return V78LossPermissions(
            stratum,
            "closure_ce" if has_valid_target else "none",
            bool(
                allow_rkl
                and has_valid_target
                and teacher_action_support
                and has_action_boundary
            ),
            False,
            (
                "safe_decision_closure_bridge_with_optional_action_opd"
                if has_valid_target
                else "b_without_teacher_target"
            ),
        )
    if stratum == "a":
        return V78LossPermissions(
            stratum,
            "full_target_ce" if has_valid_target else "none",
            False,
            False,
            (
                "dangerous_action_full_target_ce"
                if has_valid_target
                else "a_without_teacher_target"
            ),
        )
    if stratum == "i":
        allowed = bool(has_valid_target and model_generated_invalid)
        return V78LossPermissions(
            stratum,
            "full_target_ce" if allowed else "none",
            False,
            False,
            (
                "model_invalid_full_target_ce"
                if allowed
                else "runtime_or_target_unavailable_invalid"
            ),
        )
    raise ValueError(f"unsupported V7.8 stratum: {stratum!r}")


__all__ = [
    "V78ActionSupport",
    "V78LossPermissions",
    "V78TargetAvailability",
    "classify_v78_target_availability",
    "evaluate_v78_action_support",
    "v78_loss_permissions",
]
