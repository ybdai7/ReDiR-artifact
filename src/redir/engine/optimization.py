"""Pure helpers for V4.3 long-rollout constrained OPD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RolloutCompletionStatus:
    status: str
    hit_token_limit: bool
    action_reached: bool
    action_complete: bool
    has_eos: bool
    function_open_count: int
    function_close_count: int
    visible_text: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_p = teacher_logp.exp()
    return (teacher_p * (teacher_logp - student_logp)).sum(dim=-1).mean()


def weighted_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_p = teacher_logp.exp()
    token_losses = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
    weights = weights.to(token_losses.device, dtype=token_losses.dtype)
    return (token_losses * weights).sum() / weights.sum().clamp_min(1e-6)


def classify_rollout_completion(
    text: str,
    token_ids: list[int],
    *,
    eos_token_id: int | None,
    max_new_tokens: int,
    expect_thinking_close: bool,
    native_protocol: str = "qwen",
) -> RolloutCompletionStatus:
    """Classify whether a generated assistant response is safe to train on."""
    raw_text = text or ""
    stripped = raw_text.strip()
    has_eos = bool(token_ids) and eos_token_id is not None and token_ids[-1] == eos_token_id
    hit_token_limit = max_new_tokens > 0 and len(token_ids) >= max_new_tokens and not has_eos
    protocol = str(native_protocol or "qwen").strip().lower()
    if protocol in {"ministral3", "mistral", "mistral_tool_calls_args"}:
        opens = raw_text.count("[TOOL_CALLS]")
        closes = sum(
            1
            for chunk in raw_text.split("[TOOL_CALLS]")[1:]
            if "[ARGS]" in chunk and chunk.split("[ARGS]", 1)[0].strip()
        )
    elif protocol in {"gemma4", "gemma4_tool_call_channel"}:
        opens = raw_text.count("<|tool_call>")
        closes = raw_text.count("<tool_call|>")
    else:
        function_opens = len(
            re.findall(r"<function\s*=\s*[^>]+>", raw_text, re.IGNORECASE)
        )
        function_closes = len(
            re.findall(r"</function\s*>", raw_text, re.IGNORECASE)
        )
        # Qwen3 emits JSON directly inside <tool_call>; Qwen3.5 emits the
        # historical nested <function=...> form. Prefer the inner XML count
        # when it exists so Qwen3.5 is not double-counted.
        if function_opens or function_closes:
            opens, closes = function_opens, function_closes
        else:
            opens = len(re.findall(r"<tool_call\s*>", raw_text, re.IGNORECASE))
            closes = len(re.findall(r"</tool_call\s*>", raw_text, re.IGNORECASE))
    last_think_close = raw_text.lower().rfind("</think>")
    visible_text = raw_text[last_think_close + len("</think>") :].strip() if last_think_close >= 0 else stripped

    if not stripped:
        status = "empty_eos" if has_eos else "token_limit_incomplete" if hit_token_limit else "malformed_action"
        return RolloutCompletionStatus(status, hit_token_limit, False, False, has_eos, opens, closes, "")

    if opens or closes:
        if opens > 0 and opens == closes:
            return RolloutCompletionStatus(
                "function_complete", hit_token_limit, True, True, has_eos, opens, closes, visible_text
            )
        status = "token_limit_incomplete" if hit_token_limit else "malformed_action"
        return RolloutCompletionStatus(status, hit_token_limit, True, False, has_eos, opens, closes, visible_text)

    if hit_token_limit:
        return RolloutCompletionStatus(
            "token_limit_incomplete", True, bool(visible_text), False, has_eos, opens, closes, visible_text
        )

    if has_eos:
        if expect_thinking_close and last_think_close < 0:
            return RolloutCompletionStatus(
                "eos_complete", False, False, False, True, opens, closes, visible_text
            )
        if visible_text:
            return RolloutCompletionStatus(
                "plain_message_complete", False, True, True, True, opens, closes, visible_text
            )
        return RolloutCompletionStatus("empty_eos", False, False, False, True, opens, closes, "")

    return RolloutCompletionStatus("malformed_action", False, bool(visible_text), False, False, opens, closes, visible_text)


def retry_prefix_matches(initial_ids: list[int], retry_ids: list[int]) -> bool:
    return len(retry_ids) >= len(initial_ids) and retry_ids[: len(initial_ids)] == initial_ids


def rollout_seed(base_seed: int, epoch: int, update: int, record_index: int, rollout_index: int) -> int:
    """Stable, unique seeds for serial rollouts from one policy snapshot."""
    return int(base_seed + epoch * 1_000_000 + update * 10_000 + record_index * 100 + rollout_index)


def token_limit_gate_status(rate: float, *, warning_rate: float, hard_rate: float) -> str:
    """Classify token-limit incompleteness without stopping on a small warning rate."""
    if not 0.0 <= warning_rate <= hard_rate <= 1.0:
        raise ValueError(
            "token-limit thresholds must satisfy 0 <= warning_rate <= hard_rate <= 1"
        )
    if rate > hard_rate:
        return "hard_fail"
    if rate > warning_rate:
        return "warning"
    return "ok"


def early_non_special_token_indices(
    token_ids: Iterable[int],
    *,
    special_token_ids: Iterable[int],
    limit: int,
) -> list[int]:
    """Select the first generated, non-special token positions for early-decision KL."""
    if limit <= 0:
        return []
    special = {int(token_id) for token_id in special_token_ids}
    selected: list[int] = []
    for index, token_id in enumerate(token_ids):
        if int(token_id) in special:
            continue
        selected.append(index)
        if len(selected) >= limit:
            break
    return selected


def _gradient_sums(
    safety: Mapping[str, torch.Tensor],
    benign: Mapping[str, torch.Tensor],
) -> tuple[float, float, float]:
    safety_sq = 0.0
    benign_sq = 0.0
    dot = 0.0
    for name in set(safety) | set(benign):
        left = safety.get(name)
        right = benign.get(name)
        if left is not None:
            safety_sq += float(left.detach().float().pow(2).sum().cpu())
        if right is not None:
            benign_sq += float(right.detach().float().pow(2).sum().cpu())
        if left is not None and right is not None:
            dot += float((left.detach().float() * right.detach().float()).sum().cpu())
    return safety_sq, benign_sq, dot


def constrained_gradient_metrics(
    safety: Mapping[str, torch.Tensor],
    benign: Mapping[str, torch.Tensor],
) -> dict[str, float | bool]:
    safety_sq, benign_sq, dot = _gradient_sums(safety, benign)
    safety_norm = math.sqrt(max(safety_sq, 0.0))
    benign_norm = math.sqrt(max(benign_sq, 0.0))
    cosine = dot / max(safety_norm * benign_norm, 1e-12)
    total_sq = safety_sq + benign_sq + 2.0 * dot
    total_norm = math.sqrt(max(total_sq, 0.0))
    total_dot_safety = safety_sq + dot
    total_vs_safety = total_dot_safety / max(total_norm * safety_norm, 1e-12)
    return {
        "safety_grad_norm": safety_norm,
        "benign_grad_norm": benign_norm,
        "benign_to_safety_norm_ratio": benign_norm / max(safety_norm, 1e-12),
        "safety_benign_dot": dot,
        "safety_benign_cosine": cosine,
        "total_vs_safety_cosine_unconstrained": total_vs_safety,
        "gradient_conflict": dot < 0.0,
    }


def project_and_cap_benign_gradients(
    safety: Mapping[str, torch.Tensor],
    benign: Mapping[str, torch.Tensor],
    *,
    max_norm_ratio: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float | bool]]:
    """Remove safety-conflicting benign gradient and cap its global norm."""
    safety_sq, _, dot = _gradient_sums(safety, benign)
    projection_coefficient = dot / max(safety_sq, 1e-12) if dot < 0.0 else 0.0
    projected: dict[str, torch.Tensor] = {}
    for name, benign_grad in benign.items():
        safety_grad = safety.get(name)
        value = benign_grad.detach().clone()
        if safety_grad is not None and projection_coefficient != 0.0:
            value = value - projection_coefficient * safety_grad.to(value.device, dtype=value.dtype)
        projected[name] = value

    safety_norm = math.sqrt(max(safety_sq, 0.0))
    projected_sq = sum(float(value.float().pow(2).sum().cpu()) for value in projected.values())
    projected_norm = math.sqrt(max(projected_sq, 0.0))
    max_norm = max_norm_ratio * safety_norm
    cap_scale = min(1.0, max_norm / max(projected_norm, 1e-12))
    if cap_scale < 1.0:
        projected = {name: value * cap_scale for name, value in projected.items()}

    constrained_metrics = constrained_gradient_metrics(safety, projected)
    constrained_metrics.update(
        {
            "projection_applied": dot < 0.0,
            "projection_coefficient": projection_coefficient,
            "benign_norm_before_cap": projected_norm,
            "benign_cap_scale": cap_scale,
            "benign_max_norm_ratio": max_norm_ratio,
        }
    )
    return projected, constrained_metrics
