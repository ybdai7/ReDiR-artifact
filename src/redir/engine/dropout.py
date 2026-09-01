"""Pure helpers for the V7.9.5 G0-G4 gating diagnostics."""

from __future__ import annotations

from collections import Counter
import math
from statistics import median
from typing import Any, Iterable, Mapping

import torch
from torch import nn


def vector_pair_metrics(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Measure repeatability without relying on cosine near its saturation point."""

    if set(reference) != set(candidate):
        raise ValueError("vector dictionaries must have identical keys")
    reference_sq = 0.0
    candidate_sq = 0.0
    difference_sq = 0.0
    dot = 0.0
    for name in reference:
        left = reference[name].detach().to(device="cpu", dtype=torch.float64)
        right = candidate[name].detach().to(device="cpu", dtype=torch.float64)
        if left.shape != right.shape:
            raise ValueError(f"vector shape mismatch for {name}")
        reference_sq += float(torch.sum(left * left))
        candidate_sq += float(torch.sum(right * right))
        difference = left - right
        difference_sq += float(torch.sum(difference * difference))
        dot += float(torch.sum(left * right))
    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    difference_norm = math.sqrt(difference_sq)
    if reference_norm == 0.0:
        relative_error = 0.0 if difference_norm == 0.0 else math.inf
        norm_ratio = 1.0 if candidate_norm == 0.0 else math.inf
    else:
        relative_error = difference_norm / reference_norm
        norm_ratio = candidate_norm / reference_norm
    cosine = (
        dot / (reference_norm * candidate_norm)
        if reference_norm > 0.0 and candidate_norm > 0.0
        else (1.0 if reference_norm == candidate_norm == 0.0 else 0.0)
    )
    return {
        "cosine": cosine,
        "relative_error": relative_error,
        "norm_ratio": norm_ratio,
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "difference_norm": difference_norm,
    }


def summarize_psd_gram(gram: torch.Tensor) -> dict[str, Any]:
    """Summarize a positive-semidefinite pair-gradient Gram matrix."""

    matrix = gram.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Gram matrix must be square with at least two rows")
    if not torch.allclose(matrix, matrix.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Gram matrix must be symmetric")
    diagonal = torch.diag(matrix)
    if bool(torch.any(diagonal <= 0.0)):
        raise ValueError("Gram matrix diagonal must be strictly positive")
    norms = torch.sqrt(diagonal)
    cosine = matrix / torch.outer(norms, norms)
    cosine = (cosine + cosine.T) * 0.5
    count = int(matrix.shape[0])
    mask = ~torch.eye(count, dtype=torch.bool)
    off_diagonal = cosine[mask]

    def spectrum(source: torch.Tensor) -> dict[str, Any]:
        eigenvalues = torch.linalg.eigvalsh(source).clamp_min(0.0).flip(0)
        total = float(eigenvalues.sum())
        fractions = eigenvalues / max(total, 1e-30)
        nonzero = fractions[fractions > 0.0]
        entropy = float(-(nonzero * torch.log(nonzero)).sum())
        return {
            "eigenvalues": [float(value) for value in eigenvalues],
            "top1_fraction": float(fractions[:1].sum()),
            "top3_fraction": float(fractions[:3].sum()),
            "top5_fraction": float(fractions[:5].sum()),
            "effective_rank": math.exp(entropy),
        }

    coherence_sq = float(cosine.sum()) / (count * count)
    return {
        "count": count,
        "off_diagonal_mean_cosine": float(off_diagonal.mean()),
        "off_diagonal_median_cosine": float(off_diagonal.median()),
        "off_diagonal_negative_fraction": float((off_diagonal < 0.0).double().mean()),
        "mean_unit_gradient_coherence": math.sqrt(max(coherence_sq, 0.0)),
        "raw_spectrum": spectrum(matrix),
        "cosine_spectrum": spectrum(cosine),
        "cosine_gram": cosine.tolist(),
    }


def force_dropout_zero(module: nn.Module) -> list[dict[str, Any]]:
    """Disable every active Dropout module and return an auditable inventory."""

    inventory: list[dict[str, Any]] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Dropout):
            inventory.append(
                {
                    "module": name,
                    "class": type(child).__name__,
                    "original_p": float(child.p),
                }
            )
            child.p = 0.0
    return inventory


def assert_dropout_zero(module: nn.Module) -> None:
    nonzero = {
        name: float(child.p)
        for name, child in module.named_modules()
        if isinstance(child, nn.Dropout) and float(child.p) != 0.0
    }
    if nonzero:
        raise RuntimeError(f"active dropout modules remain non-zero: {nonzero}")


def cosine_classification(value: float) -> str:
    if value >= 0.8:
        return "stable_remove_ema_minibatch"
    if value >= 0.3:
        return "partially_stable_keep_ema"
    return "unstable_run_pair_gradient_gram"


def summarize_g1(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    step_rows = list(rows)
    b_values = [
        float(row["groups"]["lora_b"]["cos_prev_functional"])
        for row in step_rows
        if row["groups"]["lora_b"].get("cos_prev_functional") is not None
    ]
    query_values = [
        float(row["groups"]["prompt_query"]["cos_prev_raw"])
        for row in step_rows
        if row["groups"]["prompt_query"].get("cos_prev_raw") is not None
    ]
    if len(b_values) != 9 or len(query_values) != 9:
        raise ValueError("G1 requires exactly nine consecutive cosine values")
    primary = float(median(b_values))
    secondary = float(median(query_values))
    return {
        "lora_b_functional_cosines": b_values,
        "lora_b_functional_median": primary,
        "query_raw_cosines": query_values,
        "query_raw_median": secondary,
        "classification": cosine_classification(primary),
    }


def _effect_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("effect summary requires at least one value")
    signed_mean = sum(values) / len(values)
    return {
        "count": len(values),
        "signed_mean": signed_mean,
        "absolute_signed_mean": abs(signed_mean),
        "mean_absolute": sum(abs(value) for value in values) / len(values),
        "rms": math.sqrt(sum(value * value for value in values) / len(values)),
        "positive": sum(value > 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
    }


def summarize_latent_lever(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate latent content from the insertion-position control."""

    material = list(rows)
    changed_with_no = [bool(row["greedy_changed_with_vs_no"]) for row in material]
    changed_with_zero = [bool(row["greedy_changed_with_vs_zero"]) for row in material]
    result = {
        "with_vs_no_early32": _effect_summary(
            [float(row["delta_logprob_early32_with_vs_no"]) for row in material]
        ),
        "content_with_vs_zero_early32": _effect_summary(
            [float(row["delta_logprob_early32_with_vs_zero"]) for row in material]
        ),
        "position_zero_vs_no_early32": _effect_summary(
            [float(row["delta_logprob_early32_zero_vs_no"]) for row in material]
        ),
        "greedy_changed_with_vs_no_rate": sum(changed_with_no) / len(material),
        "greedy_changed_with_vs_zero_rate": sum(changed_with_zero) / len(material),
    }
    magnitude = result["content_with_vs_zero_early32"]["mean_absolute"]
    changed = result["greedy_changed_with_vs_zero_rate"]
    if magnitude >= 2.0 and changed >= 0.60:
        classification = "sufficient"
    elif magnitude < 0.5 or changed < 0.30:
        classification = "negligible"
    else:
        classification = "weak_but_usable"
    result["classification"] = classification
    return result


def select_g3_pairs(
    r0_rows: Iterable[Mapping[str, Any]],
    *,
    count: int = 4,
) -> list[dict[str, Any]]:
    """Pick median-difficulty zero-C full-CE pairs from distinct tasks."""

    candidates = [
        dict(row)
        for row in r0_rows
        if not bool(row.get("r0_has_c"))
        and str(row.get("bridge_kind")) == "full_target_ce"
    ]
    if len(candidates) < count:
        raise ValueError("not enough zero-C full-target pairs for G3")
    ordered = sorted(
        candidates,
        key=lambda row: (float(row["decision_logprob"]), str(row["pair_id"])),
    )
    middle = (len(ordered) - 1) / 2.0
    search = sorted(
        enumerate(ordered),
        key=lambda item: (abs(item[0] - middle), item[0], str(item[1]["pair_id"])),
    )
    selected: list[dict[str, Any]] = []
    tasks: set[str] = set()
    for _, row in search:
        task = str(row["task_key"])
        if task in tasks:
            continue
        selected.append(row)
        tasks.add(task)
        if len(selected) == count:
            return selected
    raise ValueError("G3 median neighborhood cannot satisfy distinct-task contract")


def select_g4_states(r0_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select four evenly spaced task-key states from each zero-C difficulty half."""

    by_state: dict[str, list[dict[str, Any]]] = {}
    for source in r0_rows:
        if bool(source.get("r0_has_c")):
            continue
        row = dict(source)
        by_state.setdefault(str(row["state_id"]), []).append(row)
    states: list[dict[str, Any]] = []
    for state_id, rows in by_state.items():
        mean_decision = sum(float(row["decision_logprob"]) for row in rows) / len(rows)
        representative = min(
            rows,
            key=lambda row: (
                abs(float(row["decision_logprob"]) - mean_decision),
                str(row["pair_id"]),
            ),
        )
        states.append(
            {
                "state_id": state_id,
                "task_key": str(representative["task_key"]),
                "target_id": str(representative["target_id"]),
                "pair_id": str(representative["pair_id"]),
                "state_mean_decision_logprob": mean_decision,
            }
        )
    if len(states) != 33:
        raise ValueError(f"G4 requires exactly 33 zero-C states; got {len(states)}")
    ordered = sorted(
        states,
        key=lambda row: (
            float(row["state_mean_decision_logprob"]),
            str(row["state_id"]),
        ),
    )
    halves = (("low", ordered[: len(ordered) // 2]), ("high", ordered[len(ordered) // 2 :]))
    selected: list[dict[str, Any]] = []
    for tier, half in halves:
        lexical = sorted(half, key=lambda row: (str(row["task_key"]), str(row["state_id"])))
        indices = [round(index * (len(lexical) - 1) / 3) for index in range(4)]
        if len(set(indices)) != 4:
            raise ValueError("G4 equidistant selection produced duplicate indices")
        for index in indices:
            selected.append({**lexical[index], "decision_tier": tier})
    return selected


def majority_pattern(codes: Iterable[Mapping[str, Any]]) -> str:
    """Return the registered G4 mode, with ties conservatively non-overriding."""

    rows = list(codes)
    if len(rows) != 8:
        raise ValueError("G4 requires exactly eight coded transcripts")
    counts = Counter(
        "missing_consideration"
        if str(row.get("s1")).lower() == "no"
        else "task_misunderstanding"
        if str(row.get("s3")).lower() == "yes"
        else "self_overruled"
        if str(row.get("s2")).lower() == "yes"
        else "other"
        for row in rows
    )
    highest = max(counts.values())
    winners = sorted(key for key, value in counts.items() if value == highest)
    return winners[0] if len(winners) == 1 and highest > len(rows) / 2 else "no_majority"


def exact_tensor_dict_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


__all__ = [
    "assert_dropout_zero",
    "cosine_classification",
    "exact_tensor_dict_equal",
    "force_dropout_zero",
    "majority_pattern",
    "select_g3_pairs",
    "select_g4_states",
    "summarize_g1",
    "summarize_latent_lever",
    "summarize_psd_gram",
    "vector_pair_metrics",
]
