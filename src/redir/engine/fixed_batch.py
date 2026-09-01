"""Frozen contracts and automatic decisions for MT-AgentRisk V7.9.1."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from typing import Any, Iterable


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bridge_kind_for_frozen_route(route: str) -> str:
    if route == "b":
        return "closure_ce"
    if route in {"a", "i"}:
        return "full_target_ce"
    if route == "c":
        return "none"
    raise ValueError(f"unsupported frozen route: {route!r}")


def build_fixed_pair_manifest(
    records: Iterable[dict[str, Any]],
    frozen_routes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten all supported state/target pairs with task-balanced weights."""

    supported: list[dict[str, Any]] = []
    states_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        targets = list(record.get("v78_native_refusal_targets") or [])
        if not targets:
            continue
        state_id = str(record.get("state_id") or "")
        task_key = str(record.get("task_key") or "")
        if not state_id or not task_key or state_id not in frozen_routes:
            raise ValueError(f"missing V7.9.1 frozen route for {state_id!r}")
        row = {"record": record, "targets": targets}
        supported.append(row)
        states_by_task[task_key].append(row)
    if not supported or not states_by_task:
        raise ValueError("V7.9.1 fixed batch has no target-supported states")

    active_tasks: set[str] = set()
    for task_key, rows in states_by_task.items():
        if any(
            bridge_kind_for_frozen_route(
                str(frozen_routes[str(row["record"]["state_id"])]["stratum"])
            )
            != "none"
            for row in rows
        ):
            active_tasks.add(task_key)
    if not active_tasks:
        raise ValueError("V7.9.1 fixed batch has no active bridge-CE tasks")

    manifest: list[dict[str, Any]] = []
    for task_key in sorted(states_by_task):
        task_rows = states_by_task[task_key]
        active_state_rows = [
            row
            for row in task_rows
            if bridge_kind_for_frozen_route(
                str(frozen_routes[str(row["record"]["state_id"])]["stratum"])
            )
            != "none"
        ]
        for row in sorted(
            task_rows,
            key=lambda value: str(value["record"].get("state_id") or ""),
        ):
            record = row["record"]
            state_id = str(record["state_id"])
            route = frozen_routes[state_id]
            bridge_kind = bridge_kind_for_frozen_route(str(route["stratum"]))
            targets = sorted(
                row["targets"], key=lambda target: str(target.get("target_id") or "")
            )
            active = bridge_kind != "none"
            weight = (
                1.0
                / len(active_tasks)
                / len(active_state_rows)
                / len(targets)
                if active
                else 0.0
            )
            for target in targets:
                manifest.append(
                    {
                        "task_key": task_key,
                        "state_id": state_id,
                        "target_id": str(target.get("target_id") or ""),
                        "frozen_route": str(route["stratum"]),
                        "frozen_route_seed": int(route["seed"]),
                        "bridge_kind": bridge_kind,
                        "active": active,
                        "weight": weight,
                        "target": target,
                    }
                )
    weight_sum = sum(float(row["weight"]) for row in manifest)
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"V7.9.1 active pair weights do not sum to one: {weight_sum}")
    return manifest


def plateau_reached(
    losses: Iterable[float],
    *,
    threshold: float = 5.0e-4,
    consecutive: int = 5,
) -> bool:
    values = list(losses)
    if len(values) < consecutive + 1:
        return False
    improvements = [
        (left - right) / max(abs(left), 1.0e-12)
        for left, right in zip(values[-consecutive - 1 : -1], values[-consecutive:])
    ]
    return all(value < threshold for value in improvements)


def a_unlock_allowed(
    *,
    outer_index: int,
    current_gradient_norm: float,
    o1_final_gradient_norm: float,
    recent_b_ema_cosines: Iterable[float],
    minimum_gradient_ratio: float = 1.05,
    minimum_mean_cosine: float = 0.5,
    window: int = 5,
) -> bool:
    values = list(recent_b_ema_cosines)
    return bool(
        outer_index >= 2
        and o1_final_gradient_norm > 0.0
        and current_gradient_norm / o1_final_gradient_norm >= minimum_gradient_ratio
        and len(values) >= window
        and sum(values[-window:]) / window > minimum_mean_cosine
    )


def route_counts(routes: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("stratum") or "i") for row in routes.values())
    return {key: int(counts.get(key, 0)) for key in ("c", "b", "a", "i")}


__all__ = [
    "a_unlock_allowed",
    "bridge_kind_for_frozen_route",
    "build_fixed_pair_manifest",
    "plateau_reached",
    "route_counts",
    "stable_json_sha256",
]
