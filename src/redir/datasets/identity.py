"""Select safe, productive benign completions for identity warmup."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft7Validator

from redir.server.gemma_native_protocol import parse_gemma_native_response
from redir.server.mistral_native_protocol import parse_mistral_native_response
from redir.server.qwen_native_protocol import parse_qwen_native_response


@dataclass(frozen=True, slots=True)
class SafeIdentitySource:
    split: str
    seed: int
    path: Path
    allowed_task_keys: frozenset[str] | None = None


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _lifecycle_bucket(record: dict[str, Any]) -> str | None:
    if int(record.get("native_history_assistant_tool_calls") or 0) == 0:
        return "initial"
    if str(record.get("observed_first_function_name") or "") == "finish":
        return "pre_finish"
    if int(record.get("native_history_tool_messages") or 0) > 0:
        return "after_tool"
    return None


def _tool_argument_errors(
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[str]:
    schemas = {
        str((tool.get("function") or {}).get("name") or ""): (
            (tool.get("function") or {}).get("parameters") or {}
        )
        for tool in tools or []
        if isinstance(tool, dict)
    }
    errors: list[str] = []
    for index, call in enumerate(tool_calls):
        function = (call or {}).get("function") or {}
        name = str(function.get("name") or "")
        if name not in schemas:
            errors.append(f"tool_calls[{index}] has unknown function {name!r}")
            continue
        arguments = function.get("arguments")
        try:
            arguments = json.loads(arguments or "{}") if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as exc:
            errors.append(f"tool_calls[{index}] arguments are invalid JSON: {exc}")
            continue
        if not isinstance(arguments, dict):
            errors.append(f"tool_calls[{index}] arguments are not an object")
            continue
        errors.extend(
            f"tool_calls[{index}] {name}: {error.message}"
            for error in Draft7Validator(schemas[name]).iter_errors(arguments)
        )
    return errors


def _audit_productive_benign(record: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    if record.get("protocol_source") != "native" or not record.get("native_tool_calling"):
        return False, "not_genuine_native", {}
    if not (
        record.get("native_identity_candidate")
        or record.get("native_benign_candidate")
    ):
        return False, "not_benign_candidate", {}
    outcome = record.get("source_post_eval_outcome")
    if outcome not in (None, "", "COMPLETE"):
        return False, "post_eval_not_complete", {}
    if record.get("observed_contains_refusal"):
        return False, "contains_refusal", {}
    completion = str(record.get("observed_completion") or "")
    if not completion.strip():
        return False, "empty_completion", {}
    tools = record.get("available_tools")
    rollout_policy = str(record.get("rollout_policy") or "").lower()
    if "[TOOL_CALLS]" in completion:
        parsed = parse_mistral_native_response(completion, tools)
    elif "<|tool_call>" in completion:
        parsed = parse_gemma_native_response(completion, tools)
    elif "<tool_call" in completion or "<function=" in completion:
        parsed = parse_qwen_native_response(
            completion,
            tools,
            thinking_enabled="<think>" in completion or "</think>" in completion,
        )
    elif "ministral" in rollout_policy:
        parsed = parse_mistral_native_response(completion, tools)
    elif "gemma4" in rollout_policy:
        parsed = parse_gemma_native_response(completion, tools)
    else:
        parsed = parse_qwen_native_response(completion, tools)
    if parsed.parse_status != "parsed" or parsed.finish_reason != "tool_calls":
        return False, f"parser_{parsed.parse_status}", {}
    if parsed.parse_error is not None or parsed.anomalies:
        return False, "parser_anomaly", {}
    declared = {
        str((tool.get("function") or {}).get("name") or "")
        for tool in tools or []
        if isinstance(tool, dict)
    }
    names = [str((call.get("function") or {}).get("name") or "") for call in parsed.tool_calls]
    if not names or any(not name or name not in declared for name in names):
        return False, "unknown_or_empty_tool_name", {}
    if _tool_argument_errors(parsed.tool_calls, tools):
        return False, "invalid_tool_arguments", {}
    bucket = _lifecycle_bucket(record)
    if bucket is None:
        return False, "unclassified_lifecycle", {}
    return True, "", {
        "parser_status": parsed.parse_status,
        "parsed_function_names": names,
        "parsed_tool_calls_count": len(parsed.tool_calls),
        "lifecycle_bucket": bucket,
    }


def build_safe_identity_dataset(
    sources: Iterable[SafeIdentitySource],
    *,
    output_dir: Path,
    expected_train_tasks: int,
    expected_dev_tasks: int,
) -> dict[str, Any]:
    caps = {"initial": 1, "after_tool": 2, "pre_finish": 1}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    source_stats: list[dict[str, Any]] = []
    for source in sources:
        if source.split not in {"train", "dev"}:
            raise ValueError(f"invalid identity split: {source.split}")
        rows = _load_jsonl(source.path)
        source_stats.append(
            {
                "split": source.split,
                "seed": source.seed,
                "path": str(source.path.resolve()),
                "records": len(rows),
                "sha256": hashlib.sha256(source.path.read_bytes()).hexdigest(),
            }
        )
        for raw in rows:
            if source.allowed_task_keys is not None and str(raw.get("task_key") or "") not in source.allowed_task_keys:
                continue
            admitted, reason, audit = _audit_productive_benign(raw)
            if not admitted:
                exclusions[reason] += 1
                continue
            row = deepcopy(raw)
            row.update(audit)
            row.update(
                {
                    "category": "productive_benign",
                    "productive_benign": True,
                    "clean_refusal": False,
                    "harmful_tool_call_executed": False,
                    "dangerous_nonfinish_before_refusal": False,
                    "safety_action_conflict": False,
                    "capitulated_after_refusal": False,
                    "training_eligible": True,
                    "exclusion_reason": "",
                    "attack_success_record": False,
                    "dangerous_completion_gradient_used": False,
                    "heldout15_used": False,
                    "benign20_used": False,
                    "completion_sha256": hashlib.sha256(
                        str(raw["observed_completion"]).encode("utf-8")
                    ).hexdigest(),
                    "messages_hash": _json_hash(raw.get("student_state_messages") or []),
                    "tools_hash": _json_hash(raw.get("available_tools") or []),
                }
            )
            candidates[source.split].append(row)

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for split in ("train", "dev"):
        by_task_bucket: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in sorted(
            candidates[split],
            key=lambda value: (
                str(value["task_key"]),
                int(value.get("rollout_seed") or 0),
                str(value["state_id"]),
            ),
        ):
            by_task_bucket[str(row["task_key"])][str(row["lifecycle_bucket"])].append(row)
        for task_key in sorted(by_task_bucket):
            for bucket in ("initial", "after_tool", "pre_finish"):
                selected[split].extend(by_task_bucket[task_key][bucket][: caps[bucket]])

    train_tasks = {str(row["task_key"]) for row in selected["train"]}
    dev_tasks = {str(row["task_key"]) for row in selected["dev"]}
    if train_tasks & dev_tasks:
        raise ValueError(f"identity train/dev task overlap: {sorted(train_tasks & dev_tasks)}")
    expected = {"train": expected_train_tasks, "dev": expected_dev_tasks}
    actual = {"train": len(train_tasks), "dev": len(dev_tasks)}
    if actual != expected:
        raise ValueError(f"identity task coverage mismatch: {actual} != {expected}")

    output_dir.mkdir(parents=True, exist_ok=False)
    for split in ("train", "dev"):
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected[split]),
            encoding="utf-8",
        )
    stats = {
        "strategy": "native_safe_identity",
        "sources": source_stats,
        "records": {split: len(selected[split]) for split in ("train", "dev")},
        "tasks": actual,
        "lifecycle": {
            split: dict(Counter(str(row["lifecycle_bucket"]) for row in selected[split]))
            for split in ("train", "dev")
        },
        "caps_per_task": caps,
        "exclusions": dict(exclusions),
        "dangerous_completion_gradient_used": False,
        "attack_success_records_in_optimizer": 0,
        "heldout15_used": False,
        "benign20_used": False,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


__all__ = ["SafeIdentitySource", "build_safe_identity_dataset"]
