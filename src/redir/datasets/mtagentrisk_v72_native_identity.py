"""Build a protocol-audited V7.2 native identity-warmup dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class NativeIdentitySource:
    split: str
    seed: int
    path: Path
    domain: str = "identity"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _kind_limit(kind: str, *, assistant: int, tool: int, finish: int) -> int:
    if kind == "assistant_message":
        return assistant
    if kind == "tool_action":
        return tool
    if kind == "finish_action":
        return finish
    return 0


def build_v72_native_identity_dataset(
    sources: Iterable[NativeIdentitySource],
    *,
    output_dir: Path,
    expected_train_tasks: int | None = None,
    expected_dev_tasks: int | None = None,
    max_assistant_states_per_task_seed: int = 1,
    max_tool_states_per_task_seed: int = 3,
    max_finish_states_per_task_seed: int = 1,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output_dir}")
    source_rows = list(sources)
    if not source_rows:
        raise ValueError("at least one native identity source is required")
    if any(source.split not in {"train", "dev"} for source in source_rows):
        raise ValueError("native identity sources must use split=train|dev")
    if any(source.domain not in {"identity", "benign"} for source in source_rows):
        raise ValueError("native identity source domain must be identity or benign")
    if any(
        source.domain == "benign" and source.split != "train"
        for source in source_rows
    ):
        raise ValueError("benign retention sources are train-only")

    admitted: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    source_stats: list[dict[str, Any]] = []
    duplicate_prompts: Counter[str] = Counter()
    seen_prompts: dict[str, set[str]] = defaultdict(set)

    for source in sorted(
        source_rows,
        key=lambda item: (item.split, item.domain, item.seed, str(item.path)),
    ):
        path = source.path.resolve()
        rows = _load_jsonl(path)
        source_stats.append(
            {
                "split": source.split,
                "domain": source.domain,
                "seed": source.seed,
                "path": str(path),
                "sha256": _file_sha256(path),
                "records": len(rows),
            }
        )
        per_task_kind: Counter[tuple[str, str]] = Counter()
        for raw in sorted(
            rows,
            key=lambda row: (
                str(row.get("task_key") or ""),
                str(row.get("state_id") or ""),
            ),
        ):
            if raw.get("protocol_source") != "native" or not raw.get("native_tool_calling"):
                raise ValueError(f"identity source is not genuine native: {raw.get('state_id')}")
            if raw.get("v6_native_domain") != source.domain:
                raise ValueError(
                    f"native source has wrong domain: {raw.get('state_id')}"
                )
            if source.domain == "identity":
                if not raw.get("v72_native_identity_candidate"):
                    raise ValueError(
                        f"identity candidate flag is missing: {raw.get('state_id')}"
                    )
                role = "identity_main"
            else:
                if (
                    not raw.get("v6_native_benign_candidate")
                    or raw.get("source_post_eval_outcome") != "COMPLETE"
                    or raw.get("observed_contains_refusal")
                ):
                    raise ValueError(
                        f"benign retention source is not admitted: {raw.get('state_id')}"
                    )
                role = "benign_retention"
            task_key = str(raw.get("task_key") or "")
            state_kind = str(raw.get("state_kind") or "")
            if not task_key:
                raise ValueError("identity record is missing task_key")
            limit = _kind_limit(
                state_kind,
                assistant=max_assistant_states_per_task_seed,
                tool=max_tool_states_per_task_seed,
                finish=max_finish_states_per_task_seed,
            )
            if limit <= 0:
                continue
            key = (task_key, state_kind)
            if per_task_kind[key] >= limit:
                continue
            prompt_hash = str(raw.get("source_prompt_sha256") or "")
            if not prompt_hash:
                raise ValueError(f"identity record misses prompt hash: {raw.get('state_id')}")
            if prompt_hash in seen_prompts[source.split]:
                duplicate_prompts[source.split] += 1
                continue
            completion = str(raw.get("observed_completion") or "")
            messages = raw.get("student_state_messages")
            tools = raw.get("available_tools")
            if not completion.strip() or not isinstance(messages, list) or not messages:
                raise ValueError(f"identity record is incomplete: {raw.get('state_id')}")
            if not isinstance(tools, list) or not tools:
                raise ValueError(f"identity record has no native tools: {raw.get('state_id')}")

            row = dict(raw)
            row.update(
                {
                    "split": source.split,
                    "source_split": source.split,
                    "identity_rollout_seed": source.seed,
                    "v72_native_identity_role": role,
                    "raw_completion": completion,
                    "raw_nonfncall_completion": completion,
                    "v72_native_identity_candidate": source.domain == "identity",
                    "raw_nonfncall_warmup_candidate": True,
                    "is_training_candidate": True,
                    "sample_weight": 1.0,
                    "state_weight": 1.0,
                    "heldout15_used": False,
                }
            )
            admitted[source.split].append(row)
            seen_prompts[source.split].add(prompt_hash)
            per_task_kind[key] += 1

    for split in admitted:
        admitted[split].sort(
            key=lambda row: (
                str(row.get("task_key") or ""),
                int(row.get("identity_rollout_seed", 0)),
                str(row.get("state_id") or ""),
            )
        )
        position0_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in admitted[split]:
            row["v72_position0_candidate"] = False
            if row["v72_native_identity_role"] == "identity_main":
                position0_groups[
                    (str(row["task_key"]), int(row["identity_rollout_seed"]))
                ].append(row)
        for rows in position0_groups.values():
            selected = min(
                rows,
                key=lambda row: (
                    int(row.get("native_history_assistant_tool_calls") or 0),
                    len(row.get("student_state_messages") or []),
                    str(row.get("state_id") or ""),
                ),
            )
            selected["v72_position0_candidate"] = True

    task_counts = {
        split: len(
            {
                str(row["task_key"])
                for row in rows
                if row["v72_native_identity_role"] == "identity_main"
            }
        )
        for split, rows in admitted.items()
    }
    expected = {"train": expected_train_tasks, "dev": expected_dev_tasks}
    for split, count in task_counts.items():
        if expected[split] is not None and count != expected[split]:
            raise ValueError(
                f"native identity {split} task coverage is {count}, expected {expected[split]}"
            )
    if not admitted["train"]:
        raise ValueError("native identity dataset has no train records")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in admitted.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "strategy": "v72_genuine_native_identity_warmup_v1",
        "protocol_source": "native",
        "native_tool_calling": True,
        "records": {split: len(rows) for split, rows in admitted.items()},
        "tasks": task_counts,
        "benign_retention_tasks": len(
            {
                str(row["task_key"])
                for row in admitted["train"]
                if row["v72_native_identity_role"] == "benign_retention"
            }
        ),
        "roles": {
            split: dict(
                Counter(str(row["v72_native_identity_role"]) for row in rows)
            )
            for split, rows in admitted.items()
        },
        "position0_candidates": {
            split: sum(bool(row["v72_position0_candidate"]) for row in rows)
            for split, rows in admitted.items()
        },
        "state_kinds": {
            split: dict(Counter(str(row["state_kind"]) for row in rows))
            for split, rows in admitted.items()
        },
        "seeds": {
            split: sorted({int(row["identity_rollout_seed"]) for row in rows})
            for split, rows in admitted.items()
        },
        "duplicate_prompts_dropped": dict(duplicate_prompts),
        "sources": source_stats,
        "caps_per_task_seed": {
            "assistant_message": max_assistant_states_per_task_seed,
            "tool_action": max_tool_states_per_task_seed,
            "finish_action": max_finish_states_per_task_seed,
        },
        "train_jsonl_sha256": _file_sha256(output_dir / "train.jsonl"),
        "dev_jsonl_sha256": _file_sha256(output_dir / "dev.jsonl"),
        "dataset_records_sha256": _json_sha256(admitted),
        "heldout15_used": False,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


__all__ = ["NativeIdentitySource", "build_v72_native_identity_dataset"]
