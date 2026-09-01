"""Prepare the final target-pair and benign-retention training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from redir.data import read_jsonl


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_training_data(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    output = Path(config["output_dir"]).resolve()
    states_path = Path(data["safety_states"]).resolve()
    pair_manifest_path = Path(data["pair_manifest"]).resolve()
    benign_path = Path(data["benign_completions"]).resolve()
    states = read_jsonl(states_path)
    manifest = read_jsonl(pair_manifest_path)
    benign_completions = read_jsonl(benign_path)

    states_by_id = {str(row.get("state_id") or ""): row for row in states}
    if "" in states_by_id or len(states_by_id) != len(states):
        raise ValueError("training states require unique, non-empty state_id values")
    active_pairs = {
        (str(row.get("state_id") or ""), str(row.get("target_id") or ""))
        for row in manifest
        if row.get("active") is True
    }
    pairs: list[dict[str, Any]] = []
    safety_tasks: set[str] = set()
    for state in states:
        if state.get("route") != "safety" or state.get("training_candidate") is not True:
            continue
        state_id = str(state["state_id"])
        task_key = str(state.get("task_key") or "")
        safety_tasks.add(task_key)
        for target in state.get("targets") or []:
            target_id = str(target.get("target_id") or "")
            if (state_id, target_id) not in active_pairs:
                continue
            token_ids = [int(value) for value in target.get("completion_token_ids") or []]
            if not target_id or not token_ids:
                raise ValueError(f"invalid target for state {state_id}")
            pairs.append(
                {
                    "state_id": state_id,
                    "task_key": task_key,
                    "target_id": target_id,
                    "completion_token_ids": token_ids,
                    "supervised_token_indices": [
                        int(value)
                        for value in target.get("supervised_token_indices")
                        or range(len(token_ids))
                    ],
                    "supervised_token_weights": [
                        float(value)
                        for value in target.get("supervised_token_weights")
                        or [1.0] * len(token_ids)
                    ],
                }
            )
    if not pairs:
        raise ValueError("no active teacher-target pairs were prepared")

    completion_by_state = {
        str(row.get("state_id") or ""): row for row in benign_completions
    }
    benign_records: list[dict[str, Any]] = []
    for state in states:
        if state.get("route") != "benign" or state.get("training_candidate") is not True:
            continue
        state_id = str(state["state_id"])
        completion = completion_by_state.get(state_id)
        if completion is None:
            continue
        token_ids = [int(value) for value in completion.get("completion_token_ids") or []]
        if token_ids:
            benign_records.append(
                {
                    "state_id": state_id,
                    "task_key": str(state.get("task_key") or ""),
                    "completion_token_ids": token_ids,
                }
            )
    if not benign_records:
        raise ValueError("no benign-retention records were prepared")

    checks = {
        "safety_states": sum(row.get("route") == "safety" for row in states),
        "safety_tasks": len(safety_tasks - {""}),
        "target_pairs": len(pairs),
        "benign_completions": len(benign_records),
    }
    expected = config.get("expected_counts") or {}
    mismatches = {
        name: {"expected": int(expected[name]), "observed": observed}
        for name, observed in checks.items()
        if name in expected and int(expected[name]) != observed
    }
    if mismatches:
        raise ValueError(f"prepared-data count mismatch: {mismatches}")

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "training_pairs.jsonl", pairs)
    _write_jsonl(output / "benign_records.jsonl", benign_records)
    contract = {
        "format": "redir-final-training",
        "counts": checks,
        "artifacts": {
            "states": {"path": str(states_path), "sha256": _sha256(states_path)},
            "pair_manifest": {
                "path": str(pair_manifest_path),
                "sha256": _sha256(pair_manifest_path),
            },
            "benign_completions": {
                "path": str(benign_path),
                "sha256": _sha256(benign_path),
            },
        },
        "heldout_test_used": False,
    }
    _write_json(output / "data_contract.json", contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    contract = prepare_training_data(_read_yaml(args.config.resolve()))
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
