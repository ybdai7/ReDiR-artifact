"""Portable ReDiR training-data schemas, builders, and validation."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Iterator, Mapping


class DataContractError(ValueError):
    """Raised when a ReDiR data bundle violates its frozen contract."""


PRIVATE_METADATA_KEYS = {
    "source_completion_log",
    "source_post_eval_path",
    "source_task_path",
    "source_post_eval_judge_base_url",
    "checkpoint_path",
}
FORBIDDEN_RELEASE_PATTERNS = {
    "local_user_path": re.compile(r"/Users/"),
    "server_data_path": re.compile(r"(?:^|[\s\"'`:(])/data/"),
}


@dataclass(frozen=True)
class BundlePaths:
    root: Path
    identity_train: Path
    identity_dev: Path
    safety_states: Path
    teacher_targets: Path
    pair_manifest: Path
    benign_completions: Path
    margin_family0: Path
    margin_family123: Path
    manifest: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "BundlePaths":
        root = Path(root).resolve()
        return cls(
            root=root,
            identity_train=root / "identity_train.jsonl.gz",
            identity_dev=root / "identity_dev.jsonl.gz",
            safety_states=root / "safety_states.jsonl.gz",
            teacher_targets=root / "teacher_targets.jsonl.gz",
            pair_manifest=root / "pair_manifest.jsonl.gz",
            benign_completions=root / "benign_completions.jsonl.gz",
            margin_family0=root / "margin_bank_family0.jsonl.gz",
            margin_family123=root / "margin_bank_family123.jsonl.gz",
            manifest=root / "manifest.json",
        )

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.identity_train,
            self.identity_dev,
            self.safety_states,
            self.teacher_targets,
            self.pair_manifest,
            self.benign_completions,
            self.margin_family0,
            self.margin_family123,
            self.manifest,
        )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write deterministic gzip-compressed JSONL without a checksum sidecar."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                payload = json.dumps(
                    dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                zipped.write(payload.encode("utf-8") + b"\n")


def strip_private_metadata(value: Any, path: str = "") -> Any:
    """Remove collection-machine provenance that is not used by training."""

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if key in PRIVATE_METADATA_KEYS:
                continue
            if key == "base_url" and "semantic_audit" in path:
                continue
            child = f"{path}.{key}" if path else key
            output[key] = strip_private_metadata(item, child)
        return output
    if isinstance(value, list):
        return [strip_private_metadata(item, path + "[]") for item in value]
    return value


def _sensitive_matches(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            yield from _sensitive_matches(item, child)
    elif isinstance(value, list):
        for item in value:
            yield from _sensitive_matches(item, path + "[]")
    elif isinstance(value, str):
        for label, pattern in FORBIDDEN_RELEASE_PATTERNS.items():
            if pattern.search(value):
                yield path, label


def _copy_jsonl(source: Path, destination: Path) -> list[dict[str, Any]]:
    rows = [strip_private_metadata(row) for row in read_jsonl(source)]
    write_jsonl_gz(destination, rows)
    return rows


def merge_candidate_states(
    candidates: list[dict[str, Any]],
    audited_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach audited targets to safety states while retaining benign candidates."""

    target_by_state = {
        str(row.get("state_id") or ""): row for row in audited_targets
    }
    if "" in target_by_state or len(target_by_state) != len(audited_targets):
        raise DataContractError("teacher targets must have unique, non-empty state_id")

    merged: list[dict[str, Any]] = []
    observed_safety_states: set[str] = set()
    for candidate in candidates:
        row = strip_private_metadata(dict(candidate))
        state_id = str(row.get("state_id") or "")
        route = str(row.get("v485_route") or row.get("route") or "")
        if route == "protocol_safety":
            teacher = target_by_state.get(state_id)
            if teacher is None:
                raise DataContractError(
                    f"safety state has no audited target record: {state_id}"
                )
            row.update(strip_private_metadata(teacher))
            observed_safety_states.add(state_id)
        row.setdefault("task_id", row.get("task_key"))
        row.setdefault("messages", row.get("student_state_messages"))
        row.setdefault("tools", row.get("available_tools"))
        row.setdefault(
            "route", "safety" if route == "protocol_safety" else "benign"
        )
        row.setdefault("targets", row.get("v78_native_refusal_targets") or [])
        merged.append(row)

    missing = sorted(set(target_by_state) - observed_safety_states)
    if missing:
        raise DataContractError(
            f"audited target states are absent from candidates: {missing[:5]}"
        )
    return merged


def _clean_teacher_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for source in rows:
        row = strip_private_metadata(dict(source))
        row["task_id"] = source.get("task_key")
        row["teacher_messages"] = source.get("v48_teacher_exact_state_messages") or []
        row["targets"] = source.get("v78_native_refusal_targets") or []
        row["availability"] = source.get("v78_target_availability") or {}
        row["heldout_test_used"] = bool(source.get("heldout15_used"))
        cleaned.append(row)
    return cleaned


def _clean_manifest_rows(
    manifest: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    siblings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = {
        (str(row.get("state_id")), str(row.get("target_id"))) for row in selected
    }
    sibling_ids = {
        (str(row.get("state_id")), str(row.get("target_id"))) for row in siblings
    }
    output: list[dict[str, Any]] = []
    for source in manifest:
        row = strip_private_metadata(dict(source))
        pair = (str(row.get("state_id")), str(row.get("target_id")))
        row["training_role"] = (
            "selected"
            if pair in selected_ids
            else "sibling"
            if pair in sibling_ids
            else "inactive"
        )
        output.append(row)
    return output


def _clean_margin_rows(
    rows: list[dict[str, Any]], *, expected_rounds: set[int]
) -> list[dict[str, Any]]:
    rounds = {int(row.get("round_index", -1)) for row in rows}
    if rounds != expected_rounds:
        raise DataContractError(
            f"margin bank rounds {rounds} != {expected_rounds}"
        )
    output: list[dict[str, Any]] = []
    for source in rows:
        row = strip_private_metadata(dict(source))
        row["source_checkpoint"] = row.get("checkpoint", "reference")
        row["probe_kind"] = row.get("stage", "fixed_prefix")
        output.append(row)
    return output


def build_bundle(
    *,
    identity_train: Path,
    identity_dev: Path,
    candidate_states: Path,
    audited_targets: Path,
    pair_manifest: Path,
    benign_completions: Path,
    selected_manifest: Path | None,
    sibling_targets: Path | None,
    margin_family0: Path,
    margin_family123: Path,
    output_dir: Path,
    source: str,
) -> dict[str, Any]:
    paths = BundlePaths.from_root(output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    identity_train_rows = _copy_jsonl(identity_train, paths.identity_train)
    identity_dev_rows = _copy_jsonl(identity_dev, paths.identity_dev)
    candidate_rows = read_jsonl(candidate_states)
    target_rows = read_jsonl(audited_targets)
    merged_rows = merge_candidate_states(candidate_rows, target_rows)
    write_jsonl_gz(paths.safety_states, merged_rows)
    write_jsonl_gz(paths.teacher_targets, _clean_teacher_rows(target_rows))

    manifest_rows = read_jsonl(pair_manifest)
    selected_rows = read_jsonl(selected_manifest) if selected_manifest else []
    sibling_rows = read_jsonl(sibling_targets) if sibling_targets else []
    write_jsonl_gz(
        paths.pair_manifest,
        _clean_manifest_rows(manifest_rows, selected_rows, sibling_rows),
    )
    benign_rows = _copy_jsonl(benign_completions, paths.benign_completions)

    family0_rows = _clean_margin_rows(
        read_jsonl(margin_family0), expected_rounds={0}
    )
    family123_rows = _clean_margin_rows(
        read_jsonl(margin_family123), expected_rounds={1, 2, 3}
    )
    write_jsonl_gz(paths.margin_family0, family0_rows)
    write_jsonl_gz(paths.margin_family123, family123_rows)

    safety_rows = [row for row in merged_rows if row.get("route") == "safety"]
    benign_candidates = [row for row in merged_rows if row.get("route") == "benign"]
    target_bearing = [row for row in safety_rows if row.get("targets")]
    unique_targets = {
        str(target.get("target_id"))
        for row in target_bearing
        for target in row.get("targets", [])
    }
    manifest_payload = {
        "format": "redir-training-data",
        "format_version": 1,
        "source": source,
        "base_model_family": "qwen3.5-9b",
        "counts": {
            "identity_train": len(identity_train_rows),
            "identity_train_tasks": len(
                {str(row.get("task_key") or "") for row in identity_train_rows}
                - {""}
            ),
            "identity_dev": len(identity_dev_rows),
            "identity_dev_tasks": len(
                {str(row.get("task_key") or "") for row in identity_dev_rows}
                - {""}
            ),
            "candidate_states": len(merged_rows),
            "safety_states": len(safety_rows),
            "safety_tasks": len(
                {str(row.get("task_key") or "") for row in safety_rows} - {""}
            ),
            "benign_candidates": len(benign_candidates),
            "target_bearing_states": len(target_bearing),
            "target_supported_tasks": len(
                {str(row.get("task_key") or "") for row in target_bearing}
                - {""}
            ),
            "unique_targets": len(unique_targets),
            "target_pairs": sum(len(row.get("targets", [])) for row in target_bearing),
            "pair_manifest": len(manifest_rows),
            "active_pairs": sum(bool(row.get("active")) for row in manifest_rows),
            "benign_completions": len(benign_rows),
            "margin_family0": len(family0_rows),
            "margin_family123": len(family123_rows),
        },
        "notes": {
            "base_model_weights_included": False,
            "heldout_test_used": False,
        },
    }
    paths.manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_bundle(paths.root)
    return manifest_payload


def validate_bundle(root: str | Path) -> dict[str, Any]:
    paths = BundlePaths.from_root(root)
    missing = [str(path) for path in paths.required_files() if not path.is_file()]
    if missing:
        raise DataContractError(f"training bundle is missing files: {missing}")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    counts = dict(manifest.get("counts") or {})
    observed = {
        "identity_train": len(read_jsonl(paths.identity_train)),
        "identity_dev": len(read_jsonl(paths.identity_dev)),
        "candidate_states": len(read_jsonl(paths.safety_states)),
        "pair_manifest": len(read_jsonl(paths.pair_manifest)),
        "benign_completions": len(read_jsonl(paths.benign_completions)),
        "margin_family0": len(read_jsonl(paths.margin_family0)),
        "margin_family123": len(read_jsonl(paths.margin_family123)),
    }
    mismatches = {
        key: {"expected": counts.get(key), "observed": value}
        for key, value in observed.items()
        if counts.get(key) != value
    }
    if mismatches:
        raise DataContractError(f"training bundle count mismatch: {mismatches}")
    if manifest.get("base_model_family") != "qwen3.5-9b":
        raise DataContractError("this release supports only qwen3.5-9b")
    sensitive: list[dict[str, Any]] = []
    for file_path in paths.required_files():
        if file_path.suffix != ".gz":
            continue
        for row_number, row in enumerate(iter_jsonl(file_path), start=1):
            for field, label in _sensitive_matches(row):
                sensitive.append(
                    {
                        "file": file_path.name,
                        "row": row_number,
                        "field": field,
                        "kind": label,
                    }
                )
                if len(sensitive) >= 20:
                    break
            if len(sensitive) >= 20:
                break
        if len(sensitive) >= 20:
            break
    if sensitive:
        raise DataContractError(
            "collected data contains private paths or endpoints: "
            + json.dumps(sensitive, ensure_ascii=False)
        )
    return manifest


def materialize_engine_inputs(root: str | Path, output: str | Path) -> BundlePaths:
    """Decompress a validated bundle for the current training engine."""

    paths = BundlePaths.from_root(root)
    validate_bundle(paths.root)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping = {
        paths.identity_train: output / "identity_train.jsonl",
        paths.identity_dev: output / "identity_dev.jsonl",
        paths.safety_states: output / "safety_states.jsonl",
        paths.teacher_targets: output / "teacher_targets.jsonl",
        paths.pair_manifest: output / "pair_manifest.jsonl",
        paths.benign_completions: output / "benign_completions.jsonl",
        paths.margin_family0: output / "margin_family0.jsonl",
        paths.margin_family123: output / "margin_family123.jsonl",
    }
    for source, destination in mapping.items():
        with gzip.open(source, "rb") as compressed, destination.open("wb") as raw:
            shutil.copyfileobj(compressed, raw)
    return BundlePaths(
        root=output,
        identity_train=mapping[paths.identity_train],
        identity_dev=mapping[paths.identity_dev],
        safety_states=mapping[paths.safety_states],
        teacher_targets=mapping[paths.teacher_targets],
        pair_manifest=mapping[paths.pair_manifest],
        benign_completions=mapping[paths.benign_completions],
        margin_family0=mapping[paths.margin_family0],
        margin_family123=mapping[paths.margin_family123],
        manifest=paths.manifest,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    result = validate_bundle(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
