"""Build a ReDiR data bundle from newly collected JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redir.data import build_bundle


def _resolve_file(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"none of {names!r} exists under {root}")


def collected_sources(root: Path) -> dict[str, Path | None]:
    data = root / "data" if (root / "data").is_dir() else root
    return {
        "identity_train": _resolve_file(data, "identity_train.jsonl.gz", "identity_train.jsonl"),
        "identity_dev": _resolve_file(data, "identity_dev.jsonl.gz", "identity_dev.jsonl"),
        "candidate_states": _resolve_file(
            data,
            "formal_candidate_train.jsonl.gz",
            "candidate_states.jsonl.gz",
            "candidate_states.jsonl",
            "safety_states.jsonl.gz",
            "safety_states.jsonl",
        ),
        "audited_targets": _resolve_file(
            data, "teacher_audited_targets.jsonl.gz", "teacher_targets.jsonl.gz", "teacher_targets.jsonl"
        ),
        "pair_manifest": _resolve_file(
            data, "fixed_batch_manifest.jsonl.gz", "pair_manifest.jsonl.gz", "pair_manifest.jsonl"
        ),
        "benign_completions": _resolve_file(
            data, "frozen_benign_completions.jsonl.gz", "benign_completions.jsonl.gz", "benign_completions.jsonl"
        ),
        "selected_manifest": next(
            (
                path
                for path in (
                    data / "selected_training_manifest.jsonl.gz",
                    data / "selected_manifest.jsonl.gz",
                    data / "selected_manifest.jsonl",
                )
                if path.is_file()
            ),
            None,
        ),
        "sibling_targets": next(
            (path for path in (data / "sibling_targets.jsonl.gz", data / "sibling_targets.jsonl") if path.is_file()),
            None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    sources = collected_sources(args.source.resolve())
    result = build_bundle(
        **sources,
        output_dir=args.output.resolve(),
        source="user-collected-qwen3.5-9b-data",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
