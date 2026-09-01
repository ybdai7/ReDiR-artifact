"""Assemble newly collected ReDiR rows into a validated training bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redir.build import collected_sources
from redir.data import build_bundle


def _margin_path(root: Path, stem: str) -> Path:
    for suffix in (".jsonl.gz", ".jsonl"):
        path = root / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {stem}.jsonl[.gz] under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing newly collected component JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the normalized collected bundle.",
    )
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output.resolve()
    sources = collected_sources(source)
    result = build_bundle(
        **sources,
        margin_family0=_margin_path(source, "margin_bank_family0"),
        margin_family123=_margin_path(source, "margin_bank_family123"),
        output_dir=output,
        source="user-collected-qwen3.5-9b-data",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
