"""Centralized path resolution for ToolShield.

Resolution order: environment variable → CLI argument → default relative to repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

from redir.paths import MTAGENTRISK_EVAL_ROOT, OUTPUTS_ROOT, REPO_ROOT


def repo_root() -> Path:
    """Return the repository root directory."""
    env = os.getenv("REDIR_REPO_ROOT") or os.getenv("TOOLSHIELD_REPO_ROOT")
    if env:
        return Path(env)
    return REPO_ROOT


def default_agent_config() -> Path:
    """Return the default agent config TOML path."""
    return MTAGENTRISK_EVAL_ROOT / "agent_config" / "config.toml"


def default_eval_dir() -> Path:
    """Return the default evaluation directory."""
    return MTAGENTRISK_EVAL_ROOT


def default_output_dir() -> Path:
    """Return the default output directory for generated tasks."""
    return OUTPUTS_ROOT / "toolshield"


def default_seed_sql() -> Path:
    """Return the path to the bundled Postgres seed SQL."""
    return Path(__file__).resolve().parent / "data" / "seed.sql"
