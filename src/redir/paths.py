from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

PACKAGE_ROOT = REPO_ROOT / "src" / "redir"
MTAGENTRISK_EVAL_ROOT = PACKAGE_ROOT / "eval" / "mtagentrisk"
TOOLSHIELD_BASELINE_ROOT = PACKAGE_ROOT / "baselines" / "toolshield"
LATENT_MEMORY_MODELS_ROOT = PACKAGE_ROOT / "models" / "latent_memory"

CONFIGS_ROOT = REPO_ROOT / "configs"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
DATA_ROOT = REPO_ROOT / "data"
