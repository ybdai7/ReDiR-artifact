from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

PACKAGE_ROOT = REPO_ROOT / "src" / "latent_safety"
MTAGENTRISK_EVAL_ROOT = PACKAGE_ROOT / "eval" / "mtagentrisk"
TOOLSHIELD_BASELINE_ROOT = PACKAGE_ROOT / "baselines" / "toolshield"
LATENT_MEMORY_MODELS_ROOT = PACKAGE_ROOT / "models" / "latent_memory"
LATENT_MEMORY_TRAINING_ROOT = PACKAGE_ROOT / "training" / "latent_memory"

UPSTREAM_TOOLSHIELD_ROOT = WORKSPACE_ROOT / "ToolShield"
UPSTREAM_MEMGEN_ROOT = WORKSPACE_ROOT / "MemGen"

CONFIGS_ROOT = REPO_ROOT / "configs"
LATENT_MEMORY_CONFIGS_ROOT = CONFIGS_ROOT / "latent_memory"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
DATA_ROOT = REPO_ROOT / "data"
