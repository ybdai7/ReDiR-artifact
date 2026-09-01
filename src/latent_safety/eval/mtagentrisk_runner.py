from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from latent_safety.paths import MTAGENTRISK_EVAL_ROOT, REPO_ROOT


@dataclass(frozen=True)
class MTAgentRiskRun:
    task_path: Path
    outputs_path: Path
    config_path: Path = REPO_ROOT / "configs" / "mtagentrisk_local.toml"
    use_experience: Path | None = None
    server_hostname: str = "localhost"
    agent_llm_config: str = "agent"
    env_llm_config: str = "env"

    def command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "latent_safety.eval.mtagentrisk.run_eval",
            "--task-path",
            str(self.task_path),
            "--agent-llm-config",
            self.agent_llm_config,
            "--agent-llm-config-file",
            str(self.config_path),
            "--env-llm-config",
            self.env_llm_config,
            "--env-llm-config-file",
            str(self.config_path),
            "--outputs-path",
            str(self.outputs_path),
            "--server-hostname",
            self.server_hostname,
        ]
        if self.use_experience is not None:
            cmd.extend(["--use-experience", str(self.use_experience)])
        return cmd


def build_env(extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"
    if extra_env:
        env.update(extra_env)
    return env


def run_mtagentrisk(run: MTAgentRiskRun, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    run.outputs_path.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        run.command(),
        cwd=REPO_ROOT,
        env=build_env(),
        text=True,
        check=check,
    )


def format_command(command: Sequence[str]) -> str:
    return " ".join(_quote_arg(arg) for arg in command)


def _quote_arg(arg: str) -> str:
    if not arg or any(ch.isspace() for ch in arg):
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    return arg
