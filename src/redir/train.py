"""Run the complete final ReDiR training pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

from redir.data import materialize_engine_inputs, validate_bundle


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _model_config(
    semantic: dict[str, Any],
    *,
    base_model: str,
    checkpoint: Path | None,
    lora_dropout: float,
) -> dict[str, Any]:
    model = semantic["model"]
    return {
        "base_model": base_model,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "latent_tokens": int(model["latent_tokens"]),
        "lora_rank": int(model["lora_rank"]),
        "lora_alpha": int(model["lora_alpha"]),
        "lora_dropout": float(lora_dropout),
    }


def _warmup_config(
    semantic: dict[str, Any],
    *,
    base_model: str,
    data: Any,
    output: Path,
) -> dict[str, Any]:
    warmup = semantic["warmup"]
    return {
        "stage": "identity_warmup",
        "seed": 42,
        "output_dir": str(output),
        "distributed": False,
        "model": _model_config(
            semantic,
            base_model=base_model,
            checkpoint=None,
            lora_dropout=0.1,
        ),
        "data": {
            "identity_train": str(data.identity_train),
            "identity_dev": str(data.identity_dev),
        },
        "optimization": {
            "epochs": int(warmup["epochs"]),
            "learning_rate": float(warmup["learning_rate"]),
            "records_per_update": int(warmup["records_per_update"]),
            "gradient_clip": float(semantic["training"]["gradient_clip"]),
            "trainable_scope": "all_weaver_lora",
        },
    }


def _prepare_config(
    *,
    data: Any,
    output: Path,
    counts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "prepare",
        "output_dir": str(output),
        "data": {
            "safety_states": str(data.safety_states),
            "pair_manifest": str(data.pair_manifest),
            "benign_completions": str(data.benign_completions),
        },
        "expected_counts": {
            "safety_states": int(counts["safety_states"]),
            "safety_tasks": int(counts["safety_tasks"]),
            "target_pairs": int(counts["target_pairs"]),
            "benign_completions": int(counts["benign_completions"]),
        },
    }


def _safety_config(
    semantic: dict[str, Any],
    *,
    stage: str,
    base_model: str,
    checkpoint: Path,
    data: Any,
    prepared: Path,
    output: Path,
    learning_rate: float,
    max_updates: int,
) -> dict[str, Any]:
    training = semantic["training"]
    return {
        "stage": stage,
        "seed": 42,
        "output_dir": str(output),
        "distributed": True,
        "model": _model_config(
            semantic,
            base_model=base_model,
            checkpoint=checkpoint,
            lora_dropout=0.0,
        ),
        "data": {
            "safety_states": str(data.safety_states),
            "training_pairs": str(prepared / "training_pairs.jsonl"),
            "benign_records": str(prepared / "benign_records.jsonl"),
        },
        "optimization": {
            "learning_rate": float(learning_rate),
            "max_updates": int(max_updates),
            "safety_per_batch": int(training["safety_per_batch"]),
            "benign_per_batch": int(training["benign_per_batch"]),
            "benign_weight": float(training["benign_weight"]),
            "gradient_clip": float(training["gradient_clip"]),
            "checkpoint_interval": int(training["checkpoint_interval"]),
            "trainable_scope": "last_four_attention_lora",
        },
    }


def _trainer_command(config: Path, *, distributed: bool, workers: int) -> list[str]:
    if not distributed:
        return [sys.executable, "-m", "redir.engine.trainer", "--config", str(config)]
    torchrun = Path(sys.executable).with_name("torchrun")
    executable = str(torchrun) if torchrun.is_file() else shutil.which("torchrun")
    return [
        executable or "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={workers}",
        "-m",
        "redir.engine.trainer",
        "--config",
        str(config),
    ]


def _run_stage(
    command: list[str],
    *,
    repo_root: Path,
    cuda_devices: str,
    dry_run: bool,
) -> None:
    print("+", " ".join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = cuda_devices
    environment["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    subprocess.run(command, cwd=repo_root, env=environment, check=True)


def _select_learning_rate(
    calibration_dirs: dict[float, Path],
    semantic: dict[str, Any],
) -> float:
    rules = semantic["calibration"]
    passing: list[float] = []
    for learning_rate, directory in calibration_dirs.items():
        summary = json.loads(
            (directory / "calibration_summary.json").read_text(encoding="utf-8")
        )
        if (
            float(summary["final_displacement"])
            <= float(rules["maximum_displacement"])
            and float(summary["final_target_ce"])
            <= float(rules["maximum_target_ce_ratio"])
            * float(summary["baseline_target_ce"])
            and float(summary["final_benign_kl"])
            <= float(rules["maximum_benign_loss_ratio"])
            * max(float(summary["baseline_benign_kl"]), 1e-12)
        ):
            passing.append(learning_rate)
    if not passing:
        raise RuntimeError("no calibration arm passed the acceptance gate")
    return max(passing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--collected-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/redir"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--cuda-devices")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output = (
        (repo_root / args.output).resolve()
        if not args.output.is_absolute()
        else args.output.resolve()
    )
    config_path = (
        (repo_root / args.config).resolve()
        if not args.config.is_absolute()
        else args.config.resolve()
    )
    semantic = _load_yaml(config_path)
    if str(semantic["model"].get("family")) != "qwen3.5-9b":
        raise ValueError("the final ReDiR recipe supports only qwen3.5-9b")
    cuda_devices = args.cuda_devices or str(semantic["runtime"]["cuda_devices"])
    gpu_list = [value.strip() for value in cuda_devices.split(",") if value.strip()]
    workers = int(semantic["runtime"]["workers"])
    gpus_per_worker = int(semantic["runtime"]["gpus_per_worker"])
    if len(gpu_list) != workers * gpus_per_worker or gpus_per_worker != 2:
        raise ValueError(
            "ReDiR requires two workers with two GPUs each (reasoner + Weaver)"
        )

    source_bundle = args.collected_dir.resolve()
    data_manifest = validate_bundle(source_bundle)
    counts = dict(data_manifest["counts"])
    output.mkdir(parents=True, exist_ok=True)
    data = materialize_engine_inputs(source_bundle, output / "training_data")
    configs = output / "configs"

    warmup_output = output / "warmup"
    warmup = _warmup_config(
        semantic,
        base_model=args.model,
        data=data,
        output=warmup_output,
    )
    warmup_path = configs / "warmup.yaml"
    _write_yaml(warmup_path, warmup)
    _run_stage(
        _trainer_command(warmup_path, distributed=False, workers=1),
        repo_root=repo_root,
        cuda_devices=",".join(gpu_list[:2]),
        dry_run=args.dry_run,
    )
    parent = warmup_output / "model"

    prepared = output / "prepare"
    prepare = _prepare_config(data=data, output=prepared, counts=counts)
    prepare_path = configs / "prepare.yaml"
    _write_yaml(prepare_path, prepare)
    _run_stage(
        [sys.executable, "-m", "redir.engine.prepare", "--config", str(prepare_path)],
        repo_root=repo_root,
        cuda_devices=",".join(gpu_list[:2]),
        dry_run=args.dry_run,
    )

    calibration_dirs: dict[float, Path] = {}
    for index, raw_learning_rate in enumerate(
        semantic["calibration"]["learning_rates"], start=1
    ):
        learning_rate = float(raw_learning_rate)
        stage_output = output / "calibration" / f"arm_{index}"
        calibration_dirs[learning_rate] = stage_output
        calibration = _safety_config(
            semantic,
            stage="calibration",
            base_model=args.model,
            checkpoint=parent,
            data=data,
            prepared=prepared,
            output=stage_output,
            learning_rate=learning_rate,
            max_updates=int(semantic["calibration"]["updates_per_arm"]),
        )
        calibration_path = configs / f"calibration_{index}.yaml"
        _write_yaml(calibration_path, calibration)
        _run_stage(
            _trainer_command(calibration_path, distributed=True, workers=workers),
            repo_root=repo_root,
            cuda_devices=cuda_devices,
            dry_run=args.dry_run,
        )

    selected_learning_rate = (
        max(float(value) for value in semantic["calibration"]["learning_rates"])
        if args.dry_run
        else _select_learning_rate(calibration_dirs, semantic)
    )
    if not args.dry_run:
        _write_yaml(
            output / "calibration" / "selection.yaml",
            {"selected_learning_rate": selected_learning_rate},
        )

    training_output = output / "train"
    formal = _safety_config(
        semantic,
        stage="formal",
        base_model=args.model,
        checkpoint=parent,
        data=data,
        prepared=prepared,
        output=training_output,
        learning_rate=selected_learning_rate,
        max_updates=int(semantic["training"]["max_updates"]),
    )
    formal_path = configs / "formal.yaml"
    _write_yaml(formal_path, formal)
    _run_stage(
        _trainer_command(formal_path, distributed=True, workers=workers),
        repo_root=repo_root,
        cuda_devices=cuda_devices,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("dry-run complete; all final ReDiR stage configs were generated")
        return

    selected_step = int(semantic["selection"]["checkpoint_step"])
    selected_checkpoint = training_output / f"model_update_{selected_step}"
    if not selected_checkpoint.is_dir():
        raise RuntimeError(f"selected checkpoint was not produced: {selected_checkpoint}")
    final_link = output / "final"
    if final_link.is_symlink():
        final_link.unlink()
    elif final_link.exists():
        raise FileExistsError(f"refusing to replace non-symlink final path: {final_link}")
    final_link.symlink_to(selected_checkpoint.relative_to(output), target_is_directory=True)
    _write_yaml(
        output / "checkpoint_selection.yaml",
        {
            "selected_step": selected_step,
            "selected_checkpoint": str(selected_checkpoint),
            "selected_learning_rate": selected_learning_rate,
        },
    )
    print(f"final checkpoint: {final_link}")


if __name__ == "__main__":
    main()
