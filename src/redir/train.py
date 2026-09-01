"""Run the complete ReDiR pipeline from an external Qwen3.5-9B base model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

from redir.build import collected_sources
from redir.data import build_bundle, materialize_engine_inputs, validate_bundle


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


def _set_model_paths(config: dict[str, Any], model: str, parent: str | None) -> None:
    model_config = config["model"]
    model_config["model_name"] = model
    model_config["load_model_path"] = parent
    model_config["weaver"]["model_name"] = model
    model_config["trigger"]["model_name"] = model


def _disable_versioned_training(config: dict[str, Any]) -> None:
    for key, value in config.items():
        if key.startswith("v") and isinstance(value, dict) and "enabled" in value:
            value["enabled"] = False


def _remove_environment_interpolations(config: dict[str, Any]) -> None:
    """Replace release-template environment placeholders with local values."""

    judge = config["v78_canonical_target_action_support"]["target_audit"][
        "semantic_judge"
    ]
    judge.update(
        {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1-mini",
            "api_key_file": None,
        }
    )
    config["run"]["repo_root"] = str(Path(__file__).resolve().parents[2])


def _warmup_config(
    template: dict[str, Any],
    *,
    model: str,
    data: Any,
    output: Path,
    semantic: dict[str, Any],
    data_manifest: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(template)
    counts = dict(data_manifest["counts"])
    _disable_versioned_training(config)
    _remove_environment_interpolations(config)
    _set_model_paths(config, model, None)
    config["model"]["weaver"]["lora_config"]["lora_dropout"] = 0.1
    config["dataset"].update(
        {
            "train_path": str(data.identity_train),
            "dev_path": str(data.identity_dev),
            "candidate_field": "training_eligible",
            "require_training_candidate": True,
        }
    )
    config["v78_canonical_target_action_support"]["teacher_views_path"] = None
    config["v79_ce_mode_creation"].update(
        {
            "frozen_target_dir": None,
            "base_target_dir": None,
            "recoverable_audit_path": None,
        }
    )
    config["v796_displacement_frontier"].update(
        {
            "manifest_path": None,
            "r0_audit_path": None,
            "frozen_benign_path": None,
            "r0_metrics_path": None,
            "gram_summary_path": None,
        }
    )
    config["v73_native_identity_warmup"] = {
        "enabled": True,
        "records_per_update": int(semantic["warmup"]["records_per_update"]),
        "max_clean_refusal_fraction": 0.20,
        "expected_train_tasks": int(counts["identity_train_tasks"]),
        "expected_dev_tasks": int(counts["identity_dev_tasks"]),
        "min_train_records": int(counts["identity_train"]),
        "min_dev_records": int(counts["identity_dev"]),
        "record_floor_contract": "portable",
        "attack_audit_path": str(data.teacher_targets),
    }
    run = config["run"]
    run.update(
        {
            "output_dir": str(output),
            "reasoner_device": "cuda:0",
            "weaver_device": "cuda:1",
            "train_weaver": True,
            "train_trigger": False,
        }
    )
    run["distributed"].update(
        {"enabled": False, "expected_workers": 1, "gpus_per_worker": 2}
    )
    training = run["training"]
    training.update(
        {
            "training_loop": "v73_native_identity_warmup",
            "branch_name": "identity_warmup",
            "rollout_source": "dataset_source",
            "completion_source": "audited_safe_native_completion",
            "teacher_source": "student_state_no_latent",
            "loss_mode": "reverse_kl",
            "state_source": "saved_final_agent_state",
            "mask_mode": "schema_dynamic",
            "mask_profile": "full_completion",
            "mask_unknown_policy": "skip",
            "v6_protocol_mode": "native",
            "num_epochs": int(semantic["warmup"]["epochs"]),
            "learning_rate": float(semantic["warmup"]["learning_rate"]),
            "optimizer_name": "group_normalized_sgd",
            "optimizer_weight_decay": 0.0,
            "optimizer_momentum": 0.0,
            "group_normalized_sgd_target_update_norms": {
                "prompt_query": 1.0e-4,
                "lora": 2.5e-4,
            },
            "v5_trainable_scope": "prompt_query_plus_all_lora_ab",
            "max_new_tokens": 0,
            "initial_max_new_tokens": 4096,
            "retry_max_new_tokens": 4096,
            "assistant_generation_prefix": "<think>\n",
            "do_sample": False,
            "temperature": 0.0,
            "max_train_examples": 0,
            "max_dev_examples": 0,
            "skip_static_coverage_check": True,
            "min_parsed_rate_for_full_train": 0.0,
            "min_non_empty_rate_for_full_train": 1.0,
            "fail_on_zero_step": True,
        }
    )
    return config


def _safety_config(
    template: dict[str, Any],
    *,
    model: str,
    parent: Path,
    data: Any,
    prepare_dir: Path,
    output: Path,
    semantic: dict[str, Any],
    data_manifest: dict[str, Any],
    mode: str,
    learning_rate: float,
    max_updates: int,
) -> dict[str, Any]:
    config = deepcopy(template)
    _remove_environment_interpolations(config)
    counts = dict(data_manifest["counts"])
    _set_model_paths(config, model, str(parent))
    config["model"]["weaver"]["lora_config"]["lora_dropout"] = 0.0
    config["dataset"].update(
        {
            "train_path": str(data.safety_states),
            "dev_path": None,
            "candidate_field": "v5_precise_mask_candidate",
            "require_training_candidate": True,
        }
    )
    config["v5_precise_mask"]["enabled"] = True
    config["v72_stratified_opd"]["enabled"] = True
    config["v76_state_conditioned_opd"]["enabled"] = False
    config["v78_canonical_target_action_support"].update(
        {
            "enabled": True,
            "teacher_views_path": str(data.safety_states),
        }
    )
    config["v78_canonical_target_action_support"]["target_audit"][
        "semantic_judge"
    ]["api_key_file"] = None
    config["v79_ce_mode_creation"].update(
        {
            "enabled": True,
            "frozen_target_dir": str(prepare_dir),
            "base_target_dir": str(prepare_dir),
            "recoverable_audit_path": None,
            "expected_target_count": int(counts["unique_targets"]),
            "expected_supported_task_count": int(
                counts["target_supported_tasks"]
            ),
            "expected_elicited_target_count": 0,
            "expected_format_instructed_task_count": 0,
        }
    )
    config["v796_displacement_frontier"].update(
        {
            "enabled": False,
            "manifest_path": str(prepare_dir / "v797_portable_manifest.jsonl"),
            "r0_audit_path": str(prepare_dir / "v797_portable_r0_pair_audit.jsonl"),
            "frozen_benign_path": str(
                prepare_dir / "v797_portable_frozen_benign.jsonl"
            ),
            "r0_metrics_path": str(prepare_dir / "v797_portable_r0_metrics.jsonl"),
            "gram_summary_path": None,
        }
    )
    final = config["v797_fully_fitted_offline_control"]
    final.update(
        {
            "enabled": True,
            "mode": mode,
            "max_updates": max_updates,
            "warmup_updates": int(semantic["training"]["warmup_updates"]),
            "grad_clip_norm": float(semantic["training"]["gradient_clip"]),
            "safety_per_batch": int(semantic["training"]["safety_per_batch"]),
            "benign_per_batch": int(semantic["training"]["benign_per_batch"]),
            "benign_lambda": float(semantic["training"]["benign_weight"]),
            "portable_data_contract": True,
            "preaudited_target_contract": True,
            "fixed_margin_enabled": True,
            "allow_cross_task_benign_replacement": False,
            "expected_safety_states": int(counts["safety_states"]),
            "expected_safety_tasks": int(counts["safety_tasks"]),
            "expected_manifest_pairs": int(counts["target_pairs"]),
            "expected_active_pairs": -1,
            "expected_supported_states": int(counts["target_bearing_states"]),
            "expected_selected_pairs": -1,
            "expected_sibling_pairs": -1,
        }
    )
    final["fixed_margin_enabled"] = False
    config["v796_posthoc_diagnostics"].update(
        {"enabled": False, "probe_paths": []}
    )
    run = config["run"]
    run["output_dir"] = str(output)
    run["distributed"].update(
        {"enabled": True, "expected_workers": 2, "gpus_per_worker": 2}
    )
    training = run["training"]
    training.update(
        {
            "training_loop": "v5_precise_mask_opd",
            "branch_name": "redir_safety_training",
            "loss_mode": "v797_weighted_target_ce_plus_fixed_benign",
            "learning_rate": learning_rate,
            "v5_trainable_scope": "prompt_query_plus_lora_ab_last4_attn",
            "v6_expected_safety_states": int(counts["safety_states"]),
            "v6_expected_safety_tasks": int(counts["safety_tasks"]),
        }
    )
    return config


def _run_stage(
    *,
    command: list[str],
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


def _trainer_command(config: Path, *, distributed: bool, workers: int) -> list[str]:
    if distributed:
        torchrun = Path(sys.executable).with_name("torchrun")
        executable = str(torchrun) if torchrun.is_file() else shutil.which("torchrun")
        executable = executable or "torchrun"
        return [
            executable,
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={workers}",
            "-m",
            "redir.engine.trainer",
            "--config",
            str(config),
        ]
    return [
        sys.executable,
        "-m",
        "redir.engine.trainer",
        "--config",
        str(config),
    ]


def _select_learning_rate(calibration_dirs: dict[float, Path], config: dict[str, Any]) -> float:
    passing: list[float] = []
    rules = config["calibration"]
    for learning_rate, directory in calibration_dirs.items():
        summary_path = directory / "v797_calibration_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline_ce = float(summary["baseline_selected_ce"])
        final_ce = float(summary["final_selected_ce"])
        baseline_benign = float(summary["baseline_benign_loss"])
        final_benign = float(summary["final_benign_loss"])
        displacement = float(summary["final_displacement"])
        if (
            displacement <= float(rules["maximum_displacement"])
            and final_ce <= float(rules["maximum_target_ce_ratio"]) * baseline_ce
            and final_benign
            <= float(rules["maximum_benign_loss_ratio"]) * baseline_benign
        ):
            passing.append(learning_rate)
    if not passing:
        raise RuntimeError("no calibration arm passed the frozen acceptance gate")
    return max(passing)


def _build_collected_bundle(source: Path, output: Path) -> None:
    sources = collected_sources(source)
    build_bundle(
        **sources,
        output_dir=output,
        source="user-collected-qwen3.5-9b-data",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/redir"))
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--collected-dir", type=Path, required=True)
    parser.add_argument("--cuda-devices")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output = (repo_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    semantic_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    semantic = _load_yaml(semantic_path)
    cuda_devices = args.cuda_devices or str(semantic["runtime"]["cuda_devices"])
    gpu_list = [value.strip() for value in cuda_devices.split(",") if value.strip()]
    if len(gpu_list) != 4:
        raise ValueError("the full ReDiR pipeline requires exactly four visible GPUs")

    output.mkdir(parents=True, exist_ok=True)
    data_root = output / "collected_data"
    _build_collected_bundle(args.collected_dir.resolve(), data_root)
    data_manifest = validate_bundle(data_root)
    engine_data = materialize_engine_inputs(data_root, output / "engine_data")

    template = _load_yaml(repo_root / "src" / "redir" / "engine_template.yaml")
    configs_dir = output / "configs"

    warmup_dir = output / "warmup"
    warmup_config = _warmup_config(
        template,
        model=args.model,
        data=engine_data,
        output=warmup_dir,
        semantic=semantic,
        data_manifest=data_manifest,
    )
    warmup_config_path = configs_dir / "warmup.yaml"
    _write_yaml(warmup_config_path, warmup_config)
    _run_stage(
        command=_trainer_command(warmup_config_path, distributed=False, workers=1),
        repo_root=repo_root,
        cuda_devices=",".join(gpu_list[:2]),
        dry_run=args.dry_run,
    )
    parent = warmup_dir / "model"

    prepare_dir = output / "prepare"
    prepare_config = _safety_config(
        template,
        model=args.model,
        parent=parent,
        data=engine_data,
        prepare_dir=prepare_dir,
        output=prepare_dir,
        semantic=semantic,
        data_manifest=data_manifest,
        mode="prepare",
        learning_rate=float(semantic["calibration"]["learning_rates"][0]),
        max_updates=int(semantic["calibration"]["updates_per_arm"]),
    )
    prepare_config_path = configs_dir / "prepare.yaml"
    _write_yaml(prepare_config_path, prepare_config)
    _run_stage(
        command=_trainer_command(prepare_config_path, distributed=True, workers=2),
        repo_root=repo_root,
        cuda_devices=cuda_devices,
        dry_run=args.dry_run,
    )

    calibration_dirs: dict[float, Path] = {}
    for index, learning_rate in enumerate(semantic["calibration"]["learning_rates"], start=1):
        learning_rate = float(learning_rate)
        stage_dir = output / "calibration" / f"arm_{index}"
        calibration_dirs[learning_rate] = stage_dir
        stage_config = _safety_config(
            template,
            model=args.model,
            parent=parent,
            data=engine_data,
            prepare_dir=prepare_dir,
            output=stage_dir,
            semantic=semantic,
            data_manifest=data_manifest,
            mode="calibration",
            learning_rate=learning_rate,
            max_updates=int(semantic["calibration"]["updates_per_arm"]),
        )
        stage_config_path = configs_dir / f"calibration_{index}.yaml"
        _write_yaml(stage_config_path, stage_config)
        _run_stage(
            command=_trainer_command(stage_config_path, distributed=True, workers=2),
            repo_root=repo_root,
            cuda_devices=cuda_devices,
            dry_run=args.dry_run,
        )

    selected_lr = (
        max(float(value) for value in semantic["calibration"]["learning_rates"])
        if args.dry_run
        else _select_learning_rate(calibration_dirs, semantic)
    )
    if not args.dry_run:
        selection_path = output / "calibration" / "selection.json"
        selection_path.write_text(
            json.dumps({"selected_learning_rate": selected_lr}, indent=2) + "\n",
            encoding="utf-8",
        )

    train_dir = output / "train"
    formal_config = _safety_config(
        template,
        model=args.model,
        parent=parent,
        data=engine_data,
        prepare_dir=prepare_dir,
        output=train_dir,
        semantic=semantic,
        data_manifest=data_manifest,
        mode="formal",
        learning_rate=selected_lr,
        max_updates=int(semantic["training"]["max_updates"]),
    )
    formal_config_path = configs_dir / "formal.yaml"
    _write_yaml(formal_config_path, formal_config)
    _run_stage(
        command=_trainer_command(formal_config_path, distributed=True, workers=2),
        repo_root=repo_root,
        cuda_devices=cuda_devices,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("dry-run complete; all stage configs were generated")
        return

    selected_step = int(semantic["selection"]["checkpoint_step"])
    selected_checkpoint = train_dir / f"model_update_{selected_step}"
    if not selected_checkpoint.is_dir():
        raise RuntimeError(f"selected checkpoint was not produced: {selected_checkpoint}")
    final_link = output / "final"
    if final_link.exists() or final_link.is_symlink():
        final_link.unlink()
    final_link.symlink_to(selected_checkpoint.relative_to(output), target_is_directory=True)
    (output / "checkpoint_selection.json").write_text(
        json.dumps(
            {
                "selected_step": selected_step,
                "selected_checkpoint": str(selected_checkpoint),
                "selected_learning_rate": selected_lr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"final checkpoint: {final_link}")


if __name__ == "__main__":
    main()
