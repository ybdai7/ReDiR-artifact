"""The single training engine used by the public ReDiR artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
import yaml

from redir.models.latent_memory.modeling import MemGenModel
from redir.models.latent_memory.native_boundary import (
    render_generation_boundary_prompt,
)
from redir.models.native_model_compat import text_num_hidden_layers


def normalize_assistant_generation_prefix(value: Any) -> str:
    """Normalize the optional native assistant prefix used by training and serving."""

    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in {"", "none", "null", "false"}:
        return ""
    if text.strip().lower() in {"qwen_thinking", "openhands_qwen_thinking"}:
        return "<think>\n"
    return text.replace("\\n", "\n").replace("\\t", "\t")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_primary() -> bool:
    return _rank() == 0


def _initialize_distributed(enabled: bool) -> None:
    if not enabled:
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.cpu())


def _synchronize_gradients(parameters: list[torch.nn.Parameter]) -> None:
    if not dist.is_initialized():
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _devices(distributed: bool) -> tuple[torch.device, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("ReDiR training requires CUDA")
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        reasoner_index = local_rank * 2
        weaver_index = reasoner_index + 1
    else:
        reasoner_index, weaver_index = 0, 1
    if torch.cuda.device_count() <= weaver_index:
        raise RuntimeError(
            f"worker needs CUDA devices {reasoner_index},{weaver_index}; "
            f"only {torch.cuda.device_count()} visible"
        )
    torch.cuda.set_device(reasoner_index)
    return torch.device(f"cuda:{reasoner_index}"), torch.device(f"cuda:{weaver_index}")



def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    base_model = str(model.pop("base_model"))
    checkpoint = model.pop("checkpoint", None)
    latent_tokens = int(model.pop("latent_tokens"))
    lora_rank = int(model.pop("lora_rank"))
    lora_alpha = int(model.pop("lora_alpha"))
    lora_dropout = float(model.pop("lora_dropout", 0.0))
    return {
        "model_name": base_model,
        "load_model_path": checkpoint,
        "model_family": "qwen3.5",
        "attn_implementation": "sdpa",
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 0,
        "latent_injection_mode": "before_assistant_generation_boundary",
        "latent_init": {
            "mode": "token_set_average",
            "seed": int(config.get("seed", 42)),
            "noise_ratio": 0.02,
            "scale_mode": "token_norm_p50_over_sqrt_hidden",
            "apply_to_prompt_latents": True,
            "apply_to_inference_latents": True,
        },
        "latent_output_norm": {
            "enabled": True,
            "target_mode": "token_norm_p50",
            "target_scale": 1.0,
            "eps": 1.0e-6,
        },
        "latent_projection": {"mode": "identity", "require_matching_hidden_size": True},
        "weaver": {
            "model_name": base_model,
            "prompt_latents_len": latent_tokens,
            "inference_latents_len": latent_tokens,
            "lora_config": {
                "r": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "lora_dropout": lora_dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        },
        "trigger": {
            "model_name": base_model,
            "active": False,
            "lora_config": {
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["q_proj"],
                "lora_dropout": 0.0,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        },
    }


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _configure_trainable_parameters(
    model: MemGenModel,
    *,
    scope: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    model.open_component("weaver")
    model.fix_component("trigger")
    if scope not in {"all_weaver_lora", "last_four_attention_lora"}:
        raise ValueError(f"unsupported trainable scope: {scope}")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter.requires_grad = bool(
            "prompt_query_latents" in name or "lora_" in name
        )
    if scope == "last_four_attention_lora":
        layer_count = text_num_hidden_layers(model.weaver.model.config)
        selected_layers = set(range(max(0, layer_count - 4), layer_count))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or "lora_" not in name:
                continue
            match = _LAYER_RE.search(name)
            attention = any(
                marker in name
                for marker in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            parameter.requires_grad = bool(
                match is not None
                and int(match.group(1)) in selected_layers
                and attention
            )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("the final ReDiR scope selected no trainable parameters")
    if any(parameter.requires_grad for parameter in model.reasoner.parameters()):
        raise RuntimeError("the frozen reasoner unexpectedly has trainable parameters")
    return trainable


def _load_model(
    config: dict[str, Any],
    *,
    distributed: bool,
) -> tuple[MemGenModel, list[tuple[str, torch.nn.Parameter]], torch.device]:
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reasoner_device, weaver_device = _devices(distributed)
    model = MemGenModel.from_config(_model_config(config))
    model.place_components(reasoner_device, weaver_device, weaver_device)
    configured_dropout = float(config["model"].get("lora_dropout", 0.0))
    for name, module in model.weaver.named_modules():
        if "lora_dropout" in name and isinstance(module, torch.nn.Dropout):
            module.p = configured_dropout
    trainable = _configure_trainable_parameters(
        model,
        scope=str(config["optimization"]["trainable_scope"]),
    )
    model.reasoner.eval()
    model.weaver.train()
    return model, trainable, reasoner_device


def _record_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row.get("state_id") or ""): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError("records require unique, non-empty state_id values")
    return result


def _prompt(model: MemGenModel, record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("student_state_messages") or record.get("messages") or []
    tools = record.get("available_tools") or record.get("tools") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"state {record.get('state_id')} has no messages")
    return render_generation_boundary_prompt(
        model.tokenizer,
        messages,
        tools=tools,
        device=model.device,
        assistant_generation_prefix="<think>\n",
        allow_empty_generation_suffix=bool(
            getattr(model.config, "allow_empty_generation_suffix", False)
        ),
    )


def _completion_tensor(model: MemGenModel, token_ids: Iterable[int]) -> torch.Tensor:
    values = [int(value) for value in token_ids]
    if not values:
        raise ValueError("completion has no token ids")
    return torch.tensor([values], device=model.device, dtype=torch.long)


def _student_logits(
    model: MemGenModel,
    record: dict[str, Any],
    completion_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompt = _prompt(model, record)
    outputs = model.forward_prompt_completion_with_prompt_latent(
        prompt["input_ids"],
        prompt["attention_mask"],
        completion_ids,
        int(prompt["latent_insertion_index"]),
    )
    completion_length = completion_ids.size(1)
    augmented_prompt_length = outputs.logits.size(1) - completion_length
    logits = outputs.logits[
        :, augmented_prompt_length - 1 : augmented_prompt_length + completion_length - 1
    ]
    if logits.shape[:2] != completion_ids.shape:
        raise RuntimeError("student completion logits are misaligned")
    return logits, prompt


@torch.no_grad()
def _base_logits(
    model: MemGenModel,
    prompt: dict[str, Any],
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    prompt_ids = prompt["input_ids"].to(model.device)
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention = torch.ones_like(full_ids)
    outputs = model.reasoner(input_ids=full_ids, attention_mask=attention, use_cache=False)
    prompt_length = prompt_ids.size(1)
    logits = outputs.logits[:, prompt_length - 1 : prompt_length + completion_ids.size(1) - 1]
    if logits.shape[:2] != completion_ids.shape:
        raise RuntimeError("base completion logits are misaligned")
    return logits


def _target_loss(
    model: MemGenModel,
    record: dict[str, Any],
    pair: dict[str, Any],
) -> torch.Tensor:
    completion_ids = _completion_tensor(model, pair["completion_token_ids"])
    logits, _ = _student_logits(model, record, completion_ids)
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        completion_ids.reshape(-1),
        reduction="none",
    )
    indices = [
        int(value)
        for value in pair.get("supervised_token_indices") or range(completion_ids.size(1))
        if 0 <= int(value) < completion_ids.size(1)
    ]
    if not indices:
        raise ValueError(f"target {pair.get('target_id')} has no supervised tokens")
    index_tensor = torch.tensor(indices, device=per_token.device, dtype=torch.long)
    selected = per_token[index_tensor]
    raw_weights = pair.get("supervised_token_weights") or [1.0] * completion_ids.size(1)
    weights = torch.tensor(
        [float(raw_weights[index]) if index < len(raw_weights) else 1.0 for index in indices],
        device=selected.device,
        dtype=selected.dtype,
    )
    return (selected * weights).sum() / weights.sum().clamp_min(1e-6)


def _retention_loss(
    model: MemGenModel,
    record: dict[str, Any],
    token_ids: Iterable[int],
) -> torch.Tensor:
    completion_ids = _completion_tensor(model, token_ids)
    student, prompt = _student_logits(model, record, completion_ids)
    teacher = _base_logits(model, prompt, completion_ids)
    teacher_logp = F.log_softmax(teacher.float(), dim=-1)
    student_logp = F.log_softmax(student.float(), dim=-1)
    teacher_prob = teacher_logp.exp()
    return (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1).mean()


def _rotation(rows: list[dict[str, Any]], count: int, step: int, seed: int) -> list[dict[str, Any]]:
    if not rows or count <= 0:
        raise ValueError("rotation requires non-empty rows and a positive count")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    start = step * count
    return [rows[order[(start + offset) % len(order)]] for offset in range(count)]


def _optimizer_step(
    optimizer: AdamW,
    trainable: list[tuple[str, torch.nn.Parameter]],
    *,
    gradient_clip: float,
) -> float:
    parameters = [parameter for _, parameter in trainable]
    _synchronize_gradients(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().cpu())


@torch.no_grad()
def _evaluate(
    model: MemGenModel,
    states: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    benign: list[dict[str, Any]],
) -> dict[str, float]:
    rank, world = _rank(), _world_size()
    target_sum = 0.0
    target_count = 0
    for pair in pairs[rank::world]:
        loss = _target_loss(model, states[str(pair["state_id"])], pair)
        target_sum += float(loss.detach().cpu())
        target_count += 1
    benign_sum = 0.0
    benign_count = 0
    for row in benign[rank::world]:
        loss = _retention_loss(
            model,
            states[str(row["state_id"])],
            row["completion_token_ids"],
        )
        benign_sum += float(loss.detach().cpu())
        benign_count += 1
    device = model.device
    target_total = _reduce_sum(target_sum, device)
    target_n = _reduce_sum(target_count, device)
    benign_total = _reduce_sum(benign_sum, device)
    benign_n = _reduce_sum(benign_count, device)
    return {
        "target_ce": target_total / max(target_n, 1.0),
        "benign_kl": benign_total / max(benign_n, 1.0),
    }


def _parameter_snapshot(trainable: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}


def _displacement(
    trainable: list[tuple[str, torch.nn.Parameter]],
    initial: dict[str, torch.Tensor],
) -> float:
    squared = 0.0
    for name, parameter in trainable:
        delta = parameter.detach().float().cpu() - initial[name]
        squared += float(delta.square().sum())
    return math.sqrt(squared)


def _save_model(model: MemGenModel, path: Path) -> None:
    _barrier()
    if _is_primary():
        model.save_pretrained(str(path))
    _barrier()


def _identity_warmup(config: dict[str, Any]) -> None:
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_records = _read_jsonl(Path(config["data"]["identity_train"]))
    dev_records = _read_jsonl(Path(config["data"]["identity_dev"]))
    if not train_records or not dev_records:
        raise ValueError("identity warmup requires non-empty train and dev sets")
    model, trainable, _ = _load_model(config, distributed=False)
    optimization = config["optimization"]
    optimizer = AdamW(
        [parameter for _, parameter in trainable],
        lr=float(optimization["learning_rate"]),
        weight_decay=0.0,
    )
    epochs = int(optimization["epochs"])
    records_per_update = int(optimization["records_per_update"])
    update = 0
    for epoch in range(epochs):
        epoch_records = list(train_records)
        random.Random(int(config.get("seed", 42)) + epoch).shuffle(epoch_records)
        for start in range(0, len(epoch_records), records_per_update):
            group = epoch_records[start : start + records_per_update]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for record in group:
                token_ids = model.tokenizer(
                    str(record.get("observed_completion") or ""),
                    add_special_tokens=False,
                )["input_ids"]
                loss = _retention_loss(model, record, token_ids)
                (loss / len(group)).backward()
                total += float(loss.detach().cpu())
            gradient_norm = _optimizer_step(
                optimizer,
                trainable,
                gradient_clip=float(optimization["gradient_clip"]),
            )
            update += 1
            _append_jsonl(
                output / "metrics.jsonl",
                {
                    "stage": "identity_warmup",
                    "epoch": epoch + 1,
                    "update": update,
                    "identity_kl": total / len(group),
                    "gradient_norm": gradient_norm,
                },
            )
    _save_model(model, output / "model")
    _write_json(
        output / "summary.json",
        {
            "stage": "identity_warmup",
            "epochs": epochs,
            "updates": update,
            "train_records": len(train_records),
            "dev_records": len(dev_records),
        },
    )


def _safety_training(config: dict[str, Any]) -> None:
    distributed = bool(config.get("distributed", False))
    _initialize_distributed(distributed)
    output = Path(config["output_dir"]).resolve()
    if _is_primary():
        output.mkdir(parents=True, exist_ok=True)
    _barrier()
    states = _record_index(_read_jsonl(Path(config["data"]["safety_states"])))
    pairs = _read_jsonl(Path(config["data"]["training_pairs"]))
    benign = _read_jsonl(Path(config["data"]["benign_records"]))
    model, trainable, _ = _load_model(config, distributed=distributed)
    initial = _parameter_snapshot(trainable)
    optimization = config["optimization"]
    optimizer = AdamW(
        [parameter for _, parameter in trainable],
        lr=float(optimization["learning_rate"]),
        weight_decay=0.0,
    )
    baseline = _evaluate(model, states, pairs, benign)
    max_updates = int(optimization["max_updates"])
    safety_per_batch = int(optimization["safety_per_batch"])
    benign_per_batch = int(optimization["benign_per_batch"])
    benign_weight = float(optimization["benign_weight"])
    checkpoint_interval = int(optimization["checkpoint_interval"])
    rank, world = _rank(), _world_size()
    for update in range(max_updates):
        safety_batch = _rotation(pairs, safety_per_batch, update, int(config.get("seed", 42)))
        benign_batch = _rotation(benign, benign_per_batch, update, int(config.get("seed", 42)) + 1)
        optimizer.zero_grad(set_to_none=True)
        local_target = 0.0
        for pair in safety_batch[rank::world]:
            loss = _target_loss(model, states[str(pair["state_id"])], pair)
            (loss / len(safety_batch)).backward()
            local_target += float(loss.detach().cpu())
        local_benign = 0.0
        for row in benign_batch[rank::world]:
            loss = _retention_loss(
                model,
                states[str(row["state_id"])],
                row["completion_token_ids"],
            )
            (benign_weight * loss / len(benign_batch)).backward()
            local_benign += float(loss.detach().cpu())
        gradient_norm = _optimizer_step(
            optimizer,
            trainable,
            gradient_clip=float(optimization["gradient_clip"]),
        )
        target_mean = _reduce_sum(local_target, model.device) / len(safety_batch)
        benign_mean = _reduce_sum(local_benign, model.device) / len(benign_batch)
        step = update + 1
        if _is_primary():
            _append_jsonl(
                output / "metrics.jsonl",
                {
                    "stage": str(config["stage"]),
                    "update": step,
                    "target_ce": target_mean,
                    "benign_kl": benign_mean,
                    "gradient_norm": gradient_norm,
                },
            )
        if config["stage"] == "formal" and (
            step % checkpoint_interval == 0 or step == max_updates
        ):
            _save_model(model, output / f"model_update_{step}")
    final = _evaluate(model, states, pairs, benign)
    displacement = _displacement(trainable, initial)
    if config["stage"] == "calibration":
        if _is_primary():
            _write_json(
                output / "calibration_summary.json",
                {
                    "baseline_target_ce": baseline["target_ce"],
                    "final_target_ce": final["target_ce"],
                    "baseline_benign_kl": baseline["benign_kl"],
                    "final_benign_kl": final["benign_kl"],
                    "final_displacement": displacement,
                    "updates": max_updates,
                },
            )
    elif _is_primary():
        _write_json(
            output / "training_summary.json",
            {
                "baseline": baseline,
                "final": final,
                "final_displacement": displacement,
                "updates": max_updates,
            },
        )
    _barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _read_yaml(args.config.resolve())
    stage = str(config.get("stage") or "")
    if stage == "identity_warmup":
        _identity_warmup(config)
    elif stage in {"calibration", "formal"}:
        _safety_training(config)
    else:
        raise ValueError(f"unsupported ReDiR stage: {stage!r}")


if __name__ == "__main__":
    main()
