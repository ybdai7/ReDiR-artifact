"""Distributed helpers for replicated two-GPU OPD workers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RolloutWorkUnit:
    """One rollout generated from a globally selected state and seed slot."""

    record_index: int
    rollout_index: int


def build_rollout_work_units(record_count: int, rollouts_per_state: int) -> list[RolloutWorkUnit]:
    if record_count < 0:
        raise ValueError("record_count must be non-negative")
    if rollouts_per_state <= 0:
        raise ValueError("rollouts_per_state must be positive")
    return [
        RolloutWorkUnit(record_index=record_index, rollout_index=rollout_index)
        for record_index in range(record_count)
        for rollout_index in range(rollouts_per_state)
    ]


def build_variable_rollout_work_units(
    rollouts_per_record: Sequence[int],
) -> list[RolloutWorkUnit]:
    """Build work units when sampled and auxiliary states need different coverage."""

    counts = [int(value) for value in rollouts_per_record]
    if any(value <= 0 for value in counts):
        raise ValueError("rollouts_per_record values must be positive")
    return [
        RolloutWorkUnit(record_index=record_index, rollout_index=rollout_index)
        for record_index, count in enumerate(counts)
        for rollout_index in range(count)
    ]


def partition_rollout_work_units(
    units: Sequence[RolloutWorkUnit],
    *,
    rank: int,
    world_size: int,
) -> list[RolloutWorkUnit]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    return [unit for index, unit in enumerate(units) if index % world_size == rank]


def worker_device_indices(local_rank: int, gpus_per_worker: int = 2) -> tuple[int, int]:
    if local_rank < 0:
        raise ValueError("local_rank must be non-negative")
    if gpus_per_worker != 2:
        raise ValueError("distributed OPD currently requires exactly two GPUs per worker")
    reasoner_index = local_rank * gpus_per_worker
    return reasoner_index, reasoner_index + 1


class DistributedOpdContext:
    """Process-group and collective operations for replicated OPD workers.

    Each process owns two GPUs, but collective communication uses CPU tensors and
    Gloo. This avoids relying on unsupported multi-device DDP behavior while the
    reasoner and weaver parameters live on different CUDA devices.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        rank: int,
        world_size: int,
        local_rank: int,
        gpus_per_worker: int,
        backend: str,
        initialized_here: bool,
    ) -> None:
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.gpus_per_worker = gpus_per_worker
        self.backend = backend
        self._initialized_here = initialized_here

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DistributedOpdContext":
        distributed = config.get("run", {}).get("distributed", {}) or {}
        enabled = bool(distributed.get("enabled", False))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        gpus_per_worker = int(distributed.get("gpus_per_worker", 2))
        expected_workers = int(distributed.get("expected_workers", world_size))
        backend = str(distributed.get("backend", "gloo"))

        if not enabled and world_size > 1:
            raise ValueError("WORLD_SIZE > 1 requires run.distributed.enabled=true")
        if enabled and expected_workers != world_size:
            raise ValueError(
                f"distributed expected_workers={expected_workers} but torchrun WORLD_SIZE={world_size}"
            )
        reasoner_index, weaver_index = worker_device_indices(local_rank, gpus_per_worker)
        if enabled and torch.cuda.is_available() and weaver_index >= torch.cuda.device_count():
            raise ValueError(
                f"worker {local_rank} needs CUDA devices {reasoner_index},{weaver_index}, "
                f"but only {torch.cuda.device_count()} visible devices exist"
            )

        initialized_here = False
        if enabled and world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=int(distributed.get("timeout_minutes", 60))),
            )
            initialized_here = True
        return cls(
            enabled=enabled,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            gpus_per_worker=gpus_per_worker,
            backend=backend,
            initialized_here=initialized_here,
        )

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def reasoner_device(self) -> torch.device:
        reasoner_index, _ = worker_device_indices(self.local_rank, self.gpus_per_worker)
        return torch.device(f"cuda:{reasoner_index}")

    @property
    def weaver_device(self) -> torch.device:
        _, weaver_index = worker_device_indices(self.local_rank, self.gpus_per_worker)
        return torch.device(f"cuda:{weaver_index}")

    def barrier(self) -> None:
        if self.enabled and self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if self._initialized_here and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized_here = False

    def local_work_units(self, record_count: int, rollouts_per_state: int) -> list[RolloutWorkUnit]:
        units = build_rollout_work_units(record_count, rollouts_per_state)
        return partition_rollout_work_units(units, rank=self.rank, world_size=self.world_size)

    def local_variable_work_units(
        self,
        rollouts_per_record: Sequence[int],
    ) -> list[RolloutWorkUnit]:
        units = build_variable_rollout_work_units(rollouts_per_record)
        return partition_rollout_work_units(units, rank=self.rank, world_size=self.world_size)

    def sum_scalars(self, values: Mapping[str, float | int]) -> dict[str, float]:
        keys = sorted(values)
        tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64)
        if self.enabled and self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return {key: float(value) for key, value in zip(keys, tensor.tolist(), strict=True)}

    def sum_counter(self, counter: Mapping[str, int]) -> Counter[str]:
        if not self.enabled or self.world_size == 1:
            return Counter(counter)
        gathered: list[dict[str, int] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, dict(counter))
        result: Counter[str] = Counter()
        for item in gathered:
            if item:
                result.update(item)
        return result

    def first_nonempty_object(self, value: Any) -> Any:
        if not self.enabled or self.world_size == 1:
            return value
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return next((item for item in gathered if item is not None), None)

    def all_objects(self, value: Any) -> list[Any]:
        if not self.enabled or self.world_size == 1:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def synchronize_parameters(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        """Broadcast trainable parameters from worker zero before the first rollout."""
        if not self.enabled or self.world_size == 1:
            return
        for _, parameter in named_parameters:
            value = parameter.detach().float().cpu()
            dist.broadcast(value, src=0)
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        self.barrier()

    def synchronize_gradient_dict(
        self,
        local_gradients: Mapping[str, torch.Tensor],
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        *,
        local_normalization_count: int,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """All-reduce gradient sums and normalize by globally valid trajectories."""
        counts = self.sum_scalars({"valid": local_normalization_count})
        global_count = int(round(counts["valid"]))
        if global_count <= 0:
            return {}, 0

        synchronized: dict[str, torch.Tensor] = {}
        for name, parameter in named_parameters:
            gradient = local_gradients.get(name)
            if gradient is None:
                value = torch.zeros(tuple(parameter.shape), dtype=torch.float32)
            else:
                value = gradient.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if self.enabled and self.world_size > 1:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(global_count)
            synchronized[name] = value
        return synchronized, global_count

    def sum_gradient_dict(
        self,
        local_gradients: Mapping[str, torch.Tensor],
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    ) -> dict[str, torch.Tensor]:
        """All-reduce already-normalized gradient contributions without rescaling."""

        synchronized: dict[str, torch.Tensor] = {}
        for name, parameter in named_parameters:
            gradient = local_gradients.get(name)
            if gradient is None:
                value = torch.zeros(tuple(parameter.shape), dtype=torch.float32)
            else:
                value = gradient.detach().to(
                    device="cpu", dtype=torch.float32
                ).contiguous()
            if self.enabled and self.world_size > 1:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            synchronized[name] = value
        return synchronized


__all__ = [
    "DistributedOpdContext",
    "RolloutWorkUnit",
    "build_rollout_work_units",
    "build_variable_rollout_work_units",
    "partition_rollout_work_units",
    "worker_device_indices",
]
