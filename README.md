# ReDiR Artifact

This repository provides the end-to-end training and evaluation artifact of
ReDiR, a frozen-base latent safety controller for multi-turn tool-using agents.
It contains neither base-model weights nor previously collected training data.

Users provide:

1. a Qwen3.5-9B model path;
2. a local checkout of the public MT-AgentRisk dataset;
3. four GPUs for the full training recipe.

The repository validates the collected data, initializes ReDiR
from the base model, runs warmup and safety training, and selects the final
checkpoint.

## Installation

Create a Python 3.12 environment and install a PyTorch build matching the local
CUDA runtime, then install ReDiR:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[model]"
python -m pip install -e ".[eval]"
SKIP_VSCODE_BUILD=1 python -m pip install -e third_party/openhands
python -m pip install -e third_party/mcpmark
```

## Collect data

Download the public benchmark without adding it to this repository:

```bash
git clone https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk \
  /path/to/MT-AgentRisk
```

Run fresh Qwen3.5-9B collection:

```bash
./scripts/collect.sh \
  --dataset-root /path/to/MT-AgentRisk \
  --model /path/to/Qwen3.5-9B \
  --output collected/redir \
  --split-seed 42
```

The collector discovers the filesystem multi-turn pool, deterministically
selects 55 tasks, and splits them into 45 train and 10 dev tasks. It separately
samples 10 filesystem benign tasks and splits them into 8 train and 2 dev
tasks. It then starts the local base-model server, filesystem MCP server, and
OpenHands; collects native decision states; uses the same Qwen3.5-9B model as a
collapsed-view safety teacher; and writes the portable bundle to
`collected/redir/bundle`.

Inspect task selection without loading the model:

```bash
./scripts/collect.sh \
  --dataset-root /path/to/MT-AgentRisk \
  --model /path/to/Qwen3.5-9B \
  --output /tmp/redir-collection-plan \
  --dry-run
```

## End-to-end training

Run the complete pipeline:

```bash
./scripts/train.sh \
  --model /path/to/Qwen3.5-9B \
  --collected-dir collected/redir/bundle \
  --output runs/redir
```

Override physical GPU indices when needed:

```bash
./scripts/train.sh \
  --model /path/to/Qwen3.5-9B \
  --collected-dir collected/redir/bundle \
  --cuda-devices 4,5,6,7 \
  --output runs/redir
```

## Pipeline stages

One invocation performs:

1. collected-data normalization and schema validation;
2. deterministic latent/Weaver initialization from the external base model;
3. identity and benign-behavior warmup;
4. portable target-pair and benign-retention preparation;
5. two 10-update learning-rate calibration arms;
6. automatic learning-rate selection;
7. 400-update safety training with benign retention;
8. final checkpoint selection and creation of `runs/redir/final`.

Only latent queries and Weaver LoRA parameters are trainable. The reasoner
weights remain frozen. Training does not rely on any checkpoint, target cache,
or trajectory bundled with the artifact.

## Data construction

The collector builds and validates the complete portable training bundle from
fresh MT-AgentRisk rollouts. The source tasks remain in the external benchmark
checkout. No previously collected trajectory is included in this repository.

## Configuration

`configs/train.yaml` is the only public training recipe. It contains latent
size, warmup settings, calibration arms, formal training budget, and checkpoint
selection policy. Machine paths are supplied only through command-line
arguments.

Generate all stage configurations without loading a model:

```bash
./scripts/train.sh \
  --model /path/to/Qwen3.5-9B \
  --collected-dir collected/redir/bundle \
  --output /tmp/redir-dry-run \
  --dry-run
```

## Evaluation

Evaluate one task with the selected checkpoint:

```bash
export REDIR_MODEL_PATH=/path/to/Qwen3.5-9B
export REDIR_CHECKPOINT_PATH="$PWD/runs/redir/final"
./scripts/test.sh /path/to/MT-AgentRisk/task outputs/test_run
```

The test script starts both the ReDiR endpoint and filesystem MCP server. By
default it places the frozen reasoner on `cuda:0` and the Weaver on `cuda:1`;
override `REDIR_DEVICE` and `REDIR_WEAVER_DEVICE` when needed.

## Source layout

```text
configs/train.yaml       Public training recipe
scripts/train.sh         End-to-end training entry point
scripts/collect.sh       Fresh MT-AgentRisk collection entry point
scripts/test.sh          Evaluation entry point
src/redir/collect.py     Task split, rollout, self-teacher, and bundle collector
src/redir/data.py        Data schema and validation
src/redir/train.py       End-to-end training orchestrator
src/redir/engine/        Final warmup and safety training engine
src/redir/models/        Latent model and native protocol integration
src/redir/server/        OpenAI-compatible local model server
src/redir/eval/          MT-AgentRisk evaluation integration
src/redir/datasets/      Fresh-rollout state and identity utilities
third_party/             Pinned collection/evaluation dependencies
```
