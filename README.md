# ReDiR Artifact

This repository provides the end-to-end training and evaluation artifact of
ReDiR, a frozen-base latent safety controller for multi-turn tool-using agents.
It contains neither base-model weights nor previously collected training data.

Users provide:

1. a Qwen3.5-9B model path;
2. a newly collected training-data directory following the documented schema;
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
```

## Collect and normalize data

Prepare a collection directory:

```text
collected/
  identity_train.jsonl[.gz]
  identity_dev.jsonl[.gz]
  candidate_states.jsonl[.gz]
  teacher_targets.jsonl[.gz]
  pair_manifest.jsonl[.gz]
  benign_completions.jsonl[.gz]
  margin_bank_family0.jsonl[.gz]
  margin_bank_family123.jsonl[.gz]

  # Optional reference annotations
  selected_manifest.jsonl[.gz]
  sibling_targets.jsonl[.gz]
```

Normalize and validate the newly collected files:

```bash
./scripts/collect.sh \
  --source-dir /path/to/raw_collection \
  --output collected/redir
```

`collect.sh` does not contain or download any previously collected ReDiR data.
It assembles the files produced by the user's own OpenHands/teacher collection
run into the bundle consumed by training.

## End-to-end training

Run the complete pipeline:

```bash
./scripts/train.sh \
  --model /path/to/Qwen3.5-9B \
  --collected-dir collected/redir \
  --output runs/redir
```

Override physical GPU indices when needed:

```bash
./scripts/train.sh \
  --model /path/to/Qwen3.5-9B \
  --collected-dir collected/redir \
  --cuda-devices 4,5,6,7 \
  --output runs/redir
```

## Pipeline stages

One invocation performs:

1. collected-data normalization and schema validation;
2. deterministic latent/Weaver initialization from the external base model;
3. identity and benign-behavior warmup;
4. portable pair-manifest and baseline-metric preparation;
5. two 10-update learning-rate calibration arms;
6. automatic learning-rate selection;
7. 400-update safety training with benign retention;
8. final checkpoint selection and creation of `runs/redir/final`.

Only latent queries and Weaver LoRA parameters are trainable. The reasoner
weights remain frozen. Training does not reuse any checkpoint, target cache, or
trajectory bundled by the artifact.

## Data construction

`src/redir/build.py` converts newly collected rows into the internal portable
bundle used by training and validates the required fields and counts.

The builder can be run independently:

```bash
python -m redir.build \
  --source /path/to/collected \
  --margin-family0 /path/to/collected/margin_bank_family0.jsonl.gz \
  --margin-family123 /path/to/collected/margin_bank_family123.jsonl.gz \
  --output /path/to/generated_bundle
```

The source tasks and collection environments are external inputs. The vendored
OpenHands, MCPMark, and ToolShield source trees provide the execution runtime;
they do not contain the submission's previously collected trajectories.

## Configuration

`configs/train.yaml` is the only public training recipe. It contains latent
size, warmup settings, calibration arms, formal training budget, and checkpoint
selection policy. Machine paths are supplied only through command-line
arguments.

Generate all stage configurations without loading a model:

```bash
./scripts/train.sh \
  --model /path/to/Qwen3.5-9B \
  --collected-dir /path/to/collected \
  --output /tmp/redir-dry-run \
  --dry-run
```

## Evaluation

Install the vendored agent runtime:

```bash
python -m pip install -e ".[eval]"
SKIP_VSCODE_BUILD=1 python -m pip install -e third_party/openhands
python -m pip install -e third_party/mcpmark
```

Evaluate one task with the selected checkpoint:

```bash
export REDIR_MODEL_PATH=/path/to/Qwen3.5-9B
export REDIR_CHECKPOINT_PATH="$PWD/runs/redir/final"
./scripts/test.sh /path/to/MT-AgentRisk/task outputs/test_run
```

## Source layout

```text
configs/train.yaml       Public training recipe
scripts/train.sh         End-to-end training entry point
scripts/collect.sh       Collected-data assembly entry point
scripts/test.sh          Evaluation entry point
src/redir/collect.py     Collection bundle orchestrator
src/redir/data.py        Data schema and validation
src/redir/build.py       User-collected data builder
src/redir/train.py       End-to-end training orchestrator
src/redir/engine/        Internal implementation modules
src/latent_safety/       Model, protocol, server, and evaluation integration
third_party/             Pinned collection/evaluation dependencies
```
