"""Collect a fresh ReDiR training bundle from the public MT-AgentRisk tasks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

from redir.data import build_bundle


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _task_directories(root: Path) -> list[Path]:
    rows = sorted(path for path in root.iterdir() if path.is_dir())
    if not rows:
        raise ValueError(f"task root has no task directories: {root}")
    return rows


def _resolve_workspace_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    if (dataset_root / "workspaces").is_dir():
        return dataset_root / "workspaces"
    if (dataset_root / "filesystem").is_dir():
        return dataset_root
    raise FileNotFoundError(
        "MT-AgentRisk root must contain workspaces/ or the workspace directories"
    )


def _select_tasks(dataset_root: Path, seed: int) -> dict[str, list[Path]]:
    workspaces = _resolve_workspace_root(dataset_root)
    attack_root = workspaces / "filesystem" / "multi-turn-tasks"
    benign_root = workspaces / "benign_tasks" / "benign_tasks_filesystem"
    attack_pool = _task_directories(attack_root)
    benign_pool = _task_directories(benign_root)
    if len(attack_pool) < 55:
        raise ValueError(f"filesystem attack pool has {len(attack_pool)} tasks, need 55")
    if len(benign_pool) < 10:
        raise ValueError(f"filesystem benign pool has {len(benign_pool)} tasks, need 10")

    attack_rng = random.Random(seed)
    attack_rng.shuffle(attack_pool)
    attack_selected = attack_pool[:55]
    benign_rng = random.Random(seed + 1)
    benign_rng.shuffle(benign_pool)
    benign_selected = benign_pool[:10]
    return {
        "attack_train": sorted(attack_selected[:45]),
        "attack_dev": sorted(attack_selected[45:]),
        "benign_train": sorted(benign_selected[:8]),
        "benign_dev": sorted(benign_selected[8:]),
    }


def _materialize_task_roots(
    selections: dict[str, list[Path]], output: Path
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for split, tasks in selections.items():
        root = output / "tasks" / split
        root.mkdir(parents=True, exist_ok=True)
        for source in tasks:
            destination = root / source.name
            if destination.exists() or destination.is_symlink():
                continue
            destination.symlink_to(source.resolve(), target_is_directory=True)
        roots[split] = root
    manifest = {
        "seed": None,
        "splits": {
            split: [path.name for path in tasks]
            for split, tasks in selections.items()
        },
        "heldout_test_used": False,
    }
    (output / "task_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return roots


def _http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_server(url: str, process: subprocess.Popen[Any], timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited before health check: {process.args}")
        try:
            _http_json(url)
            return
        except (OSError, URLError, json.JSONDecodeError):
            time.sleep(2)
    raise TimeoutError(f"service did not become healthy: {url}")


def _wait_for_port(
    host: str, port: int, process: subprocess.Popen[Any], timeout: int = 60
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited before opening port: {process.args}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"service did not open {host}:{port}")


@contextmanager
def _services(
    *,
    repo_root: Path,
    model: str,
    output: Path,
    device: str,
    model_port: int,
    filesystem_port: int,
) -> Iterator[None]:
    logs = output / "service_logs"
    raw = output / "raw_completions"
    workspace = output / "shared_workspace"
    logs.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    model_log = (logs / "base_server.log").open("w", encoding="utf-8")
    filesystem_log = (logs / "filesystem_server.log").open("w", encoding="utf-8")
    model_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "redir.server.openai_compatible",
            "--backend",
            "base",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(model_port),
            "--device",
            device,
            "--completion-log-dir",
            str(raw),
        ],
        cwd=repo_root,
        env=environment,
        stdout=model_log,
        stderr=subprocess.STDOUT,
    )
    filesystem_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "redir.eval.mtagentrisk.mcp_server.filesystem_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(filesystem_port),
            "--workspace",
            str(workspace),
        ],
        cwd=repo_root,
        env=environment,
        stdout=filesystem_log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(f"http://127.0.0.1:{model_port}/v1/models", model_process)
        _wait_for_port("127.0.0.1", filesystem_port, filesystem_process)
        yield
    finally:
        for process in (model_process, filesystem_process):
            if process.poll() is None:
                process.terminate()
        for process in (model_process, filesystem_process):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        model_log.close()
        filesystem_log.close()


def _write_eval_config(path: Path, model_port: int) -> None:
    path.write_text(
        "\n".join(
            [
                "[llm.agent]",
                'model = "openai/redir-base"',
                f'base_url = "http://127.0.0.1:{model_port}/v1"',
                'api_key = "local"',
                "max_output_tokens = 4096",
                "",
                "[llm.env]",
                'model = "openai/redir-base"',
                f'base_url = "http://127.0.0.1:{model_port}/v1"',
                'api_key = "local"',
                "max_output_tokens = 4096",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_rollouts(
    *,
    repo_root: Path,
    task_roots: dict[str, Path],
    output: Path,
    model_port: int,
    filesystem_port: int,
) -> None:
    config = output / "base_eval.toml"
    _write_eval_config(config, model_port)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repo_root / "src")
            + os.pathsep
            + str(repo_root / "third_party" / "openhands")
            + os.pathsep
            + str(repo_root / "third_party" / "mcpmark")
            + os.pathsep
            + environment.get("PYTHONPATH", ""),
            "MCP_FILESYSTEM_HOST": "127.0.0.1",
            "MCP_FILESYSTEM_PORT": str(filesystem_port),
            "MTAGENTRISK_FORCE_MCP_FILESYSTEM": "1",
            "NO_PROXY": "localhost,127.0.0.1,0.0.0.0",
            "no_proxy": "localhost,127.0.0.1,0.0.0.0",
        }
    )
    for split, root in task_roots.items():
        for task in sorted(path for path in root.iterdir() if path.is_dir()):
            task_output = output / "rollouts" / split / task.name
            task_output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "redir.eval.mtagentrisk.run_eval",
                "--task-path",
                str(task.resolve()),
                "--agent-llm-config",
                "agent",
                "--agent-llm-config-file",
                str(config),
                "--env-llm-config",
                "env",
                "--env-llm-config-file",
                str(config),
                "--outputs-path",
                str(task_output),
                "--server-hostname",
                "localhost",
            ]
            subprocess.run(command, cwd=repo_root, env=environment, check=True)


def _partition_completion_logs(
    raw_root: Path, task_roots: dict[str, Path], output: Path
) -> dict[str, Path]:
    from redir.datasets.native_states import (
        identify_task,
        load_task_specs,
    )

    specs_by_split = {split: load_task_specs(root) for split, root in task_roots.items()}
    all_specs = [spec for specs in specs_by_split.values() for spec in specs]
    split_by_task = {
        spec.task_key: split
        for split, specs in specs_by_split.items()
        for spec in specs
    }
    destinations = {split: output / "raw_by_split" / split for split in task_roots}
    for directory in destinations.values():
        directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload.get("messages") or []
        try:
            spec = identify_task(messages, all_specs)
        except ValueError:
            continue
        shutil.copy2(path, destinations[split_by_task[spec.task_key]] / path.name)
    return destinations


def _extract_states(
    *,
    raw_dirs: dict[str, Path],
    task_roots: dict[str, Path],
    output: Path,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    from redir.datasets.native_states import build_native_states

    results: dict[str, list[dict[str, Any]]] = {}
    for split in task_roots:
        domain = "safety" if split.startswith("attack") else "identity"
        rows, stats = build_native_states(
            raw_completion_dir=raw_dirs[split],
            task_root=task_roots[split],
            domain=domain,
            rollout_seed=seed,
            expected_task_count=len(_task_directories(task_roots[split])),
            min_tools=1,
            rollout_policy="qwen3.5-9b-base-native",
        )
        results[split] = rows
        _write_jsonl(output / "states" / f"{split}.jsonl", rows)
        (output / "states" / f"{split}_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return results


def _identity_dataset(
    states: dict[str, list[dict[str, Any]]], output: Path
) -> tuple[Path, Path]:
    from redir.datasets.identity import (
        SafeIdentitySource,
        build_safe_identity_dataset,
    )

    source_dir = output / "identity_sources"
    train_source = source_dir / "train.jsonl"
    dev_source = source_dir / "dev.jsonl"
    _write_jsonl(train_source, states["benign_train"])
    _write_jsonl(dev_source, states["benign_dev"])
    identity_dir = output / "identity"
    stats = build_safe_identity_dataset(
        [
            SafeIdentitySource("train", 42, train_source, None),
            SafeIdentitySource("dev", 42, dev_source, None),
        ],
        output_dir=identity_dir,
        expected_train_tasks=8,
        expected_dev_tasks=2,
    )
    if not stats["records"]["train"] or not stats["records"]["dev"]:
        raise RuntimeError("identity collection produced an empty split")
    return identity_dir / "train.jsonl", identity_dir / "dev.jsonl"


def _teacher_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    user_texts = [
        str(message.get("content") or "")
        for message in record.get("student_state_messages") or []
        if message.get("role") == "user"
    ]
    collapsed = "\n\n".join(text for text in user_texts if text)
    return [
        {
            "role": "system",
            "content": (
                "You are the safety teacher for a tool-using agent. Reconstruct the "
                "overall user goal across all turns. If it is harmful, refuse by "
                "calling finish with a concise explanation and do not call any "
                "non-finish tool."
            ),
        },
        {"role": "user", "content": collapsed},
    ]


def _normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for call in value if isinstance(value, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        output.append(
            {
                "id": str(call.get("id") or f"call_{len(output)}"),
                "type": "function",
                "function": {
                    "name": str(function.get("name") or ""),
                    "arguments": arguments or {},
                },
            }
        )
    return output


def _completion_target(
    *,
    tokenizer: Any,
    record: dict[str, Any],
    response_message: dict[str, Any],
) -> dict[str, Any] | None:
    calls = _normalize_tool_calls(response_message.get("tool_calls"))
    if not calls or str(calls[0]["function"]["name"]) != "finish":
        return None
    assistant = {
        "role": "assistant",
        "content": str(response_message.get("content") or ""),
        "tool_calls": calls,
    }
    state_messages = list(record.get("student_state_messages") or [])
    tools = list(record.get("available_tools") or [])
    prompt = tokenizer.apply_chat_template(
        state_messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    full = tokenizer.apply_chat_template(
        state_messages + [assistant],
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    )
    if not isinstance(prompt, str) or not isinstance(full, str) or not full.startswith(prompt):
        return None
    completion_text = full[len(prompt) :]
    token_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        return None
    supervised = list(range(len(token_ids)))
    target_id = hashlib.sha256(
        (str(record["task_key"]) + "\n" + completion_text).encode("utf-8")
    ).hexdigest()
    return {
        "available": True,
        "target_id": target_id,
        "task_key": str(record["task_key"]),
        "target_kind": "native_refusal_finish",
        "target_protocol": "native",
        "source_teacher_completion_exact": True,
        "heldout_test_used": False,
        "completion_text": completion_text,
        "completion_token_ids": [int(value) for value in token_ids],
        "supervised_token_indices": supervised,
        "supervised_token_weights": [1.0] * len(supervised),
        "reasoning_token_indices": [],
        "weighted_reasoning_token_indices": [],
        "action_onset_token_indices": supervised[: min(8, len(supervised))],
        "finish_function_name_token_indices": [],
        "message_token_indices": supervised,
        "weighting_profile": "full_completion",
    }


def _teacher_targets(
    *,
    records: list[dict[str, Any]],
    model_url: str,
    model: str,
    output: Path,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        teacher_messages = _teacher_messages(record)
        payload = {
            "model": "redir-base",
            "messages": teacher_messages,
            "tools": record.get("available_tools") or [],
            "temperature": 0.0,
            "max_tokens": 1024,
            "seed": 42,
        }
        response = _http_json(model_url + "/v1/chat/completions", payload)
        choices = response.get("choices") or []
        message = (choices[0] if choices else {}).get("message") or {}
        target = _completion_target(
            tokenizer=tokenizer, record=record, response_message=message
        )
        row = dict(record)
        row["teacher_messages"] = teacher_messages
        row["targets"] = [] if target is None else [target]
        row["target_availability"] = {
            "has_valid_target": target is not None,
            "selected_target_ids": [] if target is None else [target["target_id"]],
            "selected_unique_targets": int(target is not None),
            "total_rollouts": 1,
            "task_key": str(record["task_key"]),
            "heldout_test_used": False,
        }
        rows.append(row)
    _write_jsonl(output, rows)
    if not any(row["targets"] for row in rows):
        raise RuntimeError("self-teacher produced zero native refusal targets")
    return rows


def _formal_components(
    *,
    safety_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    tokenizer: Any,
    output: Path,
) -> None:
    teacher_by_state = {str(row["state_id"]): row for row in teacher_rows}
    candidates: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for source in safety_rows:
        row = dict(teacher_by_state[str(source["state_id"])])
        row.update(
            {
                "route": "safety",
                "training_candidate": True,
            }
        )
        candidates.append(row)
        for target in row.get("targets") or []:
            manifest.append(
                {
                    "state_id": row["state_id"],
                    "task_key": row["task_key"],
                    "target_id": target["target_id"],
                    "target": target,
                    "active": True,
                    "weight": 1.0,
                    "bridge_kind": "full_target_ce",
                    "frozen_route": "c",
                    "frozen_route_seed": 42,
                }
            )
    benign_first: dict[str, dict[str, Any]] = {}
    for source in benign_rows:
        row = dict(source)
        row.update(
            {
                "route": "benign",
                "training_candidate": True,
            }
        )
        candidates.append(row)
        benign_first.setdefault(str(row["task_key"]), row)
    benign_completions: list[dict[str, Any]] = []
    for row in benign_first.values():
        completion = str(row.get("observed_completion") or "")
        token_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        benign_completions.append(
            {
                "state_id": row["state_id"],
                "task_key": row["task_key"],
                "raw_completion": completion,
                "completion_token_ids": [int(value) for value in token_ids],
                "seed": 42,
                "heldout_test_used": False,
                "generation": {"source": "fresh_base_collection"},
            }
        )
    _write_jsonl(output / "candidate_states.jsonl", candidates)
    _write_jsonl(output / "teacher_targets.jsonl", teacher_rows)
    _write_jsonl(output / "pair_manifest.jsonl", manifest)
    _write_jsonl(output / "benign_completions.jsonl", benign_completions)


def _run_collection(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selections = _select_tasks(args.dataset_root, args.split_seed)
    task_roots = _materialize_task_roots(selections, output)
    manifest = json.loads((output / "task_manifest.json").read_text(encoding="utf-8"))
    manifest["seed"] = args.split_seed
    (output / "task_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    with _services(
        repo_root=repo_root,
        model=args.model,
        output=output,
        device=args.device,
        model_port=args.model_port,
        filesystem_port=args.filesystem_port,
    ):
        _run_rollouts(
            repo_root=repo_root,
            task_roots=task_roots,
            output=output,
            model_port=args.model_port,
            filesystem_port=args.filesystem_port,
        )
        raw_dirs = _partition_completion_logs(
            output / "raw_completions", task_roots, output
        )
        states = _extract_states(
            raw_dirs=raw_dirs,
            task_roots=task_roots,
            output=output,
            seed=args.split_seed,
        )
        identity_train, identity_dev = _identity_dataset(states, output)
        teacher_rows = _teacher_targets(
            records=states["attack_train"],
            model_url=f"http://127.0.0.1:{args.model_port}",
            model=args.model,
            output=output / "components" / "teacher_targets.jsonl",
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        _formal_components(
            safety_rows=states["attack_train"],
            benign_rows=states["benign_train"] + states["benign_dev"],
            teacher_rows=teacher_rows,
            tokenizer=tokenizer,
            output=output / "components",
        )
        shutil.copy2(identity_train, output / "components" / "identity_train.jsonl")
        shutil.copy2(identity_dev, output / "components" / "identity_dev.jsonl")

    components = output / "components"
    sources = {
        "identity_train": components / "identity_train.jsonl",
        "identity_dev": components / "identity_dev.jsonl",
        "candidate_states": components / "candidate_states.jsonl",
        "audited_targets": components / "teacher_targets.jsonl",
        "pair_manifest": components / "pair_manifest.jsonl",
        "benign_completions": components / "benign_completions.jsonl",
        "selected_manifest": None,
        "sibling_targets": None,
    }
    bundle = output / "bundle"
    result = build_bundle(
        **sources,
        output_dir=bundle,
        source="fresh-mtagentrisk-filesystem-collection",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"collected bundle: {bundle}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-port", type=int, default=8011)
    parser.add_argument("--filesystem-port", type=int, default=19090)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _run_collection(args)


if __name__ == "__main__":
    main()
