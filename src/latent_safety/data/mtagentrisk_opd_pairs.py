"""Build masked-OPD pairs from MT-AgentRisk filesystem trajectories."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SINGLE_ROOT = (
    "outputs/mtagentrisk_qwen35_9b_p3_filesystem_single_70_20260628_204427/"
    "filesystem_single"
)
DEFAULT_MULTI_ROOT = (
    "outputs/mtagentrisk_qwen35_9b_p3_filesystem_multiturn_70_20260629_025346/"
    "filesystem_multiturn"
)
DEFAULT_SINGLE_POST_EVAL = (
    "outputs/mtagentrisk_qwen35_9b_p3_filesystem_single_70_20260628_204427/"
    "filesystem_single/post_eval_openrouter_gpt41mini_with_failed7_rerun.json"
)
DEFAULT_MULTI_POST_EVAL = (
    "outputs/mtagentrisk_qwen35_9b_p3_filesystem_multiturn_70_20260629_025346/"
    "filesystem_multiturn/post_eval_openrouter_gpt41mini.json"
)
DEFAULT_SINGLE_EXTRA_ROOTS = (
    "outputs/mtagentrisk_qwen35_9b_p3_filesystem_single_failed7_rerun_20260629_015542/"
    "filesystem_single",
)
DEFAULT_MULTI_EXTRA_ROOTS = (
    "outputs/mtagentrisk_qwen35_9b_p3_filesystem_multiturn_failed1_rerun_20260629_105000/"
    "filesystem_multiturn",
)
DEFAULT_SINGLE_TASKS_ROOT = "../ToolShield/workspaces/filesystem/single-turn-tasks"
DEFAULT_MULTI_TASKS_ROOT = "../ToolShield/workspaces/filesystem/multi-turn-tasks"
DEFAULT_OUTPUT_ROOT = "outputs/latent_training/mtagentrisk_qwen35_filesystem_p3_masked_opd"

RETRIEVAL_PREFIX = "Retrieving content for:"
CONTINUE_PREFIX = "Please continue working on the task"
FINAL_WRAP_UP_PREFIX = "Thanks, that covers everything I needed"
MCP_ACTION_RE = re.compile(
    r"I am interacting with the MCP server with name:\s*```(?P<tool>.*?)```",
    re.DOTALL,
)
REFUSAL_RE = re.compile(
    r"\b(cannot|can't|will not|won't|unable to|refuse|not able to|cannot fulfill)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BuildConfig:
    repo_root: Path
    single_root: Path
    multi_root: Path
    single_extra_roots: tuple[Path, ...]
    multi_extra_roots: tuple[Path, ...]
    single_post_eval: Path
    multi_post_eval: Path
    single_tasks_root: Path
    multi_tasks_root: Path
    output_root: Path
    seed: int = 42
    train_size: int = 45
    dev_size: int = 10
    test_size: int = 15


@dataclass
class OpdPair:
    task_key: str
    split: str
    single_task_id: str
    multi_task_id: str
    single_outcome: str
    multi_outcome: str
    canonical_single_messages: list[dict[str, str]]
    raw_multiturn_messages: list[dict[str, str]]
    student_action_prefix_source: str
    decision_mask_spans: list[dict[str, int | str]]
    single_traj_path: str
    multi_traj_path: str
    single_task_path: str
    multi_task_path: str
    selected_event_id: int | None
    selected_event_index: int
    selected_event_kind: str
    is_training_candidate: bool


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def normalize_task_id(task_id: str) -> str:
    task_id = task_id.replace("_", "-")
    for prefix in ("safety-", "multi-turn-"):
        if task_id.startswith(prefix):
            task_id = task_id[len(prefix) :]
    return task_id


def load_post_eval(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"post-eval file has no results dict: {path}")
    outcomes = {}
    for task_id, result in results.items():
        outcome = result.get("outcome") if isinstance(result, dict) else None
        if not isinstance(outcome, str):
            raise ValueError(f"missing outcome for {task_id} in {path}")
        outcomes[task_id] = outcome.upper()
    return outcomes


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"trajectory is not a list: {path}")
    return data


def find_task_dirs(root: Path) -> dict[str, Path]:
    dirs = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            dirs[normalize_task_id(child.name)] = child
    return dirs


def find_traj_dirs(root: Path) -> dict[str, Path]:
    dirs = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        traj = child / f"traj_{child.name}.json"
        if traj.exists():
            dirs[normalize_task_id(child.name)] = child
    return dirs


def find_traj_dirs_many(roots: Iterable[Path]) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for key, path in find_traj_dirs(root).items():
            merged.setdefault(key, path)
    return merged


def task_text(task_dir: Path, task_id: str, *, single: bool) -> str:
    if single:
        path = task_dir / "task.md"
        return path.read_text().strip()
    turn_files = sorted(
        task_dir.glob("task-turn-*.md"),
        key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)],
    )
    if not turn_files:
        raise FileNotFoundError(f"no task-turn files for {task_id}: {task_dir}")
    return "\n\n".join(path.read_text().strip() for path in turn_files)


def first_system_message(traj: list[dict[str, Any]]) -> str:
    for event in traj:
        if event.get("source") == "agent" and event.get("id") == 0:
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                return message
    for event in traj:
        if event.get("source") == "agent":
            message = event.get("message")
            if isinstance(message, str) and "<ROOT_SAFETY_POLICY>" in message:
                return message
    raise ValueError("could not find system prompt in trajectory")


def is_retrieval_message(message: str) -> bool:
    return message.startswith(RETRIEVAL_PREFIX)


def is_continue_message(message: str) -> bool:
    return message.startswith(CONTINUE_PREFIX) or message.startswith(FINAL_WRAP_UP_PREFIX)


def looks_like_observation(message: str) -> bool:
    stripped = message.strip()
    if not stripped.startswith("{"):
        return False
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and ("content" in value or "isError" in value)


def is_empty_assistant_wait(message: str) -> bool:
    return message.strip() == ""


def event_message(event: dict[str, Any]) -> str:
    message = event.get("message")
    return message if isinstance(message, str) else ""


def event_kind(message: str) -> str:
    if MCP_ACTION_RE.search(message):
        return "mcp_action"
    if REFUSAL_RE.search(message):
        return "refusal"
    if looks_like_observation(message):
        return "observation"
    return "assistant_message"


def select_student_action_event(traj: list[dict[str, Any]]) -> tuple[int, dict[str, Any], str]:
    candidates: list[tuple[int, dict[str, Any], str]] = []
    for idx, event in enumerate(traj):
        if event.get("source") != "agent" or event.get("id") == 0:
            continue
        message = event_message(event)
        if not message or is_empty_assistant_wait(message) or looks_like_observation(message):
            continue
        kind = event_kind(message)
        candidates.append((idx, event, kind))

    if not candidates:
        raise ValueError("no assistant action/message candidates in multiturn trajectory")

    for idx, event, kind in reversed(candidates):
        if kind == "mcp_action":
            return idx, event, kind
    for idx, event, kind in candidates:
        if kind == "refusal":
            return idx, event, kind
    return candidates[-1]


def history_messages_before(
    traj: list[dict[str, Any]],
    selected_index: int,
    system_content: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for event in traj[1:selected_index]:
        source = event.get("source")
        message = event_message(event)
        if not message or is_empty_assistant_wait(message):
            continue
        if is_retrieval_message(message):
            continue
        if source == "user":
            messages.append({"role": "user", "content": message})
        elif source == "environment":
            messages.append({"role": "tool", "content": message})
        elif source == "agent":
            if looks_like_observation(message):
                messages.append({"role": "tool", "content": message})
            else:
                messages.append({"role": "assistant", "content": message})
    return messages


def canonical_messages(system_content: str, single_task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": single_task},
    ]


def raw_messages_from_multiturn_task(system_content: str, task_dir: Path) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_content}]
    turn_files = sorted(
        task_dir.glob("task-turn-*.md"),
        key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)],
    )
    for path in turn_files:
        messages.append({"role": "user", "content": path.read_text().strip()})
    return messages


def decision_mask_spans(text: str) -> list[dict[str, int | str]]:
    match = MCP_ACTION_RE.search(text)
    if match:
        start = 0
        end = match.end("tool")
        return [{"start": start, "end": end, "kind": "action_head"}]

    stripped_len = len(text) - len(text.lstrip())
    start = stripped_len
    sentence_end = len(text)
    for sep in (".", "\n"):
        pos = text.find(sep, start)
        if pos != -1:
            sentence_end = min(sentence_end, pos + 1)
    end = min(sentence_end, start + 160)
    return [{"start": start, "end": end, "kind": "refusal_or_message_head"}]


def allocate_splits(
    task_keys: list[str],
    groups: dict[str, tuple[str, str]],
    *,
    seed: int,
    train_size: int,
    dev_size: int,
    test_size: int,
) -> dict[str, str]:
    if len(task_keys) != train_size + dev_size + test_size:
        raise ValueError(
            f"split sizes do not add up: {train_size}+{dev_size}+{test_size} "
            f"!= {len(task_keys)}"
        )

    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in task_keys:
        grouped[groups[key]].append(key)
    for values in grouped.values():
        values.sort()
        rng.shuffle(values)

    train_alloc = proportional_alloc(grouped, train_size, len(task_keys))
    remaining: dict[tuple[str, str], list[str]] = {}
    split: dict[str, str] = {}
    for group, keys in grouped.items():
        n_train = train_alloc[group]
        for key in keys[:n_train]:
            split[key] = "train"
        remaining[group] = keys[n_train:]

    dev_alloc = proportional_alloc(remaining, dev_size, len(task_keys) - train_size)
    for group, keys in remaining.items():
        n_dev = dev_alloc[group]
        for key in keys[:n_dev]:
            split[key] = "dev"
        for key in keys[n_dev:]:
            split[key] = "test"

    counts = Counter(split.values())
    expected = {"train": train_size, "dev": dev_size, "test": test_size}
    if counts != expected:
        raise AssertionError(f"bad split counts: got {counts}, expected {expected}")
    return split


def proportional_alloc(
    grouped: dict[tuple[str, str], list[str]],
    target: int,
    denominator: int,
) -> dict[tuple[str, str], int]:
    raw = []
    alloc: dict[tuple[str, str], int] = {}
    for group, keys in grouped.items():
        exact = len(keys) * target / denominator if denominator else 0.0
        base = int(exact)
        base = min(base, len(keys))
        alloc[group] = base
        raw.append((exact - base, len(keys), group))

    remaining = target - sum(alloc.values())
    for _, _, group in sorted(raw, reverse=True):
        if remaining <= 0:
            break
        if alloc[group] < len(grouped[group]):
            alloc[group] += 1
            remaining -= 1

    if remaining != 0:
        for _, _, group in sorted(raw, reverse=True):
            while remaining > 0 and alloc[group] < len(grouped[group]):
                alloc[group] += 1
                remaining -= 1
            if remaining == 0:
                break
    if remaining != 0:
        raise AssertionError(f"could not allocate {target} examples")
    return alloc


def relative_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats(path: Path, records: list[OpdPair]) -> None:
    rows = ["metric\tvalue"]
    rows.append(f"total_pairs\t{len(records)}")
    for split, count in sorted(Counter(r.split for r in records).items()):
        rows.append(f"split.{split}\t{count}")
    for key, count in sorted(Counter((r.single_outcome, r.multi_outcome) for r in records).items()):
        rows.append(f"outcome.{key[0]}->{key[1]}\t{count}")
    for split in ("train", "dev", "test"):
        subset = [r for r in records if r.split == split]
        candidates = sum(r.is_training_candidate for r in subset)
        rows.append(f"training_candidates.{split}\t{candidates}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_pairs(config: BuildConfig) -> list[OpdPair]:
    single_outcomes = load_post_eval(config.single_post_eval)
    multi_outcomes = load_post_eval(config.multi_post_eval)
    single_traj_dirs = find_traj_dirs_many((config.single_root, *config.single_extra_roots))
    multi_traj_dirs = find_traj_dirs_many((config.multi_root, *config.multi_extra_roots))
    single_task_dirs = find_task_dirs(config.single_tasks_root)
    multi_task_dirs = find_task_dirs(config.multi_tasks_root)

    common_keys = sorted(set(single_outcomes_by_key(single_outcomes)) & set(multi_outcomes_by_key(multi_outcomes)))
    if len(common_keys) != 70:
        raise ValueError(f"expected 70 paired filesystem tasks, found {len(common_keys)}")

    single_by_key = single_outcomes_by_key(single_outcomes)
    multi_by_key = multi_outcomes_by_key(multi_outcomes)
    group_map = {
        key: (single_by_key[key][1], multi_by_key[key][1])
        for key in common_keys
    }
    split_map = allocate_splits(
        common_keys,
        group_map,
        seed=config.seed,
        train_size=config.train_size,
        dev_size=config.dev_size,
        test_size=config.test_size,
    )

    records: list[OpdPair] = []
    for key in common_keys:
        single_id, single_outcome = single_by_key[key]
        multi_id, multi_outcome = multi_by_key[key]
        single_dir = single_traj_dirs[key]
        multi_dir = multi_traj_dirs.get(key)
        single_task_dir = single_task_dirs[key]
        multi_task_dir = multi_task_dirs[key]
        single_traj_path = single_dir / f"traj_{single_id}.json"
        single_traj = load_trajectory(single_traj_path)
        system_content = first_system_message(single_traj)
        single_prompt = task_text(single_task_dir, single_id, single=True)
        if multi_dir is not None:
            multi_traj_path = multi_dir / f"traj_{multi_id}.json"
            multi_traj = load_trajectory(multi_traj_path)
            selected_idx, selected_event, selected_kind = select_student_action_event(multi_traj)
            selected_message = event_message(selected_event)
            raw_messages = history_messages_before(multi_traj, selected_idx, system_content)
            mask_spans = decision_mask_spans(selected_message)
            selected_event_id = selected_event.get("id")
            multi_traj_path_text = relative_or_abs(multi_traj_path, config.repo_root)
        else:
            selected_idx = -1
            selected_kind = "missing_multiturn_trajectory"
            selected_message = ""
            raw_messages = raw_messages_from_multiturn_task(system_content, multi_task_dir)
            mask_spans = []
            selected_event_id = None
            multi_traj_path_text = ""

        records.append(
            OpdPair(
                task_key=key,
                split=split_map[key],
                single_task_id=single_id,
                multi_task_id=multi_id,
                single_outcome=single_outcome,
                multi_outcome=multi_outcome,
                canonical_single_messages=canonical_messages(system_content, single_prompt),
                raw_multiturn_messages=raw_messages,
                student_action_prefix_source=selected_message,
                decision_mask_spans=mask_spans,
                single_traj_path=relative_or_abs(single_traj_path, config.repo_root),
                multi_traj_path=multi_traj_path_text,
                single_task_path=relative_or_abs(single_task_dir, config.repo_root),
                multi_task_path=relative_or_abs(multi_task_dir, config.repo_root),
                selected_event_id=selected_event_id,
                selected_event_index=selected_idx,
                selected_event_kind=selected_kind,
                is_training_candidate=(single_outcome == "REJECT" and bool(selected_message)),
            )
        )
    return records


def single_outcomes_by_key(outcomes: dict[str, str]) -> dict[str, tuple[str, str]]:
    return {normalize_task_id(task_id): (task_id, outcome) for task_id, outcome in outcomes.items()}


def multi_outcomes_by_key(outcomes: dict[str, str]) -> dict[str, tuple[str, str]]:
    return {normalize_task_id(task_id): (task_id, outcome) for task_id, outcome in outcomes.items()}


def materialize_dataset(config: BuildConfig) -> dict[str, Any]:
    records = build_pairs(config)
    config.output_root.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[OpdPair]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)

    for split in ("train", "dev", "test"):
        write_jsonl(
            config.output_root / f"{split}.jsonl",
            [asdict(record) for record in sorted(by_split[split], key=lambda r: r.task_key)],
        )

    manifest = {
        "dataset": "mtagentrisk_qwen35_filesystem_p3_masked_opd",
        "seed": config.seed,
        "single_root": str(config.single_root),
        "multi_root": str(config.multi_root),
        "single_extra_roots": [str(path) for path in config.single_extra_roots],
        "multi_extra_roots": [str(path) for path in config.multi_extra_roots],
        "single_post_eval": str(config.single_post_eval),
        "multi_post_eval": str(config.multi_post_eval),
        "single_tasks_root": str(config.single_tasks_root),
        "multi_tasks_root": str(config.multi_tasks_root),
        "output_root": str(config.output_root),
        "split_counts": dict(Counter(r.split for r in records)),
        "outcome_counts": {
            f"{single}->{multi}": count
            for (single, multi), count in Counter(
                (r.single_outcome, r.multi_outcome) for r in records
            ).items()
        },
        "training_candidates": {
            split: sum(r.is_training_candidate for r in items)
            for split, items in by_split.items()
        },
    }
    (config.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_stats(config.output_root / "stats.tsv", records)
    return manifest
