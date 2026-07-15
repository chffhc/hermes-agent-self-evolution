"""Read-only dry-run seam to the current Hermes ``batch_runner.py``."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.capability.schema import SchemaError
from benchmarks.capability.suite import CapabilitySuite


def build_batch_runner_plan(
    suite: CapabilitySuite,
    *,
    hermes_repo: str | Path,
    dataset_path: str | Path,
    model: str,
    run_name: str,
    max_turns: int = 20,
    batch_size: int = 1,
    num_workers: int = 1,
) -> dict[str, object]:
    """Write a dataset and return a non-executable, non-evidence plan.

    Current Hermes batch_runner records trajectories but does not inject an
    isolated per-task workspace/artifact or run these deterministic verifiers.
    The plan is therefore deliberately marked non-executable until a live
    adapter implements that boundary.
    """
    repo = Path(hermes_repo).resolve()
    runner = repo / "batch_runner.py"
    if not runner.is_file():
        raise SchemaError(f"Hermes batch_runner.py not found under {repo}")
    if not model.strip() or not run_name.strip():
        raise SchemaError("model and run_name must be non-empty")
    if max_turns < 1 or batch_size < 1 or num_workers < 1:
        raise SchemaError("max_turns, batch_size, and num_workers must be positive")
    output = Path(dataset_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in suite.tasks:
            handle.write(
                json.dumps(
                    {"prompt": task.prompt, "task_id": task.task_id},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    argv = [
        "<python>",
        str(runner),
        f"--dataset_file={output}",
        f"--batch_size={batch_size}",
        f"--run_name={run_name}",
        f"--model={model}",
        f"--max_turns={max_turns}",
        f"--num_workers={num_workers}",
    ]
    return {
        "execution_mode": "dry_run",
        "capability_evidence": False,
        "executable": False,
        "runner": str(runner),
        "dataset": str(output),
        "argv": argv,
        "task_count": len(suite.tasks),
        "blocking_gaps": [
            "isolated per-task workspace mounting",
            "baseline/candidate artifact injection",
            "trajectory-to-workspace/result attribution",
            "post-run deterministic verifier invocation",
            "cost extraction and hard budget enforcement",
        ],
        "note": "Command shape only. Do not execute as capability evaluation yet.",
    }
