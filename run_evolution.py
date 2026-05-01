#!/usr/bin/env python
"""Run evolution with CLI arguments."""
import os
import sys

import click

from evolution.skills.evolve_skill import evolve

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)  # noqa: E402
sys.stderr.reconfigure(line_buffering=True)  # noqa: E402
os.environ["PYTHONUNBUFFERED"] = "1"  # noqa: E402


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=50, help="Number of GEPA iterations")
@click.option(
    "--eval-source",
    default="golden",
    type=click.Choice(["synthetic", "golden", "sessiondb"]),
    help="Source for evaluation dataset",
)
@click.option("--dataset-path", default=None, help="Path to existing eval dataset")
@click.option(
    "--optimizer-model",
    default="qwen3.6-plus",
    help="Model for GEPA reflections (default: qwen3.6-plus)",
)
@click.option(
    "--eval-model",
    default="qwen3.6-plus",
    help="Model for LLM-as-judge evaluation (default: qwen3.6-plus)",
)
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model):
    """Run skill evolution with configurable parameters."""
    print("=== Starting Evolution ===", flush=True)
    print(f"Skill: {skill}", flush=True)
    print(f"Iterations: {iterations}", flush=True)
    print(f"Eval source: {eval_source}", flush=True)
    print(f"Optimizer model: {optimizer_model}", flush=True)
    print(f"Eval model: {eval_model}", flush=True)

    kwargs = {
        "skill_name": skill,
        "iterations": iterations,
        "eval_source": eval_source,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
    }
    if dataset_path:
        kwargs["dataset_path"] = dataset_path

    evolve(**kwargs)

    # Auto-diff: show the difference between baseline and evolved skill
    _print_diff(skill)

    print("=== Evolution Complete ===", flush=True)


def _print_diff(skill_name: str):
    """Print a unified diff between baseline and evolved skill."""
    import difflib
    from pathlib import Path

    output_dir = Path("output") / skill_name
    if not output_dir.exists():
        return

    # Find the latest run directory
    subdirs = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not subdirs:
        return

    latest = subdirs[0]
    baseline_file = latest / "baseline_skill.md"
    evolved_file = latest / "evolved_skill.md"

    if not baseline_file.exists() or not evolved_file.exists():
        return

    baseline = baseline_file.read_text().splitlines()
    evolved = evolved_file.read_text().splitlines()

    diff = list(
        difflib.unified_diff(
            baseline, evolved, fromfile="baseline", tofile="evolved", lineterm=""
        )
    )

    if diff:
        print("\n" + "=" * 50)
        print("  Evolved Skill Diff")
        print("=" * 50)
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"\033[32m{line}\033[0m")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"\033[31m{line}\033[0m")
            else:
                print(line)
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
