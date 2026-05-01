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
def main(skill, iterations, eval_source, dataset_path):
    """Run skill evolution with configurable parameters."""
    print("=== Starting Evolution ===", flush=True)
    print(f"Skill: {skill}", flush=True)
    print(f"Iterations: {iterations}", flush=True)
    print(f"Eval source: {eval_source}", flush=True)

    kwargs = {
        "skill_name": skill,
        "iterations": iterations,
        "eval_source": eval_source,
        "optimizer_model": "qwen3.6-plus",
        "eval_model": "qwen3.6-plus",
    }
    if dataset_path:
        kwargs["dataset_path"] = dataset_path

    evolve(**kwargs)

    print("=== Evolution Complete ===", flush=True)


if __name__ == "__main__":
    main()
