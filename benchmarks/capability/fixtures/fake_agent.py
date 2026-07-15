#!/usr/bin/env python3
"""Deterministic local fake agent for capability-harness integration tests.

This script is NOT an agent and never produces capability evidence. It
replays scripted filesystem effects so the executor's subprocess seam —
workspace isolation, run/task attribution, usage parsing, budget gating,
timeout handling, and post-run verification — can be exercised end-to-end
without any model call. Failure flags exist solely so tests can prove the
harness fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--usage-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--solutions", help="directory copied over the workspace")
    parser.add_argument("--sleep", type=float, default=0.0, help="stall to trigger timeouts")
    parser.add_argument("--cost-usd", type=float, default=0.0)
    parser.add_argument("--no-usage", action="store_true", help="skip the usage report")
    parser.add_argument("--malformed-usage", action="store_true", help="write invalid JSON usage")
    parser.add_argument("--symlink-escape", action="store_true", help="plant an escaping symlink")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    if args.sleep:
        time.sleep(args.sleep)
    if args.solutions:
        solutions = Path(args.solutions)
        if not solutions.is_dir():
            print(f"fake-agent: solutions directory missing: {solutions}", file=sys.stderr)
            return 3
        shutil.copytree(solutions, workspace, dirs_exist_ok=True)
    if args.symlink_escape:
        os.symlink(os.path.abspath(os.sep), workspace / "escape-link")

    print(f"fake-agent run_id={args.run_id} task_id={args.task_id} (not capability evidence)")
    usage_file = Path(args.usage_file)
    if args.malformed_usage:
        usage_file.write_text("{not json", encoding="utf-8")
    elif not args.no_usage:
        usage_file.write_text(
            json.dumps(
                {"cost_usd": args.cost_usd, "input_tokens": 10, "output_tokens": 10},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return args.exit_code


if __name__ == "__main__":
    sys.exit(main())
