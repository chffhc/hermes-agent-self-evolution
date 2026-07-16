"""CLI for validating and exercising the capability benchmark foundation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from benchmarks.capability.batch_adapter import build_batch_runner_plan
from benchmarks.capability.compare import compare_runs
from benchmarks.capability.executor import BudgetConfig, build_fake_agent_invoker, run_local
from benchmarks.capability.hermes_adapter import (
    LiveExecutionApproval,
    build_live_hermes_invoker,
    build_stub_hermes_invoker,
    probe_hermes_checkout,
)
from benchmarks.capability.replay import digest_artifact, run_replay
from benchmarks.capability.schema import RunFingerprint, SchemaError, load_run_result
from benchmarks.capability.suite import load_suite


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _load_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot load config JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError("config JSON must contain an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capability-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a suite and all verifier fixtures")
    validate.add_argument("--suite", required=True)

    replay = sub.add_parser(
        "replay", help="exercise deterministic fixtures; never capability evidence"
    )
    replay.add_argument("--suite", required=True)
    replay.add_argument("--role", choices=("baseline", "candidate"), required=True)
    replay.add_argument("--artifact", required=True)
    replay.add_argument("--model", required=True)
    replay.add_argument("--config-json")
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--environment", required=True)
    replay.add_argument("--apply-solution", action="store_true")
    replay.add_argument("--output", required=True)

    run_fake = sub.add_parser(
        "run-fake",
        help=(
            "end-to-end isolated-workspace run with the bundled local fake agent; "
            "free, deterministic, and never capability evidence"
        ),
    )
    run_fake.add_argument("--suite", required=True)
    run_fake.add_argument("--role", choices=("baseline", "candidate"), required=True)
    run_fake.add_argument("--artifact", required=True)
    run_fake.add_argument("--artifact-digest", help="expected sha256; mismatch fails closed")
    run_fake.add_argument("--model", required=True)
    run_fake.add_argument("--config-json")
    run_fake.add_argument("--seed", type=int, default=0)
    run_fake.add_argument("--environment", required=True)
    run_fake.add_argument(
        "--solve", action="store_true", help="fake agent applies the checked-in replay solution"
    )
    run_fake.add_argument(
        "--budget-usd", type=float, default=0.0, help="post-run accounting ceiling"
    )
    run_fake.add_argument("--task-budget-usd", type=float, help="per-task accounting ceiling")
    run_fake.add_argument("--run-id")
    run_fake.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="retain per-task workspaces for debugging (path recorded in notes)",
    )
    run_fake.add_argument("--output", required=True)

    probe = sub.add_parser(
        "probe-hermes",
        help=(
            "fail-closed compatibility probe of a current-Hermes checkout; reports the "
            "exact seam invariants and blockers, never executes Hermes"
        ),
    )
    probe.add_argument("--hermes-repo", required=True)
    probe.add_argument("--output")

    def _add_hermes_run_args(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--suite", required=True)
        cmd.add_argument("--role", choices=("baseline", "candidate"), required=True)
        cmd.add_argument(
            "--artifact",
            required=True,
            help="skill directory containing SKILL.md (only live track)",
        )
        cmd.add_argument("--artifact-digest", help="expected sha256; mismatch fails closed")
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--max-turns", type=int, default=20)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--environment", required=True)
        cmd.add_argument("--run-id")
        cmd.add_argument("--keep-workspaces", action="store_true")
        cmd.add_argument("--output", required=True)

    run_stub = sub.add_parser(
        "run-hermes-stub",
        help=(
            "end-to-end run against the bundled current-Hermes CLI contract emulator; "
            "free, local, and never capability evidence"
        ),
    )
    _add_hermes_run_args(run_stub)
    run_stub.add_argument(
        "--solve", action="store_true", help="stub applies the checked-in replay solution"
    )
    run_stub.add_argument(
        "--budget-usd", type=float, default=0.0, help="post-run accounting ceiling"
    )
    run_stub.add_argument("--task-budget-usd", type=float, help="per-task accounting ceiling")

    run_live = sub.add_parser(
        "run-hermes-live",
        help=(
            "reserved REAL current-Hermes invocation contract. Intentionally blocked "
            "before launch until probe-hermes reports pre-spend USD enforcement and "
            "filesystem confinement; confirmation cannot bypass those blockers."
        ),
    )
    _add_hermes_run_args(run_live)
    run_live.add_argument("--hermes-repo", required=True)
    run_live.add_argument(
        "--confirm-live-spend",
        required=True,
        help="must be the exact confirmation phrase printed by a wrong attempt",
    )
    run_live.add_argument(
        "--budget-usd", type=float, required=True, help="run accounting ceiling (> 0)"
    )
    run_live.add_argument(
        "--task-budget-usd",
        type=float,
        required=True,
        help="per-task accounting ceiling (> 0)",
    )
    run_live.add_argument("--provider")
    run_live.add_argument(
        "--allow-env",
        action="append",
        default=[],
        metavar="NAME",
        help="environment variable NAME passed through to Hermes (e.g. an API key); "
        "values are never recorded",
    )

    compare = sub.add_parser("compare", help="compare paired baseline/candidate run files")
    compare.add_argument("--suite", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output")

    plan = sub.add_parser(
        "plan-batch", help="build a non-executable Hermes batch_runner dry-run plan"
    )
    plan.add_argument("--suite", required=True)
    plan.add_argument("--hermes-repo", required=True)
    plan.add_argument("--dataset", required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--run-name", required=True)
    plan.add_argument("--max-turns", type=int, default=20)
    plan.add_argument("--batch-size", type=int, default=1)
    plan.add_argument("--num-workers", type=int, default=1)
    plan.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe-hermes":
            report = probe_hermes_checkout(args.hermes_repo)
            payload = report.to_dict()
            if args.output:
                _write_json(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            return 0 if report.compatible else 2
        suite = load_suite(args.suite)
        if args.command == "validate":
            payload: dict[str, object] = {
                "valid": True,
                "suite_id": suite.suite_id,
                "suite_hash": suite.suite_hash,
                "task_count": len(suite.tasks),
                "capability_evidence": False,
            }
        elif args.command == "replay":
            fingerprint = RunFingerprint.from_config(
                args.model, _load_config(args.config_json), args.seed, args.environment
            )
            result = run_replay(
                suite,
                run_role=args.role,
                artifact_digest=digest_artifact(args.artifact),
                fingerprint=fingerprint,
                apply_solution=args.apply_solution,
            )
            payload = result.to_dict()
            _write_json(args.output, payload)
        elif args.command == "run-fake":
            fingerprint = RunFingerprint.from_config(
                args.model, _load_config(args.config_json), args.seed, args.environment
            )
            outcome = run_local(
                suite,
                invoker=build_fake_agent_invoker(solve=args.solve),
                run_role=args.role,
                artifact_path=args.artifact,
                expected_artifact_digest=args.artifact_digest,
                fingerprint=fingerprint,
                budget=BudgetConfig(max_run_usd=args.budget_usd, max_task_usd=args.task_budget_usd),
                run_id=args.run_id,
                keep_workspaces=args.keep_workspaces,
            )
            payload = outcome.result.to_dict()
            _write_json(args.output, payload)
        elif args.command == "run-hermes-stub":
            invoker = build_stub_hermes_invoker(
                args.artifact,
                solve=args.solve,
                expected_model=args.model,
                max_turns=args.max_turns,
            )
            fingerprint = RunFingerprint.from_config(
                args.model, invoker.fingerprint_config(), args.seed, args.environment
            )
            outcome = run_local(
                suite,
                invoker=invoker,
                run_role=args.role,
                artifact_path=args.artifact,
                expected_artifact_digest=args.artifact_digest,
                fingerprint=fingerprint,
                budget=BudgetConfig(max_run_usd=args.budget_usd, max_task_usd=args.task_budget_usd),
                run_id=args.run_id,
                keep_workspaces=args.keep_workspaces,
            )
            payload = outcome.result.to_dict()
            _write_json(args.output, payload)
        elif args.command == "run-hermes-live":
            approval = LiveExecutionApproval(
                confirm=args.confirm_live_spend,
                max_run_usd=args.budget_usd,
                max_task_usd=args.task_budget_usd,
                env_passthrough=tuple(args.allow_env),
            )
            invoker = build_live_hermes_invoker(
                args.artifact,
                checkout=args.hermes_repo,
                approval=approval,
                model=args.model,
                provider=args.provider,
                max_turns=args.max_turns,
            )
            fingerprint = RunFingerprint.from_config(
                args.model, invoker.fingerprint_config(), args.seed, args.environment
            )
            outcome = run_local(
                suite,
                invoker=invoker,
                run_role=args.role,
                artifact_path=args.artifact,
                expected_artifact_digest=args.artifact_digest,
                fingerprint=fingerprint,
                budget=BudgetConfig(max_run_usd=args.budget_usd, max_task_usd=args.task_budget_usd),
                run_id=args.run_id,
                keep_workspaces=args.keep_workspaces,
                live_approval=approval,
            )
            payload = outcome.result.to_dict()
            _write_json(args.output, payload)
        elif args.command == "compare":
            result = compare_runs(
                suite, load_run_result(args.baseline), load_run_result(args.candidate)
            )
            payload = result.to_dict()
            if args.output:
                _write_json(args.output, payload)
        else:
            payload = build_batch_runner_plan(
                suite,
                hermes_repo=args.hermes_repo,
                dataset_path=args.dataset,
                model=args.model,
                run_name=args.run_name,
                max_turns=args.max_turns,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            if args.output:
                _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except SchemaError as exc:
        print(json.dumps({"error": str(exc), "valid": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
