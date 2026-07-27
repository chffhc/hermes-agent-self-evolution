from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.capability import schema as capability_schema
from benchmarks.capability.cli import main as cli_main
from benchmarks.capability.compare import compare_runs
from benchmarks.capability.executor import (
    FAKE_AGENT_SCRIPT,
    ArgvAgentInvoker,
    BudgetConfig,
    build_fake_agent_invoker,
    run_local,
)
from benchmarks.capability.replay import digest_artifact
from benchmarks.capability.schema import (
    RunFingerprint,
    RunResult,
    SchemaError,
    UsageReport,
    load_run_result,
    load_usage_report,
)
from benchmarks.capability.suite import load_suite

REPO = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO / "benchmarks/capability/suites/native_v1/suite.json"


def _fingerprint(seed: int = 7) -> RunFingerprint:
    return RunFingerprint.from_config(
        "test/model", {"max_turns": 20, "tools": "default"}, seed, "fixture-env-v1"
    )


def _artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "candidate-skill.md"
    artifact.write_text("candidate artifact under test\n")
    return artifact


def _fake_invoker(*extra: str) -> ArgvAgentInvoker:
    return ArgvAgentInvoker(
        (
            "python",
            str(FAKE_AGENT_SCRIPT),
            "--workspace",
            "{workspace}",
            "--usage-file",
            "{usage_file}",
            "--run-id",
            "{run_id}",
            "--task-id",
            "{task_id}",
            *extra,
        )
    )


def _mini_suite(tmp_path: Path, *, task_count: int = 1, timeout: float = 5) -> Path:
    """Small single-verifier suite so failure-mode tests stay fast."""
    root = tmp_path / "mini-suite"
    tasks = []
    for i in range(task_count):
        task_id = f"mini-task-{i}"
        task_dir = root / "tasks" / task_id
        (task_dir / "workspace").mkdir(parents=True)
        (task_dir / "workspace" / "seed.txt").write_text("seed\n")
        (task_dir / "replay").mkdir()
        (task_dir / "replay" / "out.txt").write_text("done\n")
        (task_dir / "expected").mkdir()
        (task_dir / "expected" / "out.txt").write_text("done\n")
        tasks.append(
            {
                "task_id": task_id,
                "category": "file-editing",
                "prompt": "Write out.txt containing done.",
                "fixture": f"tasks/{task_id}",
                "verifiers": [
                    {
                        "type": "file_exact",
                        "params": {"path": "out.txt", "expected_file": "expected/out.txt"},
                    }
                ],
                "timeout_seconds": timeout,
                "critical": True,
            }
        )
    suite_doc = {
        "schema_version": 1,
        "suite_id": "mini-exec-suite",
        "description": "executor failure-mode fixture suite",
        "tasks": tasks,
    }
    suite_path = root / "suite.json"
    suite_path.write_text(json.dumps(suite_doc))
    return suite_path


def test_fake_agent_end_to_end_and_honest_comparison(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    artifact = _artifact(tmp_path)
    baseline = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=False),
        run_role="baseline",
        artifact_path=artifact,
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    candidate = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="candidate",
        artifact_path=artifact,
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    assert baseline.execution_mode == candidate.execution_mode == "fake_agent"
    assert baseline.capability_evidence is False
    assert candidate.capability_evidence is False
    assert baseline.pass_rate == 0
    assert candidate.pass_rate == 1
    assert baseline.run_id and candidate.run_id and baseline.run_id != candidate.run_id
    assert baseline.artifact_digest == digest_artifact(artifact)
    assert all(r.cost_usd == 0.0 for r in candidate.results)
    comparison = compare_runs(suite, baseline, candidate)
    assert comparison.passed_gate
    assert comparison.capability_evidence is False


def test_fake_agent_run_cannot_claim_capability_evidence(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    raw = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result.to_dict()
    raw["capability_evidence"] = True
    with pytest.raises(SchemaError, match="only valid"):
        RunResult.from_dict(raw)


def test_non_fake_invoker_modes_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="live invocation is not implemented"):
        ArgvAgentInvoker(("python", str(FAKE_AGENT_SCRIPT)), execution_mode="live")

    class LiveImpostor:
        execution_mode = "live"

        def invoke(self, invocation):  # pragma: no cover - must not be reached
            raise AssertionError("must not run")

    suite = load_suite(_mini_suite(tmp_path))
    with pytest.raises(SchemaError, match="requires the attested HermesCliInvoker"):
        run_local(
            suite,
            invoker=LiveImpostor(),
            run_role="baseline",
            artifact_path=_artifact(tmp_path),
            fingerprint=_fingerprint(),
            budget=BudgetConfig(max_run_usd=0.0),
            runs_root=tmp_path / "runs",
        )

    class UnknownMode:
        execution_mode = "telepathy"

        def invoke(self, invocation):  # pragma: no cover - must not be reached
            raise AssertionError("must not run")

    with pytest.raises(SchemaError, match="unsupported invoker execution_mode"):
        run_local(
            suite,
            invoker=UnknownMode(),
            run_role="baseline",
            artifact_path=_artifact(tmp_path),
            fingerprint=_fingerprint(),
            budget=BudgetConfig(max_run_usd=0.0),
            runs_root=tmp_path / "runs",
        )


def test_artifact_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="artifact digest mismatch"):
        run_local(
            load_suite(_mini_suite(tmp_path)),
            invoker=build_fake_agent_invoker(solve=True),
            run_role="candidate",
            artifact_path=_artifact(tmp_path),
            expected_artifact_digest="0" * 64,
            fingerprint=_fingerprint(),
            budget=BudgetConfig(max_run_usd=0.0),
            runs_root=tmp_path / "runs",
        )


def test_artifact_injected_with_digest_binding_and_retained_workspaces(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    artifact = _artifact(tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    outcome = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="candidate",
        artifact_path=artifact,
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=runs_root,
        keep_workspaces=True,
    )
    assert outcome.retained_root is not None and outcome.retained_root.is_dir()
    assert str(outcome.retained_root) in outcome.result.notes
    injected = outcome.retained_root / "tasks/mini-task-0/workspace/hermes_artifact" / artifact.name
    assert digest_artifact(injected) == outcome.result.artifact_digest
    invocation = json.loads(
        (outcome.retained_root / "tasks/mini-task-0/control/invocation.json").read_text()
    )
    assert invocation["run_id"] == outcome.result.run_id
    assert invocation["task_id"] == "mini-task-0"
    assert invocation["capability_evidence"] is False


def test_workspaces_cleaned_up_on_success_and_harness_error(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    artifact = _artifact(tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="baseline",
        artifact_path=artifact,
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=runs_root,
    )
    assert list(runs_root.iterdir()) == []
    # Harness misconfiguration mid-run must still clean the run root.
    with pytest.raises(SchemaError, match="collides with fixture content"):
        run_local(
            suite,
            invoker=build_fake_agent_invoker(solve=True),
            run_role="baseline",
            artifact_path=artifact,
            fingerprint=_fingerprint(),
            budget=BudgetConfig(max_run_usd=0.0),
            artifact_dest="seed.txt",
            runs_root=runs_root,
        )
    assert list(runs_root.iterdir()) == []


def test_artifact_dest_traversal_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="unsafe path segment"):
        run_local(
            load_suite(_mini_suite(tmp_path)),
            invoker=build_fake_agent_invoker(solve=True),
            run_role="baseline",
            artifact_path=_artifact(tmp_path),
            fingerprint=_fingerprint(),
            budget=BudgetConfig(max_run_usd=0.0),
            artifact_dest="../outside",
            runs_root=tmp_path / "runs",
        )


def test_subprocess_timeout_records_failure_and_cleans_up(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path, timeout=1))
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    result = run_local(
        suite,
        invoker=_fake_invoker("--sleep", "5", "--solutions", "{task_fixture_dir}/replay"),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=runs_root,
    ).result
    assert result.results[0].passed is False
    assert "timed out" in result.results[0].error
    assert list(runs_root.iterdir()) == []


def test_missing_usage_report_fails_closed(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path, task_count=2))
    result = run_local(
        suite,
        invoker=_fake_invoker("--no-usage", "--solutions", "{task_fixture_dir}/replay"),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    first, second = result.results
    assert first.passed is False and first.cost_usd is None
    assert first.error is not None and "usage report invalid or missing" in first.error
    assert second.passed is False
    assert second.error is not None and "not executed" in second.error
    assert "usage/cost is unknown" in second.error


def test_malformed_usage_report_fails_closed(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    result = run_local(
        suite,
        invoker=_fake_invoker("--malformed-usage", "--solutions", "{task_fixture_dir}/replay"),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    assert result.results[0].passed is False
    assert "usage report invalid or missing" in result.results[0].error


def test_per_task_budget_overrun_fails_closed(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    result = run_local(
        suite,
        invoker=_fake_invoker("--cost-usd", "2.0", "--solutions", "{task_fixture_dir}/replay"),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=10.0, max_task_usd=1.0),
        runs_root=tmp_path / "runs",
    ).result
    task = result.results[0]
    assert task.passed is False and task.cost_usd == 2.0
    assert "per-task budget" in task.error


def test_run_budget_exhaustion_blocks_remaining_tasks(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path, task_count=3))
    result = run_local(
        suite,
        invoker=_fake_invoker("--cost-usd", "2.0", "--solutions", "{task_fixture_dir}/replay"),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=3.0),
        runs_root=tmp_path / "runs",
    ).result
    by_id = {r.task_id: r for r in result.results}
    assert set(by_id) == set(suite.task_ids)
    assert by_id["mini-task-0"].passed is True  # $2 spent, within $3
    assert "exceeds run budget" in by_id["mini-task-1"].error
    assert "not executed" in by_id["mini-task-2"].error


def test_symlink_in_final_workspace_fails_closed(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    result = run_local(
        suite,
        invoker=_fake_invoker("--symlink-escape", "--solutions", "{task_fixture_dir}/replay"),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    assert result.results[0].passed is False
    assert "symlink" in result.results[0].error


def test_nonzero_agent_exit_fails_task(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    result = run_local(
        suite,
        invoker=_fake_invoker("--exit-code", "9", "--solutions", "{task_fixture_dir}/replay"),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    assert result.results[0].passed is False
    assert result.results[0].cost_usd == 0.0
    assert "exited with code 9" in result.results[0].error


def test_usage_report_schema_fails_closed() -> None:
    with pytest.raises(SchemaError, match="unknown keys"):
        UsageReport.from_dict(
            {"cost_usd": 0.0, "input_tokens": 1, "output_tokens": 1, "vibes": "good"}
        )
    with pytest.raises(SchemaError, match=">= 0"):
        UsageReport.from_dict({"cost_usd": -0.5, "input_tokens": 1, "output_tokens": 1})
    with pytest.raises(SchemaError, match="must be an integer"):
        UsageReport.from_dict({"cost_usd": 0.0, "input_tokens": 1.5, "output_tokens": 1})


def test_budget_config_fails_closed() -> None:
    with pytest.raises(SchemaError, match="finite"):
        BudgetConfig(max_run_usd=float("inf"))
    with pytest.raises(SchemaError, match=">= 0"):
        BudgetConfig(max_run_usd=1.0, max_task_usd=-1.0)


def test_cli_run_fake_and_compare_smoke(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    baseline_out = tmp_path / "baseline.json"
    candidate_out = tmp_path / "candidate.json"
    common = [
        "--suite",
        str(SUITE_PATH),
        "--artifact",
        str(artifact),
        "--model",
        "test/model",
        "--environment",
        "fixture-env-v1",
    ]
    assert cli_main(["run-fake", *common, "--role", "baseline", "--output", str(baseline_out)]) == 0
    assert (
        cli_main(
            [
                "run-fake",
                *common,
                "--role",
                "candidate",
                "--solve",
                "--output",
                str(candidate_out),
            ]
        )
        == 0
    )
    baseline = load_run_result(baseline_out)
    candidate = load_run_result(candidate_out)
    assert baseline.capability_evidence is False
    assert candidate.capability_evidence is False
    assert candidate.pass_rate == 1
    assert (
        cli_main(
            [
                "compare",
                "--suite",
                str(SUITE_PATH),
                "--baseline",
                str(baseline_out),
                "--candidate",
                str(candidate_out),
            ]
        )
        == 0
    )


def test_run_id_traversal_fails_before_workspace_creation(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    with pytest.raises(SchemaError, match="run_id must match"):
        run_local(
            load_suite(_mini_suite(tmp_path)),
            invoker=build_fake_agent_invoker(solve=True),
            run_role="baseline",
            artifact_path=_artifact(tmp_path),
            fingerprint=_fingerprint(),
            budget=BudgetConfig(max_run_usd=0.0),
            run_id="../escape",
            runs_root=runs_root,
        )
    assert not runs_root.exists()


def test_symlink_usage_report_fails_closed(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    outside = tmp_path / "outside-usage.json"
    outside.write_text('{"cost_usd": 0, "input_tokens": 0, "output_tokens": 0}')

    class SymlinkUsageInvoker:
        execution_mode = "fake_agent"

        def invoke(self, invocation):
            invocation.usage_file.symlink_to(outside)
            from benchmarks.capability.executor import InvocationOutcome

            return InvocationOutcome(exit_code=0, timed_out=False)

    result = run_local(
        suite,
        invoker=SymlinkUsageInvoker(),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    assert result.results[0].passed is False
    assert result.results[0].error is not None
    assert "must not be a symlink" in result.results[0].error


def test_duplicate_key_usage_report_fails_closed_and_halts_run(tmp_path: Path) -> None:
    """A duplicate cost_usd key must never be silently resolved last-wins:
    a human auditing control/usage.json could read the first value while the
    accounting gate charges the second."""

    class DuplicateKeyUsageInvoker:
        execution_mode = "fake_agent"

        def invoke(self, invocation):
            invocation.usage_file.write_text(
                '{"cost_usd": 9.99, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}',
                encoding="utf-8",
            )
            from benchmarks.capability.executor import InvocationOutcome

            return InvocationOutcome(exit_code=0, timed_out=False)

    result = run_local(
        load_suite(_mini_suite(tmp_path, task_count=2)),
        invoker=DuplicateKeyUsageInvoker(),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=10.0),
        runs_root=tmp_path / "runs",
    ).result
    first, second = result.results
    assert first.passed is False and first.cost_usd is None
    assert first.error is not None and "duplicate JSON key 'cost_usd'" in first.error
    assert "usage report invalid or missing" in first.error
    assert second.passed is False
    assert second.error is not None and "not executed" in second.error


def test_usage_report_ingestion_is_bounded_and_strict(tmp_path: Path) -> None:
    usage = tmp_path / "usage.json"

    usage.write_text('{"cost_usd": Infinity, "input_tokens": 1, "output_tokens": 1}')
    with pytest.raises(SchemaError, match="non-finite JSON constant"):
        load_usage_report(usage)

    usage.write_text("[" * 2000 + "0" + "]" * 2000)
    with pytest.raises(SchemaError, match="cannot read usage report"):
        load_usage_report(usage)

    usage.write_bytes(b" " * 65_537)
    with pytest.raises(SchemaError, match="exceeds 65536 bytes"):
        load_usage_report(usage)


def test_run_result_duplicate_keys_are_rejected_not_last_wins(tmp_path: Path) -> None:
    """A run file whose first run_role a reviewer reads differs from the one
    the parser keeps is a tampered document, not a valid run."""
    outcome = run_local(
        load_suite(_mini_suite(tmp_path)),
        invoker=build_fake_agent_invoker(solve=True),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    )
    text = json.dumps(outcome.result.to_dict())
    assert text.count('"run_role": "baseline"') == 1
    smuggled = text.replace(
        '"run_role": "baseline"', '"run_role": "candidate", "run_role": "baseline"'
    )
    run_file = tmp_path / "run.json"
    run_file.write_text(smuggled, encoding="utf-8")
    with pytest.raises(SchemaError, match="duplicate JSON key 'run_role'"):
        load_run_result(run_file)


def test_run_result_ingestion_is_bounded_and_types_deep_json_errors(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"

    run_file.write_text("[" * 2000 + "0" + "]" * 2000)
    with pytest.raises(SchemaError, match="cannot read run result"):
        load_run_result(run_file)

    run_file.write_bytes(b" " * 10_000_001)
    with pytest.raises(SchemaError, match="exceeds 10000000 bytes"):
        load_run_result(run_file)


def test_trust_boundary_json_rejects_symlinks_and_special_files(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "run.json"
    link.symlink_to(target)
    with pytest.raises(SchemaError, match="run result must not be a symlink"):
        load_run_result(link)

    fifo = tmp_path / "usage.json"
    os.mkfifo(fifo)
    with pytest.raises(SchemaError, match="usage report is not a regular file"):
        load_usage_report(fifo)


def test_run_result_rejects_path_replacement_between_stat_and_open(
    tmp_path: Path, monkeypatch
) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_text("{}", encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"replacement": true}', encoding="utf-8")
    original = tmp_path / "original.json"
    real_open = capability_schema.os.open
    replaced = False

    def replacing_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if Path(path) == run_file and not replaced:
            run_file.replace(original)
            replacement.replace(run_file)
            replaced = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(capability_schema.os, "open", replacing_open)
    with pytest.raises(SchemaError, match="changed before it could be read"):
        load_run_result(run_file)
    assert replaced and original.read_text(encoding="utf-8") == "{}"


def test_usage_report_rejects_same_inode_mutation_during_read(tmp_path: Path, monkeypatch) -> None:
    usage = tmp_path / "usage.json"
    usage.write_text('{"cost_usd": 0, "input_tokens": 1, "output_tokens": 1}')
    real_fstat = capability_schema.os.fstat
    calls = 0

    def mutating_fstat(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            usage.write_text(
                '{"cost_usd": 9, "input_tokens": 1, "output_tokens": 1}',
                encoding="utf-8",
            )
        return real_fstat(descriptor)

    monkeypatch.setattr(capability_schema.os, "fstat", mutating_fstat)
    with pytest.raises(SchemaError, match="changed while it was being read"):
        load_usage_report(usage)


def test_trust_boundary_json_loops_past_short_reads(tmp_path: Path, monkeypatch) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_bytes(b"{}TRAILING")
    real_fdopen = capability_schema.os.fdopen

    class ShortRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            return self.handle.read(min(size, 2))

        def seek(self, offset):
            return self.handle.seek(offset)

    monkeypatch.setattr(
        capability_schema.os,
        "fdopen",
        lambda descriptor, *args, **kwargs: ShortRead(real_fdopen(descriptor, *args, **kwargs)),
    )
    with pytest.raises(SchemaError, match="cannot read run result"):
        load_run_result(run_file)


def test_fdopen_failure_closes_untransferred_descriptor(tmp_path: Path, monkeypatch) -> None:
    usage = tmp_path / "usage.json"
    usage.write_text('{"cost_usd": 0, "input_tokens": 1, "output_tokens": 1}')
    descriptors: list[int] = []

    def failing_fdopen(descriptor, *args, **kwargs):
        descriptors.append(descriptor)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(capability_schema.os, "fdopen", failing_fdopen)
    with pytest.raises(SchemaError, match="cannot read usage report"):
        load_usage_report(usage)
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_missing_nonblocking_open_capability_fails_closed(tmp_path: Path, monkeypatch) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform has no filesystem FIFO race")
    usage = tmp_path / "usage.json"
    usage.write_text('{"cost_usd": 0, "input_tokens": 1, "output_tokens": 1}')
    monkeypatch.setattr(capability_schema.os, "O_NONBLOCK", 0)
    with pytest.raises(SchemaError, match="nonblocking regular-file open is unavailable"):
        load_usage_report(usage)


def test_double_read_detects_rewrite_when_metadata_appears_stable(
    tmp_path: Path, monkeypatch
) -> None:
    usage = tmp_path / "usage.json"
    usage.write_text('{"cost_usd": 0, "input_tokens": 1, "output_tokens": 1}')
    stable = os.stat(usage)
    real_fdopen = capability_schema.os.fdopen
    mutated = False

    class MutatingRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            nonlocal mutated
            data = self.handle.read(size)
            if data and not mutated:
                usage.write_text(
                    '{"cost_usd": 9, "input_tokens": 1, "output_tokens": 1}',
                    encoding="utf-8",
                )
                mutated = True
            return data

        def seek(self, offset):
            return self.handle.seek(offset)

    monkeypatch.setattr(
        capability_schema.os,
        "fdopen",
        lambda descriptor, *args, **kwargs: MutatingRead(real_fdopen(descriptor, *args, **kwargs)),
    )
    monkeypatch.setattr(capability_schema.os, "fstat", lambda descriptor: stable)
    monkeypatch.setattr(capability_schema.os, "stat", lambda *args, **kwargs: stable)
    with pytest.raises(SchemaError, match="changed while it was being read"):
        load_usage_report(usage)
    assert mutated


def test_compare_cli_keeps_stable_error_contract_on_pathological_run_json(
    tmp_path: Path, capsys
) -> None:
    """Nesting overflow in a run file must produce the one-line CLI error —
    and, in optimizer-feedback mode, only the holdout-safe diagnostic — not
    an uncaught RecursionError traceback."""
    bad = tmp_path / "baseline.json"
    bad.write_text("[" * 2000 + "0" + "]" * 2000)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    feedback = tmp_path / "feedback.json"
    rc = cli_main(
        [
            "compare",
            "--suite",
            str(SUITE_PATH),
            "--baseline",
            str(bad),
            "--candidate",
            str(candidate),
            "--optimizer-feedback",
            str(feedback),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["valid"] is False
    assert "holdout-aware details withheld" in error["error"]
    assert not feedback.exists()


def test_invoker_exception_records_failure_and_cleans_up(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    class BrokenInvoker:
        execution_mode = "fake_agent"

        def invoke(self, invocation):
            raise RuntimeError("fixture boom")

    result = run_local(
        load_suite(_mini_suite(tmp_path)),
        invoker=BrokenInvoker(),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=runs_root,
    ).result
    assert result.results[0].passed is False
    assert result.results[0].error is not None
    assert "invoker failure: RuntimeError: fixture boom" in result.results[0].error
    assert list(runs_root.iterdir()) == []


def test_invoker_exception_after_usage_write_still_accounts_spend(tmp_path: Path) -> None:
    class WritesUsageThenRaises:
        execution_mode = "fake_agent"

        def invoke(self, invocation):
            invocation.usage_file.write_text(
                json.dumps({"cost_usd": 2.0, "input_tokens": 10, "output_tokens": 5})
            )
            raise RuntimeError("failed after usage persistence")

    result = run_local(
        load_suite(_mini_suite(tmp_path, task_count=2)),
        invoker=WritesUsageThenRaises(),
        run_role="baseline",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=10.0, max_task_usd=5.0),
        runs_root=tmp_path / "runs",
    ).result
    assert len(result.results) == 2
    assert all(task.cost_usd == 2.0 for task in result.results)
    assert all(
        task.error and "failed after usage persistence" in task.error for task in result.results
    )
    assert all(
        task.error and "usage report invalid or missing" not in task.error
        for task in result.results
    )


def test_fake_run_fingerprint_mismatch_fails_closed(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    artifact = _artifact(tmp_path)
    baseline = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="baseline",
        artifact_path=artifact,
        fingerprint=_fingerprint(1),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    candidate = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="candidate",
        artifact_path=artifact,
        fingerprint=_fingerprint(2),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    with pytest.raises(SchemaError, match="fingerprint"):
        compare_runs(suite, baseline, candidate)


def test_fake_run_critical_regression_blocks_gate(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    artifact = _artifact(tmp_path)
    baseline = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="baseline",
        artifact_path=artifact,
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    candidate = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=False),
        run_role="candidate",
        artifact_path=artifact,
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
    ).result
    comparison = compare_runs(suite, baseline, candidate)
    assert comparison.passed_gate is False
    assert comparison.critical_regressions == ("mini-task-0",)


def test_fixture_cache_artifacts_never_enter_run_workspace(tmp_path: Path) -> None:
    suite_path = _mini_suite(tmp_path)
    cache = suite_path.parent / "tasks/mini-task-0/workspace/__pycache__"
    cache.mkdir()
    (cache / "stale.cpython-311.pyc").write_bytes(b"unbound bytes")
    (suite_path.parent / "tasks/mini-task-0/workspace/.DS_Store").write_bytes(b"finder")
    suite = load_suite(suite_path)
    outcome = run_local(
        suite,
        invoker=build_fake_agent_invoker(solve=True),
        run_role="candidate",
        artifact_path=_artifact(tmp_path),
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
        keep_workspaces=True,
    )
    assert outcome.result.pass_rate == 1
    assert outcome.retained_root is not None
    workspace = outcome.retained_root / "tasks/mini-task-0/workspace"
    assert (workspace / "seed.txt").is_file()
    assert not (workspace / "__pycache__").exists()
    assert not (workspace / ".DS_Store").exists()
