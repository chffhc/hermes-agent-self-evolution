"""Regression tests for benchmark runner discovery and the local smoke runner.

Covers: resolver priority order and fail-closed semantics, the smoke runner's
CLI/JSON contract, and BenchmarkGate running end-to-end against the
repo-owned smoke runner when hermes-agent ships no benchmark infrastructure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import evolution.core.benchmark_gate as bg
from evolution.core.benchmark_gate import (
    BenchmarkGate,
    resolve_benchmark_runner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_RUNNER = REPO_ROOT / "benchmarks" / "run_bench.py"


@pytest.fixture(autouse=True)
def clear_runner_env(monkeypatch):
    monkeypatch.delenv("EVOLUTION_BENCH_RUNNER", raising=False)


def _run_runner(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SMOKE_RUNNER), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _make_target(tmp_path: Path, files: dict[str, str]) -> Path:
    target = tmp_path / "target"
    target.mkdir()
    for name, content in files.items():
        (target / name).write_text(content)
    return target


# ── runner discovery ─────────────────────────────────────────────────────


def _make_hermes_repo(tmp_path: Path, with_runner: bool) -> Path:
    repo = tmp_path / "hermes-agent"
    bench = repo / "environments" / "benchmarks"
    bench.mkdir(parents=True)
    if with_runner:
        (bench / "run_bench.py").write_text("print('{}')\n")
    return repo


def test_resolver_prefers_hermes_agent_runner(tmp_path: Path):
    repo = _make_hermes_repo(tmp_path, with_runner=True)

    runner, source = resolve_benchmark_runner(repo)

    assert source == "hermes-agent"
    assert runner == repo / "environments" / "benchmarks" / "run_bench.py"


def test_resolver_falls_back_to_local_smoke_runner(tmp_path: Path):
    repo = _make_hermes_repo(tmp_path, with_runner=False)

    runner, source = resolve_benchmark_runner(repo)

    assert source == "local-smoke"
    assert runner == SMOKE_RUNNER
    assert runner.is_file()


def test_resolver_env_override_wins_over_hermes_runner(tmp_path: Path, monkeypatch):
    repo = _make_hermes_repo(tmp_path, with_runner=True)
    custom = tmp_path / "custom_bench.py"
    custom.write_text("print('{}')\n")
    monkeypatch.setenv("EVOLUTION_BENCH_RUNNER", str(custom))

    runner, source = resolve_benchmark_runner(repo)

    assert source == "env"
    assert runner == custom


def test_resolver_explicit_path_wins_over_everything(tmp_path: Path, monkeypatch):
    repo = _make_hermes_repo(tmp_path, with_runner=True)
    monkeypatch.setenv("EVOLUTION_BENCH_RUNNER", str(tmp_path / "env_runner.py"))
    explicit = tmp_path / "explicit.py"
    explicit.write_text("print('{}')\n")

    runner, source = resolve_benchmark_runner(repo, runner_path=explicit)

    assert source == "configured"
    assert runner == explicit


def test_resolver_missing_explicit_path_fails_closed(tmp_path: Path):
    """A configured runner that doesn't exist must not fall through."""
    repo = _make_hermes_repo(tmp_path, with_runner=True)

    assert resolve_benchmark_runner(repo, runner_path=tmp_path / "nope.py") is None


def test_resolver_missing_env_path_fails_closed(tmp_path: Path, monkeypatch):
    repo = _make_hermes_repo(tmp_path, with_runner=True)
    monkeypatch.setenv("EVOLUTION_BENCH_RUNNER", str(tmp_path / "nope.py"))

    assert resolve_benchmark_runner(repo) is None


def test_resolver_returns_none_when_nothing_exists(tmp_path: Path, monkeypatch):
    repo = _make_hermes_repo(tmp_path, with_runner=False)
    monkeypatch.setattr(bg, "LOCAL_SMOKE_RUNNER", tmp_path / "missing.py")

    assert resolve_benchmark_runner(repo) is None


# ── smoke runner CLI contract ────────────────────────────────────────────


def test_runner_passes_on_clean_target(tmp_path: Path):
    target = _make_target(tmp_path, {"a.py": "x = 1\n", "b.py": "def f():\n    return 2\n"})

    proc = _run_runner("--tasks", "10", "--target", str(target))

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    assert output["passed"] == 2
    assert output["failed"] == 0
    assert output["proxy"] is True
    assert output["runner"] == "self-evolution-local-smoke"


def test_runner_reports_real_syntax_failures(tmp_path: Path):
    target = _make_target(tmp_path, {"good.py": "x = 1\n", "broken.py": "def broken(:\n"})

    proc = _run_runner("--tasks", "10", "--target", str(target))

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    assert output["passed"] == 1
    assert output["failed"] == 1
    failed = [c for c in output["checks"] if not c["ok"]]
    assert "broken.py" in failed[0]["name"]


def test_runner_respects_task_budget(tmp_path: Path):
    target = _make_target(tmp_path, {f"f{i}.py": "x = 1\n" for i in range(5)})

    proc = _run_runner("--tasks", "2", "--target", str(target))

    output = json.loads(proc.stdout)
    assert output["passed"] + output["failed"] == 2


def test_runner_validates_skill_overrides(tmp_path: Path):
    target = _make_target(tmp_path, {"a.py": "x = 1\n"})
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"good-skill": "Use tool X first.", "corrupt-skill": ""}))

    proc = _run_runner(
        "--tasks", "10", "--target", str(target), "--skill-overrides", str(overrides)
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    failed = [c["name"] for c in output["checks"] if not c["ok"]]
    assert failed == ["skill-override:corrupt-skill"]


def test_runner_fails_nonzero_on_missing_target(tmp_path: Path):
    proc = _run_runner("--tasks", "10", "--target", str(tmp_path / "nope"))

    assert proc.returncode != 0
    assert "not a directory" in proc.stderr


def test_runner_fails_nonzero_on_unreadable_overrides(tmp_path: Path):
    target = _make_target(tmp_path, {"a.py": "x = 1\n"})
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")

    proc = _run_runner("--tasks", "10", "--target", str(target), "--skill-overrides", str(bad))

    assert proc.returncode != 0


def test_runner_fails_nonzero_when_no_checks_possible(tmp_path: Path):
    """An empty target must not produce a vacuous pass."""
    target = tmp_path / "empty"
    target.mkdir()

    proc = _run_runner("--tasks", "10", "--target", str(target))

    assert proc.returncode != 0
    assert "no checks" in proc.stderr


# ── BenchmarkGate end-to-end with the smoke runner ───────────────────────


def _make_gate(tmp_path: Path, repo: Path) -> BenchmarkGate:
    return BenchmarkGate(hermes_agent_path=repo, baseline_file=tmp_path / "baselines.json")


def test_gate_runs_smoke_runner_and_labels_result(tmp_path: Path):
    repo = _make_hermes_repo(tmp_path, with_runner=False)
    (repo / "agent.py").write_text("x = 1\n")
    gate = _make_gate(tmp_path, repo)

    result = gate.run_tblite_fast(timeout=60)

    assert result.error is None
    assert result.name == "tblite-fast[smoke]"
    assert result.score == 1.0
    assert result.details["runner_source"] == "local-smoke"
    assert result.details["proxy"] is True


def test_gate_smoke_result_fails_on_real_check_failure(tmp_path: Path):
    repo = _make_hermes_repo(tmp_path, with_runner=False)
    (repo / "broken.py").write_text("def broken(:\n")
    gate = _make_gate(tmp_path, repo)

    result = gate.run_tblite_fast(timeout=60)

    assert result.error is None
    assert result.failed_tasks >= 1
    assert result.score < 1.0


def test_gate_passes_skill_overrides_through_to_smoke_runner(tmp_path: Path):
    repo = _make_hermes_repo(tmp_path, with_runner=False)
    (repo / "agent.py").write_text("x = 1\n")
    gate = _make_gate(tmp_path, repo)

    result = gate.run_tblite_fast(skill_overrides={"evolved": ""}, timeout=60)

    assert result.error is None
    assert result.failed_tasks >= 1  # corrupt (empty) evolved skill must fail


def test_gate_errors_when_no_runner_exists(tmp_path: Path, monkeypatch):
    repo = _make_hermes_repo(tmp_path, with_runner=False)
    monkeypatch.setattr(bg, "LOCAL_SMOKE_RUNNER", tmp_path / "missing.py")
    gate = _make_gate(tmp_path, repo)

    result = gate.run_tblite_fast(timeout=60)

    assert result.error is not None
    assert "No benchmark runner available" in result.error
    assert not gate.check_gate([result]).passed  # fail-closed downstream


def test_gate_errors_when_hermes_repo_missing(tmp_path: Path):
    gate = _make_gate(tmp_path, tmp_path / "nope")

    result = gate.run_tblite_fast(timeout=60)

    assert result.error is not None
    assert "repo not found" in result.error


def test_smoke_baselines_keyed_under_smoke_namespace(tmp_path: Path, monkeypatch):
    """establish_baselines must store scores under the [smoke]-suffixed name
    so smoke and real-benchmark scores never compare against each other."""
    from evolution.core.benchmark_gate import establish_baselines

    monkeypatch.chdir(tmp_path)
    repo = _make_hermes_repo(tmp_path, with_runner=False)
    (repo / "agent.py").write_text("x = 1\n")

    baselines = establish_baselines(hermes_agent_path=repo)

    assert set(baselines) == {"tblite-fast[smoke]", "tblite-full[smoke]", "yc-bench-fast[smoke]"}

    # And a subsequent gate check against those baselines passes end-to-end.
    gate = BenchmarkGate(hermes_agent_path=repo, baseline_file=Path("benchmarks/baselines.json"))
    result = gate.run_tblite_fast(timeout=60)
    assert gate.check_gate([result]).passed
