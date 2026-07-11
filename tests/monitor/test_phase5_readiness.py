"""Tests for the Phase 5 fail-closed unattended-readiness gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evolution.core.cost_tracker import tracker
from evolution.monitor.readiness import (
    ReadinessCheck,
    ReadinessReport,
    check_phase5_readiness,
)


@pytest.fixture(autouse=True)
def restore_budget():
    """Tests mutate the global tracker budget; always restore it."""
    old = tracker.max_cost_usd
    yield
    tracker.set_budget(old)


def _make_repo(tmp_path: Path, with_runner: bool = True) -> Path:
    repo = tmp_path / "hermes-agent"
    bench = repo / "environments" / "benchmarks"
    bench.mkdir(parents=True)
    if with_runner:
        (bench / "run_bench.py").write_text("print('{}')\n")
    return repo


def _check(report: ReadinessReport, name: str) -> ReadinessCheck:
    matches = [c for c in report.checks if c.name == name]
    assert matches, f"missing check {name}"
    return matches[0]


def test_ready_when_repo_runner_and_budget_present(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = _make_repo(tmp_path)
    tracker.set_budget(5.0)

    report = check_phase5_readiness(hermes_repo=str(repo), hermes_home=tmp_path / ".hermes")

    assert report.ready
    assert report.failing() == []
    # Missing metrics sources are surfaced but advisory — they never block.
    session_db = _check(report, "session_db")
    assert not session_db.ok
    assert not session_db.required


def test_missing_hermes_repo_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker.set_budget(5.0)

    report = check_phase5_readiness(
        hermes_repo=str(tmp_path / "nope"), hermes_home=tmp_path / ".hermes"
    )

    assert not report.ready
    failing = {c.name for c in report.failing()}
    assert "hermes_repo" in failing
    assert "benchmark_runner" in failing  # cannot verify without the repo


def test_missing_benchmark_runner_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = _make_repo(tmp_path, with_runner=False)
    tracker.set_budget(5.0)

    report = check_phase5_readiness(hermes_repo=str(repo), hermes_home=tmp_path / ".hermes")

    assert not report.ready
    assert {c.name for c in report.failing()} == {"benchmark_runner"}


def test_missing_hard_budget_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = _make_repo(tmp_path)
    tracker.set_budget(None)

    report = check_phase5_readiness(hermes_repo=str(repo), hermes_home=tmp_path / ".hermes")

    assert not report.ready
    assert {c.name for c in report.failing()} == {"hard_budget"}
    assert "EVOLUTION_MAX_COST_USD" in _check(report, "hard_budget").detail


def test_checkpoint_presence_is_informational_only(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = _make_repo(tmp_path)
    tracker.set_budget(5.0)
    out_dir = tmp_path / "output" / "monitor"
    out_dir.mkdir(parents=True)
    (out_dir / "checkpoint.json").write_text("{}")

    report = check_phase5_readiness(
        hermes_repo=str(repo), hermes_home=tmp_path / ".hermes", output_dir=out_dir
    )

    assert report.ready
    checkpoint = _check(report, "checkpoint")
    assert checkpoint.ok and not checkpoint.required
    assert "resume" in checkpoint.detail


def test_report_is_json_serializable(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker.set_budget(None)

    report = check_phase5_readiness(
        hermes_repo=str(tmp_path / "nope"), hermes_home=tmp_path / ".hermes"
    )

    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["ready"] is False
    assert {c["name"] for c in payload["checks"]} >= {"hermes_repo", "hard_budget"}


# ── CLI wiring ───────────────────────────────────────────────────────────


def _not_ready_report() -> ReadinessReport:
    return ReadinessReport(
        checks=[ReadinessCheck("hard_budget", False, True, "no hard USD budget")]
    )


def _ready_report() -> ReadinessReport:
    return ReadinessReport(checks=[ReadinessCheck("hard_budget", True, True, "ok")])


def test_cli_status_exits_nonzero_when_not_ready(monkeypatch, capsys):
    import evolution.monitor.continuous_evolution as ce

    monkeypatch.setattr(ce, "check_phase5_readiness", lambda **kw: _not_ready_report())
    monkeypatch.setattr(sys, "argv", ["continuous_evolution", "--status"])

    with pytest.raises(SystemExit) as excinfo:
        ce.main()

    assert excinfo.value.code == 1
    assert '"ready": false' in capsys.readouterr().out


def test_cli_status_exits_zero_when_ready(monkeypatch, capsys):
    import evolution.monitor.continuous_evolution as ce

    monkeypatch.setattr(ce, "check_phase5_readiness", lambda **kw: _ready_report())
    monkeypatch.setattr(sys, "argv", ["continuous_evolution", "--status"])

    with pytest.raises(SystemExit) as excinfo:
        ce.main()

    assert excinfo.value.code == 0
    assert '"ready": true' in capsys.readouterr().out


def test_cli_live_cycle_refused_when_not_ready(monkeypatch, capsys):
    """A live --cycle must never construct the engine when the gate fails."""
    import evolution.monitor.continuous_evolution as ce

    def forbidden(*args, **kwargs):
        raise AssertionError("ContinuousEvolution must not be constructed when gate fails")

    monkeypatch.setattr(ce, "check_phase5_readiness", lambda **kw: _not_ready_report())
    monkeypatch.setattr(ce, "ContinuousEvolution", forbidden)
    monkeypatch.setattr(sys, "argv", ["continuous_evolution", "--cycle"])

    with pytest.raises(SystemExit) as excinfo:
        ce.main()

    assert excinfo.value.code == 1
    assert "Refusing to run a live cycle" in capsys.readouterr().out


class _StubEngine:
    def __init__(self, **kwargs):
        pass

    def run_cycle(self, dry_run: bool = False):
        return {"dry_run": dry_run}


def test_cli_dry_run_cycle_bypasses_gate(monkeypatch):
    import evolution.monitor.continuous_evolution as ce

    def forbidden(**kw):
        raise AssertionError("readiness gate must not run for --dry-run cycles")

    monkeypatch.setattr(ce, "check_phase5_readiness", forbidden)
    monkeypatch.setattr(ce, "ContinuousEvolution", _StubEngine)
    monkeypatch.setattr(sys, "argv", ["continuous_evolution", "--cycle", "--dry-run"])

    ce.main()  # must not raise


def test_cli_skip_readiness_check_overrides_gate(monkeypatch):
    import evolution.monitor.continuous_evolution as ce

    monkeypatch.setattr(ce, "check_phase5_readiness", lambda **kw: _not_ready_report())
    monkeypatch.setattr(ce, "ContinuousEvolution", _StubEngine)
    monkeypatch.setattr(sys, "argv", ["continuous_evolution", "--cycle", "--skip-readiness-check"])

    ce.main()  # must not raise
