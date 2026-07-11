"""Safety regression tests for Phase 5 continuous evolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.core.errors import BudgetExceededError, EvolutionError
from evolution.monitor.continuous_evolution import (
    AutoTriage,
    BenchmarkTrend,
    ContinuousEvolution,
    OptimizationTarget,
    PerformanceMonitor,
    ToolMetric,
)


def test_auto_triage_does_not_treat_missing_tool_correctness_as_zero_accuracy():
    triage = AutoTriage(min_usage=3)

    targets = triage.triage(
        skill_metrics=[],
        tool_metrics=[
            ToolMetric(
                name="terminal",
                selection_count=100,
                correct_selection_count=0,
                avg_param_accuracy=0.0,
            )
        ],
        benchmark_trends=[],
    )

    assert targets == []


def test_auto_triage_only_emits_supported_target_types():
    triage = AutoTriage(min_usage=3)

    targets = triage.triage(
        skill_metrics=[],
        tool_metrics=[],
        benchmark_trends=[
            BenchmarkTrend(
                name="tblite-fast",
                scores=[("2026-01-01T00:00:00", 0.9), ("2026-01-02T00:00:00", 0.5)],
            )
        ],
    )

    assert targets == []


def test_monitor_state_defaults_to_ignored_output_directory(tmp_path: Path, monkeypatch):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)

    monitor = PerformanceMonitor(repo)

    assert monitor.metrics_file == Path("output/monitor/metrics_store.json")


def _empty_result() -> dict:
    return {"success": False, "improvement": 0.0, "output_dir": "", "error": None}


def test_constraint_violating_run_is_not_counted_as_success():
    result = _empty_result()

    ContinuousEvolution._consume_run_metrics({"improvement": 0.5, "deployable": False}, result)

    assert result["improvement"] == 0.5
    assert not result["success"]


def test_deployable_improving_run_counts_as_success():
    result = _empty_result()

    ContinuousEvolution._consume_run_metrics(
        {"improvement": 0.5, "deployable": True, "output_dir": "output/demo/1"}, result
    )

    assert result["success"]
    assert result["output_dir"] == "output/demo/1"


def test_legacy_metrics_without_deployable_flag_still_count():
    result = _empty_result()

    ContinuousEvolution._consume_run_metrics({"improvement": 0.5}, result)

    assert result["success"]


def test_run_without_metrics_is_not_counted_as_success():
    result = _empty_result()

    ContinuousEvolution._consume_run_metrics(None, result)

    assert not result["success"]
    assert "no metrics" in result["error"]


def test_failed_run_error_is_surfaced():
    result = _empty_result()

    ContinuousEvolution._consume_run_metrics(
        {"improvement": 0.0, "deployable": False, "error": "pytest gate failed"}, result
    )

    assert not result["success"]
    assert result["error"] == "pytest gate failed"


def test_optimize_target_ignores_stale_output_directories(tmp_path: Path, monkeypatch):
    """A stale metrics.json from a previous/concurrent run must not be
    attributed to a run that returned no metrics of its own."""
    monkeypatch.chdir(tmp_path)
    stale_run = tmp_path / "output" / "demo" / "20260101_000000"
    stale_run.mkdir(parents=True)
    (stale_run / "metrics.json").write_text(json.dumps({"improvement": 0.9, "deployable": True}))

    engine = ContinuousEvolution(hermes_agent_path=tmp_path, benchmark_gate=False, resume=False)

    import evolution.skills.evolve_skill as evolve_skill_mod

    monkeypatch.setattr(evolve_skill_mod, "evolve", lambda **kwargs: None)

    result = engine._optimize_target(_skill_target("demo"))

    assert not result["success"]
    assert result["output_dir"] == ""


def test_optimize_target_consumes_returned_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = ContinuousEvolution(hermes_agent_path=tmp_path, benchmark_gate=False, resume=False)

    import evolution.skills.evolve_skill as evolve_skill_mod

    monkeypatch.setattr(
        evolve_skill_mod,
        "evolve",
        lambda **kwargs: {
            "improvement": 0.2,
            "deployable": True,
            "output_dir": "output/demo/20260709_120000",
        },
    )

    result = engine._optimize_target(_skill_target("demo"))

    assert result["success"]
    assert result["improvement"] == 0.2
    assert result["output_dir"] == "output/demo/20260709_120000"


def _skill_target(name: str) -> OptimizationTarget:
    return OptimizationTarget(
        target_type="skill",
        target_name=name,
        current_score=0.2,
        estimated_improvement=0.5,
        usage_frequency=10,
        priority_score=5.0,
        reason="test",
    )


def test_optimize_target_propagates_budget_exceeded(tmp_path: Path, monkeypatch):
    """A hard budget abort must escape the per-target error handling —
    swallowing it would let the cycle keep spending on remaining targets."""
    monkeypatch.chdir(tmp_path)
    engine = ContinuousEvolution(hermes_agent_path=tmp_path, benchmark_gate=False, resume=False)

    import evolution.skills.evolve_skill as evolve_skill_mod

    def boom(**kwargs):
        raise BudgetExceededError("estimated cost $5.00 exceeds budget $1.00")

    monkeypatch.setattr(evolve_skill_mod, "evolve", boom)

    with pytest.raises(BudgetExceededError):
        engine._optimize_target(_skill_target("demo"))


def test_run_cycle_aborts_on_budget_exceeded(tmp_path: Path, monkeypatch):
    """Once the budget blows, the cycle stops: no further targets are
    optimized, the summary says why, and a checkpoint remains so a resume
    with a raised budget re-runs the aborted target."""
    monkeypatch.chdir(tmp_path)
    engine = ContinuousEvolution(hermes_agent_path=tmp_path, benchmark_gate=False, resume=False)

    monkeypatch.setattr(engine.monitor, "get_skill_metrics", lambda: [])
    monkeypatch.setattr(engine.monitor, "get_tool_metrics", lambda: [])
    monkeypatch.setattr(engine.monitor, "get_benchmark_trends", lambda: [])
    monkeypatch.setattr(
        engine.triage,
        "triage",
        lambda *args, **kwargs: [_skill_target("first"), _skill_target("second")],
    )

    import evolution.skills.evolve_skill as evolve_skill_mod

    calls = []

    def boom(**kwargs):
        calls.append(kwargs["skill_name"])
        raise BudgetExceededError("estimated cost $5.00 exceeds budget $1.00")

    monkeypatch.setattr(evolve_skill_mod, "evolve", boom)

    summary = engine.run_cycle()

    assert summary["budget_exceeded"] is True
    assert calls == ["first"]  # the second target was never attempted
    assert summary["targets_optimized"] == 0
    checkpoint = json.loads(engine.checkpoint_file.read_text())
    remaining = [t["name"] for t in checkpoint["remaining_targets"]]
    assert remaining == ["first", "second"]  # aborted target is re-run on resume


def test_optimize_target_survives_per_target_evolution_error(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = ContinuousEvolution(hermes_agent_path=tmp_path, benchmark_gate=False, resume=False)

    import evolution.skills.evolve_skill as evolve_skill_mod

    def boom(**kwargs):
        raise EvolutionError("no eval dataset available")

    monkeypatch.setattr(evolve_skill_mod, "evolve", boom)

    target = OptimizationTarget(
        target_type="skill",
        target_name="demo",
        current_score=0.2,
        estimated_improvement=0.5,
        usage_frequency=10,
        priority_score=5.0,
        reason="test",
    )

    result = engine._optimize_target(target)

    assert not result["success"]
    assert "no eval dataset available" in result["error"]
