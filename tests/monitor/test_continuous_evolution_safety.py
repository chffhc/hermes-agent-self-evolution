"""Safety regression tests for Phase 5 continuous evolution."""

from __future__ import annotations

import json
from pathlib import Path

from evolution.core.errors import EvolutionError
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


def _write_run_metrics(output_dir: Path, **metrics) -> Path:
    run_dir = output_dir / "20260709_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def _empty_result() -> dict:
    return {"success": False, "improvement": 0.0, "output_dir": "", "error": None}


def test_constraint_violating_run_is_not_counted_as_success(tmp_path: Path):
    _write_run_metrics(tmp_path / "out", improvement=0.5, deployable=False)
    result = _empty_result()

    ContinuousEvolution._read_latest_run_metrics(tmp_path / "out", result)

    assert result["improvement"] == 0.5
    assert not result["success"]


def test_deployable_improving_run_counts_as_success(tmp_path: Path):
    _write_run_metrics(tmp_path / "out", improvement=0.5, deployable=True)
    result = _empty_result()

    ContinuousEvolution._read_latest_run_metrics(tmp_path / "out", result)

    assert result["success"]


def test_legacy_metrics_without_deployable_flag_still_count(tmp_path: Path):
    _write_run_metrics(tmp_path / "out", improvement=0.5)
    result = _empty_result()

    ContinuousEvolution._read_latest_run_metrics(tmp_path / "out", result)

    assert result["success"]


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
