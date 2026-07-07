"""Safety regression tests for Phase 5 continuous evolution."""

from __future__ import annotations

from pathlib import Path

from evolution.monitor.continuous_evolution import (
    AutoTriage,
    BenchmarkTrend,
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
