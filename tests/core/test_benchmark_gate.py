"""Regression tests for benchmark gate safety semantics."""

from __future__ import annotations

from datetime import datetime

from evolution.core.benchmark_gate import BenchmarkGate, BenchmarkResult


def _result_with_error() -> BenchmarkResult:
    return BenchmarkResult(
        name="tblite-fast",
        score=0.0,
        total_tasks=20,
        passed_tasks=0,
        failed_tasks=20,
        elapsed_seconds=0.1,
        timestamp=datetime.now().isoformat(),
        error="Benchmark runner not found",
    )


def test_benchmark_gate_fails_closed_on_errors_by_default(tmp_path):
    gate = BenchmarkGate(hermes_agent_path=tmp_path, baseline_file=tmp_path / "baselines.json")

    result = gate.check_gate([_result_with_error()])

    assert not result.passed
    assert "benchmark error" in result.regressions[0]


def test_benchmark_gate_can_explicitly_skip_errors_for_diagnostics(tmp_path):
    gate = BenchmarkGate(hermes_agent_path=tmp_path, baseline_file=tmp_path / "baselines.json")

    result = gate.check_gate([_result_with_error()], fail_on_error=False)

    assert result.passed
    assert result.regressions == []


def test_benchmark_gate_fails_closed_on_missing_baseline(tmp_path):
    gate = BenchmarkGate(hermes_agent_path=tmp_path, baseline_file=tmp_path / "baselines.json")
    result = BenchmarkResult(
        name="tblite-fast",
        score=0.9,
        total_tasks=10,
        passed_tasks=9,
        failed_tasks=1,
        elapsed_seconds=0.1,
        timestamp=datetime.now().isoformat(),
    )

    gate_result = gate.check_gate([result])

    assert not gate_result.passed
    assert "no stored baseline" in gate_result.regressions[0]


def test_benchmark_gate_missing_baseline_escape_hatch_is_explicit(tmp_path):
    gate = BenchmarkGate(hermes_agent_path=tmp_path, baseline_file=tmp_path / "baselines.json")
    result = BenchmarkResult(
        name="tblite-fast",
        score=0.9,
        total_tasks=10,
        passed_tasks=9,
        failed_tasks=1,
        elapsed_seconds=0.1,
        timestamp=datetime.now().isoformat(),
    )

    gate_result = gate.check_gate([result], require_baseline=False)

    assert gate_result.passed
    assert gate_result.regressions == []
