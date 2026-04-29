"""Benchmark gating for evolved artifacts.

Runs TBLite and YC-Bench to establish baselines and detect regressions
after skill evolution. Every evolved variant must not regress benchmarks
beyond the configured threshold (default: 2%).

Usage:
    python -m evolution.core.benchmark_gate --hermes-repo /path/to/hermes-agent
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    score: float
    passed: bool
    details: str = ""
    duration_seconds: float = 0.0


@dataclass
class BenchmarkGateResult:
    """Overall benchmark gate result."""
    results: list[BenchmarkResult] = field(default_factory=list)
    regression_detected: bool = False
    max_regression: float = 0.0
    summary: str = ""

    @property
    def passed(self) -> bool:
        return not self.regression_detected and all(r.passed for r in self.results)


def run_tblite(hermes_repo: Path, timeout: int = 7200) -> BenchmarkResult:
    """Run TBLite benchmark suite.

    TBLite is a fast agent capability test (~1-2 hours).
    Returns a composite score (0-1).
    """
    console.print(f"  Running TBLite benchmark (timeout: {timeout}s)...")
    start = time.time()

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "environments/benchmarks/tblite/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(hermes_repo),
        )
        duration = time.time() - start

        # Parse pytest output for pass/fail counts
        stdout = result.stdout.strip()
        lines = stdout.split("\n")
        summary_line = lines[-1] if lines else ""

        # Extract scores from output like "15 passed, 5 failed"
        import re
        passed_match = re.search(r'(\d+)\s*passed', summary_line)
        failed_match = re.search(r'(\d+)\s*failed', summary_line)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        total = passed + failed
        score = passed / total if total > 0 else 0.0

        return BenchmarkResult(
            name="tblite",
            score=score,
            passed=result.returncode == 0,
            details=summary_line,
            duration_seconds=duration,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            name="tblite",
            score=0.0,
            passed=False,
            details=f"Benchmark timed out after {timeout}s",
            duration_seconds=float(timeout),
        )
    except Exception as e:
        return BenchmarkResult(
            name="tblite",
            score=0.0,
            passed=False,
            details=f"Failed to run: {e}",
        )


def run_yc_bench_fast(hermes_repo: Path, timeout: int = 3600) -> BenchmarkResult:
    """Run YC-Bench fast_test preset.

    YC-Bench tests long-term agent coherence (~50 turns, composite score).
    """
    console.print(f"  Running YC-Bench fast_test (timeout: {timeout}s)...")
    start = time.time()

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "environments/benchmarks/yc_bench/",
             "-v", "--tb=short", "-q", "-k", "fast_test"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(hermes_repo),
        )
        duration = time.time() - start

        stdout = result.stdout.strip()
        lines = stdout.split("\n")
        summary_line = lines[-1] if lines else ""

        import re
        passed_match = re.search(r'(\d+)\s*passed', summary_line)
        failed_match = re.search(r'(\d+)\s*failed', summary_line)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        total = passed + failed
        score = passed / total if total > 0 else 0.0

        return BenchmarkResult(
            name="yc_bench_fast",
            score=score,
            passed=result.returncode == 0,
            details=summary_line,
            duration_seconds=duration,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            name="yc_bench_fast",
            score=0.0,
            passed=False,
            details=f"Benchmark timed out after {timeout}s",
            duration_seconds=float(timeout),
        )
    except Exception as e:
        return BenchmarkResult(
            name="yc_bench_fast",
            score=0.0,
            passed=False,
            details=f"Failed to run: {e}",
        )


def check_regression(baseline_score: float, evolved_score: float, threshold: float = 0.02) -> tuple[bool, float]:
    """Check if evolved score regressed beyond threshold.

    Returns (has_regression, regression_amount).
    Regression is negative (score dropped).
    """
    regression = evolved_score - baseline_score
    has_regression = regression < -threshold
    return has_regression, regression


def run_benchmark_gate(
    hermes_repo: Path,
    baseline_scores: Optional[dict[str, float]] = None,
    run_tblite: bool = True,
    run_yc_bench: bool = False,
    tblite_threshold: float = 0.02,
    timeout_tblite: int = 7200,
    timeout_yc: int = 3600,
) -> BenchmarkGateResult:
    """Run benchmark suite and check for regressions.

    Args:
        hermes_repo: Path to hermes-agent repository.
        baseline_scores: Previous benchmark scores to compare against.
        run_tblite: Whether to run TBLite.
        run_yc_bench: Whether to run YC-Bench.
        tblite_threshold: Max allowed regression (0.02 = 2%).
        timeout_tblite: Timeout for TBLite in seconds.
        timeout_yc: Timeout for YC-Bench in seconds.

    Returns:
        BenchmarkGateResult with all results and regression analysis.
    """
    gate_result = BenchmarkGateResult()
    scores: dict[str, float] = {}

    if run_tblite:
        tblite_result = run_tblite(hermes_repo, timeout=timeout_tblite)
        gate_result.results.append(tblite_result)
        scores["tblite"] = tblite_result.score

        if baseline_scores and "tblite" in baseline_scores:
            has_reg, reg_amount = check_regression(
                baseline_scores["tblite"], tblite_result.score, tblite_threshold
            )
            if has_reg:
                gate_result.regression_detected = True
                gate_result.max_regression = min(gate_result.max_regression, reg_amount)
                gate_result.summary += f"  ❌ TBLite regression: {reg_amount:+.1%}\n"
            else:
                gate_result.summary += f"  ✓ TBLite OK: {tblite_result.score:.3f} ({reg_amount:+.1%})\n"

    if run_yc_bench:
        yc_result = run_yc_bench_fast(hermes_repo, timeout=timeout_yc)
        gate_result.results.append(yc_result)
        scores["yc_bench_fast"] = yc_result.score

        if baseline_scores and "yc_bench_fast" in baseline_scores:
            yc_threshold = tblite_threshold  # Same default threshold
            has_reg, reg_amount = check_regression(
                baseline_scores["yc_bench_fast"], yc_result.score, yc_threshold
            )
            if has_reg:
                gate_result.regression_detected = True
                gate_result.max_regression = min(gate_result.max_regression, reg_amount)
                gate_result.summary += f"  ❌ YC-Bench regression: {reg_amount:+.1%}\n"
            else:
                gate_result.summary += f"  ✓ YC-Bench OK: {yc_result.score:.3f} ({reg_amount:+.1%})\n"

    if not gate_result.summary:
        gate_result.summary = "  All benchmarks passed (no baseline comparison available)\n"

    # Save scores to file
    output_dir = Path("./output/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scores_file = output_dir / f"benchmark_scores_{timestamp}.json"
    scores_file.write_text(json.dumps({
        "timestamp": timestamp,
        "scores": scores,
        "regression_detected": gate_result.regression_detected,
        "results": [
            {"name": r.name, "score": r.score, "passed": r.passed, "details": r.details}
            for r in gate_result.results
        ],
    }, indent=2))

    return gate_result


@click.command()
@click.option("--hermes-repo", required=True, help="Path to hermes-agent repo")
@click.option("--baseline", default=None, help="Path to baseline scores JSON file")
@click.option("--run-tblite/--no-tblite", default=True, help="Run TBLite benchmark")
@click.option("--run-yc-bench/--no-yc-bench", default=False, help="Run YC-Bench fast_test")
@click.option("--tblite-threshold", default=0.02, help="Max TBLite regression allowed (default: 0.02)")
def main(hermes_repo, baseline, run_tblite, run_yc_bench, tblite_threshold):
    """Run benchmark gate and check for regressions."""
    hermes_path = Path(hermes_repo)
    baseline_scores = None

    if baseline:
        baseline_path = Path(baseline)
        if baseline_path.exists():
            data = json.loads(baseline_path.read_text())
            baseline_scores = data.get("scores", {})
            console.print(f"  Loaded baseline scores from {baseline_path}")
        else:
            console.print(f"[red]✗ Baseline file not found: {baseline}[/red]")
            return

    console.print(f"\n[bold cyan]Running benchmark gate...[/bold cyan]\n")
    console.print(f"  Hermes repo: {hermes_path}")
    console.print(f"  Run TBLite: {run_tblite}")
    console.print(f"  Run YC-Bench: {run_yc_bench}")
    console.print(f"  Regression threshold: {tblite_threshold:.1%}\n")

    result = run_benchmark_gate(
        hermes_repo=hermes_path,
        baseline_scores=baseline_scores,
        run_tblite=run_tblite,
        run_yc_bench=run_yc_bench,
        tblite_threshold=tblite_threshold,
    )

    console.print(f"\n[bold]Benchmark Results:[/bold]\n")
    console.print(result.summary)

    if result.regression_detected:
        console.print(f"\n[bold red]❌ BENCHMARK GATE FAILED — regression detected[/bold red]")
    else:
        console.print(f"\n[bold green]✓ BENCHMARK GATE PASSED[/bold green]")


if __name__ == "__main__":
    main()
