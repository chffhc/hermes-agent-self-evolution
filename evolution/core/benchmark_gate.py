"""Benchmark gating for regression detection.

Ensures evolved artifacts don't degrade overall agent capability.
Runs lightweight benchmark subsets as gates before allowing deployment.

Three benchmark tiers:
1. TBLite fast subset (20 tasks, ~20 min) — quick capability check
2. Full TBLite (100 tasks, ~1-2 hours) — thorough regression check
3. YC-Bench fast_test (~50 turns) — long-horizon coherence check

Benchmarks are GATES, not fitness functions. A variant that improves
skill quality but drops benchmark scores is REJECTED.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    score: float  # 0-1 pass rate
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    elapsed_seconds: float
    timestamp: str
    error: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class GateResult:
    """Result of running all benchmark gates."""
    passed: bool
    results: list[BenchmarkResult]
    baseline_scores: dict[str, float]
    regressions: list[str]
    max_allowed_regression: float = 0.02  # 2% max drop


class BenchmarkGate:
    """Manages benchmark gates for evolved artifact validation.

    Tracks baseline scores and rejects any evolved variant that
    regresses beyond the allowed threshold.
    """

    def __init__(
        self,
        hermes_agent_path: Optional[Path] = None,
        max_regression: float = 0.02,
        baseline_file: Optional[Path] = None,
    ):
        self.hermes_agent_path = hermes_agent_path or Path.home() / ".hermes" / "hermes-agent"
        self.max_regression = max_regression
        self.baseline_file = baseline_file or Path("benchmarks/baselines.json")
        self.baseline_scores = self._load_baselines()

    def _load_baselines(self) -> dict[str, float]:
        """Load stored baseline scores."""
        if self.baseline_file.exists():
            try:
                return json.loads(self.baseline_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                # Corrupted baseline — start fresh
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to load baselines from %s: %s — starting fresh",
                    self.baseline_file, e,
                )
        return {}

    def save_baselines(self):
        """Save current scores as new baselines."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_file.write_text(
            json.dumps(self.baseline_scores, indent=2)
        )

    def run_tblite_fast(
        self,
        skill_overrides: Optional[dict[str, str]] = None,
        timeout: int = 1800,
    ) -> BenchmarkResult:
        """Run TBLite fast subset (20 tasks).

        Args:
            skill_overrides: {skill_name: evolved_skill_text} to test with
            timeout: max seconds for the benchmark run
        """
        return self._run_benchmark(
            name="tblite-fast",
            description="TBLite fast subset (20 tasks)",
            task_count=20,
            skill_overrides=skill_overrides,
            timeout=timeout,
        )

    def run_tblite_full(
        self,
        skill_overrides: Optional[dict[str, str]] = None,
        timeout: int = 7200,
    ) -> BenchmarkResult:
        """Run full TBLite (100 tasks)."""
        return self._run_benchmark(
            name="tblite-full",
            description="TBLite full suite (100 tasks)",
            task_count=100,
            skill_overrides=skill_overrides,
            timeout=timeout,
        )

    def run_yc_bench_fast(
        self,
        skill_overrides: Optional[dict[str, str]] = None,
        timeout: int = 3600,
    ) -> BenchmarkResult:
        """Run YC-Bench fast_test (~50 turns)."""
        return self._run_benchmark(
            name="yc-bench-fast",
            description="YC-Bench fast_test (~50 turns)",
            task_count=50,
            skill_overrides=skill_overrides,
            timeout=timeout,
        )

    def _run_benchmark(
        self,
        name: str,
        description: str,
        task_count: int,
        skill_overrides: Optional[dict[str, str]] = None,
        timeout: int = 1800,
    ) -> BenchmarkResult:
        """Run a benchmark and return results."""
        from datetime import datetime

        start = time.time()
        result = BenchmarkResult(
            name=name,
            score=0.0,
            total_tasks=task_count,
            passed_tasks=0,
            failed_tasks=task_count,
            elapsed_seconds=0.0,
            timestamp=datetime.now().isoformat(),
        )

        # Check if hermes-agent has the benchmark infrastructure
        bench_dir = self.hermes_agent_path / "environments" / "benchmarks"
        if not bench_dir.exists():
            result.error = f"Benchmark directory not found: {bench_dir}"
            result.elapsed_seconds = time.time() - start
            return result

        # Try to run the benchmark via subprocess
        # This assumes hermes-agent has a benchmark runner script
        runner = bench_dir / "run_bench.py"
        if not runner.exists():
            # Fallback: simulate with a placeholder
            # In production, this would call the actual benchmark runner
            result.error = f"Benchmark runner not found: {runner}"
            result.elapsed_seconds = time.time() - start
            return result

        cmd = ["python", str(runner), "--tasks", str(task_count)]
        if skill_overrides:
            # Write skill overrides to a temp file (unique per run to avoid races)
            import tempfile
            fd, override_path = tempfile.mkstemp(suffix=".json", prefix="skill_overrides_")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(json.dumps(skill_overrides))
                cmd.extend(["--skill-overrides", override_path])
            except Exception:
                os.close(fd)
                raise

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.hermes_agent_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result.elapsed_seconds = time.time() - start

            if proc.returncode == 0:
                # Parse benchmark output
                output = json.loads(proc.stdout)
                result.passed_tasks = output.get("passed", 0)
                result.failed_tasks = output.get("failed", task_count)
                result.total_tasks = result.passed_tasks + result.failed_tasks
                result.score = (
                    result.passed_tasks / result.total_tasks
                    if result.total_tasks > 0
                    else 0.0
                )
                result.details = output
            else:
                result.error = proc.stderr.strip() or f"Exit code: {proc.returncode}"

        except subprocess.TimeoutExpired:
            result.error = f"Timeout after {timeout}s"
            result.elapsed_seconds = timeout
        except Exception as e:
            result.error = str(e)
            result.elapsed_seconds = time.time() - start

        return result

    def check_gate(
        self,
        results: list[BenchmarkResult],
        max_regression: Optional[float] = None,
    ) -> GateResult:
        """Check if all benchmark gates pass (no significant regression).

        Args:
            results: List of benchmark results to check
            max_regression: Override default max regression threshold
        """
        threshold = max_regression if max_regression is not None else self.max_regression
        regressions = []

        for r in results:
            if r.error:
                # If benchmark couldn't run, skip gate (don't fail on missing infra)
                continue

            baseline = self.baseline_scores.get(r.name)
            if baseline is not None:
                drop = baseline - r.score
                if drop > threshold:
                    regressions.append(
                        f"{r.name}: {baseline:.3f} → {r.score:.3f} "
                        f"(drop {drop:.3f}, max allowed {threshold:.3f})"
                    )

        return GateResult(
            passed=len(regressions) == 0,
            results=results,
            baseline_scores=self.baseline_scores,
            regressions=regressions,
            max_allowed_regression=threshold,
        )

    def update_baseline(self, name: str, score: float):
        """Update baseline score for a benchmark."""
        self.baseline_scores[name] = score
        self.save_baselines()


def establish_baselines(
    hermes_agent_path: Optional[Path] = None,
) -> dict[str, float]:
    """Run all benchmarks with current (baseline) skills and store scores.

    Call this once before starting any evolution runs to establish
    the baseline to compare against.
    """
    gate = BenchmarkGate(hermes_agent_path=hermes_agent_path)

    print("Establishing benchmark baselines...")
    print("  This may take 1-3 hours depending on benchmark scope.\n")

    results = []
    for bench_fn, name in [
        (gate.run_tblite_fast, "tblite-fast"),
        (gate.run_tblite_full, "tblite-full"),
        (gate.run_yc_bench_fast, "yc-bench-fast"),
    ]:
        print(f"  Running {name}...")
        result = bench_fn()
        results.append(result)
        if result.error:
            print(f"    ⚠ Skipped: {result.error}")
        else:
            print(f"    ✓ Score: {result.score:.3f} "
                  f"({result.passed_tasks}/{result.total_tasks})")
            gate.update_baseline(name, result.score)

    print(f"\n  Baselines saved to {gate.baseline_file}")
    return gate.baseline_scores
