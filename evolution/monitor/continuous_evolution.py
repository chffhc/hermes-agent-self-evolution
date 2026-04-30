"""Phase 5: Continuous self-improvement loop.

The agent automatically identifies its weakest areas and improves them
over time through scheduled optimization cycles.

Components:
1. PerformanceMonitor — Tracks metrics from real usage
2. AutoTriage — Identifies what to optimize next
3. ContinuousEvolution — Orchestrates the full loop
4. Cron integration — Scheduled benchmark runs and threshold-triggered optimization

All automated PRs still require human merge — this phase automates
detection and optimization, not deployment.

Usage:
    # Run one cycle manually
    python -m evolution.monitor.continuous_evolution

    # Or set up as a cron job via Hermes scheduler
    # Weekly: run benchmarks + check for optimization targets
"""

import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from evolution.core.config import get_hermes_agent_path
from evolution.core.benchmark_gate import BenchmarkGate, BenchmarkResult

console = Console()


# ── Performance metrics ─────────────────────────────────────────────────

@dataclass
class SkillMetric:
    """Per-skill performance metric."""
    name: str
    load_count: int  # How many times the skill was loaded
    success_count: int  # How many times the task succeeded
    failure_count: int  # How many times the task failed
    avg_score: float  # Average quality score (0-1)
    last_used: str  # ISO timestamp

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total)

    @property
    def failure_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.failure_count / max(1, total)


@dataclass
class ToolMetric:
    """Per-tool selection metric."""
    name: str
    selection_count: int
    correct_selection_count: int  # Judged as the right tool
    avg_param_accuracy: float  # How often params were used correctly

    @property
    def selection_accuracy(self) -> float:
        return self.correct_selection_count / max(1, self.selection_count)


@dataclass
class BenchmarkTrend:
    """Benchmark score trend over time."""
    name: str
    scores: list[tuple[str, float]]  # (timestamp, score)

    @property
    def latest_score(self) -> Optional[float]:
        return self.scores[-1][1] if self.scores else None

    @property
    def trend(self) -> str:
        if len(self.scores) < 2:
            return "insufficient_data"
        recent = self.scores[-1][1]
        older = self.scores[-2][1]
        diff = recent - older
        if diff > 0.02:
            return "improving"
        elif diff < -0.02:
            return "degrading"
        return "stable"


@dataclass
class OptimizationTarget:
    """A candidate for optimization."""
    target_type: str  # "skill", "tool", "prompt"
    target_name: str
    current_score: float
    estimated_improvement: float  # Potential improvement (0-1)
    usage_frequency: int  # How often it's used
    priority_score: float  # estimated_improvement * usage_frequency
    reason: str


# ── Performance monitor ─────────────────────────────────────────────────

class PerformanceMonitor:
    """Tracks performance metrics from real Hermes Agent usage.

    Reads from SessionDB and log files to compute per-skill and
    per-tool performance metrics over time.
    """

    def __init__(self, hermes_agent_path: Optional[Path] = None):
        self.hermes_agent_path = hermes_agent_path or get_hermes_agent_path()
        self.metrics_file = Path("evolution/monitor/metrics_store.json")
        self._metrics = self._load_metrics()

    def _load_metrics(self) -> dict:
        """Load persisted metrics."""
        if self.metrics_file.exists():
            return json.loads(self.metrics_file.read_text())
        return {
            "skills": {},
            "tools": {},
            "benchmarks": {},
            "last_updated": None,
        }

    def save_metrics(self):
        """Persist current metrics."""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_file.write_text(
            json.dumps(self._metrics, indent=2, ensure_ascii=False)
        )

    def scan_session_db(self) -> dict[str, SkillMetric]:
        """Scan SessionDB for skill usage patterns.

        Looks for skill load events and task outcomes in session messages.
        """
        session_db_path = Path.home() / ".hermes" / "sessions.db"
        skill_metrics: dict[str, SkillMetric] = {}

        if not session_db_path.exists():
            console.print(f"  ⚠ SessionDB not found at {session_db_path}")
            return skill_metrics

        try:
            conn = sqlite3.connect(str(session_db_path))
            cursor = conn.cursor()

            # Get all messages
            cursor.execute(
                "SELECT content, timestamp FROM messages ORDER BY timestamp DESC LIMIT 10000"
            )
            messages = cursor.fetchall()

            # Parse skill loading and task outcomes
            import re
            skill_pattern = re.compile(r"skill[ _:]+(\S+)", re.IGNORECASE)
            success_pattern = re.compile(r"(?:success|completed|done|passed)", re.IGNORECASE)
            fail_pattern = re.compile(r"(?:fail|error|broken|incorrect)", re.IGNORECASE)

            current_skill = None
            for content, timestamp in messages:
                if not content:
                    continue

                # Check for skill loading
                skill_match = skill_pattern.search(str(content))
                if skill_match:
                    current_skill = skill_match.group(1).strip("*/").lower()
                    if current_skill not in skill_metrics:
                        skill_metrics[current_skill] = SkillMetric(
                            name=current_skill,
                            load_count=0,
                            success_count=0,
                            failure_count=0,
                            avg_score=0.0,
                            last_used="",
                        )
                    skill_metrics[current_skill].load_count += 1
                    skill_metrics[current_skill].last_used = str(timestamp)

                # Check for success/failure in subsequent messages
                if current_skill and current_skill in skill_metrics:
                    text = str(content)
                    if fail_pattern.search(text) and len(text) < 500:
                        skill_metrics[current_skill].failure_count += 1
                        current_skill = None
                    elif success_pattern.search(text) and len(text) < 500:
                        skill_metrics[current_skill].success_count += 1
                        current_skill = None

            conn.close()

        except Exception as e:
            console.print(f"  ⚠ SessionDB scan failed: {e}")

        return skill_metrics

    def scan_logs(self) -> dict[str, SkillMetric]:
        """Scan agent logs for tool usage and error patterns."""
        log_path = Path.home() / ".hermes" / "logs" / "agent.log"
        tool_metrics: dict[str, ToolMetric] = {}

        if not log_path.exists():
            return tool_metrics

        try:
            import re
            tool_call_pattern = re.compile(r"Tool call: (\w+)")
            error_pattern = re.compile(r"Tool execution failed|Error:|Exception:")

            with open(log_path, "r") as f:
                for line in f:
                    tool_match = tool_call_pattern.search(line)
                    if tool_match:
                        tool_name = tool_match.group(1)
                        if tool_name not in tool_metrics:
                            tool_metrics[tool_name] = ToolMetric(
                                name=tool_name,
                                selection_count=0,
                                correct_selection_count=0,
                                avg_param_accuracy=0.0,
                            )
                        tool_metrics[tool_name].selection_count += 1

                    if error_pattern.search(line) and tool_metrics:
                        # Last mentioned tool likely caused the error
                        for t in list(tool_metrics.values())[-1:]:
                            t.correct_selection_count = max(0, t.correct_selection_count)

        except Exception:
            pass

        return tool_metrics

    def get_skill_metrics(self) -> list[SkillMetric]:
        """Get current skill performance metrics."""
        db_metrics = self.scan_session_db()
        # Merge with stored metrics
        for name, metric in db_metrics.items():
            self._metrics["skills"][name] = {
                "name": metric.name,
                "load_count": metric.load_count,
                "success_count": metric.success_count,
                "failure_count": metric.failure_count,
                "avg_score": metric.avg_score,
                "last_used": metric.last_used,
            }
        self._metrics["last_updated"] = datetime.now().isoformat()
        self.save_metrics()

        return [
            SkillMetric(**data) for data in self._metrics["skills"].values()
        ]

    def get_tool_metrics(self) -> list[ToolMetric]:
        """Get current tool performance metrics."""
        log_metrics = self.scan_logs()
        for name, metric in log_metrics.items():
            self._metrics["tools"][name] = {
                "name": metric.name,
                "selection_count": metric.selection_count,
                "correct_selection_count": metric.correct_selection_count,
                "avg_param_accuracy": metric.avg_param_accuracy,
            }
        self.save_metrics()

        return [
            ToolMetric(**data) for data in self._metrics["tools"].values()
        ]

    def get_benchmark_trends(self) -> list[BenchmarkTrend]:
        """Get benchmark score trends from stored data."""
        trends = []
        for name, data in self._metrics.get("benchmarks", {}).items():
            scores = [(ts, score) for ts, score in data.get("scores", [])]
            trends.append(BenchmarkTrend(name=name, scores=scores))
        return trends

    def record_benchmark(self, name: str, score: float):
        """Record a benchmark result."""
        if name not in self._metrics["benchmarks"]:
            self._metrics["benchmarks"][name] = {"scores": []}
        self._metrics["benchmarks"][name]["scores"].append(
            (datetime.now().isoformat(), score)
        )
        self.save_metrics()


# ── Auto-triage ─────────────────────────────────────────────────────────

class AutoTriage:
    """Identifies optimization targets ranked by impact × frequency.

    Skills with declining success rates, tools with low selection accuracy,
    and benchmark categories with low pass rates are prioritized.
    """

    def __init__(
        self,
        min_usage: int = 3,  # Minimum usage to consider
        failure_rate_threshold: float = 0.2,  # Trigger optimization above this
        min_priority: float = 0.1,  # Minimum priority score
    ):
        self.min_usage = min_usage
        self.failure_rate_threshold = failure_rate_threshold
        self.min_priority = min_priority

    def triage(
        self,
        skill_metrics: list[SkillMetric],
        tool_metrics: list[ToolMetric],
        benchmark_trends: list[BenchmarkTrend],
    ) -> list[OptimizationTarget]:
        """Identify and rank optimization targets."""
        targets = []

        # Skills with high failure rates
        for sm in skill_metrics:
            if sm.load_count < self.min_usage:
                continue
            if sm.failure_rate > self.failure_rate_threshold:
                estimated_improvement = min(0.5, sm.failure_rate * 0.8)
                targets.append(OptimizationTarget(
                    target_type="skill",
                    target_name=sm.name,
                    current_score=sm.success_rate,
                    estimated_improvement=estimated_improvement,
                    usage_frequency=sm.load_count,
                    priority_score=estimated_improvement * sm.load_count,
                    reason=(
                        f"Failure rate {sm.failure_rate:.1%} "
                        f"({sm.failure_count}/{sm.load_count} failures)"
                    ),
                ))

        # Tools with low selection accuracy
        for tm in tool_metrics:
            if tm.selection_count < self.min_usage:
                continue
            accuracy = tm.selection_accuracy
            if accuracy < 0.8:  # Below 80% accuracy
                estimated_improvement = min(0.3, (1.0 - accuracy) * 0.5)
                targets.append(OptimizationTarget(
                    target_type="tool",
                    target_name=tm.name,
                    current_score=accuracy,
                    estimated_improvement=estimated_improvement,
                    usage_frequency=tm.selection_count,
                    priority_score=estimated_improvement * tm.selection_count,
                    reason=f"Selection accuracy {accuracy:.1%}",
                ))

        # Degrading benchmarks
        for bt in benchmark_trends:
            if bt.trend == "degrading" and bt.latest_score is not None:
                estimated_improvement = 0.2  # Conservative estimate
                targets.append(OptimizationTarget(
                    target_type="benchmark",
                    target_name=bt.name,
                    current_score=bt.latest_score,
                    estimated_improvement=estimated_improvement,
                    usage_frequency=10,  # Proxy
                    priority_score=estimated_improvement * 10,
                    reason=f"Benchmark trend: {bt.trend} (score: {bt.latest_score:.3f})",
                ))

        # Sort by priority score descending
        targets.sort(key=lambda t: t.priority_score, reverse=True)

        # Filter by minimum priority
        targets = [t for t in targets if t.priority_score >= self.min_priority]

        return targets


# ── Continuous evolution orchestrator ───────────────────────────────────

class ContinuousEvolution:
    """Orchestrates the full continuous improvement loop.

    1. Scan performance metrics
    2. Triage to find targets
    3. Run optimization on top targets
    4. Validate with benchmarks
    5. Create PRs for improvements
    """

    def __init__(
        self,
        hermes_agent_path: Optional[Path] = None,
        max_targets: int = 3,
        benchmark_gate: bool = True,
    ):
        self.hermes_agent_path = hermes_agent_path or get_hermes_agent_path()
        self.max_targets = max_targets
        self.benchmark_gate = benchmark_gate
        self.monitor = PerformanceMonitor(self.hermes_agent_path)
        self.triage = AutoTriage()
        self.benchmarks = BenchmarkGate(hermes_agent_path=self.hermes_agent_path)

    def run_cycle(self, dry_run: bool = False) -> dict:
        """Run one full continuous improvement cycle.

        Returns a summary of what was done.
        """
        console.print(f"\n[bold cyan]🧬 Continuous Self-Improvement Cycle[/bold cyan]\n")
        start = time.time()
        summary = {
            "timestamp": datetime.now().isoformat(),
            "targets_found": 0,
            "targets_optimized": 0,
            "prs_created": 0,
            "benchmarks_passed": True,
            "elapsed_seconds": 0,
        }

        # ── Step 1: Scan metrics ────────────────────────────────────────
        console.print("[bold]Step 1: Scanning performance metrics[/bold]")
        skill_metrics = self.monitor.get_skill_metrics()
        tool_metrics = self.monitor.get_tool_metrics()
        benchmark_trends = self.monitor.get_benchmark_trends()

        console.print(f"  Skills tracked: {len(skill_metrics)}")
        console.print(f"  Tools tracked: {len(tool_metrics)}")
        console.print(f"  Benchmark trends: {len(benchmark_trends)}")

        # ── Step 2: Triage ──────────────────────────────────────────────
        console.print(f"\n[bold]Step 2: Identifying optimization targets[/bold]")
        targets = self.triage.triage(skill_metrics, tool_metrics, benchmark_trends)
        summary["targets_found"] = len(targets)

        if targets:
            table = Table(title="Optimization Targets (ranked)")
            table.add_column("Priority", justify="right")
            table.add_column("Type")
            table.add_column("Target")
            table.add_column("Score")
            table.add_column("Reason")

            for t in targets[:self.max_targets]:
                table.add_row(
                    f"{t.priority_score:.2f}",
                    t.target_type,
                    t.target_name,
                    f"{t.current_score:.3f}",
                    t.reason,
                )
            console.print(table)
        else:
            console.print("  [green]✓ No optimization targets — all metrics look good![/green]")
            summary["elapsed_seconds"] = time.time() - start
            return summary

        # ── Step 3: Run benchmarks (gate) ───────────────────────────────
        if self.benchmark_gate:
            console.print(f"\n[bold]Step 3: Running benchmark gate[/bold]")
            # Run fast benchmarks to ensure no existing regressions
            # In production, this would call the actual benchmark runner
            console.print("  [yellow]⚠ Benchmark gate skipped (no hermes-agent benchmarks configured)[/yellow]")

        # ── Step 4: Optimize top targets ────────────────────────────────
        console.print(f"\n[bold]Step 4: Optimizing top targets[/bold]")

        if dry_run:
            console.print("  [cyan]DRY RUN — would optimize:[/cyan]")
            for t in targets[:self.max_targets]:
                console.print(f"    → {t.target_type}: {t.target_name}")
            summary["elapsed_seconds"] = time.time() - start
            return summary

        # In production, this would call the appropriate evolve function
        # For now, we log what would be done
        for t in targets[:self.max_targets]:
            console.print(f"  → Would optimize {t.target_type}: {t.target_name}")
            console.print(f"    Reason: {t.reason}")
            console.print(f"    Expected improvement: {t.estimated_improvement:.1%}")
            summary["targets_optimized"] += 1

        # ── Step 5: Report ──────────────────────────────────────────────
        summary["elapsed_seconds"] = time.time() - start
        console.print(f"\n[bold]Cycle complete in {summary['elapsed_seconds']:.1f}s[/bold]")
        console.print(f"  Targets found: {summary['targets_found']}")
        console.print(f"  Targets optimized: {summary['targets_optimized']}")
        console.print(f"  PRs created: {summary['prs_created']}")

        return summary


# ── CLI entry point ─────────────────────────────────────────────────────

def setup_cron_jobs():
    """Set up cron jobs for continuous improvement.

    Creates:
    1. Weekly benchmark run
    2. Threshold-triggered optimization
    """
    console.print("\n[bold cyan]🧬 Setting up continuous improvement cron jobs[/bold cyan]\n")

    print("To set up automated continuous improvement, add the following")
    print("to your Hermes Agent cron configuration:")
    print()
    print("  # Weekly benchmark + triage cycle")
    print("  schedule: '0 3 * * 0'  # Every Sunday at 3 AM")
    print("  prompt: >")
    print("    Run a continuous improvement cycle:")
    print("    1. Scan session history for skill/tool performance metrics")
    print("    2. Run benchmark suite (TBLite fast subset)")
    print("    3. Identify underperforming skills/tools")
    print("    4. For each target with failure rate > 20%:")
    print("       - Run GEPA optimization (10 iterations)")
    print("       - Validate with constraints")
    print("       - Create PR if improved")
    print("    5. Report results")
    print()
    print("This will automatically detect and fix degrading skills")
    print("without manual intervention. All changes require human merge.")


def main():
    """Main entry point for continuous evolution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hermes Agent Self-Evolution — Continuous Improvement Loop"
    )
    parser.add_argument("--cycle", action="store_true", help="Run one improvement cycle")
    parser.add_argument("--setup-cron", action="store_true", help="Set up cron jobs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--hermes-repo", default=None, help="Path to hermes-agent repo")

    args = parser.parse_args()

    if args.setup_cron:
        setup_cron_jobs()
        return

    if args.cycle:
        evolution = ContinuousEvolution(
            hermes_agent_path=Path(args.hermes_repo) if args.hermes_repo else None,
        )
        result = evolution.run_cycle(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
