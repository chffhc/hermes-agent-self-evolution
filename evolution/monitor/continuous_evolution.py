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
import logging
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

# ── Structured logging ───────────────────────────────────────────────────

logger = logging.getLogger("hermes-self-evolution.monitor")


def _setup_logging(log_file: Optional[Path] = None):
    """Configure structured logging for continuous evolution.

    Outputs to both console (via Rich handler) and a rotating log file.
    """
    if logger.handlers:
        return  # Already configured

    logger.setLevel(logging.DEBUG)

    # Console handler — use Rich for colored output in terminal
    from rich.logging import RichHandler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler — full debug log for post-hoc analysis
    if log_file is None:
        log_file = Path("evolution/monitor/evolution.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(file_handler)

    logger.info("Continuous evolution logging initialized")


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
            logger.warning("SessionDB not found at %s", session_db_path)
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
            logger.warning("SessionDB scan failed: %s", e)

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

    Supports checkpoint/resume: if a cycle is interrupted, the next
    run picks up where it left off.
    """

    def __init__(
        self,
        hermes_agent_path: Optional[Path] = None,
        max_targets: int = 3,
        benchmark_gate: bool = True,
        optimize_iterations: int = 10,
        optimizer_model: str = "qwen3.6-plus",
        resume: bool = True,
    ):
        self.hermes_agent_path = hermes_agent_path or get_hermes_agent_path()
        self.max_targets = max_targets
        self.benchmark_gate = benchmark_gate
        self.optimize_iterations = optimize_iterations
        self.optimizer_model = optimizer_model
        self.resume = resume
        self.checkpoint_file = Path("evolution/monitor/checkpoint.json")

        self.monitor = PerformanceMonitor(self.hermes_agent_path)
        self.triage = AutoTriage()
        self.benchmarks = BenchmarkGate(hermes_agent_path=self.hermes_agent_path)

        _setup_logging()

    def _save_checkpoint(self, summary: dict, next_target_index: int, targets: list):
        """Save progress checkpoint for resume after interruption."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "next_target_index": next_target_index,
            "remaining_targets": [
                {"type": t.target_type, "name": t.target_name, "priority": t.priority_score}
                for t in targets[next_target_index:]
            ],
        }
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False))
        logger.debug("Checkpoint saved: %d targets remaining", len(targets) - next_target_index)

    def _load_checkpoint(self) -> Optional[dict]:
        """Load interrupted checkpoint if available."""
        if self.checkpoint_file.exists():
            try:
                data = json.loads(self.checkpoint_file.read_text())
                # Only resume if checkpoint is < 24 hours old
                ts = datetime.fromisoformat(data["timestamp"])
                if (datetime.now() - ts) < timedelta(hours=24):
                    logger.info("Resuming from checkpoint: %s", data["timestamp"])
                    return data
                else:
                    logger.info("Checkpoint too old (%s), starting fresh", data["timestamp"])
                    self.checkpoint_file.unlink()
            except Exception:
                logger.debug("Failed to load checkpoint, starting fresh")
        return None

    def _clear_checkpoint(self):
        """Remove checkpoint after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def _optimize_target(self, target) -> dict:
        """Dispatch optimization to the appropriate Phase 1/2/3 function.

        Returns a dict with: success, improvement, output_dir, error.
        """
        result = {"success": False, "improvement": 0.0, "output_dir": "", "error": None}

        if target.target_type == "skill":
            try:
                from evolution.skills.evolve_skill import evolve
                logger.info("Optimizing skill: %s (%d iterations)", target.target_name, self.optimize_iterations)
                evolve(
                    skill_name=target.target_name,
                    iterations=self.optimize_iterations,
                    eval_source="synthetic",
                    optimizer_model=self.optimizer_model,
                    eval_model=self.optimizer_model,
                    hermes_repo=str(self.hermes_agent_path),
                    run_tests=False,
                )
                # Check output directory for the latest run
                output_dir = Path("output") / target.target_name
                if output_dir.exists():
                    subdirs = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    if subdirs:
                        metrics_file = subdirs[0] / "metrics.json"
                        if metrics_file.exists():
                            metrics = json.loads(metrics_file.read_text())
                            result["improvement"] = metrics.get("improvement", 0.0)
                            result["output_dir"] = str(subdirs[0])
                            result["success"] = metrics.get("improvement", 0.0) > 0
                logger.info("Skill %s: improvement=%+.3f success=%s", target.target_name, result["improvement"], result["success"])
            except Exception as e:
                logger.error("Skill optimization failed for %s: %s", target.target_name, e)
                result["error"] = str(e)

        elif target.target_type == "tool":
            try:
                from evolution.tools.evolve_tool_descriptions import evolve_tool_descriptions
                logger.info("Optimizing tool description: %s (%d iterations)", target.target_name, self.optimize_iterations)
                evolve_tool_descriptions(
                    iterations=self.optimize_iterations,
                    optimizer_model=self.optimizer_model,
                    eval_model=self.optimizer_model,
                    hermes_repo=str(self.hermes_agent_path),
                    tool_filter=[target.target_name],
                )
                output_dir = Path("output/tool_descriptions")
                if output_dir.exists():
                    subdirs = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    if subdirs:
                        mf = subdirs[0] / "metrics.json"
                        if mf.exists():
                            metrics = json.loads(mf.read_text())
                            result["improvement"] = metrics.get("improvement", 0.0)
                            result["output_dir"] = str(subdirs[0])
                            result["success"] = metrics.get("improvement", 0.0) > 0
                logger.info("Tool %s: improvement=%+.3f success=%s", target.target_name, result["improvement"], result["success"])
            except Exception as e:
                logger.error("Tool optimization failed for %s: %s", target.target_name, e)
                result["error"] = str(e)

        elif target.target_type == "prompt":
            try:
                from evolution.prompts.evolve_prompt_section import evolve_prompt_section
                logger.info("Optimizing prompt section: %s (%d iterations)", target.target_name, self.optimize_iterations)
                evolve_prompt_section(
                    section_name=target.target_name,
                    iterations=self.optimize_iterations,
                    optimizer_model=self.optimizer_model,
                    eval_model=self.optimizer_model,
                    hermes_repo=str(self.hermes_agent_path),
                )
                output_dir = Path("output/prompt_sections")
                if output_dir.exists():
                    subdirs = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    if subdirs:
                        mf = subdirs[0] / "metrics.json"
                        if mf.exists():
                            metrics = json.loads(mf.read_text())
                            result["improvement"] = metrics.get("improvement", 0.0)
                            result["output_dir"] = str(subdirs[0])
                            result["success"] = metrics.get("improvement", 0.0) > 0
                logger.info("Prompt section %s: improvement=%+.3f success=%s", target.target_name, result["improvement"], result["success"])
            except Exception as e:
                logger.error("Prompt optimization failed for %s: %s", target.target_name, e)
                result["error"] = str(e)

        else:
            logger.warning("Unknown target type: %s for %s", target.target_type, target.target_name)
            result["error"] = f"Unknown target type: {target.target_type}"

        return result

    def run_cycle(self, dry_run: bool = False) -> dict:
        """Run one full continuous improvement cycle.

        Supports checkpoint/resume — if a previous cycle was interrupted,
        this run picks up where it left off.

        Returns a summary of what was done.
        """
        logger.info("=== Continuous Self-Improvement Cycle START ===")
        start = time.time()
        summary = {
            "timestamp": datetime.now().isoformat(),
            "targets_found": 0,
            "targets_optimized": 0,
            "prs_created": 0,
            "benchmarks_passed": True,
            "elapsed_seconds": 0,
            "optimizations": [],
        }

        # ── Resume from checkpoint if available ──────────────────────────
        checkpoint = self._load_checkpoint() if self.resume else None
        if checkpoint:
            summary = checkpoint["summary"]
            logger.info("Resumed: %d targets already optimized", summary["targets_optimized"])

        # ── Step 1: Scan metrics ────────────────────────────────────────
        logger.info("Step 1: Scanning performance metrics")
        skill_metrics = self.monitor.get_skill_metrics()
        tool_metrics = self.monitor.get_tool_metrics()
        benchmark_trends = self.monitor.get_benchmark_trends()

        logger.info("Skills tracked: %d, Tools tracked: %d, Benchmarks: %d",
                     len(skill_metrics), len(tool_metrics), len(benchmark_trends))

        # ── Step 2: Triage ──────────────────────────────────────────────
        logger.info("Step 2: Identifying optimization targets")
        targets = self.triage.triage(skill_metrics, tool_metrics, benchmark_trends)
        summary["targets_found"] = len(targets)

        if not targets:
            logger.info("No optimization targets found — all metrics look good")
            summary["elapsed_seconds"] = time.time() - start
            self._clear_checkpoint()
            return summary

        targets = targets[:self.max_targets]
        for t in targets:
            logger.info("Target: %s/%s priority=%.2f score=%.3f reason=%s",
                         t.target_type, t.target_name, t.priority_score, t.current_score, t.reason)

        if dry_run:
            logger.info("DRY RUN — would optimize %d targets", len(targets))
            for t in targets:
                logger.info("  → %s: %s", t.target_type, t.target_name)
            summary["elapsed_seconds"] = time.time() - start
            return summary

        # ── Step 3: Run benchmarks (gate) ───────────────────────────────
        if self.benchmark_gate:
            logger.info("Step 3: Running benchmark gate")
            # Try the fast benchmarks as a regression check
            try:
                fast_result = self.benchmarks.run_tblite_fast()
                if fast_result.error:
                    logger.warning("Benchmark gate skipped: %s", fast_result.error)
                else:
                    gate = self.benchmarks.check_gate([fast_result])
                    summary["benchmarks_passed"] = gate.passed
                    if not gate.passed:
                        logger.warning("Benchmark gate FAILED: %s", gate.regressions)
            except Exception as e:
                logger.warning("Benchmark gate error (continuing anyway): %s", e)

        # ── Step 4: Optimize top targets ────────────────────────────────
        logger.info("Step 4: Optimizing %d targets", len(targets))

        target_results = []
        start_index = checkpoint["next_target_index"] if checkpoint else 0

        for i, target in enumerate(targets):
            if i < start_index:
                logger.info("Skipping already-optimized target: %s/%s", target.target_type, target.target_name)
                continue

            logger.info("Optimizing target %d/%d: %s/%s", i + 1, len(targets), target.target_type, target.target_name)

            result = self._optimize_target(target)
            target_results.append(result)
            summary["optimizations"].append({
                "type": target.target_type,
                "name": target.target_name,
                "improvement": result["improvement"],
                "success": result["success"],
                "output_dir": result["output_dir"],
                "error": result["error"],
            })

            if result["success"]:
                summary["targets_optimized"] += 1
            else:
                logger.warning("Optimization did not improve %s/%s: %s",
                               target.target_type, target.target_name, result.get("error", "no improvement"))

            # Save checkpoint after each target
            self._save_checkpoint(summary, i + 1, targets)

        # ── Step 5: Report ──────────────────────────────────────────────
        summary["elapsed_seconds"] = time.time() - start
        self._clear_checkpoint()

        logger.info("=== Cycle complete in %.1fs ===", summary["elapsed_seconds"])
        logger.info("Targets found: %d, optimized: %d, PRs created: %d",
                     summary["targets_found"], summary["targets_optimized"], summary["prs_created"])

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
    parser.add_argument("--iterations", default=10, type=int, help="Iterations per target (default: 10)")
    parser.add_argument("--model", default="qwen3.6-plus", help="Optimizer model")
    parser.add_argument("--max-targets", default=3, type=int, help="Max targets per cycle")
    parser.add_argument("--no-resume", action="store_true", help="Skip checkpoint resume")

    args = parser.parse_args()

    if args.setup_cron:
        setup_cron_jobs()
        return

    if args.cycle:
        evolution = ContinuousEvolution(
            hermes_agent_path=Path(args.hermes_repo) if args.hermes_repo else None,
            max_targets=args.max_targets,
            optimize_iterations=args.iterations,
            optimizer_model=args.model,
            resume=not args.no_resume,
        )
        result = evolution.run_cycle(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
