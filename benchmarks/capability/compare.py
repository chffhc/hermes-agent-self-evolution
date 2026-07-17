"""Paired baseline/candidate comparison with fail-closed comparability gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.capability.schema import RunResult, SchemaError
from benchmarks.capability.suite import CapabilitySuite


@dataclass(frozen=True)
class Comparison:
    passed_gate: bool
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float
    baseline_mean_score: float
    candidate_mean_score: float
    score_delta: float
    regressions: tuple[str, ...]
    critical_regressions: tuple[str, ...]
    improvements: tuple[str, ...]
    duration_delta_seconds: float
    cost_delta_usd: float | None
    capability_evidence: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "passed_gate": self.passed_gate,
            "baseline_pass_rate": self.baseline_pass_rate,
            "candidate_pass_rate": self.candidate_pass_rate,
            "pass_rate_delta": self.pass_rate_delta,
            "baseline_mean_score": self.baseline_mean_score,
            "candidate_mean_score": self.candidate_mean_score,
            "score_delta": self.score_delta,
            "regressions": list(self.regressions),
            "critical_regressions": list(self.critical_regressions),
            "improvements": list(self.improvements),
            "duration_delta_seconds": self.duration_delta_seconds,
            "cost_delta_usd": self.cost_delta_usd,
            "capability_evidence": self.capability_evidence,
            "note": (
                "Paired live capability evidence."
                if self.capability_evidence
                else "Harness/replay comparison only; not live capability evidence."
            ),
        }


def _assert_comparable(suite: CapabilitySuite, baseline: RunResult, candidate: RunResult) -> None:
    if baseline.capability_evidence or candidate.capability_evidence:
        raise SchemaError(
            "schema v1 comparisons refuse capability_evidence=true, including manually "
            "constructed live RunResult objects (fail closed)"
        )
    if baseline.run_role != "baseline" or candidate.run_role != "candidate":
        raise SchemaError("comparison requires baseline and candidate run roles")
    for field in ("suite_id", "suite_hash", "fingerprint", "execution_mode"):
        if getattr(baseline, field) != getattr(candidate, field):
            raise SchemaError(f"comparison mismatch: {field}")
    if baseline.suite_id != suite.suite_id or baseline.suite_hash != suite.suite_hash:
        raise SchemaError("run files do not match the supplied suite")
    if baseline.capability_evidence != candidate.capability_evidence:
        raise SchemaError("comparison mismatch: capability_evidence")
    expected = set(suite.task_ids)
    baseline_ids = {r.task_id for r in baseline.results}
    candidate_ids = {r.task_id for r in candidate.results}
    if baseline_ids != expected or candidate_ids != expected:
        raise SchemaError(
            "comparison task set mismatch: "
            f"expected={sorted(expected)} baseline={sorted(baseline_ids)} "
            f"candidate={sorted(candidate_ids)}"
        )


def compare_runs(suite: CapabilitySuite, baseline: RunResult, candidate: RunResult) -> Comparison:
    _assert_comparable(suite, baseline, candidate)
    bmap = {r.task_id: r for r in baseline.results}
    cmap = {r.task_id: r for r in candidate.results}
    regressions = tuple(sorted(tid for tid in bmap if bmap[tid].passed and not cmap[tid].passed))
    improvements = tuple(sorted(tid for tid in bmap if not bmap[tid].passed and cmap[tid].passed))
    critical_ids = {task.task_id for task in suite.tasks if task.critical}
    critical_regressions = tuple(tid for tid in regressions if tid in critical_ids)
    b_score = sum(r.score for r in baseline.results) / len(baseline.results)
    c_score = sum(r.score for r in candidate.results) / len(candidate.results)
    b_duration = sum(r.duration_seconds for r in baseline.results)
    c_duration = sum(r.duration_seconds for r in candidate.results)
    b_costs = [r.cost_usd for r in baseline.results]
    c_costs = [r.cost_usd for r in candidate.results]
    cost_delta = None
    if all(value is not None for value in b_costs + c_costs):
        cost_delta = sum(c_costs) - sum(b_costs)  # type: ignore[arg-type]
    # Milestone-1 conservative gate: no pass-rate regression and no critical
    # regression. Holdout tasks participate fully in the human-review gate and
    # aggregates; optimizer_feedback derives a development-only view and omits
    # all holdout outcomes and full-suite metrics.
    passed_gate = not critical_regressions and candidate.pass_rate >= baseline.pass_rate
    return Comparison(
        passed_gate=passed_gate,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        pass_rate_delta=candidate.pass_rate - baseline.pass_rate,
        baseline_mean_score=b_score,
        candidate_mean_score=c_score,
        score_delta=c_score - b_score,
        regressions=regressions,
        critical_regressions=critical_regressions,
        improvements=improvements,
        duration_delta_seconds=c_duration - b_duration,
        cost_delta_usd=cost_delta,
        capability_evidence=False,
    )


def optimizer_feedback(suite: CapabilitySuite, comparison: Comparison) -> dict[str, Any]:
    """Build a development-only view safe for optimizer iteration.

    Holdout outcomes must not become an adaptive oracle: this document omits
    their identities, outcome counts, full-suite gate, and full-suite metric
    deltas. The holdout-aware :class:`Comparison` remains human-review data.
    Fixtures still live in this repository, so this is isolation of the
    feedback channel rather than fixture secrecy.
    """
    if comparison.capability_evidence:
        raise SchemaError(
            "optimizer feedback refuses capability_evidence=true comparisons (fail closed)"
        )

    fields = {
        "regressions": comparison.regressions,
        "improvements": comparison.improvements,
        "critical_regressions": comparison.critical_regressions,
    }
    for name, task_ids in fields.items():
        if len(task_ids) != len(set(task_ids)):
            raise SchemaError(f"comparison contains duplicate {name} task IDs (fail closed)")

    known = set(suite.task_ids)
    referenced = set().union(*(set(task_ids) for task_ids in fields.values()))
    unknown = referenced - known
    if unknown:
        raise SchemaError(
            f"comparison references tasks outside the suite (fail closed): {sorted(unknown)}"
        )
    if set(comparison.regressions) & set(comparison.improvements):
        raise SchemaError("comparison marks a task as both regression and improvement")
    if not set(comparison.critical_regressions).issubset(comparison.regressions):
        raise SchemaError("critical regressions must also appear in regressions")
    expected_critical = {
        task.task_id
        for task in suite.tasks
        if task.critical and task.task_id in comparison.regressions
    }
    if set(comparison.critical_regressions) != expected_critical:
        raise SchemaError("comparison critical regression metadata is inconsistent with the suite")

    development = set(suite.development_task_ids)
    if not development:
        raise SchemaError("optimizer feedback requires at least one development task")
    dev_regressions = sorted(development & set(comparison.regressions))
    dev_improvements = sorted(development & set(comparison.improvements))
    dev_critical = sorted(development & set(comparison.critical_regressions))
    development_pass_rate_delta = (len(dev_improvements) - len(dev_regressions)) / len(development)
    development_gate_passed = not dev_critical and development_pass_rate_delta >= 0

    return {
        "feedback_version": 2,
        "suite_id": suite.suite_id,
        "suite_hash": suite.suite_hash,
        "capability_evidence": False,
        "development": {
            "task_count": len(development),
            "gate_passed": development_gate_passed,
            "pass_rate_delta": development_pass_rate_delta,
            "regressions": dev_regressions,
            "improvements": dev_improvements,
            "critical_regressions": dev_critical,
        },
        "holdout_outcomes_withheld": True,
        "note": (
            "Development-only optimizer feedback. Holdout identities, outcomes, counts, "
            "full-suite gate, and full-suite metric deltas are withheld for human review. "
            "Harness comparison only, never live capability evidence."
        ),
    }
