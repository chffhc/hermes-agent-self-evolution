"""Paired baseline/candidate comparison with fail-closed comparability gates."""

from __future__ import annotations

from dataclasses import dataclass

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
    # Milestone-1 conservative gate: no pass-rate regression and no critical regression.
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
        capability_evidence=baseline.capability_evidence and candidate.capability_evidence,
    )
