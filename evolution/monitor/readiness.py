"""Phase 5 unattended-readiness gate (fail-closed).

Phase 5's goal is a continuous improvement cycle that runs unattended, which
makes preflight failures expensive: a cron-launched cycle with no hard budget,
no benchmark runner, or an unresolvable hermes-agent repo either burns money
or silently no-ops week after week. This gate inspects only local state (no
network calls) and answers one question before a live cycle is allowed to
start: is this environment actually ready to run unattended?

Required checks fail closed — any error while probing counts as "not ready".
Advisory checks (metrics sources, checkpoint state) are surfaced in the
report but do not block a live run, because the cycle degrades safely
without them (it finds no targets and exits).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from evolution.core.config import resolve_hermes_agent_path
from evolution.core.cost_tracker import tracker


@dataclass(frozen=True)
class ReadinessCheck:
    """One preflight probe result."""

    name: str
    ok: bool
    required: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregated readiness verdict for a live Phase 5 cycle."""

    checks: list[ReadinessCheck] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    def failing(self) -> list[ReadinessCheck]:
        return [c for c in self.checks if c.required and not c.ok]

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "checks": [
                {"name": c.name, "ok": c.ok, "required": c.required, "detail": c.detail}
                for c in self.checks
            ],
        }


def check_phase5_readiness(
    hermes_repo: str | None = None,
    hermes_home: Path | None = None,
    output_dir: Path | None = None,
) -> ReadinessReport:
    """Run all Phase 5 preflight checks and return the report.

    Args:
        hermes_repo: explicit hermes-agent path (e.g. from ``--hermes-repo``);
            falls back to the standard discovery chain when omitted.
        hermes_home: metrics-source root, defaults to ``~/.hermes``.
        output_dir: monitor output directory, defaults to ``output/monitor``.
    """
    checks: list[ReadinessCheck] = []
    home = hermes_home if hermes_home is not None else Path.home() / ".hermes"
    out_dir = output_dir if output_dir is not None else Path("output/monitor")

    # ── hermes-agent repo (required) ────────────────────────────────────
    repo: Path | None = None
    try:
        candidate = resolve_hermes_agent_path(hermes_repo)
        if candidate.is_dir():
            repo = candidate
            checks.append(
                ReadinessCheck("hermes_repo", True, True, f"hermes-agent repo at {candidate}")
            )
        else:
            checks.append(
                ReadinessCheck(
                    "hermes_repo", False, True, f"{candidate} does not exist or is not a directory"
                )
            )
    except FileNotFoundError as e:
        checks.append(ReadinessCheck("hermes_repo", False, True, str(e)))

    # ── benchmark runner (required) ─────────────────────────────────────
    # The cycle's default benchmark gate shells out to this script; without
    # it every live cycle aborts at Step 3 with benchmarks_passed=False.
    if repo is None:
        checks.append(
            ReadinessCheck(
                "benchmark_runner", False, True, "cannot check: hermes-agent repo not found"
            )
        )
    else:
        runner = repo / "environments" / "benchmarks" / "run_bench.py"
        checks.append(
            ReadinessCheck(
                "benchmark_runner",
                runner.is_file(),
                True,
                (
                    f"benchmark runner at {runner}"
                    if runner.is_file()
                    else f"benchmark runner not found: {runner}"
                ),
            )
        )

    # ── hard budget (required) ──────────────────────────────────────────
    # Unattended optimization without a hard USD cap is an unbounded spend.
    budget = tracker.max_cost_usd
    checks.append(
        ReadinessCheck(
            "hard_budget",
            budget is not None,
            True,
            (
                f"hard budget ${budget:.2f} configured"
                if budget is not None
                else "no hard USD budget; set EVOLUTION_MAX_COST_USD or pass --max-cost-usd"
            ),
        )
    )

    # ── output directory writable (required) ────────────────────────────
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out_dir, prefix=".readiness_probe_"):
            pass
        checks.append(ReadinessCheck("output_writable", True, True, f"{out_dir} is writable"))
    except OSError as e:
        checks.append(
            ReadinessCheck("output_writable", False, True, f"{out_dir} not writable: {e}")
        )

    # ── metrics sources (advisory) ──────────────────────────────────────
    session_db = home / "sessions.db"
    checks.append(
        ReadinessCheck(
            "session_db",
            session_db.is_file(),
            False,
            (
                f"SessionDB at {session_db}"
                if session_db.is_file()
                else f"no SessionDB at {session_db}; triage will find no skill targets"
            ),
        )
    )
    agent_log = home / "logs" / "agent.log"
    checks.append(
        ReadinessCheck(
            "agent_log",
            agent_log.is_file(),
            False,
            (
                f"agent log at {agent_log}"
                if agent_log.is_file()
                else f"no agent log at {agent_log}; triage will find no tool targets"
            ),
        )
    )

    # ── checkpoint state (advisory, informational) ──────────────────────
    checkpoint = out_dir / "checkpoint.json"
    checks.append(
        ReadinessCheck(
            "checkpoint",
            True,
            False,
            (
                "interrupted-cycle checkpoint present; a live run will resume it"
                if checkpoint.is_file()
                else "no interrupted-cycle checkpoint"
            ),
        )
    )

    return ReadinessReport(checks=checks)


__all__ = [
    "ReadinessCheck",
    "ReadinessReport",
    "check_phase5_readiness",
]
