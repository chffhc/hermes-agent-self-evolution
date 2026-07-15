"""Local isolated-workspace executor seam for the capability benchmark.

For every task this executor creates a disposable per-run/per-task workspace,
injects exactly one digest-bound baseline or candidate artifact, invokes an
agent through an injectable argv/callable seam (never ``shell=True``), parses
a strict per-task usage report against a hard USD budget, and then runs the
task's deterministic verifiers against the final workspace state.

Only the ``fake_agent`` execution mode is implemented. Its output validates
the harness, not an agent, so every run emitted here is permanently
``capability_evidence=False``. A future live Hermes adapter must plug into the
same :class:`AgentInvoker` seam and satisfy explicit prerequisites (real agent
binary, real usage extraction, hard budget) before any run may ever be
labeled live evidence.

Error philosophy: harness misconfiguration (bad artifact, unsafe destination,
unsupported invoker mode, malformed budget) raises :class:`SchemaError`;
agent-side failures (timeout, nonzero exit, symlink escapes, missing or
malformed or over-budget usage) are recorded as failed task results.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from benchmarks.capability.replay import digest_artifact
from benchmarks.capability.schema import (
    RunFingerprint,
    RunResult,
    SchemaError,
    TaskResult,
    load_usage_report,
    safe_relative_path,
    utc_now_iso,
)
from benchmarks.capability.suite import CapabilitySuite
from benchmarks.capability.verifiers import VERIFIERS

FAKE_AGENT_MODE = "fake_agent"
SUPPORTED_EXECUTION_MODES = frozenset({FAKE_AGENT_MODE})
DEFAULT_ARTIFACT_DEST = "hermes_artifact"
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

FAKE_AGENT_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "fake_agent.py"


@dataclass(frozen=True)
class BudgetConfig:
    """Hard USD budget: per run, and optionally per task. Fail-closed."""

    max_run_usd: float
    max_task_usd: float | None = None

    def __post_init__(self) -> None:
        for name, value in (("max_run_usd", self.max_run_usd), ("max_task_usd", self.max_task_usd)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchemaError(f"budget: {name} must be a number, got {value!r}")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise SchemaError(f"budget: {name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True)
class AgentInvocation:
    """Everything one agent invocation may see: its own workspace and IDs."""

    run_id: str
    task_id: str
    prompt: str
    workspace: Path
    control_dir: Path
    usage_file: Path
    task_fixture_dir: Path
    timeout_seconds: float


@dataclass(frozen=True)
class InvocationOutcome:
    exit_code: int | None
    timed_out: bool
    detail: str = ""


class AgentInvoker(Protocol):
    @property
    def execution_mode(self) -> str: ...

    def invoke(self, invocation: AgentInvocation) -> InvocationOutcome: ...


@dataclass(frozen=True)
class ArgvAgentInvoker:
    """Run an agent executable via argv template substitution, no shell.

    ``argv_template[0]`` must be the literal ``"python"`` sentinel (replaced
    with the current interpreter) or an absolute path to an existing file.
    The ``{prompt}``, ``{workspace}``, ``{control_dir}``, ``{usage_file}``,
    ``{run_id}``, ``{task_id}``, and ``{task_fixture_dir}`` placeholders are
    substituted per task; run/task attribution is also exported via
    ``HERMES_BENCH_RUN_ID`` and ``HERMES_BENCH_TASK_ID`` in a scrubbed
    environment.
    """

    argv_template: tuple[str, ...]
    execution_mode: str = FAKE_AGENT_MODE

    def __post_init__(self) -> None:
        if self.execution_mode not in SUPPORTED_EXECUTION_MODES:
            raise SchemaError(
                f"invoker: execution_mode must be one of {sorted(SUPPORTED_EXECUTION_MODES)}; "
                f"live invocation is not implemented (got {self.execution_mode!r})"
            )
        argv = self.argv_template
        if not argv or not all(isinstance(a, str) and a and "\x00" not in a for a in argv):
            raise SchemaError("invoker: argv_template must be non-empty NUL-free strings")
        head = argv[0]
        if head != "python" and not (Path(head).is_absolute() and Path(head).is_file()):
            raise SchemaError(
                "invoker: argv_template[0] must be the 'python' sentinel or an "
                f"absolute path to an existing file, got {head!r}"
            )

    def build_argv(self, invocation: AgentInvocation) -> list[str]:
        substitutions = {
            "{prompt}": invocation.prompt,
            "{workspace}": str(invocation.workspace),
            "{control_dir}": str(invocation.control_dir),
            "{usage_file}": str(invocation.usage_file),
            "{run_id}": invocation.run_id,
            "{task_id}": invocation.task_id,
            "{task_fixture_dir}": str(invocation.task_fixture_dir),
        }
        argv: list[str] = []
        for element in self.argv_template:
            for token, value in substitutions.items():
                element = element.replace(token, value)
            argv.append(element)
        if argv[0] == "python":
            argv[0] = sys.executable
        return argv

    def invoke(self, invocation: AgentInvocation) -> InvocationOutcome:
        argv = self.build_argv(invocation)
        env = {
            "PATH": os.defpath,
            "HOME": str(invocation.workspace),
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "HERMES_BENCH_RUN_ID": invocation.run_id,
            "HERMES_BENCH_TASK_ID": invocation.task_id,
        }
        record = {
            "run_id": invocation.run_id,
            "task_id": invocation.task_id,
            "argv": argv,
            "execution_mode": self.execution_mode,
            "capability_evidence": False,
            "started_at": utc_now_iso(),
        }
        (invocation.control_dir / "invocation.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            proc = subprocess.run(
                argv,
                cwd=str(invocation.workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=invocation.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return InvocationOutcome(
                exit_code=None,
                timed_out=True,
                detail=f"agent invocation timed out after {invocation.timeout_seconds}s",
            )
        (invocation.control_dir / "stdout.txt").write_text(proc.stdout[-10_000:], encoding="utf-8")
        (invocation.control_dir / "stderr.txt").write_text(proc.stderr[-10_000:], encoding="utf-8")
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return InvocationOutcome(exit_code=proc.returncode, timed_out=False, detail=tail)


def build_fake_agent_invoker(*, solve: bool) -> ArgvAgentInvoker:
    """Bundled local fake-agent invoker; deterministic, free, never evidence."""
    argv = [
        "python",
        str(FAKE_AGENT_SCRIPT),
        "--workspace",
        "{workspace}",
        "--usage-file",
        "{usage_file}",
        "--run-id",
        "{run_id}",
        "--task-id",
        "{task_id}",
    ]
    if solve:
        argv += ["--solutions", "{task_fixture_dir}/replay"]
    return ArgvAgentInvoker(tuple(argv))


@dataclass(frozen=True)
class LocalRunOutcome:
    result: RunResult
    retained_root: Path | None


def _inject_artifact(
    artifact: Path, workspace: Path, dest_rel: str, expected_digest: str | None
) -> str:
    """Copy the artifact into the workspace with digest binding; return digest."""
    source_digest = digest_artifact(artifact)
    if expected_digest is not None and expected_digest != source_digest:
        raise SchemaError(
            f"artifact digest mismatch: expected {expected_digest}, got {source_digest}"
        )
    safe_relative_path(dest_rel, "artifact destination")
    dest_dir = workspace / dest_rel
    if dest_dir.exists():
        raise SchemaError(f"artifact destination collides with fixture content: {dest_rel!r}")
    dest_dir.mkdir(parents=True)
    injected = dest_dir / artifact.name
    if artifact.is_file():
        shutil.copy2(artifact, injected)
    else:
        shutil.copytree(artifact, injected, symlinks=False)
    if digest_artifact(injected) != source_digest:
        raise SchemaError(f"artifact injection corrupted content at {injected}")
    return source_digest


def _find_symlink(workspace: Path) -> Path | None:
    for root, dirs, files in os.walk(workspace, followlinks=False):
        for name in dirs + files:
            candidate = Path(root) / name
            if candidate.is_symlink():
                return candidate
    return None


def _failed(task_id: str, error: str, duration: float, cost: float | None) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        passed=False,
        score=0.0,
        duration_seconds=duration,
        tool_errors=0,
        invalid_tool_calls=0,
        cost_usd=cost,
        error=error,
    )


def run_local(
    suite: CapabilitySuite,
    *,
    invoker: AgentInvoker,
    run_role: str,
    artifact_path: str | Path,
    fingerprint: RunFingerprint,
    budget: BudgetConfig,
    expected_artifact_digest: str | None = None,
    artifact_dest: str = DEFAULT_ARTIFACT_DEST,
    run_id: str | None = None,
    runs_root: str | Path | None = None,
    keep_workspaces: bool = False,
) -> LocalRunOutcome:
    """Execute every suite task in an isolated workspace via the invoker seam."""
    if run_role not in {"baseline", "candidate"}:
        raise SchemaError("run_role must be 'baseline' or 'candidate'")
    mode = getattr(invoker, "execution_mode", None)
    if mode not in SUPPORTED_EXECUTION_MODES:
        raise SchemaError(
            f"unsupported invoker execution_mode {mode!r}: only "
            f"{sorted(SUPPORTED_EXECUTION_MODES)} are implemented, and none of them "
            "is ever capability evidence"
        )
    if run_id is None:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
    elif not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        # Validate before using the value in tempfile's prefix or exposing it
        # to an invoker. RunResult validates it again at serialization time.
        raise SchemaError(f"run_id must match {_RUN_ID_RE.pattern}, got {run_id!r}")
    artifact = Path(artifact_path)
    artifact_digest = digest_artifact(artifact)  # validate before any workspace exists
    if expected_artifact_digest is not None and expected_artifact_digest != artifact_digest:
        raise SchemaError(
            f"artifact digest mismatch: expected {expected_artifact_digest}, "
            f"got {artifact_digest}"
        )

    results: list[TaskResult] = []
    spent_usd = 0.0
    budget_exhausted = False
    if runs_root is not None:
        Path(runs_root).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"capability-{run_id}-", dir=runs_root))
    retained: Path | None = None
    try:
        for task in suite.tasks:
            if budget_exhausted:
                results.append(
                    _failed(
                        task.task_id,
                        f"not executed: run budget ${budget.max_run_usd:.4f} "
                        "already exhausted (fail closed)",
                        0.0,
                        None,
                    )
                )
                continue
            task_dir = suite.root / task.fixture
            task_root = root / "tasks" / task.task_id
            workspace = task_root / "workspace"
            control_dir = task_root / "control"
            control_dir.mkdir(parents=True)
            shutil.copytree(task_dir / "workspace", workspace, symlinks=False)
            _inject_artifact(artifact, workspace, artifact_dest, artifact_digest)
            invocation = AgentInvocation(
                run_id=run_id,
                task_id=task.task_id,
                prompt=task.prompt,
                workspace=workspace,
                control_dir=control_dir,
                usage_file=control_dir / "usage.json",
                task_fixture_dir=task_dir,
                timeout_seconds=task.timeout_seconds,
            )
            started = time.monotonic()
            try:
                outcome = invoker.invoke(invocation)
            except Exception as exc:  # invoker bugs must fail the task, not the harness
                results.append(
                    _failed(
                        task.task_id,
                        f"invoker failure: {type(exc).__name__}: {exc}",
                        time.monotonic() - started,
                        None,
                    )
                )
                continue
            duration = time.monotonic() - started

            error: str | None = None
            if outcome.timed_out:
                error = (
                    outcome.detail or f"agent invocation timed out after {task.timeout_seconds}s"
                )
            elif outcome.exit_code != 0:
                error = f"agent exited with code {outcome.exit_code}: {outcome.detail}"

            if error is None:
                escaped = _find_symlink(workspace)
                if escaped is not None:
                    error = f"symlink in final workspace (fail closed): {escaped.relative_to(workspace)}"

            cost: float | None = None
            if error is None:
                try:
                    usage = load_usage_report(invocation.usage_file)
                    cost = usage.cost_usd
                except SchemaError as exc:
                    error = f"usage report invalid or missing (fail closed): {exc}"
            if cost is not None:
                if budget.max_task_usd is not None and cost > budget.max_task_usd:
                    error = (
                        f"task cost ${cost:.4f} exceeds per-task budget "
                        f"${budget.max_task_usd:.4f} (fail closed)"
                    )
                spent_usd += cost
                if spent_usd > budget.max_run_usd:
                    budget_exhausted = True
                    if error is None:
                        error = (
                            f"cumulative cost ${spent_usd:.4f} exceeds run budget "
                            f"${budget.max_run_usd:.4f} (fail closed)"
                        )

            if error is not None:
                results.append(_failed(task.task_id, error, duration, cost))
                continue

            details: list[dict[str, object]] = []
            try:
                for spec in task.verifiers:
                    verdict = VERIFIERS[spec.type].run(
                        workspace, task_dir, spec.params, task.timeout_seconds
                    )
                    details.append(
                        {"verifier": spec.type, "ok": verdict.ok, "detail": verdict.detail}
                    )
            except Exception as exc:  # verifier runtime errors are benchmark failures
                results.append(
                    _failed(task.task_id, f"{type(exc).__name__}: {exc}", duration, cost)
                )
                continue
            passed_count = sum(1 for item in details if item["ok"])
            passed = passed_count == len(task.verifiers)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    passed=passed,
                    score=passed_count / len(task.verifiers),
                    duration_seconds=duration,
                    tool_errors=0,
                    invalid_tool_calls=0,
                    cost_usd=cost,
                    verifier_details=tuple(details),
                )
            )
    finally:
        if keep_workspaces:
            retained = root
        elif root.exists():
            # Cleanup failure is a harness failure; never silently leave a
            # workspace behind while claiming cleanup succeeded.
            shutil.rmtree(root)

    notes = (
        f"Local {mode} execution; harness validation only, never live agent "
        f"capability evidence. total_cost_usd={spent_usd:.4f} "
        f"budget_usd={budget.max_run_usd:.4f}"
    )
    if retained is not None:
        notes += f" retained_workspaces={retained}"
    result = RunResult(
        schema_version=1,
        suite_id=suite.suite_id,
        suite_hash=suite.suite_hash,
        run_role=run_role,
        artifact_digest=artifact_digest,
        fingerprint=fingerprint,
        execution_mode=mode,
        capability_evidence=False,
        created_at=utc_now_iso(),
        results=tuple(results),
        notes=notes,
        run_id=run_id,
    )
    # Round-trip through the fail-closed parser so no invalid document escapes.
    return LocalRunOutcome(result=RunResult.from_dict(result.to_dict()), retained_root=retained)
