"""Honest replay executor for exercising the benchmark harness.

Replay copies a task's initial workspace, overlays the checked-in solved replay,
and runs deterministic verifiers. It validates the harness only and therefore
always emits ``capability_evidence=False``.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from pathlib import Path

from benchmarks.capability.schema import (
    RunFingerprint,
    RunResult,
    SchemaError,
    TaskResult,
    utc_now_iso,
)
from benchmarks.capability.suite import CapabilitySuite
from benchmarks.capability.verifiers import VERIFIERS


def digest_artifact(path: str | Path) -> str:
    """Hash one file or directory deterministically, including relative names."""
    target = Path(path)
    if not target.exists():
        raise SchemaError(f"artifact does not exist: {target}")
    if target.is_symlink():
        raise SchemaError(f"artifact must not be a symlink: {target}")
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.name.encode())
        digest.update(b"\0")
        digest.update(target.read_bytes())
        return digest.hexdigest()
    descendants = list(target.rglob("*"))
    symlinks = [p for p in descendants if p.is_symlink()]
    if symlinks:
        raise SchemaError(f"artifact contains symlink: {symlinks[0]}")
    files = sorted(p for p in descendants if p.is_file())
    if not files:
        raise SchemaError(f"artifact directory has no files: {target}")
    for item in files:
        digest.update(item.relative_to(target).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_replay(
    suite: CapabilitySuite,
    *,
    run_role: str,
    artifact_digest: str,
    fingerprint: RunFingerprint,
    apply_solution: bool,
) -> RunResult:
    """Run deterministic fixtures with or without the checked-in replay overlay."""
    if run_role not in {"baseline", "candidate"}:
        raise SchemaError("run_role must be 'baseline' or 'candidate'")
    if len(artifact_digest) != 64 or any(c not in "0123456789abcdef" for c in artifact_digest):
        raise SchemaError("artifact_digest must be 64 lowercase hex characters")

    results: list[TaskResult] = []
    with tempfile.TemporaryDirectory(prefix="capability-replay-") as temp:
        temp_root = Path(temp)
        for task in suite.tasks:
            task_dir = suite.root / task.fixture
            workspace = temp_root / task.task_id
            shutil.copytree(task_dir / "workspace", workspace, symlinks=False)
            if apply_solution:
                shutil.copytree(task_dir / "replay", workspace, dirs_exist_ok=True, symlinks=False)
            started = time.monotonic()
            details: list[dict[str, object]] = []
            error: str | None = None
            try:
                for spec in task.verifiers:
                    outcome = VERIFIERS[spec.type].run(
                        workspace, task_dir, spec.params, task.timeout_seconds
                    )
                    details.append(
                        {"verifier": spec.type, "ok": outcome.ok, "detail": outcome.detail}
                    )
            except Exception as exc:  # verifier runtime errors are benchmark failures
                error = f"{type(exc).__name__}: {exc}"
            duration = time.monotonic() - started
            passed_count = sum(1 for item in details if item["ok"])
            passed = (
                error is None
                and len(details) == len(task.verifiers)
                and passed_count == len(details)
            )
            score = passed_count / len(task.verifiers) if error is None else 0.0
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    passed=passed,
                    score=score,
                    duration_seconds=duration,
                    tool_errors=0,
                    invalid_tool_calls=0,
                    error=error,
                    verifier_details=tuple(details),
                )
            )

    return RunResult(
        schema_version=1,
        suite_id=suite.suite_id,
        suite_hash=suite.suite_hash,
        run_role=run_role,
        artifact_digest=artifact_digest,
        fingerprint=fingerprint,
        execution_mode="replay",
        capability_evidence=False,
        created_at=utc_now_iso(),
        results=tuple(results),
        notes="Harness replay only; this is not live agent capability evidence.",
    )
