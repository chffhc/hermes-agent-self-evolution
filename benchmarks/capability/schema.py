"""JSON-serializable task/result/run schema with fail-closed validation.

Every constructor here rejects unknown keys, missing keys, wrong types,
out-of-range scores/counters, unsafe relative paths, and dishonest evidence
labels (``capability_evidence=True`` on anything but a live run). Malformed
input raises :class:`SchemaError` instead of producing a partial object.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.capability import SCHEMA_VERSION

EXECUTION_MODES = frozenset({"replay", "dry_run", "fake_agent", "hermes_cli_stub", "live"})
RUN_ROLES = frozenset({"baseline", "candidate"})

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class SchemaError(ValueError):
    """A task, result, or run document failed fail-closed validation."""


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for digests."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    if not isinstance(config, dict):
        raise SchemaError(f"config must be a dict, got {type(config).__name__}")
    try:
        return sha256_text(canonical_json(config))
    except (TypeError, ValueError) as e:
        raise SchemaError(f"config is not JSON-serializable: {e}") from e


def safe_relative_path(value: Any, ctx: str) -> PurePosixPath:
    """Validate an untrusted relative path; reject traversal and absolutes.

    Returns the normalized PurePosixPath. Raises SchemaError for anything
    that could escape its root: absolute paths, drive letters, backslashes,
    ``..``/``.`` segments, empty segments, or NUL bytes.
    """
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{ctx}: path must be a non-empty string, got {value!r}")
    if "\x00" in value or "\\" in value or ":" in value:
        raise SchemaError(f"{ctx}: forbidden character in path {value!r}")
    if value.startswith("/"):
        raise SchemaError(f"{ctx}: absolute path not allowed: {value!r}")
    parts = value.split("/")
    if any(seg in ("", ".", "..") for seg in parts):
        raise SchemaError(f"{ctx}: unsafe path segment in {value!r}")
    return PurePosixPath(value)


def _check_keys(obj: Any, required: frozenset, optional: frozenset, ctx: str) -> None:
    if not isinstance(obj, dict):
        raise SchemaError(f"{ctx}: expected a JSON object, got {type(obj).__name__}")
    unknown = set(obj) - required - optional
    if unknown:
        raise SchemaError(f"{ctx}: unknown keys {sorted(unknown)}")
    missing = required - set(obj)
    if missing:
        raise SchemaError(f"{ctx}: missing required keys {sorted(missing)}")


def _req_str(obj: dict, key: str, ctx: str, *, slug: bool = False) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise SchemaError(f"{ctx}: {key!r} must be a non-empty string, got {v!r}")
    if slug and not _SLUG_RE.match(v):
        raise SchemaError(f"{ctx}: {key!r} must match {_SLUG_RE.pattern}, got {v!r}")
    return v


def _req_bool(obj: dict, key: str, ctx: str) -> bool:
    v = obj.get(key)
    if not isinstance(v, bool):
        raise SchemaError(f"{ctx}: {key!r} must be a boolean, got {v!r}")
    return v


def _req_int(obj: dict, key: str, ctx: str, *, minimum: int | None = None) -> int:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise SchemaError(f"{ctx}: {key!r} must be an integer, got {v!r}")
    if minimum is not None and v < minimum:
        raise SchemaError(f"{ctx}: {key!r} must be >= {minimum}, got {v}")
    return v


def _req_number(
    obj: dict,
    key: str,
    ctx: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SchemaError(f"{ctx}: {key!r} must be a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise SchemaError(f"{ctx}: {key!r} must be finite")
    if minimum is not None and v < minimum:
        raise SchemaError(f"{ctx}: {key!r} must be >= {minimum}, got {v}")
    if maximum is not None and v > maximum:
        raise SchemaError(f"{ctx}: {key!r} must be <= {maximum}, got {v}")
    return v


@dataclass(frozen=True)
class VerifierSpec:
    """One deterministic check against final workspace state."""

    type: str
    params: dict[str, Any]

    _KEYS = frozenset({"type", "params"})

    @classmethod
    def from_dict(cls, obj: Any, ctx: str) -> VerifierSpec:
        _check_keys(obj, cls._KEYS, frozenset(), ctx)
        vtype = _req_str(obj, "type", ctx, slug=True)
        params = obj.get("params")
        if not isinstance(params, dict):
            raise SchemaError(f"{ctx}: 'params' must be an object, got {params!r}")
        return cls(type=vtype, params=params)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": self.params}


@dataclass(frozen=True)
class TaskSpec:
    """One deterministic benchmark task."""

    task_id: str
    category: str
    prompt: str
    fixture: str  # relative task-asset dir inside the suite; contains workspace/
    verifiers: tuple[VerifierSpec, ...]
    timeout_seconds: float
    critical: bool
    description: str = ""

    _REQUIRED = frozenset(
        {"task_id", "category", "prompt", "fixture", "verifiers", "timeout_seconds", "critical"}
    )
    _OPTIONAL = frozenset({"description"})

    @classmethod
    def from_dict(cls, obj: Any) -> TaskSpec:
        ctx = f"task {obj.get('task_id')!r}" if isinstance(obj, dict) else "task"
        _check_keys(obj, cls._REQUIRED, cls._OPTIONAL, ctx)
        task_id = _req_str(obj, "task_id", ctx, slug=True)
        ctx = f"task {task_id!r}"
        fixture = str(safe_relative_path(obj.get("fixture"), f"{ctx}: fixture"))
        raw_verifiers = obj.get("verifiers")
        if not isinstance(raw_verifiers, list) or not raw_verifiers:
            raise SchemaError(f"{ctx}: 'verifiers' must be a non-empty list")
        verifiers = tuple(
            VerifierSpec.from_dict(v, f"{ctx}: verifier[{i}]") for i, v in enumerate(raw_verifiers)
        )
        description = obj.get("description", "")
        if not isinstance(description, str):
            raise SchemaError(f"{ctx}: 'description' must be a string")
        return cls(
            task_id=task_id,
            category=_req_str(obj, "category", ctx, slug=True),
            prompt=_req_str(obj, "prompt", ctx),
            fixture=fixture,
            verifiers=verifiers,
            timeout_seconds=_req_number(obj, "timeout_seconds", ctx, minimum=1, maximum=3600),
            critical=_req_bool(obj, "critical", ctx),
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "category": self.category,
            "prompt": self.prompt,
            "fixture": self.fixture,
            "verifiers": [v.to_dict() for v in self.verifiers],
            "timeout_seconds": self.timeout_seconds,
            "critical": self.critical,
        }
        if self.description:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class RunFingerprint:
    """What must MATCH between a baseline and a candidate run.

    The evolved artifact under test is deliberately NOT part of the
    fingerprint — it is what differs. Everything else (model, config, seed,
    environment) must be identical or the comparison is meaningless.
    """

    model: str
    config_digest: str
    seed: int
    environment: str

    _KEYS = frozenset({"model", "config_digest", "seed", "environment"})

    @classmethod
    def from_config(
        cls, model: str, config: dict[str, Any], seed: int, environment: str
    ) -> RunFingerprint:
        return cls.from_dict(
            {
                "model": model,
                "config_digest": config_digest(config),
                "seed": seed,
                "environment": environment,
            }
        )

    @classmethod
    def from_dict(cls, obj: Any) -> RunFingerprint:
        ctx = "fingerprint"
        _check_keys(obj, cls._KEYS, frozenset(), ctx)
        digest = _req_str(obj, "config_digest", ctx)
        if not _HEX_DIGEST_RE.match(digest):
            raise SchemaError(f"{ctx}: config_digest must be 64 lowercase hex chars")
        return cls(
            model=_req_str(obj, "model", ctx),
            config_digest=digest,
            seed=_req_int(obj, "seed", ctx),
            environment=_req_str(obj, "environment", ctx),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "config_digest": self.config_digest,
            "seed": self.seed,
            "environment": self.environment,
        }


@dataclass(frozen=True)
class UsageReport:
    """Strict per-task cost/usage report emitted by an agent invocation.

    The budget gate depends on this document, so it fails closed: unknown
    keys, missing keys, negative or non-finite numbers, and non-integer
    token counts are all hard errors rather than "free" tasks.
    """

    cost_usd: float
    input_tokens: int
    output_tokens: int

    _KEYS = frozenset({"cost_usd", "input_tokens", "output_tokens"})

    @classmethod
    def from_dict(cls, obj: Any) -> UsageReport:
        ctx = "usage report"
        _check_keys(obj, cls._KEYS, frozenset(), ctx)
        return cls(
            cost_usd=_req_number(obj, "cost_usd", ctx, minimum=0.0),
            input_tokens=_req_int(obj, "input_tokens", ctx, minimum=0),
            output_tokens=_req_int(obj, "output_tokens", ctx, minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def load_usage_report(path: str | Path) -> UsageReport:
    """Parse a usage-report JSON file, failing closed on any malformation."""
    report_path = Path(path)
    if report_path.is_symlink():
        raise SchemaError(f"usage report must not be a symlink: {report_path}")
    if not report_path.is_file():
        raise SchemaError(f"usage report is missing or not a regular file: {report_path}")
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SchemaError(f"cannot read usage report {report_path}: {e}") from e
    return UsageReport.from_dict(raw)


@dataclass(frozen=True)
class TaskResult:
    """Outcome of one task in one run."""

    task_id: str
    passed: bool
    score: float
    duration_seconds: float
    tool_errors: int
    invalid_tool_calls: int
    cost_usd: float | None = None
    error: str | None = None
    verifier_details: tuple[dict[str, Any], ...] = ()

    _REQUIRED = frozenset(
        {"task_id", "passed", "score", "duration_seconds", "tool_errors", "invalid_tool_calls"}
    )
    _OPTIONAL = frozenset({"cost_usd", "error", "verifier_details"})

    @classmethod
    def from_dict(cls, obj: Any) -> TaskResult:
        ctx = f"result {obj.get('task_id')!r}" if isinstance(obj, dict) else "result"
        _check_keys(obj, cls._REQUIRED, cls._OPTIONAL, ctx)
        task_id = _req_str(obj, "task_id", ctx, slug=True)
        ctx = f"result {task_id!r}"
        passed = _req_bool(obj, "passed", ctx)
        score = _req_number(obj, "score", ctx, minimum=0.0, maximum=1.0)
        error = obj.get("error")
        if error is not None and (not isinstance(error, str) or not error):
            raise SchemaError(f"{ctx}: 'error' must be null or a non-empty string")
        cost = obj.get("cost_usd")
        if cost is not None:
            cost = _req_number(obj, "cost_usd", ctx, minimum=0.0)
        if passed and score != 1.0:
            raise SchemaError(f"{ctx}: passed=true requires score=1.0, got {score}")
        if not passed and score == 1.0:
            raise SchemaError(f"{ctx}: score=1.0 requires passed=true")
        if passed and error is not None:
            raise SchemaError(f"{ctx}: passed=true is inconsistent with error={error!r}")
        details_raw = obj.get("verifier_details", [])
        if not isinstance(details_raw, list):
            raise SchemaError(f"{ctx}: 'verifier_details' must be a list")
        details = []
        for i, d in enumerate(details_raw):
            _check_keys(
                d,
                frozenset({"verifier", "ok", "detail"}),
                frozenset(),
                f"{ctx}: verifier_details[{i}]",
            )
            _req_str(d, "verifier", f"{ctx}: verifier_details[{i}]")
            _req_bool(d, "ok", f"{ctx}: verifier_details[{i}]")
            if not isinstance(d.get("detail"), str):
                raise SchemaError(f"{ctx}: verifier_details[{i}].detail must be a string")
            details.append(dict(d))
        return cls(
            task_id=task_id,
            passed=passed,
            score=score,
            duration_seconds=_req_number(obj, "duration_seconds", ctx, minimum=0.0),
            tool_errors=_req_int(obj, "tool_errors", ctx, minimum=0),
            invalid_tool_calls=_req_int(obj, "invalid_tool_calls", ctx, minimum=0),
            cost_usd=cost,
            error=error,
            verifier_details=tuple(details),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id": self.task_id,
            "passed": self.passed,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "tool_errors": self.tool_errors,
            "invalid_tool_calls": self.invalid_tool_calls,
        }
        if self.cost_usd is not None:
            d["cost_usd"] = self.cost_usd
        if self.error is not None:
            d["error"] = self.error
        if self.verifier_details:
            d["verifier_details"] = [dict(v) for v in self.verifier_details]
        return d


@dataclass(frozen=True)
class RunResult:
    """One complete benchmark run over a suite.

    ``capability_evidence`` may be True only for ``execution_mode="live"``:
    replay/fixture/dry-run outputs exercise the harness, not the agent, and
    the schema makes mislabeling them a hard error.
    """

    schema_version: int
    suite_id: str
    suite_hash: str
    run_role: str
    artifact_digest: str
    fingerprint: RunFingerprint
    execution_mode: str
    capability_evidence: bool
    created_at: str
    results: tuple[TaskResult, ...]
    notes: str = ""
    run_id: str = ""

    _REQUIRED = frozenset(
        {
            "schema_version",
            "suite_id",
            "suite_hash",
            "run_role",
            "artifact_digest",
            "fingerprint",
            "execution_mode",
            "capability_evidence",
            "created_at",
            "results",
        }
    )
    _OPTIONAL = frozenset({"notes", "run_id"})

    @classmethod
    def from_dict(cls, obj: Any) -> RunResult:
        ctx = "run"
        _check_keys(obj, cls._REQUIRED, cls._OPTIONAL, ctx)
        version = _req_int(obj, "schema_version", ctx, minimum=1)
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"{ctx}: schema_version {version} unsupported (expected {SCHEMA_VERSION})"
            )
        run_role = _req_str(obj, "run_role", ctx)
        if run_role not in RUN_ROLES:
            raise SchemaError(
                f"{ctx}: run_role must be one of {sorted(RUN_ROLES)}, got {run_role!r}"
            )
        mode = _req_str(obj, "execution_mode", ctx)
        if mode not in EXECUTION_MODES:
            raise SchemaError(
                f"{ctx}: execution_mode must be one of {sorted(EXECUTION_MODES)}, got {mode!r}"
            )
        evidence = _req_bool(obj, "capability_evidence", ctx)
        if evidence and mode != "live":
            raise SchemaError(
                f"{ctx}: capability_evidence=true is only valid for execution_mode='live' "
                f"(got {mode!r}) — fixture/replay/fake-agent/dry-run output is never "
                "capability evidence"
            )
        suite_hash = _req_str(obj, "suite_hash", ctx)
        if not _HEX_DIGEST_RE.match(suite_hash):
            raise SchemaError(f"{ctx}: suite_hash must be 64 lowercase hex chars")
        created_at = _req_str(obj, "created_at", ctx)
        try:
            parsed_at = datetime.fromisoformat(created_at)
        except ValueError as e:
            raise SchemaError(f"{ctx}: created_at is not ISO-8601: {e}") from e
        if parsed_at.tzinfo is None:
            raise SchemaError(f"{ctx}: created_at must include a timezone")
        artifact_digest = _req_str(obj, "artifact_digest", ctx)
        if not _HEX_DIGEST_RE.match(artifact_digest):
            raise SchemaError(f"{ctx}: artifact_digest must be 64 lowercase hex chars")
        raw_results = obj.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            raise SchemaError(f"{ctx}: 'results' must be a non-empty list")
        results = tuple(TaskResult.from_dict(r) for r in raw_results)
        seen: set[str] = set()
        for r in results:
            if r.task_id in seen:
                raise SchemaError(f"{ctx}: duplicate task_id {r.task_id!r} in results")
            seen.add(r.task_id)
        notes = obj.get("notes", "")
        if not isinstance(notes, str):
            raise SchemaError(f"{ctx}: 'notes' must be a string")
        run_id = obj.get("run_id", "")
        if run_id and (not isinstance(run_id, str) or not _SLUG_RE.match(run_id)):
            raise SchemaError(f"{ctx}: 'run_id' must match {_SLUG_RE.pattern}, got {run_id!r}")
        if not isinstance(run_id, str):
            raise SchemaError(f"{ctx}: 'run_id' must be a string")
        return cls(
            schema_version=version,
            suite_id=_req_str(obj, "suite_id", ctx, slug=True),
            suite_hash=suite_hash,
            run_role=run_role,
            artifact_digest=artifact_digest,
            fingerprint=RunFingerprint.from_dict(obj.get("fingerprint")),
            execution_mode=mode,
            capability_evidence=evidence,
            created_at=created_at,
            results=results,
            notes=notes,
            run_id=run_id,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "suite_hash": self.suite_hash,
            "run_role": self.run_role,
            "artifact_digest": self.artifact_digest,
            "fingerprint": self.fingerprint.to_dict(),
            "execution_mode": self.execution_mode,
            "capability_evidence": self.capability_evidence,
            "created_at": self.created_at,
            "results": [r.to_dict() for r in self.results],
        }
        if self.notes:
            d["notes"] = self.notes
        if self.run_id:
            d["run_id"] = self.run_id
        return d

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / len(self.results)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run_result(path) -> RunResult:
    """Parse a run-result JSON file, failing closed on any malformation."""
    try:
        raw = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError) as e:
        raise SchemaError(f"cannot read run result {path}: {e}") from e
    return RunResult.from_dict(raw)
