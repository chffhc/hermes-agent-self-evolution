"""JSON-serializable task/result/run schema with fail-closed validation.

Every constructor here rejects unknown keys, missing keys, wrong types,
out-of-range scores/counters, unsafe relative paths, and dishonest evidence
labels. Schema v1 refuses ``capability_evidence=True`` for every execution
mode, including externally supplied ``live`` JSON. Malformed input raises
:class:`SchemaError` instead of producing a partial object.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.capability import SCHEMA_VERSION

EXECUTION_MODES = frozenset({"replay", "dry_run", "fake_agent", "hermes_cli_stub", "live"})
RUN_ROLES = frozenset({"baseline", "candidate"})
TASK_SPLITS = frozenset({"development", "holdout"})

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


def is_ignored_fixture_cache_path(relative: PurePosixPath | Path) -> bool:
    """Return whether a copied-fixture path is transient cache/OS data."""
    return any(
        part == "__pycache__" or part == ".DS_Store" or part.endswith(".pyc")
        for part in relative.parts
    )


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
    split: str = "development"

    _REQUIRED = frozenset(
        {"task_id", "category", "prompt", "fixture", "verifiers", "timeout_seconds", "critical"}
    )
    _OPTIONAL = frozenset({"description", "split"})

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
        split = obj.get("split", "development")
        if not isinstance(split, str) or split not in TASK_SPLITS:
            raise SchemaError(f"{ctx}: 'split' must be one of {sorted(TASK_SPLITS)}, got {split!r}")
        return cls(
            task_id=task_id,
            category=_req_str(obj, "category", ctx, slug=True),
            prompt=_req_str(obj, "prompt", ctx),
            fixture=fixture,
            verifiers=verifiers,
            timeout_seconds=_req_number(obj, "timeout_seconds", ctx, minimum=1, maximum=3600),
            critical=_req_bool(obj, "critical", ctx),
            description=description,
            split=split,
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
        if self.split != "development":
            d["split"] = self.split
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


# Ingestion caps: a usage report holds three scalar fields; a run result grows
# with suite size and verifier details. Both stay far under these bounds, and
# the bounded read means the cap also bounds harness memory use.
_MAX_USAGE_REPORT_BYTES = 65_536
_MAX_RUN_RESULT_BYTES = 10_000_000


def _read_bounded_strict_json(path: Path, max_bytes: int, ctx: str) -> Any:
    """Read one trust-boundary JSON document with fail-closed parsing.

    Bind the read to one regular-file inode, reject symlinks/special files,
    require two consecutive complete fd reads to agree, reject
    metadata-visible replacement/change, and cap bytes before strict
    UTF-8/JSON parsing. Duplicate keys, non-finite constants, and nesting
    overflow are typed failures rather than ambiguous values or raw
    exceptions. This captures a checked byte snapshot; it does not prove
    provenance or that the pathname remains unchanged after return.
    """

    def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError(f"{ctx} {path}: duplicate JSON key {key!r} (fail closed)")
            result[key] = value
        return result

    def _no_non_finite(constant: str) -> float:
        raise SchemaError(f"{ctx} {path}: non-finite JSON constant {constant!r} (fail closed)")

    def _state(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def _read_to_eof(handle) -> bytearray:
        encoded = bytearray()
        while len(encoded) <= max_bytes:
            chunk = handle.read(max_bytes + 1 - len(encoded))
            if not chunk:
                break
            encoded.extend(chunk)
        return encoded

    def _digest_to_eof(handle) -> tuple[int, bytes]:
        digest = hashlib.sha256()
        total = 0
        while total <= max_bytes:
            chunk = handle.read(min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        return total, digest.digest()

    try:
        nonblock = getattr(os, "O_NONBLOCK", 0)
        if not nonblock and hasattr(os, "mkfifo"):
            raise SchemaError(
                f"cannot safely read {ctx}: nonblocking regular-file open is unavailable"
            )
        named_before = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(named_before.st_mode):
            raise SchemaError(f"{ctx} must not be a symlink: {path}")
        if not stat.S_ISREG(named_before.st_mode):
            raise SchemaError(f"{ctx} is not a regular file: {path}")
        if named_before.st_size > max_bytes:
            raise SchemaError(f"{ctx} {path} exceeds {max_bytes} bytes (fail closed)")

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | nonblock
        descriptor = os.open(path, flags)
        try:
            handle = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            opened_before = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened_before.st_mode):
                raise SchemaError(f"{ctx} is not a regular file: {path}")
            if _state(opened_before) != _state(named_before):
                raise SchemaError(f"{ctx} changed before it could be read: {path}")
            if opened_before.st_size > max_bytes:
                raise SchemaError(f"{ctx} {path} exceeds {max_bytes} bytes (fail closed)")
            encoded = _read_to_eof(handle)
            if len(encoded) != opened_before.st_size:
                raise SchemaError(f"{ctx} changed or was incompletely read: {path}")

            # Metadata is filesystem-dependent, so independently confirm that
            # a second complete read through the same fd yields identical
            # bytes. This detects a same-inode rewrite between the reads even
            # when timestamp reporting is too coarse to expose the change.
            handle.seek(0)
            confirmed_size, confirmed_digest = _digest_to_eof(handle)
            if (
                confirmed_size != len(encoded)
                or confirmed_digest != hashlib.sha256(encoded).digest()
            ):
                raise SchemaError(f"{ctx} changed while it was being read: {path}")
            opened_after = os.fstat(handle.fileno())

        if len(encoded) > max_bytes:
            raise SchemaError(f"{ctx} {path} exceeds {max_bytes} bytes (fail closed)")
        named_after = os.stat(path, follow_symlinks=False)
        if (
            _state(opened_before) != _state(opened_after)
            or _state(named_after) != _state(opened_after)
            or not stat.S_ISREG(named_after.st_mode)
        ):
            raise SchemaError(f"{ctx} changed while it was being read: {path}")

        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_non_finite,
        )
    except SchemaError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as e:
        raise SchemaError(f"cannot read {ctx} {path}: {e}") from e


def load_usage_report(path: str | Path) -> UsageReport:
    """Parse a usage-report JSON file, failing closed on any malformation.

    The spend-accounting gate consumes this agent-written document, so the
    read is bounded and strict: a duplicate ``cost_usd`` key or a
    multi-gigabyte file is a hard error, never an ambiguity.
    """
    raw = _read_bounded_strict_json(Path(path), _MAX_USAGE_REPORT_BYTES, "usage report")
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
        if evidence:
            raise SchemaError(
                f"{ctx}: capability_evidence=true is only valid after an attested live "
                "evidence schema is implemented; schema v1 refuses it for every execution "
                "mode, including 'live', to prevent forged external JSON"
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


def load_run_result(path: str | Path) -> RunResult:
    """Parse a run-result JSON file, failing closed on any malformation.

    Run files feed the human comparison gate and the optimizer-feedback
    derivation, so they get the same bounded strict-JSON ingestion as suite
    and feedback documents.
    """
    raw = _read_bounded_strict_json(Path(path), _MAX_RUN_RESULT_BYTES, "run result")
    return RunResult.from_dict(raw)
