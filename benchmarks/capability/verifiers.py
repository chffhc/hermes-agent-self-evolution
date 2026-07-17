"""Deterministic verifiers over final workspace state.

Each verifier inspects the disposable workspace a task ran in — never the
transcript, never keyword overlap, never an LLM judge. Two failure channels:

- ``VerifierConfigError``: the verifier itself is misconfigured (bad params,
  missing expected file). Raised, so suites fail validation up front and a
  run records an error instead of a silent pass.
- an ``ok=False`` outcome: the workspace does not satisfy the check.

Registry entries expose ``validate(params, task_dir)`` for suite-load-time
checks and ``run(workspace, task_dir, params, timeout_seconds)`` for
execution. Unknown verifier types fail suite validation closed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.capability.schema import (
    SchemaError,
    is_ignored_fixture_cache_path,
    safe_relative_path,
)


class VerifierConfigError(SchemaError):
    """Verifier params are invalid or reference missing task assets."""


@dataclass(frozen=True)
class VerifierOutcome:
    ok: bool
    detail: str


def _resolve_in(root: Path, rel: str, ctx: str) -> Path:
    """Resolve a validated relative path and confirm it stays under root."""
    safe_relative_path(rel, ctx)
    resolved = (root / rel).resolve()
    root_resolved = root.resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise VerifierConfigError(f"{ctx}: {rel!r} escapes {root}")
    return resolved


# --- file_exact -------------------------------------------------------------
# params: {"path": <workspace-relative>, "expected_file": <task-dir-relative>}


def _validate_file_exact(params: dict[str, Any], task_dir: Path) -> None:
    _check_param_keys(params, {"path", "expected_file"}, "file_exact")
    safe_relative_path(params.get("path"), "file_exact: path")
    expected_relative = safe_relative_path(params.get("expected_file"), "file_exact: expected_file")
    if expected_relative.parts[0] in {"workspace", "replay"} and is_ignored_fixture_cache_path(
        Path(*expected_relative.parts[1:])
    ):
        raise VerifierConfigError(
            "file_exact: expected_file may not reference a cache-excluded " "workspace/replay asset"
        )
    expected = _resolve_in(task_dir, str(expected_relative), "file_exact: expected_file")
    if not expected.is_file():
        raise VerifierConfigError(f"file_exact: expected_file missing: {expected}")


def _run_file_exact(
    workspace: Path, task_dir: Path, params: dict[str, Any], timeout_seconds: float
) -> VerifierOutcome:
    target = _resolve_in(workspace, params["path"], "file_exact: path")
    expected = _resolve_in(task_dir, params["expected_file"], "file_exact: expected_file")
    if not target.is_file():
        return VerifierOutcome(False, f"missing file: {params['path']}")
    if target.read_bytes() != expected.read_bytes():
        return VerifierOutcome(False, f"content mismatch: {params['path']}")
    return VerifierOutcome(True, f"exact match: {params['path']}")


# --- json_subset ------------------------------------------------------------
# params: {"path": <workspace-relative>, "expected": <JSON value>}
# Dicts: every expected key must be present and match (extra keys allowed).
# Lists: same length, elementwise subset match. Scalars: equality.


def _json_subset_match(expected: Any, actual: Any, path: str) -> str | None:
    """Return None on match, else a human-readable mismatch description."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected object, got {type(actual).__name__}"
        for key, sub in expected.items():
            if key not in actual:
                return f"{path}.{key}: missing key"
            mismatch = _json_subset_match(sub, actual[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}: expected array, got {type(actual).__name__}"
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} items, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            mismatch = _json_subset_match(e, a, f"{path}[{i}]")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return None if expected is actual else f"{path}: expected {expected!r}, got {actual!r}"
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def _validate_json_subset(params: dict[str, Any], task_dir: Path) -> None:
    _check_param_keys(params, {"path", "expected"}, "json_subset")
    safe_relative_path(params.get("path"), "json_subset: path")
    if "expected" not in params:
        raise VerifierConfigError("json_subset: 'expected' is required")
    try:
        json.dumps(params["expected"])
    except (TypeError, ValueError) as e:
        raise VerifierConfigError(f"json_subset: 'expected' not JSON-serializable: {e}") from e


def _run_json_subset(
    workspace: Path, task_dir: Path, params: dict[str, Any], timeout_seconds: float
) -> VerifierOutcome:
    target = _resolve_in(workspace, params["path"], "json_subset: path")
    if not target.is_file():
        return VerifierOutcome(False, f"missing file: {params['path']}")
    try:
        actual = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return VerifierOutcome(False, f"invalid JSON in {params['path']}: {e}")
    mismatch = _json_subset_match(params["expected"], actual, "$")
    if mismatch:
        return VerifierOutcome(False, mismatch)
    return VerifierOutcome(True, f"JSON subset match: {params['path']}")


# --- json_exact -------------------------------------------------------------
# params: {"path": <workspace-relative>, "expected": <JSON value>}
# Objects must have exactly the expected keys; scalar JSON types are strict.


def _validate_json_exact(params: dict[str, Any], task_dir: Path) -> None:
    _check_param_keys(params, {"path", "expected"}, "json_exact")
    safe_relative_path(params.get("path"), "json_exact: path")
    if "expected" not in params:
        raise VerifierConfigError("json_exact: 'expected' is required")
    try:
        json.dumps(params["expected"], allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VerifierConfigError(f"json_exact: 'expected' not JSON-serializable: {exc}") from exc


def _json_exact_mismatch(expected: Any, actual: Any, path: str) -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: expected type {type(expected).__name__}, " f"got {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}: key mismatch; missing={sorted(expected_keys - actual_keys)} "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key, sub in expected.items():
            mismatch = _json_exact_mismatch(sub, actual[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            mismatch = _json_exact_mismatch(expected_item, actual_item, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    return None if expected == actual else f"{path}: expected {expected!r}, got {actual!r}"


def _load_json_without_duplicate_keys(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _run_json_exact(
    workspace: Path, task_dir: Path, params: dict[str, Any], timeout_seconds: float
) -> VerifierOutcome:
    target = _resolve_in(workspace, params["path"], "json_exact: path")
    if not target.is_file():
        return VerifierOutcome(False, f"missing file: {params['path']}")
    try:
        actual = _load_json_without_duplicate_keys(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return VerifierOutcome(False, f"invalid JSON in {params['path']}: {exc}")
    mismatch = _json_exact_mismatch(params["expected"], actual, "$")
    if mismatch:
        return VerifierOutcome(False, mismatch)
    return VerifierOutcome(True, f"exact JSON match: {params['path']}")


# --- command_exit -----------------------------------------------------------
# params: {"argv": ["python", ...], "expected_exit": 0}
# argv[0] must be the literal sentinel "python" (replaced with the current
# interpreter) so suites cannot smuggle in arbitrary host binaries. Runs with
# cwd=workspace, a scrubbed environment, no shell, and the task timeout.


def _validate_command_exit(params: dict[str, Any], task_dir: Path) -> None:
    _check_param_keys(params, {"argv", "expected_exit"}, "command_exit", optional={"expected_exit"})
    argv = params.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 2
        or not all(isinstance(a, str) and a for a in argv)
    ):
        raise VerifierConfigError("command_exit: 'argv' must be a non-empty list of strings")
    if argv[0] != "python":
        raise VerifierConfigError(
            f"command_exit: argv[0] must be the 'python' sentinel, got {argv[0]!r}"
        )
    if any("\x00" in a for a in argv):
        raise VerifierConfigError("command_exit: NUL byte in argv")
    script = safe_relative_path(argv[1], "command_exit: script")
    if is_ignored_fixture_cache_path(Path(script)):
        raise VerifierConfigError(
            "command_exit: script may not reference a cache-excluded workspace asset"
        )
    if script.suffix != ".py":
        raise VerifierConfigError("command_exit: argv[1] must be a workspace-relative .py file")
    script_path = _resolve_in(task_dir / "workspace", str(script), "command_exit: script")
    if not script_path.is_file():
        raise VerifierConfigError(f"command_exit: script missing from fixture workspace: {script}")
    expected = params.get("expected_exit", 0)
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise VerifierConfigError("command_exit: 'expected_exit' must be an integer")


def _run_command_exit(
    workspace: Path, task_dir: Path, params: dict[str, Any], timeout_seconds: float
) -> VerifierOutcome:
    argv = [sys.executable] + list(params["argv"][1:])
    expected = params.get("expected_exit", 0)
    env = {
        "PATH": os.defpath,
        "HOME": str(workspace),
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        proc = subprocess.run(
            argv,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return VerifierOutcome(False, f"command timed out after {timeout_seconds}s")
    if proc.returncode != expected:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return VerifierOutcome(False, f"exit {proc.returncode} (expected {expected}): {tail}")
    return VerifierOutcome(True, f"exit {proc.returncode} as expected")


# --- protected_unchanged ----------------------------------------------------
# params: {"paths": [<workspace-relative>, ...]}
# Anti-leakage invariant: listed files must be byte-identical to the fixture
# originals (e.g. a code-repair task's test file must not be edited to pass).


def _validate_protected_unchanged(params: dict[str, Any], task_dir: Path) -> None:
    _check_param_keys(params, {"paths"}, "protected_unchanged")
    paths = params.get("paths")
    if not isinstance(paths, list) or not paths:
        raise VerifierConfigError("protected_unchanged: 'paths' must be a non-empty list")
    for p in paths:
        relative = safe_relative_path(p, "protected_unchanged: paths")
        if is_ignored_fixture_cache_path(Path(relative)):
            raise VerifierConfigError(
                "protected_unchanged: paths may not reference cache-excluded assets"
            )
        original = _resolve_in(task_dir / "workspace", str(relative), "protected_unchanged: paths")
        if not original.is_file():
            raise VerifierConfigError(
                f"protected_unchanged: {p!r} does not exist in the fixture workspace"
            )


def _run_protected_unchanged(
    workspace: Path, task_dir: Path, params: dict[str, Any], timeout_seconds: float
) -> VerifierOutcome:
    for p in params["paths"]:
        original = _resolve_in(task_dir / "workspace", p, "protected_unchanged: paths")
        current = _resolve_in(workspace, p, "protected_unchanged: paths")
        if not current.is_file():
            return VerifierOutcome(False, f"protected file deleted: {p}")
        if current.read_bytes() != original.read_bytes():
            return VerifierOutcome(False, f"protected file modified: {p}")
    return VerifierOutcome(True, f"{len(params['paths'])} protected file(s) unchanged")


# --- registry ---------------------------------------------------------------


def _check_param_keys(
    params: dict[str, Any], allowed: set, name: str, optional: set | None = None
) -> None:
    if not isinstance(params, dict):
        raise VerifierConfigError(f"{name}: params must be an object")
    unknown = set(params) - allowed
    if unknown:
        raise VerifierConfigError(f"{name}: unknown params {sorted(unknown)}")
    missing = allowed - (optional or set()) - set(params)
    if missing:
        raise VerifierConfigError(f"{name}: missing params {sorted(missing)}")


@dataclass(frozen=True)
class VerifierType:
    name: str
    validate: Callable[[dict[str, Any], Path], None]
    run: Callable[[Path, Path, dict[str, Any], float], VerifierOutcome]


VERIFIERS: dict[str, VerifierType] = {
    v.name: v
    for v in (
        VerifierType("file_exact", _validate_file_exact, _run_file_exact),
        VerifierType("json_subset", _validate_json_subset, _run_json_subset),
        VerifierType("json_exact", _validate_json_exact, _run_json_exact),
        VerifierType("command_exit", _validate_command_exit, _run_command_exit),
        VerifierType(
            "protected_unchanged", _validate_protected_unchanged, _run_protected_unchanged
        ),
    )
}
