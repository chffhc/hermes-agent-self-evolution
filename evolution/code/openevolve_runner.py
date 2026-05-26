"""OpenEvolve adapter for Phase 4 code evolution.

This module intentionally treats OpenEvolve as an external engine and runs it in an
isolated scratch directory. The production Hermes repo is never passed as the
working directory and never edited in place; callers receive artifacts and a
reviewable patch instead.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True)
class OpenEvolveRunnerConfig:
    """Configuration for a single isolated OpenEvolve run."""

    initial_program: Path
    evaluator: Path
    config_file: Path | None = None
    iterations: int = 3
    output_root: Path = Path("/tmp")
    openevolve_cmd: str = "openevolve-run"
    timeout_seconds: int = 3600
    env: dict[str, str] = field(default_factory=dict)
    unset_proxy_env: bool = True
    keep_workdir: bool = True


@dataclass(frozen=True)
class OpenEvolveRunResult:
    """Artifacts produced by an isolated OpenEvolve run."""

    success: bool
    command: list[str]
    returncode: int
    work_dir: Path
    output_dir: Path
    baseline_program: Path
    evaluator: Path
    config_file: Path | None
    best_program: Path | None
    best_info: Path | None
    best_metrics: dict[str, Any]
    patch_text: str
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def improved(self) -> bool:
        """Whether OpenEvolve produced a non-empty patch."""
        return bool(self.patch_text.strip())


def _copy_required_file(path: Path, destination: Path) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Required file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _build_env(extra_env: dict[str, str], unset_proxy_env: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra_env)
    if unset_proxy_env:
        for key in _PROXY_ENV_VARS:
            env.pop(key, None)
    return env


def _read_metrics(best_info: Path | None) -> dict[str, Any]:
    if best_info is None or not best_info.exists():
        return {}
    try:
        payload = json.loads(best_info.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _make_patch(original: Path, evolved: Path | None) -> str:
    if evolved is None or not evolved.exists():
        return ""
    original_lines = original.read_text(encoding="utf-8").splitlines(keepends=True)
    evolved_lines = evolved.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            original_lines,
            evolved_lines,
            fromfile=str(original.name),
            tofile="best_program.py",
        )
    )


def run_openevolve_isolated(config: OpenEvolveRunnerConfig) -> OpenEvolveRunResult:
    """Run OpenEvolve in a scratch directory and return patch-only artifacts.

    The function copies `initial_program`, `evaluator`, and optionally
    `config_file` into a newly-created temporary work directory, executes
    OpenEvolve there, and reads artifacts from `<work_dir>/openevolve-out`.
    The original input files are never modified.
    """

    if config.iterations < 0:
        raise ValueError("iterations must be >= 0")

    output_root = Path(config.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="hermes-openevolve-", dir=str(output_root)))

    baseline_program = _copy_required_file(
        config.initial_program, work_dir / Path(config.initial_program).name
    )
    evaluator = _copy_required_file(config.evaluator, work_dir / Path(config.evaluator).name)
    copied_config = None
    if config.config_file is not None:
        copied_config = _copy_required_file(config.config_file, work_dir / Path(config.config_file).name)

    output_dir = work_dir / "openevolve-out"
    command = [
        config.openevolve_cmd,
        str(baseline_program.name),
        str(evaluator.name),
        "--output",
        str(output_dir),
        "--iterations",
        str(config.iterations),
    ]
    if copied_config is not None:
        command.extend(["--config", str(copied_config.name)])

    env = _build_env(config.env, config.unset_proxy_env)

    try:
        proc = subprocess.run(
            command,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        error = None
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        error = f"OpenEvolve timed out after {config.timeout_seconds}s"
    except FileNotFoundError as exc:
        returncode = 127
        stdout = ""
        stderr = str(exc)
        error = f"OpenEvolve command not found: {config.openevolve_cmd}"

    best_program = output_dir / "best" / "best_program.py"
    best_info = output_dir / "best" / "best_program_info.json"
    best_program_path = best_program if best_program.exists() else None
    best_info_path = best_info if best_info.exists() else None
    patch_text = _make_patch(baseline_program, best_program_path)
    best_metrics = _read_metrics(best_info_path)
    success = returncode == 0 and best_program_path is not None and best_info_path is not None

    if not config.keep_workdir and not success:
        shutil.rmtree(work_dir, ignore_errors=True)

    return OpenEvolveRunResult(
        success=success,
        command=command,
        returncode=returncode,
        work_dir=work_dir,
        output_dir=output_dir,
        baseline_program=baseline_program,
        evaluator=evaluator,
        config_file=copied_config,
        best_program=best_program_path,
        best_info=best_info_path,
        best_metrics=best_metrics,
        patch_text=patch_text,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )


__all__ = [
    "OpenEvolveRunnerConfig",
    "OpenEvolveRunResult",
    "run_openevolve_isolated",
]
