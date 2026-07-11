#!/usr/bin/env python3
"""Local smoke benchmark runner — proxy evidence only, NOT TBLite/YC-Bench.

This is the self-evolution repo's fallback runner, used by BenchmarkGate when
the hermes-agent checkout has no ``environments/benchmarks/run_bench.py``. It
runs real, deterministic local checks and reports honest pass/fail counts:

1. Skill-override validation (when ``--skill-overrides`` is given): every
   evolved skill must be a non-empty string within the size cap. A corrupt
   evolved artifact fails its check.
2. Python syntax checks over the target repo's source files: each file must
   compile. A broken checkout fails its checks.

It never fabricates success: check failures are reported in the JSON output
(reducing the score BenchmarkGate computes), and operational errors — missing
target, unreadable overrides, nothing to check — exit nonzero so the gate
fails closed.

CLI contract (matches BenchmarkGate's expectations):
    python run_bench.py --tasks N [--skill-overrides PATH] [--target DIR]
    stdout: JSON with at least {"passed": int, "failed": int}
    exit 0: checks ran (failures reported in JSON); nonzero: could not run

Stdlib-only on purpose: it must run from any cwd (BenchmarkGate invokes it
with cwd = the hermes-agent repo) without this package installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNNER_NAME = "self-evolution-local-smoke"

# Mirrors EvolutionConfig.max_skill_size; hardcoded so the runner stays
# importable without the evolution package on sys.path.
MAX_SKILL_BYTES = 50_000

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox"}


def _python_files(target: Path) -> list[Path]:
    """Deterministically ordered Python source files under the target."""
    return sorted(
        p for p in target.rglob("*.py") if not any(part in _SKIP_DIRS for part in p.parts)
    )


def _check_override(name: str, value: object) -> dict:
    if not isinstance(value, str) or not value.strip():
        detail = "override is empty or not a string"
        ok = False
    elif len(value.encode("utf-8", errors="replace")) > MAX_SKILL_BYTES:
        detail = f"override exceeds size cap ({MAX_SKILL_BYTES} bytes)"
        ok = False
    else:
        detail = "non-empty string within size cap"
        ok = True
    return {"name": f"skill-override:{name}", "ok": ok, "detail": detail}


def _check_syntax(path: Path) -> dict:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        compile(source, str(path), "exec")
        return {"name": f"py-syntax:{path.name}", "ok": True, "detail": str(path)}
    except SyntaxError as e:
        return {
            "name": f"py-syntax:{path.name}",
            "ok": False,
            "detail": f"{path}: {e.msg} (line {e.lineno})",
        }
    except OSError as e:
        return {"name": f"py-syntax:{path.name}", "ok": False, "detail": f"{path}: {e}"}


def run_smoke_checks(target: Path, max_tasks: int, overrides: dict | None) -> list[dict]:
    """Run up to ``max_tasks`` checks. Override checks come first so a corrupt
    evolved skill always surfaces even under a small task budget."""
    checks: list[dict] = []
    if overrides is not None:
        for name in sorted(overrides):
            if len(checks) >= max_tasks:
                return checks
            checks.append(_check_override(name, overrides[name]))
    for path in _python_files(target):
        if len(checks) >= max_tasks:
            break
        checks.append(_check_syntax(path))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=20, help="max checks to run")
    parser.add_argument("--skill-overrides", help="JSON file of {skill: text} to validate")
    parser.add_argument(
        "--target",
        default=".",
        help="repo to smoke-check (default: cwd, which BenchmarkGate sets to hermes-agent)",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).expanduser().resolve()
    if not target.is_dir():
        print(f"target is not a directory: {target}", file=sys.stderr)
        return 2
    if args.tasks < 1:
        print(f"--tasks must be >= 1, got {args.tasks}", file=sys.stderr)
        return 2

    overrides = None
    if args.skill_overrides:
        try:
            overrides = json.loads(Path(args.skill_overrides).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"cannot read skill overrides {args.skill_overrides}: {e}", file=sys.stderr)
            return 2
        if not isinstance(overrides, dict):
            print("skill overrides must be a JSON object", file=sys.stderr)
            return 2

    checks = run_smoke_checks(target, args.tasks, overrides)
    if not checks:
        # No evidence either way — fail closed rather than report a vacuous pass.
        print(f"no checks could be run against {target}", file=sys.stderr)
        return 2

    passed = sum(1 for c in checks if c["ok"])
    print(
        json.dumps(
            {
                "passed": passed,
                "failed": len(checks) - passed,
                "runner": RUNNER_NAME,
                "proxy": True,
                "note": "smoke/proxy evidence only — not TBLite or YC-Bench",
                "target": str(target),
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
