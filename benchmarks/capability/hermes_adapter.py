"""Current-Hermes CLI adapter foundation for the capability benchmark.

The concrete integration seam, discovered against Hermes 0.18.2
(read-only checkout, see docs/CAPABILITY_BENCHMARK.md for citations):

- ``cli.py:main`` single-query mode: ``--query <prompt> --quiet`` prints the
  final response to stdout, ``session_id: <id>`` to stderr, and exits 0/1.
- ``--skills <name>`` preloads ``$HERMES_HOME/skills/<name>/SKILL.md`` via
  ``agent/skill_commands.py:build_preloaded_skills_prompt``, embedding the
  skill body verbatim into the session system prompt; when every requested
  skill is missing the CLI hard-fails (fail closed).
- ``hermes_constants.py:get_hermes_home`` honors the ``HERMES_HOME`` env var,
  isolating skills, config, and ``state.db`` per invocation;
  ``tools/terminal_tool.py`` honors ``TERMINAL_CWD`` for tool cwd scoping.
- ``hermes_state.py`` persists per-session attribution in
  ``$HERMES_HOME/state.db``: ``sessions`` carries ``system_prompt``, token
  counters, ``estimated_cost_usd``/``cost_status``/``cost_source``, ``model``,
  and ``cwd``; ``messages`` carries the trajectory.

The first live track is deliberately narrow: exactly one artifact type —
a skill markdown directory — with a documented injection contract and a
post-run consumption proof (the skill body must appear in the persisted
system prompt). Generic artifact injection into current Hermes is NOT
pretended to work.

Evidence honesty: the bundled stub mode (``hermes_cli_stub``) emulates this
contract locally and is permanently ``capability_evidence=False``. The live
argv/state contract is implemented, but construction is intentionally blocked
before launch until current Hermes gains pre-spend USD enforcement and real
filesystem confinement; see docs/CAPABILITY_BENCHMARK.md.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from benchmarks.capability.executor import (
    HERMES_CLI_STUB_MODE,
    LIVE_MODE,
    AgentInvocation,
    InvocationOutcome,
)
from benchmarks.capability.live_gate import structural_live_blockers
from benchmarks.capability.replay import digest_artifact
from benchmarks.capability.schema import SchemaError, utc_now_iso

HERMES_CLI_STUB_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "hermes_cli_stub.py"

LIVE_CONFIRM_PHRASE = "I-UNDERSTAND-LIVE-HERMES-SPENDS-REAL-MONEY"

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SESSION_LINE_RE = re.compile(r"^session_id:\s*(\S+)\s*$", re.MULTILINE)
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)

_MIN_SKILL_BODY_CHARS = 40
_TRAJECTORY_MESSAGE_CAP = 1000
_TRAJECTORY_CONTENT_CHARS = 10_000

# Structural blockers, not transient setup errors. Current Hermes can report
# spend only after an invocation, and TERMINAL_CWD scopes the default directory
# without confining absolute filesystem access. Paid execution stays fail-closed
# until both invariants have real enforcement. The strings come from the
# static live_gate requirement definitions; no runtime probe result can
# shrink this tuple (see benchmarks/capability/live_gate.py).
_LIVE_EXECUTION_BLOCKERS = structural_live_blockers()

# Every invariant the adapter depends on, checked as literal markers against
# the read-only checkout. A missing marker means the seam moved and the
# adapter must fail closed instead of guessing.
_PROBE_SPECS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "package-identity",
        "pyproject.toml",
        ('name = "hermes-agent"',),
        "pyproject.toml [project] name",
    ),
    (
        "single-query-cli",
        "cli.py",
        ("def main(", "quiet: bool = False", 'session_id: {cli.session_id}"'),
        "cli.py:main single-query quiet path printing 'session_id:' to stderr",
    ),
    (
        "skill-preload-flag",
        "cli.py",
        ("build_preloaded_skills_prompt(", "Unknown skill(s)"),
        "cli.py:main --skills preload with hard failure on unknown skills",
    ),
    (
        "skill-prompt-builder",
        "agent/skill_commands.py",
        ("def build_preloaded_skills_prompt(", "def _build_skill_message("),
        "agent/skill_commands.py skill body embedded into the system prompt",
    ),
    (
        "hermes-home-env",
        "hermes_constants.py",
        ("def get_hermes_home", '"HERMES_HOME"'),
        "hermes_constants.py:get_hermes_home HERMES_HOME env override",
    ),
    (
        "skills-dir-contract",
        "tools/skills_tool.py",
        ('SKILLS_DIR = HERMES_HOME / "skills"',),
        "tools/skills_tool.py SKILLS_DIR under HERMES_HOME",
    ),
    (
        "session-store",
        "hermes_state.py",
        (
            '"state.db"',
            "system_prompt TEXT",
            "input_tokens INTEGER",
            "estimated_cost_usd REAL",
            "cost_status TEXT",
            "cwd TEXT",
        ),
        "hermes_state.py state.db sessions schema with usage/cost/prompt columns",
    ),
    (
        "terminal-cwd-scoping",
        "tools/terminal_tool.py",
        ("TERMINAL_CWD",),
        "tools/terminal_tool.py TERMINAL_CWD working-directory pin",
    ),
    (
        "oneshot-usage-report",
        "hermes_cli/oneshot.py",
        ("def run_oneshot(", '"estimated_cost_usd"', '"cost_status"'),
        "hermes_cli/oneshot.py -z --usage-file JSON usage contract (no --skills)",
    ),
)


@dataclass(frozen=True)
class InvariantCheck:
    name: str
    ok: bool
    evidence: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "evidence": self.evidence, "detail": self.detail}


@dataclass(frozen=True)
class HermesCompatibilityReport:
    """Fail-closed probe result over a read-only current-Hermes checkout."""

    checkout: str
    version: str | None
    checks: tuple[InvariantCheck, ...]

    @property
    def compatible(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(f"{c.name}: {c.detail} [{c.evidence}]" for c in self.checks if not c.ok)

    @property
    def live_executable(self) -> bool:
        """Whether a paid run can meet the benchmark's hard safety contract."""
        return self.compatible and not _LIVE_EXECUTION_BLOCKERS

    @property
    def live_blockers(self) -> tuple[str, ...]:
        if not self.compatible:
            return self.blockers
        return _LIVE_EXECUTION_BLOCKERS

    def to_dict(self) -> dict[str, object]:
        return {
            "checkout": self.checkout,
            "hermes_version": self.version,
            "compatible": self.compatible,
            "blockers": list(self.blockers),
            "live_executable": self.live_executable,
            "live_blockers": list(self.live_blockers),
            "checks": [c.to_dict() for c in self.checks],
            "capability_evidence": False,
        }


def probe_hermes_checkout(checkout: str | Path) -> HermesCompatibilityReport:
    """Verify the exact CLI/skill/state.db invariants this adapter relies on."""
    root = Path(checkout)
    if not root.is_dir():
        raise SchemaError(f"hermes checkout is not a directory: {root}")
    checks: list[InvariantCheck] = []
    for name, rel, markers, evidence in _PROBE_SPECS:
        target = root / rel
        if not target.is_file():
            checks.append(InvariantCheck(name, False, evidence, f"missing file {rel}"))
            continue
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            checks.append(InvariantCheck(name, False, evidence, f"unreadable {rel}: {exc}"))
            continue
        absent = [marker for marker in markers if marker not in text]
        if absent:
            checks.append(
                InvariantCheck(name, False, evidence, f"{rel} lost expected marker {absent[0]!r}")
            )
        else:
            checks.append(InvariantCheck(name, True, evidence, "present"))
    version: str | None = None
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        match = _VERSION_RE.search(pyproject.read_text(encoding="utf-8", errors="replace"))
        version = match.group(1) if match else None
    return HermesCompatibilityReport(checkout=str(root), version=version, checks=tuple(checks))


@dataclass(frozen=True)
class SkillArtifact:
    """A validated skill-markdown directory — the only live artifact type."""

    path: Path
    name: str
    description: str
    body: str
    digest: str


def _parse_skill_frontmatter(text: str, ctx: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise SchemaError(f"{ctx}: SKILL.md must start with a '---' frontmatter fence")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise SchemaError(f"{ctx}: SKILL.md frontmatter fence is not closed")
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if sep and not key.startswith((" ", "\t")):
            fields[key.strip()] = value.strip().strip("'\"")
    return fields, text[end + 5 :]


def validate_skill_artifact(artifact_path: str | Path) -> SkillArtifact:
    """Fail-closed validation of the documented skill injection contract.

    Requirements mirror what current Hermes actually loads
    (``$HERMES_HOME/skills/<name>/SKILL.md`` with name/description
    frontmatter), plus benchmark-specific constraints: no symlinks, no
    ``{{`` template tokens (Hermes substitutes them before embedding, which
    would break the byte-exact consumption proof), and a body long enough
    that a substring match against the system prompt is meaningful.
    """
    path = Path(artifact_path)
    ctx = f"skill artifact {path}"
    if path.is_symlink() or not path.is_dir():
        raise SchemaError(f"{ctx}: must be a real directory (the skill dir), not a symlink/file")
    digest = digest_artifact(path)  # also rejects symlinks anywhere inside
    skill_md = path / "SKILL.md"
    if skill_md.is_symlink() or not skill_md.is_file():
        raise SchemaError(f"{ctx}: missing SKILL.md")
    try:
        text = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SchemaError(f"{ctx}: cannot read SKILL.md: {exc}") from exc
    if "\x00" in text:
        raise SchemaError(f"{ctx}: SKILL.md contains NUL bytes")
    fields, body = _parse_skill_frontmatter(text, ctx)
    name = fields.get("name", "")
    if not _SKILL_NAME_RE.match(name):
        raise SchemaError(f"{ctx}: frontmatter 'name' must match {_SKILL_NAME_RE.pattern}")
    if name != path.name:
        raise SchemaError(
            f"{ctx}: frontmatter name {name!r} must equal the artifact directory name "
            f"{path.name!r} (Hermes resolves skills/<name>/SKILL.md)"
        )
    description = fields.get("description", "")
    if not description or len(description) > 1024:
        raise SchemaError(f"{ctx}: frontmatter 'description' required, max 1024 chars")
    body = body.strip()
    if len(body) < _MIN_SKILL_BODY_CHARS:
        raise SchemaError(
            f"{ctx}: skill body must be at least {_MIN_SKILL_BODY_CHARS} chars so the "
            "system-prompt consumption proof is meaningful"
        )
    if "{{" in body:
        raise SchemaError(
            f"{ctx}: '{{{{' template tokens are not allowed — Hermes template-var "
            "substitution would alter the body before embedding and break the "
            "consumption proof"
        )
    return SkillArtifact(path=path, name=name, description=description, body=body, digest=digest)


@dataclass(frozen=True)
class LiveExecutionApproval:
    """Explicit, default-deny authorization for a paid live Hermes run.

    ``env_passthrough`` lists environment variable NAMES (typically provider
    API keys) copied by name from the parent environment into the scrubbed
    subprocess environment; values are never logged or recorded.
    """

    confirm: str
    max_run_usd: float
    max_task_usd: float
    env_passthrough: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.confirm != LIVE_CONFIRM_PHRASE:
            raise SchemaError(
                "live execution is default-deny: confirmation must be the exact phrase "
                f"{LIVE_CONFIRM_PHRASE!r}"
            )
        usd = self.max_run_usd
        if isinstance(usd, bool) or not isinstance(usd, (int, float)):
            raise SchemaError("live approval: max_run_usd must be a number")
        if not math.isfinite(float(usd)) or float(usd) <= 0:
            raise SchemaError("live approval: max_run_usd must be finite and > 0")
        task_usd = self.max_task_usd
        if isinstance(task_usd, bool) or not isinstance(task_usd, (int, float)):
            raise SchemaError("live approval: max_task_usd must be a number")
        if not math.isfinite(float(task_usd)) or float(task_usd) <= 0:
            raise SchemaError("live approval: max_task_usd must be finite and > 0")
        if float(task_usd) > float(usd):
            raise SchemaError("live approval: max_task_usd cannot exceed max_run_usd")
        for name in self.env_passthrough:
            if not isinstance(name, str) or not _ENV_NAME_RE.match(name):
                raise SchemaError(f"live approval: invalid env passthrough name {name!r}")


def _require_live_python(checkout: Path) -> Path:
    for rel in ("venv/bin/python3", "venv/bin/python"):
        candidate = checkout / rel
        if candidate.is_file():
            return candidate
    raise SchemaError(
        f"hermes checkout {checkout} has no venv/bin/python3 — cannot derive a trusted "
        "interpreter for live invocation (fail closed)"
    )


@dataclass(frozen=True)
class HermesCliInvoker:
    """AgentInvoker for the current-Hermes single-query CLI contract.

    Never constructed with an arbitrary executable: stub mode is pinned to
    the bundled emulator script, and live mode requires a passing
    compatibility probe with argv derived from the validated checkout.
    """

    artifact: SkillArtifact
    execution_mode: str
    argv_head: tuple[str, ...]
    expected_model: str
    toolsets: tuple[str, ...] = ("terminal",)
    max_turns: int = 20
    stub_behavior: tuple[str, ...] = ()
    provider: str | None = None
    env_passthrough: tuple[str, ...] = ()
    compatibility: HermesCompatibilityReport | None = field(default=None, repr=False)
    checkout: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, SkillArtifact):
            raise SchemaError("invoker: artifact must be a validated SkillArtifact")
        if not self.expected_model or not isinstance(self.expected_model, str):
            raise SchemaError("invoker: expected_model must be a non-empty string")
        if not self.toolsets or not all(
            isinstance(t, str) and re.fullmatch(r"[a-z0-9_-]+", t) for t in self.toolsets
        ):
            raise SchemaError("invoker: toolsets must be non-empty lowercase slugs")
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int):
            raise SchemaError("invoker: max_turns must be an integer")
        if not 1 <= self.max_turns <= 100:
            raise SchemaError("invoker: max_turns must be between 1 and 100")
        if self.execution_mode == HERMES_CLI_STUB_MODE:
            if self.argv_head != ("python", str(HERMES_CLI_STUB_SCRIPT)):
                raise SchemaError(
                    "invoker: hermes_cli_stub mode is pinned to the bundled stub script; "
                    "arbitrary executables are rejected (fail closed)"
                )
            if not HERMES_CLI_STUB_SCRIPT.is_file():
                raise SchemaError(f"bundled stub script missing: {HERMES_CLI_STUB_SCRIPT}")
            if self.env_passthrough:
                raise SchemaError("invoker: env passthrough is live-only; stubs get no secrets")
        elif self.execution_mode == LIVE_MODE:
            if self.compatibility is None or not self.compatibility.compatible:
                blockers = self.compatibility.blockers if self.compatibility else ("no probe run",)
                raise SchemaError(
                    "invoker: live mode requires a passing compatibility probe; blockers: "
                    + "; ".join(blockers)
                )
            if not self.compatibility.live_executable:
                raise SchemaError(
                    "invoker: live mode is blocked by structural safety requirements: "
                    + "; ".join(self.compatibility.live_blockers)
                )
            if not self.checkout or self.compatibility.checkout != self.checkout:
                raise SchemaError("invoker: live checkout must match the probed checkout")
            checkout = Path(self.checkout)
            head = tuple(str(a) for a in self.argv_head)
            if len(head) != 2 or head[1] != str(checkout / "cli.py"):
                raise SchemaError("invoker: live argv must target <checkout>/cli.py")
            interpreter = Path(head[0])
            if not interpreter.is_file() or not interpreter.is_relative_to(checkout):
                raise SchemaError(
                    "invoker: live interpreter must be the probed checkout's venv python "
                    "(fail closed; arbitrary executables are rejected)"
                )
            if self.stub_behavior:
                raise SchemaError("invoker: stub behavior flags are invalid in live mode")
            for name in self.env_passthrough:
                if not _ENV_NAME_RE.match(name):
                    raise SchemaError(f"invoker: invalid env passthrough name {name!r}")
        else:
            raise SchemaError(f"invoker: unsupported hermes execution_mode {self.execution_mode!r}")

    @property
    def is_live(self) -> bool:
        return self.execution_mode == LIVE_MODE

    @property
    def artifact_digest(self) -> str:
        return self.artifact.digest

    def fingerprint_config(self) -> dict[str, object]:
        """Exact behavior knobs applied by this adapter and bound into results."""
        return {
            "adapter": "current-hermes-cli-v1",
            "max_turns": self.max_turns,
            "provider": self.provider or "auto",
            "toolsets": list(self.toolsets),
        }

    # ── invocation ──

    def _install_skill(self, hermes_home: Path) -> None:
        dest = hermes_home / "skills" / self.artifact.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.artifact.path, dest, symlinks=False)
        if digest_artifact(dest) != self.artifact.digest:
            raise SchemaError(f"skill install corrupted artifact content at {dest}")

    def _build_argv(self, invocation: AgentInvocation) -> list[str]:
        argv = [
            sys.executable if self.argv_head[0] == "python" else self.argv_head[0],
            self.argv_head[1],
            "--query",
            invocation.prompt,
            "--quiet",
            "--skills",
            self.artifact.name,
            "--toolsets",
            ",".join(self.toolsets),
            "--model",
            self.expected_model,
            "--max_turns",
            str(self.max_turns),
        ]
        if self.is_live:
            argv += ["--ignore_user_config", "--ignore_rules"]
            if self.provider:
                argv += ["--provider", self.provider]
        else:
            for flag in self.stub_behavior:
                argv.append(flag.replace("{task_fixture_dir}", str(invocation.task_fixture_dir)))
        return argv

    def _build_env(self, invocation: AgentInvocation, hermes_home: Path) -> dict[str, str]:
        env = {
            "PATH": os.defpath,
            "HOME": str(invocation.workspace),
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "HERMES_BENCH_RUN_ID": invocation.run_id,
            "HERMES_BENCH_TASK_ID": invocation.task_id,
            "HERMES_HOME": str(hermes_home),
            "TERMINAL_CWD": str(invocation.workspace),
        }
        for name in self.env_passthrough:
            value = os.environ.get(name)
            if value is not None:
                env[name] = value
        return env

    def invoke(self, invocation: AgentInvocation) -> InvocationOutcome:
        hermes_home = invocation.control_dir / "hermes_home"
        hermes_home.mkdir(parents=True)
        self._install_skill(hermes_home)
        argv = self._build_argv(invocation)
        env = self._build_env(invocation, hermes_home)
        record = {
            "run_id": invocation.run_id,
            "task_id": invocation.task_id,
            "argv": argv,
            "env_keys": sorted(env),  # names only; values are never recorded
            "execution_mode": self.execution_mode,
            "capability_evidence": False,
            "artifact_digest": self.artifact.digest,
            "skill_name": self.artifact.name,
            "hermes_home": str(hermes_home),
            "started_at": utc_now_iso(),
        }
        _write_json(invocation.control_dir / "invocation.json", record)

        # Hermes spawns worker threads and child processes; run it in its own
        # process group so a timeout kills the whole tree, not just the leader.
        proc = subprocess.Popen(
            argv,
            cwd=str(invocation.workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=invocation.timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
            (invocation.control_dir / "stdout.txt").write_text(stdout[-10_000:], encoding="utf-8")
            (invocation.control_dir / "stderr.txt").write_text(stderr[-10_000:], encoding="utf-8")
            # Timeout does not imply zero spend: Hermes may have committed a
            # billable session before hanging. Preserve the killed process's
            # output and attribute any valid state before returning failure.
            try:
                attribution_failure = self._attribute(invocation, hermes_home, stderr)
            except Exception as exc:  # usage may already have been written
                attribution_failure = f"post-timeout attribution error: {type(exc).__name__}: {exc}"
            detail = f"hermes invocation timed out after {invocation.timeout_seconds}s"
            if attribution_failure:
                detail += f"; attribution: {attribution_failure}"
            return InvocationOutcome(
                exit_code=None,
                timed_out=True,
                detail=detail,
            )
        (invocation.control_dir / "stdout.txt").write_text(stdout[-10_000:], encoding="utf-8")
        (invocation.control_dir / "stderr.txt").write_text(stderr[-10_000:], encoding="utf-8")
        if proc.returncode != 0:
            # A failed invocation may still have reached the provider and
            # persisted billable usage. Attribute it when possible before
            # returning the process failure.
            attribution_failure = self._attribute(invocation, hermes_home, stderr)
            tail = (stderr or stdout or "").strip()[-400:]
            if attribution_failure:
                tail = (
                    f"{tail}; attribution: {attribution_failure}" if tail else attribution_failure
                )
            return InvocationOutcome(exit_code=proc.returncode, timed_out=False, detail=tail)

        failure = self._attribute(invocation, hermes_home, stderr)
        if failure is not None:
            return InvocationOutcome(exit_code=0, timed_out=False, failure=failure)
        return InvocationOutcome(exit_code=0, timed_out=False)

    # ── post-run attribution: session, consumption proof, usage ──

    def _attribute(self, invocation: AgentInvocation, hermes_home: Path, stderr: str) -> str | None:
        """Extract and verify attribution; return a fail-closed error or None."""
        matches = _SESSION_LINE_RE.findall(stderr or "")
        if not matches:
            return "hermes did not report a session_id on stderr (fail closed)"
        session_id = matches[-1]
        db_path = hermes_home / "state.db"
        if db_path.is_symlink() or not db_path.is_file():
            return f"hermes session store missing: {db_path} (fail closed)"
        try:
            session, trajectory = _read_session(db_path, session_id)
        except sqlite3.Error as exc:
            return f"cannot read hermes session store: {exc} (fail closed)"
        if session is None:
            return f"session {session_id!r} not found in state.db (fail closed)"

        # Persist attributable spend before checking the remaining evidence
        # invariants. A model mismatch, missing skill proof, or cwd violation
        # does not make an already-billed invocation free.
        usage_error = _validate_usage(session)
        if usage_error is not None:
            return usage_error
        invocation.usage_file.write_text(
            json.dumps(
                {
                    "cost_usd": float(cast(int | float, session["estimated_cost_usd"])),
                    "input_tokens": cast(int, session["input_tokens"]),
                    "output_tokens": cast(int, session["output_tokens"]),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if session["source"] != "cli":
            return f"session source {session['source']!r} is not 'cli' (fail closed)"
        # Compare resolved paths: the OS may report a physical cwd for a
        # symlinked temp root (macOS /var -> /private/var).
        session_cwd = session["cwd"]
        try:
            cwd_bound = isinstance(session_cwd, str) and (
                Path(session_cwd).resolve() == invocation.workspace.resolve()
            )
        except OSError:
            cwd_bound = False
        if not cwd_bound:
            return (
                f"session cwd {session_cwd!r} is not the isolated task workspace "
                "(fail closed: run not proven bound to the injected workspace)"
            )
        if session["model"] != self.expected_model:
            return (
                f"session model {session['model']!r} != expected {self.expected_model!r} "
                "(fail closed: fingerprint would be violated, e.g. by model fallback)"
            )
        system_prompt = session["system_prompt"]
        if not isinstance(system_prompt, str) or self.artifact.body not in system_prompt:
            return (
                f"candidate skill {self.artifact.name!r} not proven loaded: SKILL.md body "
                "is absent from the persisted session system prompt (fail closed)"
            )

        _write_json(invocation.control_dir / "session.json", {**session, "id": session_id})
        _write_json(invocation.control_dir / "trajectory.json", trajectory)
        _write_json(
            invocation.control_dir / "attestation.json",
            {
                "run_id": invocation.run_id,
                "task_id": invocation.task_id,
                "session_id": session_id,
                "execution_mode": self.execution_mode,
                "capability_evidence": False,
                "skill_name": self.artifact.name,
                "artifact_digest": self.artifact.digest,
                "consumption_proof": "SKILL.md body found in sessions.system_prompt",
                "usage_source": "state.db sessions row (estimated_cost_usd)",
                "model": session["model"],
                "cost_status": session["cost_status"],
                "cost_source": session["cost_source"],
            },
        )
        return None


_SESSION_COLUMNS = (
    "source",
    "model",
    "cwd",
    "system_prompt",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "cost_status",
    "cost_source",
)


def _read_session(
    db_path: Path, session_id: str
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            f"SELECT {', '.join(_SESSION_COLUMNS)} FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None, []
        session = dict(zip(_SESSION_COLUMNS, row, strict=True))
        trajectory = [
            {
                "role": role,
                "content": (content or "")[:_TRAJECTORY_CONTENT_CHARS],
                "timestamp": timestamp,
            }
            for role, content, timestamp in conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? "
                "ORDER BY id LIMIT ?",
                (session_id, _TRAJECTORY_MESSAGE_CAP),
            )
        ]
        return session, trajectory
    finally:
        conn.close()


def _validate_usage(session: dict[str, object]) -> str | None:
    """Fail closed on any unattributable usage/cost in the session row."""
    token_total = 0
    for key in ("input_tokens", "output_tokens"):
        value = session[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"session {key} is not a non-negative integer: {value!r} (fail closed)"
        token_total += value
    cost = session["estimated_cost_usd"]
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return f"session estimated_cost_usd is not a number: {cost!r} (fail closed)"
    if not math.isfinite(float(cost)) or float(cost) < 0:
        return f"session estimated_cost_usd is not finite and >= 0: {cost!r} (fail closed)"
    if token_total <= 0:
        return "session reported no input/output tokens for a successful invocation (fail closed)"
    status = session["cost_status"]
    source = session["cost_source"]
    if status not in {"actual", "estimated", "included"}:
        return f"session cost_status {status!r} is not attributable (fail closed)"
    if not isinstance(source, str) or source in {"", "none", "unknown"}:
        return f"session cost_source {source!r} is not attributable (fail closed)"
    return None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_stub_hermes_invoker(
    artifact_path: str | Path,
    *,
    solve: bool,
    expected_model: str,
    max_turns: int = 20,
    behavior: tuple[str, ...] = (),
) -> HermesCliInvoker:
    """Bundled contract-emulating stub invoker; free, local, never evidence."""
    artifact = validate_skill_artifact(artifact_path)
    flags = tuple(behavior)
    if solve:
        flags += ("--solutions", "{task_fixture_dir}/replay")
    return HermesCliInvoker(
        artifact=artifact,
        execution_mode=HERMES_CLI_STUB_MODE,
        argv_head=("python", str(HERMES_CLI_STUB_SCRIPT)),
        expected_model=expected_model,
        max_turns=max_turns,
        stub_behavior=flags,
    )


def build_live_hermes_invoker(
    artifact_path: str | Path,
    *,
    checkout: str | Path,
    approval: LiveExecutionApproval,
    model: str,
    provider: str | None = None,
    toolsets: tuple[str, ...] = ("terminal",),
    max_turns: int = 20,
) -> HermesCliInvoker:
    """Reserved real-Hermes constructor; currently fails closed before launch.

    Requires a :class:`LiveExecutionApproval` (exact confirmation phrase,
    positive accounting ceilings) and a fully passing compatibility probe.
    Structural safety blockers cannot be bypassed by approval. If a future
    checkout satisfies them, argv is derived from that checkout, never caller-supplied.
    """
    if not isinstance(approval, LiveExecutionApproval):
        raise SchemaError("live execution is default-deny: a LiveExecutionApproval is required")
    artifact = validate_skill_artifact(artifact_path)
    root = Path(checkout).resolve()
    report = probe_hermes_checkout(root)
    if not report.compatible:
        raise SchemaError(
            "hermes checkout failed the compatibility probe; blockers: "
            + "; ".join(report.blockers)
        )
    if not report.live_executable:
        raise SchemaError(
            "live Hermes execution is blocked because the current interface cannot satisfy "
            "the benchmark safety contract: " + "; ".join(report.live_blockers)
        )
    interpreter = _require_live_python(root)
    if not model or not isinstance(model, str):
        raise SchemaError("live execution requires an explicit model (no config fallback)")
    return HermesCliInvoker(
        artifact=artifact,
        execution_mode=LIVE_MODE,
        argv_head=(str(interpreter), str(root / "cli.py")),
        expected_model=model,
        toolsets=tuple(toolsets),
        max_turns=max_turns,
        provider=provider,
        env_passthrough=approval.env_passthrough,
        compatibility=report,
        checkout=str(root),
    )
