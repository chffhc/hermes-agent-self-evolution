"""Regression tests for the structural live-readiness gate."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.capability import live_gate
from benchmarks.capability.cli import main as cli_main
from benchmarks.capability.hermes_adapter import probe_hermes_checkout
from benchmarks.capability.live_gate import (
    CONFINEMENT_REQUIREMENT_ID,
    LIVE_REQUIREMENTS,
    PRE_SPEND_REQUIREMENT_ID,
    NoConfinementBackend,
    PreSpendAttestation,
    evaluate_live_requirements,
    probe_confinement,
    structural_live_blockers,
    verify_pre_spend_attestation,
)
from benchmarks.capability.schema import SchemaError


def _attestation(**overrides: object) -> PreSpendAttestation:
    fields: dict[str, object] = {
        "mechanism": "provider-proxy-v1",
        "enforcement_point": "before-provider-call",
        "max_usd": 5.0,
        "verified_by": "independent-reviewer",
        "evidence_uri": "file:///reviews/pre-spend-proxy-v1.md",
        "verified_at": "2026-07-16T00:00:00Z",
    }
    fields.update(overrides)
    return PreSpendAttestation(**fields)  # type: ignore[arg-type]


# ── requirement definitions ──


def test_structural_blockers_are_the_adapter_blockers(tmp_path: Path) -> None:
    blockers = structural_live_blockers()
    assert len(blockers) == 2
    assert any("pre-spend USD ceiling" in item for item in blockers)
    assert any("does not sandbox" in item for item in blockers)
    ids = [req.requirement_id for req in LIVE_REQUIREMENTS]
    assert ids == [PRE_SPEND_REQUIREMENT_ID, CONFINEMENT_REQUIREMENT_ID]


# ── pre-spend attestation contract ──


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"mechanism": "Bad Slug!"}, "mechanism"),
        ({"mechanism": ""}, "mechanism"),
        ({"enforcement_point": "post-run-accounting"}, "post-run accounting is not enforcement"),
        ({"max_usd": 0.0}, "finite and > 0"),
        ({"max_usd": -1.0}, "finite and > 0"),
        ({"max_usd": float("inf")}, "finite and > 0"),
        ({"max_usd": True}, "must be a number"),
        ({"max_usd": "5"}, "must be a number"),
        ({"verified_by": "  "}, "verified_by"),
        ({"verified_by": "Bad Reviewer"}, "verified_by must match"),
        ({"evidence_uri": ""}, "evidence_uri"),
        ({"evidence_uri": "not-a-uri"}, "absolute, control-free URI"),
        ({"evidence_uri": "https://example.invalid/review\n"}, "absolute, control-free URI"),
        ({"verified_at": ""}, "verified_at"),
        ({"verified_at": "yesterday"}, "not ISO-8601"),
        ({"verified_at": "2026-07-16T00:00:00"}, "include a timezone"),
    ],
)
def test_pre_spend_attestation_rejects_invalid_fields(overrides: dict, match: str) -> None:
    with pytest.raises(SchemaError, match=match):
        _attestation(**overrides)


def test_valid_attestation_still_fails_closed_without_execution_binding() -> None:
    failure = verify_pre_spend_attestation(_attestation())
    assert failure is not None
    assert "not execution-bound" in failure
    assert "provider, model" in failure
    assert "approved run/task ceilings" in failure
    assert "reviewed code change" in failure


def test_verifier_registry_is_empty_and_read_only() -> None:
    registry = live_gate._REGISTERED_PRE_SPEND_VERIFIERS
    assert len(registry) == 0
    with pytest.raises(TypeError):
        registry[("independent-reviewer", "provider-proxy-v1")] = lambda attestation: None  # type: ignore[index]


def test_context_binding_guard_precedes_even_a_replaced_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_gate,
        "_REGISTERED_PRE_SPEND_VERIFIERS",
        {
            ("independent-reviewer", "provider-proxy-v1"): lambda attestation: None,
        },
    )
    assert "not execution-bound" in str(verify_pre_spend_attestation(_attestation()))


def test_verify_rejects_non_attestation_objects() -> None:
    with pytest.raises(SchemaError, match="PreSpendAttestation"):
        verify_pre_spend_attestation({"mechanism": "provider-proxy-v1"})  # type: ignore[arg-type]


# ── confinement backend probe ──


def test_no_confinement_backend_provably_fails_probe() -> None:
    result = probe_confinement(NoConfinementBackend())
    assert result.confined is False
    assert "sibling escape write+fsync succeeded" in result.detail
    assert result.backend_id == "none"
    assert result.to_dict()["capability_evidence"] is False


@dataclass(frozen=True)
class _BrokenBackend:
    """Drops the probe child entirely; must fail the allowed-write check."""

    backend_id: str = "broken"

    def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
        return [sys.executable, "-c", "pass"]


def test_probe_requires_valid_child_result_protocol() -> None:
    result = probe_confinement(_BrokenBackend())
    assert result.confined is False
    assert "invalid result protocol" in result.detail


@dataclass(frozen=True)
class _SyntheticCanaryBackend:
    """Test-only protocol fixture; it is not evidence of OS confinement."""

    backend_id: str = "synthetic-canary"

    def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
        payload = {
            "probe_version": 1,
            "writes": [
                {"label": "inside", "success": True, "error": None},
                {"label": "sibling", "success": False, "error": "PermissionError"},
                {"label": "symlink", "success": False, "error": "PermissionError"},
                {"label": "external", "success": False, "error": "PermissionError"},
            ],
        }
        wrapper = (
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[1]).write_text('inside')\n"
            "print(sys.argv[2])\n"
        )
        return [sys.executable, "-c", wrapper, argv[-4], json.dumps(payload)]


def test_canary_protocol_pass_is_not_confinement_readiness() -> None:
    result = probe_confinement(_SyntheticCanaryBackend())
    assert result.canary_passed is True
    assert result.lifecycle_verified is False
    assert result.confined is False
    assert "lifecycle is not verified" in result.detail


@dataclass(frozen=True)
class _WriteThenDeleteBackend:
    backend_id: str = "write-then-delete"

    def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
        wrapper = (
            "import os, subprocess, sys\n"
            "subprocess.run(sys.argv[1:], check=False)\n"
            "for escape in sys.argv[-3:]:\n"
            "    if os.path.exists(escape):\n"
            "        os.remove(escape)\n"
        )
        return [sys.executable, "-c", wrapper, *argv]


def test_probe_rejects_successful_escape_even_when_backend_deletes_canary() -> None:
    result = probe_confinement(_WriteThenDeleteBackend())
    assert result.canary_passed is False
    assert result.confined is False
    assert "write+fsync succeeded" in result.detail
    assert "deleting its final file" in result.detail


def test_probe_rejects_filter_that_blocks_sibling_but_allows_external_path() -> None:
    @dataclass(frozen=True)
    class _SiblingOnlyFilteringBackend:
        backend_id: str = "sibling-only"

        def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
            payload = {
                "probe_version": 1,
                "writes": [
                    {"label": "inside", "success": True, "error": None},
                    {"label": "sibling", "success": False, "error": "PermissionError"},
                    {"label": "symlink", "success": False, "error": "PermissionError"},
                    {"label": "external", "success": True, "error": None},
                ],
            }
            wrapper = (
                "import pathlib, sys\n"
                "pathlib.Path(sys.argv[1]).write_text('inside')\n"
                "pathlib.Path(sys.argv[2]).write_text('external')\n"
                "print(sys.argv[3])\n"
            )
            return [
                sys.executable,
                "-c",
                wrapper,
                argv[-4],
                argv[-1],
                json.dumps(payload),
            ]

    result = probe_confinement(_SiblingOnlyFilteringBackend())
    assert result.canary_passed is False
    assert result.confined is False
    assert "external escape write+fsync succeeded" in result.detail


@dataclass(frozen=True)
class _InvalidArgvBackend:
    backend_id: str = "invalid-argv"

    def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
        return []


@dataclass(frozen=True)
class _NulArgvBackend:
    backend_id: str = "nul-argv"

    def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
        return ["bad\x00argv"]


def test_probe_rejects_backend_contract_violations() -> None:
    with pytest.raises(SchemaError, match="invalid argv"):
        probe_confinement(_InvalidArgvBackend())
    with pytest.raises(SchemaError, match="invalid argv"):
        probe_confinement(_NulArgvBackend())
    with pytest.raises(SchemaError, match="backend_id"):
        probe_confinement(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "1"])
def test_probe_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(SchemaError, match="timeout_seconds must be finite and > 0"):
        probe_confinement(NoConfinementBackend(), timeout_seconds=timeout)  # type: ignore[arg-type]


def test_probe_timeout_fails_closed_and_terminates_child() -> None:
    @dataclass(frozen=True)
    class _SleepingBackend:
        backend_id: str = "sleeping"

        def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
            return [sys.executable, "-c", "import time; time.sleep(10)"]

    result = probe_confinement(_SleepingBackend(), timeout_seconds=0.05)
    assert result.confined is False
    assert "timed out" in result.detail


def test_probe_timeout_kills_same_group_descendants_before_delayed_write(tmp_path: Path) -> None:
    marker = tmp_path / "delayed-escape.txt"

    @dataclass(frozen=True)
    class _SpawningBackend:
        backend_id: str = "spawning"

        def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
            child = (
                "import pathlib,sys,time; "
                "time.sleep(0.3); pathlib.Path(sys.argv[1]).write_text('escaped')"
            )
            wrapper = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
                "time.sleep(10)"
            )
            return [sys.executable, "-c", wrapper, child, str(marker)]

    result = probe_confinement(_SpawningBackend(), timeout_seconds=0.1)
    assert result.confined is False
    assert "timed out" in result.detail
    time.sleep(0.4)
    assert not marker.exists()


def test_timeout_is_bounded_when_detached_descendant_holds_pipes(tmp_path: Path) -> None:
    pid_file = tmp_path / "detached.pid"

    @dataclass(frozen=True)
    class _DetachedBackend:
        backend_id: str = "detached"

        def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
            wrapper = (
                "import pathlib,subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], "
                "start_new_session=True); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
                "time.sleep(10)"
            )
            return [sys.executable, "-c", wrapper, str(pid_file)]

    started = time.monotonic()
    try:
        result = probe_confinement(_DetachedBackend(), timeout_seconds=0.05)
        elapsed = time.monotonic() - started
        assert result.confined is False
        assert "timed out" in result.detail
        assert elapsed < 0.75
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_timeout_reports_process_group_quiescence_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class _SleepingBackend:
        backend_id: str = "sleeping-unquiesced"

        def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
            return [sys.executable, "-c", "import time; time.sleep(10)"]

    monkeypatch.setattr(live_gate, "_terminate_same_process_group", lambda proc: False)
    result = probe_confinement(_SleepingBackend(), timeout_seconds=0.05)
    assert result.confined is False
    assert "did not quiesce" in result.detail


def test_probe_fails_closed_on_child_error() -> None:
    @dataclass(frozen=True)
    class _CrashingBackend:
        backend_id: str = "crashing"

        def confine(self, argv: Sequence[str], allowed_roots: Sequence[Path]) -> list[str]:
            return [sys.executable, "-c", "raise SystemExit(3)"]

    result = probe_confinement(_CrashingBackend())
    assert result.confined is False
    assert "exited 3" in result.detail


# ── readiness evaluation ──


def test_default_readiness_report_is_blocked_and_never_evidence() -> None:
    report = evaluate_live_requirements()
    assert report.live_ready is False
    assert report.blockers == structural_live_blockers()
    payload = report.to_dict()
    assert payload["live_ready"] is False
    assert payload["capability_evidence"] is False
    assert "cannot be unlocked by this report" in str(payload["note"])
    by_id = {status.requirement_id: status for status in report.statuses}
    assert by_id[PRE_SPEND_REQUIREMENT_ID].satisfied is False
    assert "no pre-spend enforcement attestation" in by_id[PRE_SPEND_REQUIREMENT_ID].detail
    assert by_id[CONFINEMENT_REQUIREMENT_ID].satisfied is False
    assert "escape write" in by_id[CONFINEMENT_REQUIREMENT_ID].detail


def test_valid_attestation_does_not_satisfy_pre_spend_requirement() -> None:
    report = evaluate_live_requirements(pre_spend_attestation=_attestation())
    by_id = {status.requirement_id: status for status in report.statuses}
    assert by_id[PRE_SPEND_REQUIREMENT_ID].satisfied is False
    assert "not execution-bound" in by_id[PRE_SPEND_REQUIREMENT_ID].detail
    assert report.live_ready is False


def test_canary_satisfaction_alone_cannot_satisfy_confinement_requirement() -> None:
    report = evaluate_live_requirements(confinement_backend=_SyntheticCanaryBackend())
    by_id = {status.requirement_id: status for status in report.statuses}
    assert by_id[CONFINEMENT_REQUIREMENT_ID].satisfied is False
    assert "lifecycle is not verified" in by_id[CONFINEMENT_REQUIREMENT_ID].detail
    assert by_id[PRE_SPEND_REQUIREMENT_ID].satisfied is False
    assert report.live_ready is False
    assert report.blockers == structural_live_blockers()


def test_adapter_live_gate_is_static_regardless_of_probe_results(tmp_path: Path) -> None:
    # Even a report with a passing confinement probe cannot influence the
    # adapter: live_executable derives from the static blocker list.
    checkout = tmp_path / "hermes-checkout"
    files = {
        "pyproject.toml": '[project]\nname = "hermes-agent"\nversion = "0.18.2"\n',
        "cli.py": (
            "def main(query=None, quiet: bool = False, skills=None):\n"
            "    build_preloaded_skills_prompt(...)\n"
            "    # raise ValueError('Unknown skill(s)')\n"
            '    print(f"\\nsession_id: {cli.session_id}", file=sys.stderr)\n'
        ),
        "agent/skill_commands.py": (
            "def build_preloaded_skills_prompt(x):\n    pass\n"
            "def _build_skill_message(x):\n    pass\n"
        ),
        "hermes_constants.py": (
            "def get_hermes_home():\n" '    return os.environ.get("HERMES_HOME", "")\n'
        ),
        "tools/skills_tool.py": 'SKILLS_DIR = HERMES_HOME / "skills"\n',
        "hermes_state.py": (
            'DB = "state.db"\n'
            "SCHEMA = '''system_prompt TEXT, input_tokens INTEGER,\n"
            "estimated_cost_usd REAL, cost_status TEXT, cwd TEXT'''\n"
        ),
        "tools/terminal_tool.py": 'cwd = os.environ.get("TERMINAL_CWD")\n',
        "hermes_cli/oneshot.py": (
            "def run_oneshot(args):\n"
            '    report = {"estimated_cost_usd": 0, "cost_status": "estimated"}\n'
        ),
    }
    for rel, text in files.items():
        target = checkout / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    report = probe_hermes_checkout(checkout)
    assert report.compatible is True
    assert report.live_executable is False
    assert report.live_blockers == structural_live_blockers()


# ── CLI ──


def test_cli_probe_live_readiness_exits_2_while_blocked(tmp_path: Path, capsys) -> None:
    out = tmp_path / "readiness.json"
    assert cli_main(["probe-live-readiness", "--output", str(out)]) == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["live_ready"] is False
    assert payload["capability_evidence"] is False
    assert len(payload["requirements"]) == 2
    assert any("pre-spend USD ceiling" in item for item in payload["blockers"])
    assert any("does not sandbox" in item for item in payload["blockers"])
    printed = json.loads(capsys.readouterr().out)
    assert printed == payload
