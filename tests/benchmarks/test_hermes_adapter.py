from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.capability.cli import main as cli_main
from benchmarks.capability.compare import compare_runs
from benchmarks.capability.executor import BudgetConfig, run_local
from benchmarks.capability.hermes_adapter import (
    HERMES_CLI_STUB_SCRIPT,
    LIVE_CONFIRM_PHRASE,
    HermesCliInvoker,
    LiveExecutionApproval,
    build_live_hermes_invoker,
    build_stub_hermes_invoker,
    probe_hermes_checkout,
    validate_skill_artifact,
)
from benchmarks.capability.replay import digest_artifact
from benchmarks.capability.schema import RunFingerprint, RunResult, SchemaError
from benchmarks.capability.suite import load_suite

REPO = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO / "benchmarks/capability/suites/native_v1/suite.json"

SKILL_BODY = (
    "# Careful edits\n\n"
    "Always read the target file before editing, keep changes minimal, and "
    "re-run the relevant verifier before declaring success."
)


def _fingerprint(seed: int = 7) -> RunFingerprint:
    return RunFingerprint.from_config(
        "stub/model",
        {
            "adapter": "current-hermes-cli-v1",
            "max_turns": 20,
            "provider": "auto",
            "toolsets": ["terminal"],
        },
        seed,
        "hermes-stub-contract-v1",
    )


def _skill_artifact(tmp_path: Path, *, name: str = "careful-edits", body: str = SKILL_BODY) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Benchmark candidate skill\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _mini_suite(tmp_path: Path, *, timeout: float = 30) -> Path:
    root = tmp_path / "mini-suite"
    task_dir = root / "tasks" / "mini-task-0"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "workspace" / "seed.txt").write_text("seed\n")
    (task_dir / "replay").mkdir()
    (task_dir / "replay" / "out.txt").write_text("done\n")
    (task_dir / "expected").mkdir()
    (task_dir / "expected" / "out.txt").write_text("done\n")
    suite_doc = {
        "schema_version": 1,
        "suite_id": "mini-hermes-suite",
        "description": "hermes adapter fixture suite",
        "tasks": [
            {
                "task_id": "mini-task-0",
                "category": "file-editing",
                "prompt": "Write out.txt containing done.",
                "fixture": "tasks/mini-task-0",
                "verifiers": [
                    {
                        "type": "file_exact",
                        "params": {"path": "out.txt", "expected_file": "expected/out.txt"},
                    }
                ],
                "timeout_seconds": timeout,
                "critical": True,
            }
        ],
    }
    suite_path = root / "suite.json"
    suite_path.write_text(json.dumps(suite_doc))
    return suite_path


def _fake_checkout(tmp_path: Path) -> Path:
    """Synthetic checkout carrying every probe marker (probe tests only)."""
    root = tmp_path / "hermes-checkout"
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
            'DEFAULT_DB_PATH = get_hermes_home() / "state.db"\n'
            "SCHEMA = '''system_prompt TEXT, input_tokens INTEGER DEFAULT 0,\n"
            "estimated_cost_usd REAL, cost_status TEXT, cwd TEXT'''\n"
        ),
        "tools/terminal_tool.py": 'cwd = os.getenv("TERMINAL_CWD", default_cwd)\n',
        "hermes_cli/oneshot.py": (
            "def run_oneshot(prompt):\n"
            '    report = {"estimated_cost_usd": 0, "cost_status": None}\n'
        ),
    }
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


# ── compatibility probe ──


def test_probe_passes_on_synthetic_checkout_and_reports_version(tmp_path: Path) -> None:
    report = probe_hermes_checkout(_fake_checkout(tmp_path))
    assert report.compatible
    assert report.blockers == ()
    assert report.version == "0.18.2"
    assert report.live_executable is False
    assert any("pre-spend USD ceiling" in item for item in report.live_blockers)
    assert any("does not sandbox" in item for item in report.live_blockers)
    assert report.to_dict()["capability_evidence"] is False


def test_probe_reports_exact_blocker_when_seam_moves(tmp_path: Path) -> None:
    checkout = _fake_checkout(tmp_path)
    (checkout / "tools/skills_tool.py").write_text("SKILLS_DIR = somewhere_else\n")
    (checkout / "hermes_cli/oneshot.py").unlink()
    report = probe_hermes_checkout(checkout)
    assert not report.compatible
    blockers = " ".join(report.blockers)
    assert "skills-dir-contract" in blockers and "lost expected marker" in blockers
    assert "oneshot-usage-report" in blockers and "missing file" in blockers


def test_probe_missing_directory_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="not a directory"):
        probe_hermes_checkout(tmp_path / "nope")


@pytest.mark.skipif(
    not os.environ.get("HERMES_AGENT_CHECKOUT"),
    reason="set HERMES_AGENT_CHECKOUT to a real read-only hermes-agent checkout",
)
def test_probe_against_real_checkout() -> None:
    report = probe_hermes_checkout(os.environ["HERMES_AGENT_CHECKOUT"])
    assert report.compatible, report.blockers


# ── skill artifact contract ──


def test_skill_artifact_validates(tmp_path: Path) -> None:
    artifact = validate_skill_artifact(_skill_artifact(tmp_path))
    assert artifact.name == "careful-edits"
    assert artifact.body.startswith("# Careful edits")
    assert artifact.digest == digest_artifact(tmp_path / "careful-edits")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: (d / "SKILL.md").unlink(), "missing SKILL.md|no files"),
        (lambda d: (d / "SKILL.md").write_text("no frontmatter\n"), "frontmatter"),
        (
            lambda d: (d / "SKILL.md").write_text(
                "---\nname: other-name\ndescription: x\n---\n\n" + SKILL_BODY
            ),
            "must equal the artifact directory name",
        ),
        (
            lambda d: (d / "SKILL.md").write_text(
                "---\nname: careful-edits\ndescription: x\n---\n\nshort"
            ),
            "at least",
        ),
        (
            lambda d: (d / "SKILL.md").write_text(
                "---\nname: careful-edits\ndescription: x\n---\n\n"
                + SKILL_BODY
                + " uses {{skill_dir}}"
            ),
            "template tokens",
        ),
        (lambda d: (d / "evil").symlink_to("/etc"), "symlink"),
    ],
)
def test_skill_artifact_fails_closed(tmp_path: Path, mutate, match: str) -> None:
    skill_dir = _skill_artifact(tmp_path)
    mutate(skill_dir)
    with pytest.raises(SchemaError, match=match):
        validate_skill_artifact(skill_dir)


def test_skill_artifact_must_be_directory(tmp_path: Path) -> None:
    lone_file = tmp_path / "careful-edits.md"
    lone_file.write_text("x")
    with pytest.raises(SchemaError, match="real directory"):
        validate_skill_artifact(lone_file)


# ── stub end-to-end ──


def _run_stub(tmp_path: Path, suite_path: Path, *, solve: bool, behavior: tuple[str, ...] = ()):
    return run_local(
        load_suite(suite_path),
        invoker=build_stub_hermes_invoker(
            _skill_artifact(tmp_path, name=f"skill-{'c' if solve else 'b'}"),
            solve=solve,
            expected_model="stub/model",
            behavior=behavior,
        ),
        run_role="candidate" if solve else "baseline",
        artifact_path=tmp_path / f"skill-{'c' if solve else 'b'}",
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=tmp_path / "runs",
        keep_workspaces=bool(behavior) or solve,
    )


def test_stub_end_to_end_consumption_proof_and_attribution(tmp_path: Path) -> None:
    suite_path = _mini_suite(tmp_path)
    outcome = _run_stub(tmp_path, suite_path, solve=True)
    result = outcome.result
    assert result.execution_mode == "hermes_cli_stub"
    assert result.capability_evidence is False
    assert result.pass_rate == 1
    assert result.results[0].cost_usd == 0.0

    control = outcome.retained_root / "tasks/mini-task-0/control"
    invocation = json.loads((control / "invocation.json").read_text())
    assert invocation["capability_evidence"] is False
    assert invocation["skill_name"] == "skill-c"
    assert "--skills" in invocation["argv"] and "--quiet" in invocation["argv"]
    # env is recorded by key name only, and contains no inherited secrets
    assert "PATH" in invocation["env_keys"]

    attestation = json.loads((control / "attestation.json").read_text())
    assert attestation["capability_evidence"] is False
    assert attestation["session_id"].startswith("stub-")
    assert attestation["artifact_digest"] == result.artifact_digest
    session = json.loads((control / "session.json").read_text())
    assert "# Careful edits" in session["system_prompt"]
    trajectory = json.loads((control / "trajectory.json").read_text())
    assert [m["role"] for m in trajectory] == ["user", "assistant"]
    usage = json.loads((control / "usage.json").read_text())
    assert usage == {"cost_usd": 0.0, "input_tokens": 120, "output_tokens": 80}


def test_stub_paired_comparison_is_honest(tmp_path: Path) -> None:
    suite_path = _mini_suite(tmp_path)
    suite = load_suite(suite_path)
    skill = _skill_artifact(tmp_path)
    common = {
        "artifact_path": skill,
        "fingerprint": _fingerprint(),
        "budget": BudgetConfig(max_run_usd=0.0),
        "runs_root": tmp_path / "runs",
    }
    baseline = run_local(
        suite,
        invoker=build_stub_hermes_invoker(skill, solve=False, expected_model="stub/model"),
        run_role="baseline",
        **common,
    ).result
    candidate = run_local(
        suite,
        invoker=build_stub_hermes_invoker(skill, solve=True, expected_model="stub/model"),
        run_role="candidate",
        **common,
    ).result
    assert baseline.pass_rate == 0 and candidate.pass_rate == 1
    comparison = compare_runs(suite, baseline, candidate)
    assert comparison.passed_gate
    assert comparison.capability_evidence is False


def test_stub_run_cannot_claim_capability_evidence(tmp_path: Path) -> None:
    raw = _run_stub(tmp_path, _mini_suite(tmp_path), solve=True).result.to_dict()
    raw["capability_evidence"] = True
    with pytest.raises(SchemaError, match="only valid"):
        RunResult.from_dict(raw)


def test_live_json_and_direct_objects_cannot_forge_capability_evidence(tmp_path: Path) -> None:
    suite_path = _mini_suite(tmp_path)
    baseline = _run_stub(tmp_path, suite_path, solve=False).result
    candidate = _run_stub(tmp_path, suite_path, solve=True).result

    forged_json = candidate.to_dict()
    forged_json["execution_mode"] = "live"
    forged_json["capability_evidence"] = True
    with pytest.raises(SchemaError, match="schema v1 refuses.*including 'live'"):
        RunResult.from_dict(forged_json)

    forged_baseline = replace(baseline, execution_mode="live", capability_evidence=True)
    forged_candidate = replace(candidate, execution_mode="live", capability_evidence=True)
    with pytest.raises(SchemaError, match="manually constructed live"):
        compare_runs(load_suite(suite_path), forged_baseline, forged_candidate)


def test_stub_invoker_binding_mismatches_fail_before_execution(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    artifact_a = _skill_artifact(tmp_path, name="skill-a")
    artifact_b = _skill_artifact(tmp_path, name="skill-b")
    invoker = build_stub_hermes_invoker(
        artifact_a, solve=True, expected_model="stub/model", max_turns=20
    )
    common = {
        "suite": suite,
        "invoker": invoker,
        "run_role": "candidate",
        "budget": BudgetConfig(max_run_usd=0.0),
        "runs_root": tmp_path / "runs",
    }
    with pytest.raises(SchemaError, match="invoker artifact digest does not match"):
        run_local(artifact_path=artifact_b, fingerprint=_fingerprint(), **common)
    wrong_model = RunFingerprint.from_config(
        "other/model", invoker.fingerprint_config(), 7, "hermes-stub-contract-v1"
    )
    with pytest.raises(SchemaError, match="invoker model.*fingerprint model"):
        run_local(artifact_path=artifact_a, fingerprint=wrong_model, **common)
    wrong_config = RunFingerprint.from_config(
        "stub/model", {"max_turns": 999}, 7, "hermes-stub-contract-v1"
    )
    with pytest.raises(SchemaError, match="applied-config digest"):
        run_local(artifact_path=artifact_a, fingerprint=wrong_config, **common)


def test_stub_subprocess_env_contains_no_inherited_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "hunter2")
    outcome = _run_stub(tmp_path, _mini_suite(tmp_path), solve=True)
    hermes_home = outcome.retained_root / "tasks/mini-task-0/control/hermes_home"
    env_keys = set(json.loads((hermes_home / "env_keys.json").read_text()))
    assert "SUPER_SECRET_TOKEN" not in env_keys
    assert {
        "HERMES_HOME",
        "TERMINAL_CWD",
        "HERMES_BENCH_RUN_ID",
        "HERMES_BENCH_TASK_ID",
    } <= env_keys


# ── stub failure paths: every attribution gap fails closed ──


@pytest.mark.parametrize(
    ("behavior", "match"),
    [
        (("--omit-session-line",), "did not report a session_id"),
        (("--skip-skill-load",), "not proven loaded"),
        (("--omit-cost",), "estimated_cost_usd is not a number"),
        (("--text-cost",), "estimated_cost_usd is not a number"),
        (("--input-tokens", "0", "--output-tokens", "0"), "reported no input/output tokens"),
        (("--cost-status", "unknown"), "cost_status.*not attributable"),
        (("--cost-source", "none"), "cost_source.*not attributable"),
        (("--wrong-cwd",), "not the isolated task workspace"),
        (("--report-model", "other/model"), "fingerprint would be violated"),
    ],
)
def test_stub_attribution_failures_fail_closed(
    tmp_path: Path, behavior: tuple[str, ...], match: str
) -> None:
    result = _run_stub(tmp_path, _mini_suite(tmp_path), solve=True, behavior=behavior).result
    task = result.results[0]
    assert task.passed is False
    assert task.error is not None
    assert re.search(match, task.error), task.error


def test_stub_attribution_failure_still_accounts_spend(tmp_path: Path) -> None:
    skill = _skill_artifact(tmp_path)
    invoker = build_stub_hermes_invoker(
        skill,
        solve=True,
        expected_model="stub/model",
        behavior=("--wrong-cwd", "--cost-usd", "2.0"),
    )
    result = run_local(
        load_suite(_mini_suite(tmp_path)),
        invoker=invoker,
        run_role="candidate",
        artifact_path=skill,
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=10.0, max_task_usd=5.0),
        runs_root=tmp_path / "runs",
    ).result
    task = result.results[0]
    assert task.passed is False and task.cost_usd == 2.0
    assert task.error is not None and "not the isolated task workspace" in task.error


def test_stub_script_fails_hard_on_unknown_skill_like_hermes_cli(tmp_path: Path) -> None:
    """Contract fidelity: a skill missing from HERMES_HOME/skills exits 1."""
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(HERMES_CLI_STUB_SCRIPT),
            "--query",
            "do the task",
            "--quiet",
            "--skills",
            "not-installed",
        ],
        cwd=tmp_path,
        env={"PATH": os.defpath, "HERMES_HOME": str(hermes_home)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "Unknown skill(s): not-installed" in proc.stdout
    assert "session_id:" not in proc.stderr


def test_stub_timeout_kills_process_group_and_cleans_up(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    result = run_local(
        load_suite(_mini_suite(tmp_path, timeout=1)),
        invoker=build_stub_hermes_invoker(
            _skill_artifact(tmp_path),
            solve=True,
            expected_model="stub/model",
            behavior=("--sleep", "10"),
        ),
        run_role="candidate",
        artifact_path=tmp_path / "careful-edits",
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=0.0),
        runs_root=runs_root,
    ).result
    assert result.results[0].passed is False
    assert "timed out" in result.results[0].error
    assert list(runs_root.iterdir()) == []


def test_stub_timeout_after_session_still_accounts_attributable_spend(tmp_path: Path) -> None:
    result = run_local(
        load_suite(_mini_suite(tmp_path, timeout=1)),
        invoker=build_stub_hermes_invoker(
            _skill_artifact(tmp_path),
            solve=True,
            expected_model="stub/model",
            behavior=("--cost-usd", "2.0", "--sleep-after-session", "10"),
        ),
        run_role="candidate",
        artifact_path=tmp_path / "careful-edits",
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=10.0, max_task_usd=5.0),
        runs_root=tmp_path / "runs",
    ).result
    task = result.results[0]
    assert task.passed is False and task.cost_usd == 2.0
    assert task.error is not None and "timed out" in task.error
    assert "usage report invalid or missing" not in task.error


def test_stub_costs_feed_budget_gate(tmp_path: Path) -> None:
    result = run_local(
        load_suite(_mini_suite(tmp_path)),
        invoker=build_stub_hermes_invoker(
            _skill_artifact(tmp_path),
            solve=True,
            expected_model="stub/model",
            behavior=("--cost-usd", "2.0"),
        ),
        run_role="candidate",
        artifact_path=tmp_path / "careful-edits",
        fingerprint=_fingerprint(),
        budget=BudgetConfig(max_run_usd=10.0, max_task_usd=1.0),
        runs_root=tmp_path / "runs",
    ).result
    task = result.results[0]
    assert task.passed is False and task.cost_usd == 2.0
    assert "per-task budget" in task.error


# ── live gating: default deny, no execution ──


def test_live_approval_is_default_deny() -> None:
    with pytest.raises(SchemaError, match="default-deny"):
        LiveExecutionApproval(confirm="yes please", max_run_usd=1.0, max_task_usd=0.5)
    with pytest.raises(SchemaError, match="> 0"):
        LiveExecutionApproval(confirm=LIVE_CONFIRM_PHRASE, max_run_usd=0.0, max_task_usd=0.5)
    with pytest.raises(SchemaError, match="finite"):
        LiveExecutionApproval(
            confirm=LIVE_CONFIRM_PHRASE, max_run_usd=float("inf"), max_task_usd=0.5
        )
    with pytest.raises(SchemaError, match="max_task_usd must be finite and > 0"):
        LiveExecutionApproval(confirm=LIVE_CONFIRM_PHRASE, max_run_usd=1.0, max_task_usd=0.0)
    with pytest.raises(SchemaError, match="cannot exceed"):
        LiveExecutionApproval(confirm=LIVE_CONFIRM_PHRASE, max_run_usd=1.0, max_task_usd=2.0)
    with pytest.raises(SchemaError, match="env passthrough"):
        LiveExecutionApproval(
            confirm=LIVE_CONFIRM_PHRASE,
            max_run_usd=1.0,
            max_task_usd=0.5,
            env_passthrough=("lower case",),
        )


def test_live_invoker_requires_probe_and_safety_contract(tmp_path: Path) -> None:
    approval = LiveExecutionApproval(confirm=LIVE_CONFIRM_PHRASE, max_run_usd=1.0, max_task_usd=0.5)
    skill = _skill_artifact(tmp_path)
    incompatible = tmp_path / "empty-checkout"
    incompatible.mkdir()
    with pytest.raises(SchemaError, match="compatibility probe; blockers"):
        build_live_hermes_invoker(
            skill, checkout=incompatible, approval=approval, model="anthropic/claude"
        )
    # Even a checkout that passes every CLI marker remains non-executable:
    # current Hermes has neither pre-spend USD enforcement nor filesystem confinement.
    checkout = _fake_checkout(tmp_path)
    with pytest.raises(SchemaError, match="pre-spend USD ceiling.*does not sandbox"):
        build_live_hermes_invoker(
            skill, checkout=checkout, approval=approval, model="anthropic/claude"
        )
    with pytest.raises(SchemaError, match="LiveExecutionApproval is required"):
        build_live_hermes_invoker(skill, checkout=checkout, approval=None, model="anthropic/claude")


def test_live_mode_cannot_be_fabricated_with_arbitrary_executable(tmp_path: Path) -> None:
    skill = validate_skill_artifact(_skill_artifact(tmp_path))
    with pytest.raises(SchemaError, match="passing compatibility probe"):
        HermesCliInvoker(
            artifact=skill,
            execution_mode="live",
            argv_head=("/bin/sh", "/tmp/evil.py"),
            expected_model="anthropic/claude",
        )


def test_stub_mode_is_pinned_to_bundled_script(tmp_path: Path) -> None:
    skill = validate_skill_artifact(_skill_artifact(tmp_path))
    with pytest.raises(SchemaError, match="pinned to the bundled stub"):
        HermesCliInvoker(
            artifact=skill,
            execution_mode="hermes_cli_stub",
            argv_head=("python", "/tmp/evil.py"),
            expected_model="stub/model",
        )


def test_run_local_rejects_live_approval_for_non_live_invoker(tmp_path: Path) -> None:
    suite = load_suite(_mini_suite(tmp_path))
    skill = _skill_artifact(tmp_path)
    invoker = build_stub_hermes_invoker(skill, solve=True, expected_model="stub/model")
    with pytest.raises(SchemaError, match="refusing ambiguous intent"):
        run_local(
            suite,
            invoker=invoker,
            run_role="candidate",
            artifact_path=skill,
            fingerprint=_fingerprint(),
            budget=BudgetConfig(max_run_usd=0.0),
            runs_root=tmp_path / "runs",
            live_approval=LiveExecutionApproval(
                confirm=LIVE_CONFIRM_PHRASE, max_run_usd=1.0, max_task_usd=0.5
            ),
        )


# ── CLI ──


def test_cli_probe_hermes_reports_blockers_and_exit_codes(tmp_path: Path) -> None:
    checkout = _fake_checkout(tmp_path)
    out = tmp_path / "probe.json"
    assert cli_main(["probe-hermes", "--hermes-repo", str(checkout), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["compatible"] is True
    (checkout / "cli.py").unlink()
    assert cli_main(["probe-hermes", "--hermes-repo", str(checkout)]) == 2


def test_cli_run_hermes_stub_end_to_end(tmp_path: Path) -> None:
    skill = _skill_artifact(tmp_path)
    out = tmp_path / "candidate.json"
    code = cli_main(
        [
            "run-hermes-stub",
            "--suite",
            str(SUITE_PATH),
            "--role",
            "candidate",
            "--artifact",
            str(skill),
            "--model",
            "stub/model",
            "--environment",
            "hermes-stub-contract-v1",
            "--solve",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["execution_mode"] == "hermes_cli_stub"
    assert payload["capability_evidence"] is False
    assert all(r["passed"] for r in payload["results"])


def test_cli_run_hermes_live_wrong_phrase_is_denied(tmp_path: Path) -> None:
    skill = _skill_artifact(tmp_path)
    code = cli_main(
        [
            "run-hermes-live",
            "--suite",
            str(SUITE_PATH),
            "--role",
            "candidate",
            "--artifact",
            str(skill),
            "--model",
            "anthropic/claude",
            "--environment",
            "live-v1",
            "--hermes-repo",
            str(_fake_checkout(tmp_path)),
            "--confirm-live-spend",
            "yes",
            "--budget-usd",
            "1.0",
            "--task-budget-usd",
            "0.5",
            "--output",
            str(tmp_path / "live.json"),
        ]
    )
    assert code == 2
    assert not (tmp_path / "live.json").exists()


def test_cli_run_hermes_live_exact_phrase_still_cannot_bypass_safety_blockers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skill = _skill_artifact(tmp_path)
    out = tmp_path / "live.json"
    code = cli_main(
        [
            "run-hermes-live",
            "--suite",
            str(SUITE_PATH),
            "--role",
            "candidate",
            "--artifact",
            str(skill),
            "--model",
            "anthropic/claude",
            "--environment",
            "live-v1",
            "--hermes-repo",
            str(_fake_checkout(tmp_path)),
            "--confirm-live-spend",
            LIVE_CONFIRM_PHRASE,
            "--budget-usd",
            "1.0",
            "--task-budget-usd",
            "0.5",
            "--output",
            str(out),
        ]
    )
    assert code == 2
    assert not out.exists()
    message = capsys.readouterr().out
    assert "pre-spend USD ceiling" in message
    assert "does not sandbox" in message
