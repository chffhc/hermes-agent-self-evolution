from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.capability.batch_adapter import build_batch_runner_plan
from benchmarks.capability.compare import compare_runs
from benchmarks.capability.replay import run_replay
from benchmarks.capability.schema import RunFingerprint, RunResult, SchemaError
from benchmarks.capability.suite import load_suite

REPO = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO / "benchmarks/capability/suites/native_v1/suite.json"


def _fingerprint(seed: int = 7) -> RunFingerprint:
    return RunFingerprint.from_config(
        "test/model", {"max_turns": 20, "tools": "default"}, seed, "fixture-env-v1"
    )


def _copy_suite(tmp_path: Path) -> Path:
    target = tmp_path / "suite"
    shutil.copytree(SUITE_PATH.parent, target)
    return target / "suite.json"


def test_native_suite_and_replay_are_honestly_labeled() -> None:
    suite = load_suite(SUITE_PATH)
    baseline = run_replay(
        suite,
        run_role="baseline",
        artifact_digest="a" * 64,
        fingerprint=_fingerprint(),
        apply_solution=False,
    )
    candidate = run_replay(
        suite,
        run_role="candidate",
        artifact_digest="b" * 64,
        fingerprint=_fingerprint(),
        apply_solution=True,
    )
    assert baseline.capability_evidence is False
    assert candidate.capability_evidence is False
    assert baseline.execution_mode == candidate.execution_mode == "replay"
    assert baseline.pass_rate == 0
    assert candidate.pass_rate == 1
    comparison = compare_runs(suite, baseline, candidate)
    assert comparison.passed_gate
    assert set(comparison.improvements) == set(suite.task_ids)
    assert comparison.capability_evidence is False


def test_duplicate_task_ids_fail_closed(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    raw["tasks"].append(dict(raw["tasks"][0]))
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="duplicate task_id"):
        load_suite(suite_path)


def test_unknown_verifier_fails_closed(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    raw["tasks"][0]["verifiers"][0]["type"] = "llm_keyword_judge"
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="unknown verifier"):
        load_suite(suite_path)


def test_fixture_traversal_fails_closed(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    raw["tasks"][0]["fixture"] = "../outside"
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="unsafe path segment"):
        load_suite(suite_path)


def test_command_verifier_rejects_python_code_switch(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    raw["tasks"][2]["verifiers"][0]["params"]["argv"] = ["python", "-c", "print(1)"]
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="unsafe path|workspace-relative"):
        load_suite(suite_path)


def test_fingerprint_mismatch_fails_closed() -> None:
    suite = load_suite(SUITE_PATH)
    baseline = run_replay(
        suite,
        run_role="baseline",
        artifact_digest="a" * 64,
        fingerprint=_fingerprint(1),
        apply_solution=True,
    )
    candidate = run_replay(
        suite,
        run_role="candidate",
        artifact_digest="b" * 64,
        fingerprint=_fingerprint(2),
        apply_solution=True,
    )
    with pytest.raises(SchemaError, match="fingerprint"):
        compare_runs(suite, baseline, candidate)


def test_critical_regression_blocks_gate() -> None:
    suite = load_suite(SUITE_PATH)
    baseline = run_replay(
        suite,
        run_role="baseline",
        artifact_digest="a" * 64,
        fingerprint=_fingerprint(),
        apply_solution=True,
    )
    candidate = run_replay(
        suite,
        run_role="candidate",
        artifact_digest="b" * 64,
        fingerprint=_fingerprint(),
        apply_solution=False,
    )
    comparison = compare_runs(suite, baseline, candidate)
    assert not comparison.passed_gate
    assert "repair-calculator" in comparison.critical_regressions


def test_non_live_run_cannot_claim_capability_evidence() -> None:
    suite = load_suite(SUITE_PATH)
    run = run_replay(
        suite,
        run_role="baseline",
        artifact_digest="a" * 64,
        fingerprint=_fingerprint(),
        apply_solution=True,
    ).to_dict()
    run["capability_evidence"] = True
    with pytest.raises(SchemaError, match="only valid"):
        RunResult.from_dict(run)


def test_infinite_numeric_fields_fail_closed() -> None:
    suite = load_suite(SUITE_PATH)
    raw = run_replay(
        suite,
        run_role="baseline",
        artifact_digest="a" * 64,
        fingerprint=_fingerprint(),
        apply_solution=True,
    ).to_dict()
    raw["results"][0]["duration_seconds"] = float("inf")
    with pytest.raises(SchemaError, match="finite"):
        RunResult.from_dict(raw)


def test_batch_plan_is_non_executable_and_non_evidence(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    (hermes / "batch_runner.py").write_text("# stub\n")
    dataset = tmp_path / "plan/tasks.jsonl"
    plan = build_batch_runner_plan(
        suite,
        hermes_repo=hermes,
        dataset_path=dataset,
        model="test/model",
        run_name="capability-dry-run",
    )
    assert plan["execution_mode"] == "dry_run"
    assert plan["capability_evidence"] is False
    assert plan["executable"] is False
    assert len(dataset.read_text().splitlines()) == len(suite.tasks)
    gaps = plan["blocking_gaps"]
    assert isinstance(gaps, list)
    assert "artifact injection" in " ".join(str(item) for item in gaps)


def test_suite_hash_binds_fixture_content(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    before = load_suite(suite_path).suite_hash
    fixture = suite_path.parent / "tasks/edit-release-note/workspace/release.txt"
    fixture.write_text("changed\n")
    after = load_suite(suite_path).suite_hash
    assert before != after


def test_suite_hash_ignores_python_cache_artifacts(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    before = load_suite(suite_path).suite_hash
    cache = suite_path.parent / "tasks/repair-calculator/workspace/__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "calculator.cpython-311.pyc").write_bytes(b"transient")
    assert load_suite(suite_path).suite_hash == before
