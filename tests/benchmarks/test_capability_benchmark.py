from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import benchmarks.capability.cli as capability_cli
from benchmarks.capability.batch_adapter import build_batch_runner_plan
from benchmarks.capability.cli import main as cli_main
from benchmarks.capability.compare import Comparison, compare_runs, optimizer_feedback
from benchmarks.capability.replay import copy_fixture_tree, run_replay
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


def test_suite_hash_binds_cache_named_assets_used_outside_copied_roots(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    before = load_suite(suite_path).suite_hash
    expected_cache = suite_path.parent / "tasks/edit-release-note/expected/__pycache__"
    expected_cache.mkdir()
    (expected_cache / "expected.txt").write_text("verifier-side bytes\n")
    assert load_suite(suite_path).suite_hash != before


@pytest.mark.parametrize(
    "expected_file",
    [
        "workspace/oracle.pyc",
        "workspace/__pycache__/oracle.txt",
        "replay/.DS_Store",
    ],
)
def test_file_exact_rejects_cache_excluded_oracle_assets(
    tmp_path: Path, expected_file: str
) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    task = raw["tasks"][0]
    oracle = suite_path.parent / task["fixture"] / expected_file
    oracle.parent.mkdir(parents=True, exist_ok=True)
    oracle.write_bytes(b"unbound oracle")
    task["verifiers"] = [
        {
            "type": "file_exact",
            "params": {"path": "result.txt", "expected_file": expected_file},
        }
    ]
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="cache-excluded"):
        load_suite(suite_path)


def test_protected_unchanged_rejects_cache_excluded_oracle_asset(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    task = raw["tasks"][0]
    oracle = suite_path.parent / task["fixture"] / "workspace/oracle.pyc"
    oracle.write_bytes(b"unbound protected original")
    task["verifiers"] = [{"type": "protected_unchanged", "params": {"paths": ["oracle.pyc"]}}]
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="cache-excluded"):
        load_suite(suite_path)


@pytest.mark.parametrize(
    "script",
    ["__pycache__/oracle.py", "ignored.pyc/oracle.py", ".DS_Store/oracle.py"],
)
def test_command_exit_rejects_cache_excluded_script(tmp_path: Path, script: str) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    task = raw["tasks"][0]
    oracle = suite_path.parent / task["fixture"] / "workspace" / script
    oracle.parent.mkdir(parents=True, exist_ok=True)
    oracle.write_text("raise SystemExit(0)\n")
    task["verifiers"] = [{"type": "command_exit", "params": {"argv": ["python", script]}}]
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="cache-excluded"):
        load_suite(suite_path)


def test_native_suite_split_namespacing() -> None:
    suite = load_suite(SUITE_PATH)
    assert set(suite.holdout_task_ids) == {"dedupe-visitor-log", "migrate-settings-schema"}
    assert set(suite.development_task_ids) == {
        "edit-release-note",
        "transform-inventory-json",
        "repair-calculator",
    }
    assert set(suite.development_task_ids) | set(suite.holdout_task_ids) == set(suite.task_ids)


@pytest.mark.parametrize(
    "invalid_output",
    [
        '{"schema":2,"color":"blue","retries":3,"verbose":true,"colour":"blue"}',
        '{"schema":2,"color":"blue","retries":3.0,"verbose":true}',
        '{"schema":2,"color":"blue","color":"red","retries":3,"verbose":true}',
        '{"schema":2,"color":"blue","retries":NaN,"verbose":true}',
    ],
)
def test_schema_migration_holdout_requires_strict_unambiguous_json(
    tmp_path: Path, invalid_output: str
) -> None:
    suite_path = _copy_suite(tmp_path)
    replay = suite_path.parent / "tasks/migrate-settings-schema/replay/settings_v2.json"
    replay.write_text(invalid_output)
    suite = load_suite(suite_path)
    result = run_replay(
        suite,
        run_role="candidate",
        artifact_digest="a" * 64,
        fingerprint=_fingerprint(),
        apply_solution=True,
    )
    task = next(item for item in result.results if item.task_id == "migrate-settings-schema")
    assert task.passed is False


def test_json_exact_rejects_nonfinite_expected_values(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    migration = next(task for task in raw["tasks"] if task["task_id"] == "migrate-settings-schema")
    migration["verifiers"][0]["params"]["expected"]["retries"] = float("nan")
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="non-finite JSON constant"):
        load_suite(suite_path)


def test_split_defaults_to_development(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    for task in raw["tasks"]:
        task.pop("split", None)
    suite_path.write_text(json.dumps(raw))
    suite = load_suite(suite_path)
    assert suite.holdout_task_ids == ()
    assert set(suite.development_task_ids) == set(suite.task_ids)


@pytest.mark.parametrize("invalid_split", ["secret", [], 7, None])
def test_invalid_split_fails_closed(tmp_path: Path, invalid_split) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    raw["tasks"][0]["split"] = invalid_split
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="'split' must be one of"):
        load_suite(suite_path)


def test_overlapping_fixture_directories_fail_closed(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    clone = dict(raw["tasks"][0])
    clone["task_id"] = "edit-release-note-again"
    raw["tasks"].append(clone)
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="overlapping fixture directory"):
        load_suite(suite_path)

    raw["tasks"][-1]["fixture"] = f"{raw['tasks'][0]['fixture']}/workspace"
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="overlapping fixture directory"):
        load_suite(suite_path)


def test_symlink_fixture_path_fails_closed(tmp_path: Path) -> None:
    suite_path = _copy_suite(tmp_path)
    raw = json.loads(suite_path.read_text())
    target = suite_path.parent / raw["tasks"][0]["fixture"]
    link = suite_path.parent / "tasks/symlinked-fixture"
    link.symlink_to(target, target_is_directory=True)
    raw["tasks"][0]["fixture"] = "tasks/symlinked-fixture"
    suite_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaError, match="symlink fixture path not allowed"):
        load_suite(suite_path)


def test_suite_loader_rejects_duplicate_keys_and_non_finite_json(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        '{"schema_version":1,"schema_version":1,"suite_id":"x",' '"description":"x","tasks":[]}',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="duplicate JSON key"):
        load_suite(suite_path)

    suite_path.write_text(
        '{"schema_version":1,"suite_id":"x","description":NaN,"tasks":[]}',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="non-finite JSON constant"):
        load_suite(suite_path)


def test_suite_loader_is_bounded_and_types_deep_json_errors(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_bytes(b" " * 1_000_001)
    with pytest.raises(SchemaError, match="document exceeds 1000000 bytes"):
        load_suite(suite_path)

    suite_path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
    with pytest.raises(SchemaError, match="invalid strict JSON"):
        load_suite(suite_path)


def _assert_no_holdout_leakage(suite, payload: dict) -> None:
    serialized = json.dumps(payload)
    holdout_tasks = [t for t in suite.tasks if t.split == "holdout"]
    assert holdout_tasks, "native suite must retain a holdout slice"
    for task in holdout_tasks:
        assert task.task_id not in serialized
        assert task.prompt not in serialized
        assert task.fixture not in serialized
        assert task.category not in serialized
    assert "passed_gate" not in payload
    assert "pass_rate_delta" not in payload
    assert "score_delta" not in payload
    assert payload["holdout_outcomes_withheld"] is True
    assert "holdout" not in payload


def test_optimizer_feedback_redacts_holdout_identities() -> None:
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
    feedback = optimizer_feedback(suite, compare_runs(suite, baseline, candidate))
    _assert_no_holdout_leakage(suite, feedback)
    assert feedback["feedback_version"] == 2
    assert feedback["suite_hash"] == suite.suite_hash
    assert feedback["development"]["gate_passed"] is True
    assert feedback["development"]["pass_rate_delta"] == 1.0
    assert feedback["capability_evidence"] is False
    assert set(feedback["development"]["improvements"]) == set(suite.development_task_ids)
    assert feedback["development"]["task_count"] == len(suite.development_task_ids)


def test_optimizer_feedback_is_independent_of_holdout_outcomes() -> None:
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
    full = compare_runs(suite, baseline, candidate)
    development_only = replace(
        full,
        passed_gate=not full.passed_gate,
        pass_rate_delta=-123.0,
        score_delta=456.0,
        improvements=tuple(suite.development_task_ids),
    )
    assert optimizer_feedback(suite, full) == optimizer_feedback(suite, development_only)


def test_optimizer_feedback_rejects_inconsistent_task_lists() -> None:
    suite = load_suite(SUITE_PATH)
    base = Comparison(
        passed_gate=True,
        baseline_pass_rate=0.0,
        candidate_pass_rate=0.0,
        pass_rate_delta=0.0,
        baseline_mean_score=0.0,
        candidate_mean_score=0.0,
        score_delta=0.0,
        regressions=(),
        critical_regressions=(),
        improvements=(),
        duration_delta_seconds=0.0,
        cost_delta_usd=None,
        capability_evidence=False,
    )
    task_id = suite.development_task_ids[0]
    with pytest.raises(SchemaError, match="duplicate regressions"):
        optimizer_feedback(suite, replace(base, regressions=(task_id, task_id)))
    with pytest.raises(SchemaError, match="both regression and improvement"):
        optimizer_feedback(
            suite,
            replace(base, regressions=(task_id,), improvements=(task_id,)),
        )
    with pytest.raises(SchemaError, match="must also appear"):
        optimizer_feedback(suite, replace(base, critical_regressions=(task_id,)))
    critical_task_id = next(task.task_id for task in suite.tasks if task.critical)
    with pytest.raises(SchemaError, match="critical regression metadata is inconsistent"):
        optimizer_feedback(suite, replace(base, regressions=(critical_task_id,)))


def test_optimizer_feedback_withholds_holdout_regressions() -> None:
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
    feedback = optimizer_feedback(suite, compare_runs(suite, baseline, candidate))
    _assert_no_holdout_leakage(suite, feedback)
    assert feedback["development"]["gate_passed"] is False
    assert feedback["development"]["pass_rate_delta"] == -1.0
    assert set(feedback["development"]["regressions"]) == set(suite.development_task_ids)
    assert feedback["development"]["critical_regressions"] == ["repair-calculator"]


def test_optimizer_feedback_rejects_foreign_task_ids() -> None:
    suite = load_suite(SUITE_PATH)
    foreign = Comparison(
        passed_gate=True,
        baseline_pass_rate=0.0,
        candidate_pass_rate=1.0,
        pass_rate_delta=1.0,
        baseline_mean_score=0.0,
        candidate_mean_score=1.0,
        score_delta=1.0,
        regressions=(),
        critical_regressions=(),
        improvements=("not-a-suite-task",),
        duration_delta_seconds=0.0,
        cost_delta_usd=None,
        capability_evidence=False,
    )
    with pytest.raises(SchemaError, match="outside the suite"):
        optimizer_feedback(suite, foreign)


def test_copy_fixture_tree_excludes_unbound_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "__pycache__").mkdir(parents=True)
    (source / "__pycache__" / "stale.cpython-311.pyc").write_bytes(b"stale")
    (source / "stray.pyc").write_bytes(b"stray")
    (source / ".DS_Store").write_bytes(b"finder")
    (source / "ignored.pyc").mkdir()
    (source / "ignored.pyc" / "payload.txt").write_text("must not copy")
    (source / "keep.txt").write_text("bound content\n")
    destination = tmp_path / "destination"
    copy_fixture_tree(source, destination)
    assert (destination / "keep.txt").read_text() == "bound content\n"
    assert not (destination / "__pycache__").exists()
    assert not (destination / "stray.pyc").exists()
    assert not (destination / ".DS_Store").exists()
    assert not (destination / "ignored.pyc").exists()
    overlay_src = tmp_path / "overlay"
    overlay_src.mkdir()
    (overlay_src / "keep.txt").write_text("overlaid\n")
    (overlay_src / "late.pyc").write_bytes(b"late")
    copy_fixture_tree(overlay_src, destination, overlay=True)
    assert (destination / "keep.txt").read_text() == "overlaid\n"
    assert not (destination / "late.pyc").exists()


def test_json_output_transaction_restores_pair_on_second_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "comparison.json"
    second = tmp_path / "feedback.json"
    first.write_text("old-comparison\n")
    second.write_text("old-feedback\n")
    real_replace = capability_cli.os.replace
    calls = 0

    def fail_second_replace(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(capability_cli.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected second replace failure"):
        capability_cli._write_json_transaction(
            [(first, {"new": "comparison"}), (second, {"new": "feedback"})]
        )
    assert first.read_text() == "old-comparison\n"
    assert second.read_text() == "old-feedback\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_json_output_transaction_rejects_aliasing_paths(tmp_path: Path) -> None:
    output = tmp_path / "same.json"
    with pytest.raises(SchemaError, match="must remain distinct"):
        capability_cli._write_json_transaction(
            [(output, {"kind": "comparison"}), (output, {"kind": "feedback"})]
        )
    assert not output.exists()


def test_json_output_transaction_rejects_symlink_parent(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(actual, target_is_directory=True)
    destination = link / "comparison.json"
    with pytest.raises(SchemaError, match="symlink component"):
        capability_cli._write_json_transaction([(destination, {"kind": "comparison"})])
    assert not (actual / "comparison.json").exists()


def test_json_output_transaction_rejects_casefold_aliases(tmp_path: Path) -> None:
    upper = tmp_path / "Result.json"
    lower = tmp_path / "result.json"
    with pytest.raises(SchemaError, match="case-insensitive"):
        capability_cli._write_json_transaction(
            [(upper, {"kind": "comparison"}), (lower, {"kind": "feedback"})]
        )
    assert not upper.exists()
    assert not lower.exists()


def test_optimizer_feedback_suite_error_is_sanitized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite_path = _copy_suite(tmp_path)
    suite = load_suite(suite_path)
    holdout = next(task for task in suite.tasks if task.split == "holdout")
    shutil.rmtree(suite_path.parent / holdout.fixture / "replay")
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text("old-feedback\n")
    assert (
        cli_main(
            [
                "compare",
                "--suite",
                str(suite_path),
                "--baseline",
                str(tmp_path / "missing-baseline.json"),
                "--candidate",
                str(tmp_path / "missing-candidate.json"),
                "--optimizer-feedback",
                str(feedback_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error == {
        "error": (
            "optimizer feedback unavailable: suite validation failed "
            "(holdout-aware details withheld)"
        ),
        "valid": False,
    }
    serialized = json.dumps(error)
    for task in suite.tasks:
        if task.split == "holdout":
            assert task.task_id not in serialized
            assert task.fixture not in serialized
    assert feedback_path.read_text() == "old-feedback\n"


def test_optimizer_feedback_suite_oserror_is_sanitized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = _copy_suite(tmp_path)
    suite = load_suite(suite_path)
    holdout = next(task for task in suite.tasks if task.task_id == "migrate-settings-schema")
    real_read_bytes = Path.read_bytes

    def fail_on_holdout_asset(path: Path) -> bytes:
        if holdout.task_id in path.parts and path.name == "settings_v2.json":
            raise OSError(f"cannot read secret holdout asset {path}")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_on_holdout_asset)
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text("old-feedback\n")
    assert (
        cli_main(
            [
                "compare",
                "--suite",
                str(suite_path),
                "--baseline",
                str(tmp_path / "missing-baseline.json"),
                "--candidate",
                str(tmp_path / "missing-candidate.json"),
                "--optimizer-feedback",
                str(feedback_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error"] == (
        "optimizer feedback unavailable: suite validation failed "
        "(holdout-aware details withheld)"
    )
    serialized = json.dumps(error)
    assert holdout.task_id not in serialized
    assert holdout.fixture not in serialized
    assert "settings_v2.json" not in serialized
    assert feedback_path.read_text() == "old-feedback\n"


def test_compare_cli_writes_only_optimizer_feedback_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = load_suite(SUITE_PATH)
    artifact = tmp_path / "artifact.md"
    artifact.write_text("artifact under test\n")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    common = ["--suite", str(SUITE_PATH), "--artifact", str(artifact), "--model", "test/model"]
    assert (
        cli_main(
            [
                "replay",
                *common,
                "--role",
                "baseline",
                "--environment",
                "e",
                "--output",
                str(baseline_path),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "replay",
                *common,
                "--role",
                "candidate",
                "--environment",
                "e",
                "--apply-solution",
                "--output",
                str(candidate_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    feedback_path = tmp_path / "feedback.json"
    full_comparison_path = tmp_path / "human-review-comparison.json"
    assert (
        cli_main(
            [
                "compare",
                "--suite",
                str(SUITE_PATH),
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(full_comparison_path),
                "--optimizer-feedback",
                str(feedback_path),
            ]
        )
        == 0
    )
    feedback = json.loads(feedback_path.read_text())
    stdout_payload = json.loads(capsys.readouterr().out)
    _assert_no_holdout_leakage(suite, feedback)
    assert stdout_payload == feedback
    assert feedback["capability_evidence"] is False
    full_comparison = full_comparison_path.read_text()
    for task_id in suite.holdout_task_ids:
        assert task_id in full_comparison

    malformed = json.loads(candidate_path.read_text())
    malformed["results"][0]["task_id"] = "foreign-task"
    candidate_path.write_text(json.dumps(malformed))
    original_feedback = feedback_path.read_bytes()
    capsys.readouterr()
    assert (
        cli_main(
            [
                "compare",
                "--suite",
                str(SUITE_PATH),
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--optimizer-feedback",
                str(feedback_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    error_output = captured.err
    assert "holdout-aware details withheld" in error_output
    for task_id in suite.holdout_task_ids:
        assert task_id not in error_output
    assert feedback_path.read_bytes() == original_feedback
