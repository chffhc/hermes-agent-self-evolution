"""Fail-closed consumer boundary for capability optimizer-feedback documents.

The evolution optimizer may only ever consume the development-only
``optimizer_feedback`` document. These tests prove the consumer boundary
(``evolution/core/capability_feedback.py``):

* round-trips the real producer output (``benchmarks.capability.compare``);
* refuses a full ``Comparison.to_dict()`` payload with a dedicated error;
* rejects holdout/count/oracle-shaped extra fields, missing fields, wrong
  types, non-finite numbers, duplicate IDs/JSON keys, and documents whose
  gate/delta are inconsistent with their own regression lists;
* is wired into ``evolve()`` fail-closed, before any billable work.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from benchmarks.capability.compare import compare_runs, optimizer_feedback
from benchmarks.capability.replay import run_replay
from benchmarks.capability.schema import RunFingerprint
from benchmarks.capability.suite import load_suite
from evolution.core.capability_feedback import (
    EXPECTED_NOTE,
    CapabilityFeedback,
    CapabilityFeedbackError,
    CapabilityFeedbackPolicy,
    DevelopmentFeedback,
)
from evolution.core.capability_feedback import (
    load_optimizer_feedback as _load_optimizer_feedback,
)
from evolution.core.capability_feedback import (
    parse_optimizer_feedback as _parse_optimizer_feedback,
)

REPO = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO / "benchmarks/capability/suites/native_v1/suite.json"
NATIVE_SUITE = load_suite(SUITE_PATH)


def _policy(
    *,
    known_task_ids: set[str] | frozenset[str] | None = None,
    expected_suite_id: str | None = None,
    expected_suite_hash: str | None = None,
    critical_task_ids: set[str] | frozenset[str] | None = None,
) -> CapabilityFeedbackPolicy:
    development_ids = frozenset(
        NATIVE_SUITE.development_task_ids if known_task_ids is None else known_task_ids
    )
    if critical_task_ids is None:
        critical_ids = frozenset(
            task.task_id
            for task in NATIVE_SUITE.tasks
            if task.split == "development" and task.critical and task.task_id in development_ids
        )
    else:
        critical_ids = frozenset(critical_task_ids)
    return CapabilityFeedbackPolicy(
        suite_id=expected_suite_id or NATIVE_SUITE.suite_id,
        suite_hash=expected_suite_hash or NATIVE_SUITE.suite_hash,
        development_task_ids=development_ids,
        critical_development_task_ids=critical_ids,
    )


def parse_optimizer_feedback(
    document: object,
    *,
    known_task_ids: set[str] | frozenset[str] | None = None,
    expected_suite_id: str | None = None,
    expected_suite_hash: str | None = None,
    critical_task_ids: set[str] | frozenset[str] | None = None,
) -> CapabilityFeedback:
    return _parse_optimizer_feedback(
        document,
        policy=_policy(
            known_task_ids=known_task_ids,
            expected_suite_id=expected_suite_id,
            expected_suite_hash=expected_suite_hash,
            critical_task_ids=critical_task_ids,
        ),
    )


def load_optimizer_feedback(
    path: str | Path,
    *,
    known_task_ids: set[str] | frozenset[str] | None = None,
    expected_suite_id: str | None = None,
    expected_suite_hash: str | None = None,
    critical_task_ids: set[str] | frozenset[str] | None = None,
) -> CapabilityFeedback:
    return _load_optimizer_feedback(
        path,
        policy=_policy(
            known_task_ids=known_task_ids,
            expected_suite_id=expected_suite_id,
            expected_suite_hash=expected_suite_hash,
            critical_task_ids=critical_task_ids,
        ),
    )


def _fingerprint(seed: int = 7) -> RunFingerprint:
    return RunFingerprint.from_config(
        "test/model", {"max_turns": 20, "tools": "default"}, seed, "fixture-env-v1"
    )


def _real_feedback_and_suite():
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
    comparison = compare_runs(suite, baseline, candidate)
    return suite, comparison, optimizer_feedback(suite, comparison)


def _valid_document() -> dict:
    """A hand-rolled document matching the producer contract exactly."""
    return {
        "feedback_version": 2,
        "suite_id": NATIVE_SUITE.suite_id,
        "suite_hash": NATIVE_SUITE.suite_hash,
        "capability_evidence": False,
        "development": {
            "task_count": 3,
            "gate_passed": True,
            "pass_rate_delta": 1 / 3,
            "regressions": [],
            "improvements": ["edit-release-note"],
            "critical_regressions": [],
        },
        "holdout_outcomes_withheld": True,
        "note": EXPECTED_NOTE,
    }


# ── round trip with the real producer ──────────────────────────────────────


def test_round_trip_with_real_producer_output(tmp_path: Path) -> None:
    suite, _comparison, document = _real_feedback_and_suite()
    parsed = parse_optimizer_feedback(
        document,
        known_task_ids=set(suite.development_task_ids),
        expected_suite_id=suite.suite_id,
    )
    assert parsed.suite_id == suite.suite_id
    assert parsed.development.gate_passed is True
    assert set(parsed.development.improvements) == set(suite.development_task_ids)
    # Exact re-emission: nothing added, nothing dropped.
    assert parsed.to_document() == document

    # File loader round trip through strict JSON.
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(document))
    loaded = load_optimizer_feedback(path, expected_suite_id=suite.suite_id)
    assert loaded == parsed


def test_prompt_section_exposes_only_development_data() -> None:
    suite, _comparison, document = _real_feedback_and_suite()
    parsed = parse_optimizer_feedback(document)
    section = parsed.prompt_section()
    for task_id in suite.holdout_task_ids:
        assert task_id not in section
    for task_id in suite.development_task_ids:
        assert task_id in section
    assert "withheld" in section
    assert "never live agent capability evidence" in section
    # No full-comparison metrics leak into the rendered text.
    for marker in ("passed_gate", "score_delta", "baseline", "candidate"):
        assert marker not in section


# ── full Comparison payloads are refused ───────────────────────────────────


def test_rejects_full_comparison_to_dict() -> None:
    _suite, comparison, _document = _real_feedback_and_suite()
    with pytest.raises(CapabilityFeedbackError, match="never.*Comparison\\.to_dict"):
        parse_optimizer_feedback(comparison.to_dict())


def test_rejects_comparison_document_from_file(tmp_path: Path) -> None:
    _suite, comparison, _document = _real_feedback_and_suite()
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison.to_dict()))
    with pytest.raises(CapabilityFeedbackError, match="full-comparison fields"):
        load_optimizer_feedback(path)


def test_rejects_hybrid_document_with_comparison_fields() -> None:
    # A valid feedback document with even one comparison field grafted on
    # must fail with the dedicated comparison error, not be silently accepted.
    for key, value in [
        ("passed_gate", True),
        ("score_delta", 0.25),
        ("baseline_pass_rate", 0.4),
        ("candidate_mean_score", 0.9),
        ("cost_delta_usd", None),
    ]:
        document = _valid_document()
        document[key] = value
        with pytest.raises(
            CapabilityFeedbackError, match="full-comparison|holdout/comparison-shaped"
        ):
            parse_optimizer_feedback(document)


# ── holdout/count/oracle-shaped extra fields ───────────────────────────────


def test_rejects_holdout_shaped_top_level_fields() -> None:
    for key, value in [
        ("holdout", {"task_count": 2}),
        ("holdout_task_count", 2),
        ("holdout_outcomes", ["dedupe-visitor-log"]),
        ("oracle_hint", "pass"),
    ]:
        document = _valid_document()
        document[key] = value
        with pytest.raises(CapabilityFeedbackError, match="holdout/comparison-shaped"):
            parse_optimizer_feedback(document)


def test_rejects_holdout_shaped_development_fields() -> None:
    for key, value in [
        ("holdout_regressions", []),
        ("holdout_count", 2),
        ("baseline_pass_rate", 1.0),
        ("candidate_pass_rate", 1.0),
        ("oracle", "candidate wins"),
    ]:
        document = _valid_document()
        document["development"][key] = value
        with pytest.raises(CapabilityFeedbackError, match="holdout/comparison-shaped"):
            parse_optimizer_feedback(document)


def test_rejects_unknown_fields_generically() -> None:
    document = _valid_document()
    document["extra"] = 1
    with pytest.raises(CapabilityFeedbackError, match="unknown fields"):
        parse_optimizer_feedback(document)

    document = _valid_document()
    document["development"]["extra"] = 1
    with pytest.raises(CapabilityFeedbackError, match="unknown fields"):
        parse_optimizer_feedback(document)


# ── missing fields / wrong types / dishonest labels ────────────────────────


@pytest.mark.parametrize(
    "key",
    [
        "feedback_version",
        "suite_id",
        "suite_hash",
        "capability_evidence",
        "development",
        "holdout_outcomes_withheld",
        "note",
    ],
)
def test_rejects_missing_top_level_field(key: str) -> None:
    document = _valid_document()
    del document[key]
    with pytest.raises(CapabilityFeedbackError, match="missing required fields"):
        parse_optimizer_feedback(document)


@pytest.mark.parametrize(
    "key",
    [
        "task_count",
        "gate_passed",
        "pass_rate_delta",
        "regressions",
        "improvements",
        "critical_regressions",
    ],
)
def test_rejects_missing_development_field(key: str) -> None:
    document = _valid_document()
    del document["development"][key]
    with pytest.raises(CapabilityFeedbackError, match="missing required fields"):
        parse_optimizer_feedback(document)


def test_rejects_capability_evidence_true() -> None:
    document = _valid_document()
    document["capability_evidence"] = True
    with pytest.raises(CapabilityFeedbackError, match="capability_evidence must be false"):
        parse_optimizer_feedback(document)


def test_rejects_non_redacted_or_mislabeled_documents() -> None:
    for key, value, match in [
        ("capability_evidence", 0, "capability_evidence must be false"),
        ("capability_evidence", "false", "capability_evidence must be false"),
        ("holdout_outcomes_withheld", False, "holdout_outcomes_withheld must be true"),
        ("holdout_outcomes_withheld", 1, "holdout_outcomes_withheld must be true"),
        ("note", "trust me", "fixed producer disclaimer"),
        ("note", EXPECTED_NOTE + " plus holdout hints", "fixed producer disclaimer"),
        ("note", 7, "fixed producer disclaimer"),
        ("feedback_version", 1, "feedback_version"),
        ("feedback_version", True, "feedback_version"),
        ("feedback_version", "1", "feedback_version"),
        ("suite_id", "", "slug"),
        ("suite_id", "Bad Suite!", "slug"),
        ("suite_id", 5, "must be a string"),
        ("development", ["not", "a", "dict"], "'development' must be an object"),
    ]:
        document = _valid_document()
        document[key] = value
        with pytest.raises(CapabilityFeedbackError, match=match):
            parse_optimizer_feedback(document)


def test_rejects_bad_development_types() -> None:
    for key, value, match in [
        ("task_count", 0, "positive integer"),
        ("task_count", -1, "positive integer"),
        ("task_count", True, "positive integer"),
        ("task_count", 3.0, "positive integer"),
        ("gate_passed", 1, "boolean"),
        ("gate_passed", "true", "boolean"),
        ("pass_rate_delta", "0.33", "finite number"),
        ("pass_rate_delta", True, "finite number"),
        ("regressions", "edit-release-note", "must be a list"),
        ("improvements", [1], "must be a string"),
        ("improvements", ["Bad Slug"], "slug"),
        ("improvements", [""], "must be a string|slug"),
    ]:
        document = _valid_document()
        document["development"][key] = value
        with pytest.raises(CapabilityFeedbackError, match=match):
            parse_optimizer_feedback(document)


def test_rejects_non_dict_documents() -> None:
    for bad in [None, [], "feedback", 3]:
        with pytest.raises(CapabilityFeedbackError, match="JSON object"):
            parse_optimizer_feedback(bad)


# ── duplicate IDs, overlaps, foreign IDs, inconsistency ────────────────────


def test_rejects_duplicate_and_unsorted_task_ids() -> None:
    document = _valid_document()
    document["development"]["task_count"] = 3
    document["development"]["improvements"] = ["edit-release-note", "edit-release-note"]
    document["development"]["pass_rate_delta"] = 2 / 3
    with pytest.raises(CapabilityFeedbackError, match="duplicate task IDs"):
        parse_optimizer_feedback(document)

    document = _valid_document()
    document["development"]["improvements"] = ["b-task", "a-task"]
    document["development"]["pass_rate_delta"] = 2 / 3
    with pytest.raises(CapabilityFeedbackError, match="sorted"):
        parse_optimizer_feedback(document)


def test_rejects_regression_improvement_overlap() -> None:
    document = _valid_document()
    document["development"]["regressions"] = ["edit-release-note"]
    document["development"]["pass_rate_delta"] = 0.0
    document["development"]["gate_passed"] = True
    with pytest.raises(CapabilityFeedbackError, match="both regression and improvement"):
        parse_optimizer_feedback(document)


def test_rejects_critical_not_subset_of_regressions() -> None:
    document = _valid_document()
    document["development"]["critical_regressions"] = ["repair-calculator"]
    with pytest.raises(CapabilityFeedbackError, match="subset of regressions"):
        parse_optimizer_feedback(document)


def test_rejects_task_lists_exceeding_task_count() -> None:
    document = _valid_document()
    document["development"]["task_count"] = 1
    document["development"]["regressions"] = ["a-task"]
    document["development"]["improvements"] = ["b-task"]
    document["development"]["pass_rate_delta"] = 0.0
    with pytest.raises(CapabilityFeedbackError, match="exceed the development task count"):
        parse_optimizer_feedback(document)


def test_rejects_inconsistent_delta_and_gate() -> None:
    document = _valid_document()
    document["development"]["pass_rate_delta"] = 0.9
    with pytest.raises(CapabilityFeedbackError, match="pass_rate_delta is inconsistent"):
        parse_optimizer_feedback(document)

    document = _valid_document()
    document["development"]["gate_passed"] = False
    with pytest.raises(CapabilityFeedbackError, match="gate_passed is inconsistent"):
        parse_optimizer_feedback(document)


def test_rejects_foreign_task_ids_and_wrong_suite() -> None:
    document = _valid_document()
    with pytest.raises(CapabilityFeedbackError, match="outside the trusted development suite"):
        parse_optimizer_feedback(
            document,
            known_task_ids={"some-other-task", "second-task", "third-task"},
        )
    with pytest.raises(CapabilityFeedbackError, match="does not match the trusted suite"):
        parse_optimizer_feedback(document, expected_suite_id="different-suite")


def test_rejects_stale_suite_hash_and_unbound_task_count() -> None:
    document = _valid_document()
    document["suite_hash"] = "b" * 64
    with pytest.raises(CapabilityFeedbackError, match="suite_hash does not match"):
        parse_optimizer_feedback(document)

    document = _valid_document()
    document["suite_hash"] = "not-a-digest"
    with pytest.raises(CapabilityFeedbackError, match="SHA-256"):
        parse_optimizer_feedback(document)

    document = _valid_document()
    document["development"]["task_count"] = 4
    document["development"]["pass_rate_delta"] = 0.25
    with pytest.raises(CapabilityFeedbackError, match="trusted suite development task count"):
        parse_optimizer_feedback(document)


def test_rejects_forged_critical_regression_metadata() -> None:
    document = _valid_document()
    document["development"].update(
        {
            "gate_passed": True,
            "pass_rate_delta": 0.0,
            "regressions": ["repair-calculator"],
            "improvements": ["transform-inventory-json"],
            "critical_regressions": [],
        }
    )
    with pytest.raises(CapabilityFeedbackError, match="trusted suite critical task policy"):
        parse_optimizer_feedback(document)


def test_rejects_malformed_trusted_policy() -> None:
    with pytest.raises(CapabilityFeedbackError, match="critical task IDs must be development"):
        parse_optimizer_feedback(
            _valid_document(),
            critical_task_ids={"foreign-critical-task"},
        )


# ── strict JSON at the file boundary ───────────────────────────────────────


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    document = _valid_document()
    text = json.dumps(document)
    text = text.replace('"feedback_version": 2,', '"feedback_version": 2, "feedback_version": 2,')
    path = tmp_path / "feedback.json"
    path.write_text(text)
    with pytest.raises(CapabilityFeedbackError, match="duplicate JSON key"):
        load_optimizer_feedback(path)


def test_loader_rejects_non_finite_constants(tmp_path: Path) -> None:
    for constant in ("NaN", "Infinity", "-Infinity"):
        document = _valid_document()
        text = json.dumps(document).replace(
            f'"pass_rate_delta": {document["development"]["pass_rate_delta"]!r}',
            f'"pass_rate_delta": {constant}',
        )
        assert constant in text
        path = tmp_path / "feedback.json"
        path.write_text(text)
        with pytest.raises(CapabilityFeedbackError, match="non-finite JSON constant"):
            load_optimizer_feedback(path)


def test_loader_rejects_invalid_json_and_wrong_roots(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    path.write_text("{not json")
    with pytest.raises(CapabilityFeedbackError, match="invalid strict JSON document"):
        load_optimizer_feedback(path)
    path.write_text("[1, 2, 3]")
    with pytest.raises(CapabilityFeedbackError, match="JSON object"):
        load_optimizer_feedback(path)
    with pytest.raises(CapabilityFeedbackError, match="cannot read"):
        load_optimizer_feedback(tmp_path / "missing.json")


def test_loader_rejects_oversized_documents(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    path.write_text('{"pad": "' + "x" * 1_000_001 + '"}')
    with pytest.raises(CapabilityFeedbackError, match="exceeds"):
        load_optimizer_feedback(path)


# ── typed exceptions, no sys.exit ──────────────────────────────────────────


def test_errors_are_typed_evolution_errors() -> None:
    from evolution.core.errors import EvolutionError

    assert issubclass(CapabilityFeedbackError, EvolutionError)
    with pytest.raises(EvolutionError):
        parse_optimizer_feedback({"passed_gate": True})


def test_to_document_is_pure_and_reparses() -> None:
    parsed = parse_optimizer_feedback(_valid_document())
    assert isinstance(parsed, CapabilityFeedback)
    assert isinstance(parsed.development, DevelopmentFeedback)
    document = parsed.to_document()
    # Mutating the emitted document must not corrupt the parsed object.
    mutated = copy.deepcopy(document)
    mutated["development"]["regressions"].append("evil-task")
    assert parse_optimizer_feedback(document) == parsed


# ── evolve() wiring (fail closed before billable work) ─────────────────────


def _make_skill_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "my-hermes"
    skill_dir = repo / "skills" / "testing" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo\n\n1. Do the thing.\n"
    )
    return repo


def test_evolve_cli_dry_run_accepts_valid_feedback(tmp_path, monkeypatch) -> None:
    from evolution.skills.evolve_skill import main

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    repo = _make_skill_repo(tmp_path)
    _suite, _comparison, document = _real_feedback_and_suite()
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps(document))

    result = CliRunner().invoke(
        main,
        [
            "--skill",
            "demo",
            "--hermes-repo",
            str(repo),
            "--dry-run",
            "--capability-feedback",
            str(feedback_path),
            "--capability-suite",
            str(SUITE_PATH),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Capability harness feedback" in result.output
    assert "development gate: passed" in result.output
    assert "Capability feedback validated" in result.output


def test_evolve_cli_fails_closed_on_comparison_document(tmp_path, monkeypatch) -> None:
    from evolution.skills.evolve_skill import main

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    repo = _make_skill_repo(tmp_path)
    _suite, comparison, _document = _real_feedback_and_suite()
    feedback_path = tmp_path / "comparison.json"
    feedback_path.write_text(json.dumps(comparison.to_dict()))

    result = CliRunner().invoke(
        main,
        [
            "--skill",
            "demo",
            "--hermes-repo",
            str(repo),
            "--dry-run",
            "--capability-feedback",
            str(feedback_path),
            "--capability-suite",
            str(SUITE_PATH),
        ],
    )

    assert result.exit_code == 1
    assert "full-comparison fields" in result.output
    assert "DRY RUN" not in result.output


def test_evolve_rejects_feedback_before_any_other_work(tmp_path, monkeypatch) -> None:
    """Validation happens before repo resolution, budgets, or LLM setup."""
    from evolution.skills.evolve_skill import evolve

    feedback_path = tmp_path / "bad.json"
    feedback_path.write_text(json.dumps({"passed_gate": True}))
    # No hermes repo exists at all — the feedback gate must fire first.
    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    with pytest.raises(CapabilityFeedbackError, match="full-comparison fields"):
        evolve(
            skill_name="demo",
            capability_feedback=feedback_path,
            capability_suite=SUITE_PATH,
        )


def test_evolve_requires_feedback_and_suite_together(tmp_path: Path) -> None:
    from evolution.skills.evolve_skill import evolve

    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps(_valid_document()))
    with pytest.raises(CapabilityFeedbackError, match="must be supplied together"):
        evolve(skill_name="demo", capability_feedback=feedback_path)
    with pytest.raises(CapabilityFeedbackError, match="must be supplied together"):
        evolve(skill_name="demo", capability_suite=SUITE_PATH)


def test_evolve_rejects_stale_suite_provenance_before_other_work(tmp_path: Path) -> None:
    from evolution.skills.evolve_skill import evolve

    document = _valid_document()
    document["suite_hash"] = "b" * 64
    feedback_path = tmp_path / "stale-feedback.json"
    feedback_path.write_text(json.dumps(document))
    with pytest.raises(CapabilityFeedbackError, match="suite_hash does not match"):
        evolve(
            skill_name="demo",
            capability_feedback=feedback_path,
            capability_suite=SUITE_PATH,
        )


def test_run_evolution_cli_threads_feedback_and_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_evolution

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_evolution, "evolve", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(run_evolution, "_print_diff", lambda _skill: None)
    result = CliRunner().invoke(
        run_evolution.main,
        [
            "--skill",
            "demo",
            "--capability-feedback",
            "feedback.json",
            "--capability-suite",
            "suite.json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["capability_feedback"] == "feedback.json"
    assert captured["capability_suite"] == "suite.json"


def test_deep_json_is_typed_and_both_clis_fail_closed(tmp_path: Path) -> None:
    import run_evolution
    from evolution.skills.evolve_skill import main as evolve_main

    feedback_path = tmp_path / "deep.json"
    feedback_path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
    with pytest.raises(CapabilityFeedbackError, match="invalid strict JSON document"):
        load_optimizer_feedback(feedback_path)

    args = [
        "--skill",
        "demo",
        "--capability-feedback",
        str(feedback_path),
        "--capability-suite",
        str(SUITE_PATH),
    ]
    evolve_result = CliRunner().invoke(evolve_main, args)
    assert evolve_result.exit_code == 1
    assert "invalid strict JSON document" in evolve_result.output
    assert not isinstance(evolve_result.exception, RecursionError)

    run_result = CliRunner().invoke(run_evolution.main, args)
    assert run_result.exit_code == 1
    assert "invalid strict JSON document" in run_result.output
    assert not isinstance(run_result.exception, RecursionError)


def test_failed_metrics_persist_validated_feedback(tmp_path: Path) -> None:
    from evolution.skills.evolve_skill import _failed_run_metrics

    document = _valid_document()
    metrics = _failed_run_metrics(
        "demo",
        str(tmp_path),
        "evolved skill failed constraint gates",
        capability_feedback=document,
    )
    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert persisted == metrics
    assert persisted["deployable"] is False
    assert persisted["capability_feedback"] == document
