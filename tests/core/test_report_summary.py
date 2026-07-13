"""Tests for the pure run-metrics summary helper used by generate_report.py.

The helper must be importable and correct without reportlab installed, and
must fail closed (return None) on missing/malformed metrics rather than
producing a partial summary that could overstate results.
"""

from __future__ import annotations

from evolution.core.report_summary import PROXY_CAVEAT, build_run_summary


def _skill_metrics(**overrides) -> dict:
    metrics = {
        "skill_name": "arxiv",
        "timestamp": "20260713_101500",
        "iterations": 10,
        "baseline_score": 0.408,
        "evolved_score": 0.569,
        "improvement": 0.161,
        "holdout_examples": 2,
        "elapsed_seconds": 61.5,
        "deployable": True,
        "output_dir": "output/arxiv/20260713_101500",
    }
    metrics.update(overrides)
    return metrics


def test_skill_metrics_summary():
    summary = build_run_summary(_skill_metrics())

    assert summary is not None
    assert "arxiv" in summary["title"]
    assert "20260713_101500" in summary["title"]
    rows = dict(summary["rows"])
    assert rows["Baseline score (proxy)"] == "0.408"
    assert rows["Evolved score (proxy)"] == "0.569"
    assert rows["Change"] == "+0.161"
    assert rows["Passed local gates (deployable)"] == "yes"
    assert rows["Iterations"] == "10"
    assert rows["Holdout examples"] == "2"
    assert rows["Elapsed"] == "61.5s"
    assert rows["Artifacts"] == "output/arxiv/20260713_101500"
    assert summary["caveat"] == PROXY_CAVEAT
    assert "proxy" in summary["caveat"]
    assert "not validated production benchmarks" in summary["caveat"]


def test_accuracy_keyed_metrics_supported():
    summary = build_run_summary(
        {
            "num_tools": 12,
            "baseline_accuracy": 0.6,
            "evolved_accuracy": 0.7,
            "improvement": 0.1,
            "deployable": True,
        }
    )

    assert summary is not None
    assert "12 tool descriptions" in summary["title"]
    rows = dict(summary["rows"])
    assert rows["Baseline score (proxy)"] == "0.600"
    assert rows["Evolved score (proxy)"] == "0.700"


def test_prompt_sections_title():
    summary = build_run_summary(
        {
            "sections": ["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"],
            "baseline_score": 0.5,
            "evolved_score": 0.55,
            "deployable": False,
        }
    )

    assert summary is not None
    assert "MEMORY_GUIDANCE, SKILLS_GUIDANCE" in summary["title"]
    assert dict(summary["rows"])["Passed local gates (deployable)"] == "no"


def test_improvement_derived_when_missing():
    summary = build_run_summary({"baseline_score": 0.4, "evolved_score": 0.5, "deployable": True})
    assert dict(summary["rows"])["Change"] == "+0.100"


def test_missing_scores_fail_closed():
    assert build_run_summary({"improvement": 0.1, "deployable": True}) is None
    assert build_run_summary({"baseline_score": 0.4}) is None


def test_non_dict_and_malformed_scores_fail_closed():
    assert build_run_summary(None) is None
    assert build_run_summary(["not", "a", "dict"]) is None
    assert build_run_summary({"baseline_score": "0.4", "evolved_score": 0.5}) is None
    assert build_run_summary({"baseline_score": True, "evolved_score": 0.5}) is None


def test_generate_report_importable_without_reportlab():
    # generate_report defers its reportlab imports so metrics plumbing stays
    # testable in environments (like CI) without reportlab installed.
    import importlib.util
    import sys
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[2] / "generate_report.py"
    spec = importlib.util.spec_from_file_location("generate_report_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.build_report)
    sys.modules.pop("generate_report_under_test", None)
