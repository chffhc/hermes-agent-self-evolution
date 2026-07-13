"""Tests for the generate_report CLI (main) without requiring reportlab.

The markdown path must be fully exercisable stdlib-only, and every metrics
failure mode must fail closed with a clear SystemExit instead of writing a
report from missing/partial data.
"""

from __future__ import annotations

import json

import pytest

import generate_report


def _metrics(**overrides) -> dict:
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


def _write_metrics(tmp_path, metrics: dict) -> str:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return str(path)


def test_markdown_report_from_metrics(tmp_path, capsys):
    metrics_path = _write_metrics(tmp_path, _metrics())
    output_path = tmp_path / "reports" / "run.md"

    result = generate_report.main(
        ["--format", "markdown", "--metrics", metrics_path, "--output", str(output_path)]
    )

    assert result == str(output_path)
    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("# Measured run — skill 'arxiv'")
    assert "| Baseline score (proxy) | 0.408 |" in content
    assert "| Evolved score (proxy) | 0.569 |" in content
    assert "not validated production benchmarks" in content
    assert str(output_path) in capsys.readouterr().out


def test_markdown_report_contains_no_historical_claims(tmp_path):
    metrics_path = _write_metrics(tmp_path, _metrics())
    output_path = tmp_path / "run.md"

    generate_report.main(
        ["--format", "markdown", "--metrics", metrics_path, "--output", str(output_path)]
    )

    content = output_path.read_text(encoding="utf-8")
    # The hardcoded smoke-test narrative belongs to the PDF only.
    assert "+39.5%" not in content
    assert "Executive Summary" not in content


def test_markdown_without_metrics_fails_closed(tmp_path):
    with pytest.raises(SystemExit, match="requires --metrics"):
        generate_report.main(["--format", "markdown", "--output", str(tmp_path / "run.md")])
    assert not (tmp_path / "run.md").exists()


def test_missing_metrics_file_fails_closed(tmp_path):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(SystemExit, match="Could not load run metrics"):
        generate_report.main(
            ["--format", "markdown", "--metrics", missing, "--output", str(tmp_path / "run.md")]
        )
    assert not (tmp_path / "run.md").exists()


def test_malformed_metrics_json_fails_closed(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="Could not load run metrics"):
        generate_report.main(
            ["--format", "markdown", "--metrics", str(path), "--output", str(tmp_path / "run.md")]
        )


def test_metrics_without_scores_refused(tmp_path):
    metrics_path = _write_metrics(
        tmp_path, {"skill_name": "arxiv", "deployable": False, "improvement": 0.0}
    )
    with pytest.raises(ValueError, match="baseline/evolved score pair"):
        generate_report.main(
            [
                "--format",
                "markdown",
                "--metrics",
                metrics_path,
                "--output",
                str(tmp_path / "run.md"),
            ]
        )
    assert not (tmp_path / "run.md").exists()


def test_default_markdown_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_path = _write_metrics(tmp_path, _metrics())

    result = generate_report.main(["--format", "markdown", "--metrics", metrics_path])

    assert result == "reports/measured_run_summary.md"
    assert (tmp_path / "reports" / "measured_run_summary.md").is_file()
