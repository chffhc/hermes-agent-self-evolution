"""Tests for the fail-closed run-metrics loader used by reporting."""

from __future__ import annotations

import json

from evolution.core.run_metrics import find_latest_run_metrics, load_run_metrics


def _write_run(output_root, name, timestamp, metrics):
    run_dir = output_root / name / timestamp
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


def test_load_run_metrics_reads_valid_file(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"improvement": 0.05, "deployable": True}), encoding="utf-8")

    metrics = load_run_metrics(path)

    assert metrics == {"improvement": 0.05, "deployable": True}


def test_load_run_metrics_missing_file_is_none(tmp_path):
    assert load_run_metrics(tmp_path / "nope.json") is None


def test_load_run_metrics_invalid_json_is_none(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_run_metrics(path) is None


def test_load_run_metrics_non_dict_json_is_none(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert load_run_metrics(path) is None


def test_find_latest_run_metrics_picks_newest_timestamp(tmp_path):
    _write_run(tmp_path, "arxiv", "20260101_000000", {"improvement": 0.01})
    _write_run(tmp_path, "arxiv", "20260301_000000", {"improvement": 0.03})
    _write_run(tmp_path, "arxiv", "20260201_000000", {"improvement": 0.02})

    metrics = find_latest_run_metrics(tmp_path, "arxiv")

    assert metrics == {"improvement": 0.03}


def test_find_latest_run_metrics_skips_broken_newest_run(tmp_path):
    _write_run(tmp_path, "arxiv", "20260101_000000", {"improvement": 0.01})
    broken = tmp_path / "arxiv" / "20260401_000000"
    broken.mkdir(parents=True)
    (broken / "metrics.json").write_text("{corrupt", encoding="utf-8")
    # A run dir without metrics.json at all is also skipped.
    (tmp_path / "arxiv" / "20260501_000000").mkdir(parents=True)

    metrics = find_latest_run_metrics(tmp_path, "arxiv")

    assert metrics == {"improvement": 0.01}


def test_find_latest_run_metrics_missing_name_is_none(tmp_path):
    assert find_latest_run_metrics(tmp_path, "does-not-exist") is None
