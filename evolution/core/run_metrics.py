"""Load run metrics from evolution output artifacts.

Every evolution entrypoint writes ``output/<name>/<timestamp>/metrics.json``.
These helpers give reporting code (e.g. generate_report.py) a fail-closed way
to read real run numbers instead of hardcoding them: anything missing or
malformed returns None rather than a partially-trusted dict.

Kept dependency-free (stdlib only) so report tooling can import it without
pulling in dspy or reportlab.
"""

import json
from pathlib import Path


def load_run_metrics(metrics_path: Path) -> dict | None:
    """Load one run's metrics.json; None when missing or invalid (fail closed)."""
    path = Path(metrics_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def find_latest_run_metrics(output_root: Path, name: str) -> dict | None:
    """Metrics of the most recent run under ``output_root/name/<timestamp>/``.

    Run directories are named %Y%m%d_%H%M%S, so lexicographic order is
    chronological. Runs whose metrics.json is missing or unparseable are
    skipped rather than treated as the latest result.
    """
    runs_dir = Path(output_root) / name
    if not runs_dir.is_dir():
        return None
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        metrics = load_run_metrics(run_dir / "metrics.json")
        if metrics is not None:
            return metrics
    return None
