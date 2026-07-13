"""Build display-ready summary data from evolution run metrics.

Pure, stdlib-only helpers so report tooling (generate_report.py) can render
real measured numbers without importing reportlab or dspy. Fail closed: if
the required score fields are missing or malformed, return None rather than a
partial summary that could overstate results.
"""

# Different phases name their scores differently; each pair is
# (baseline_key, evolved_key) and the first fully-present pair wins.
_SCORE_KEY_PAIRS = (
    ("baseline_score", "evolved_score"),
    ("baseline_accuracy", "evolved_accuracy"),
)

PROXY_CAVEAT = (
    "Scores are local proxy-evaluation signals (LLM-as-judge or heuristic "
    "scoring on a small holdout set), not validated production benchmarks. "
    "Any improvement is a candidate signal requiring human review before "
    "deployment."
)


def _as_float(value) -> float | None:
    """Strictly numeric (bools rejected); None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_run_summary(metrics) -> dict | None:
    """Summarize one run's metrics dict for display.

    Returns ``{"title": str, "rows": list[tuple[str, str]], "caveat": str}``,
    or None when the metrics lack a usable baseline/evolved score pair.
    """
    if not isinstance(metrics, dict):
        return None

    baseline = evolved = None
    for baseline_key, evolved_key in _SCORE_KEY_PAIRS:
        if baseline_key in metrics and evolved_key in metrics:
            baseline = _as_float(metrics[baseline_key])
            evolved = _as_float(metrics[evolved_key])
            break
    if baseline is None or evolved is None:
        return None

    improvement = _as_float(metrics.get("improvement"))
    if improvement is None:
        improvement = evolved - baseline

    if metrics.get("skill_name"):
        subject = f"skill '{metrics['skill_name']}'"
    elif metrics.get("sections"):
        subject = f"prompt sections {', '.join(metrics['sections'])}"
    elif "num_tools" in metrics:
        subject = f"{metrics['num_tools']} tool descriptions"
    else:
        subject = "evolution run"
    timestamp = metrics.get("timestamp")
    title = f"Measured run — {subject}" + (f" ({timestamp})" if timestamp else "")

    rows: list[tuple[str, str]] = [
        ("Baseline score (proxy)", f"{baseline:.3f}"),
        ("Evolved score (proxy)", f"{evolved:.3f}"),
        ("Change", f"{improvement:+.3f}"),
        ("Passed local gates (deployable)", "yes" if metrics.get("deployable") else "no"),
    ]
    if isinstance(metrics.get("iterations"), int):
        rows.append(("Iterations", str(metrics["iterations"])))
    if isinstance(metrics.get("holdout_examples"), int):
        rows.append(("Holdout examples", str(metrics["holdout_examples"])))
    elapsed = _as_float(metrics.get("elapsed_seconds"))
    if elapsed is not None:
        rows.append(("Elapsed", f"{elapsed:.1f}s"))
    if metrics.get("output_dir"):
        rows.append(("Artifacts", str(metrics["output_dir"])))

    return {"title": title, "rows": rows, "caveat": PROXY_CAVEAT}


def _md_cell(text: str) -> str:
    """Escape a value for use inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown_summary(summary: dict) -> str:
    """Render a build_run_summary result as a standalone Markdown document.

    Stdlib-only counterpart to the PDF measured-run section: title, the
    metric/value table, and the proxy caveat as a blockquote. Contains no
    narrative claims beyond what the metrics artifact actually recorded.
    """
    lines = [
        f"# {summary['title']}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for label, value in summary["rows"]:
        lines.append(f"| {_md_cell(label)} | {_md_cell(value)} |")
    lines += ["", f"> {summary['caveat']}", ""]
    return "\n".join(lines)
