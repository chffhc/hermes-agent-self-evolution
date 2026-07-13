"""Shared opt-in PR step for evolution entrypoints.

Phases 2/3 evolve text that lives *inside* hermes-agent source files (tool
description strings, prompt section constants), so their PR changes are built
by exact-snippet replacement in those files. The invariants mirror the Phase 1
opt-in PR step:

- Nothing here runs unless the caller explicitly passed --create-pr or
  --pr-dry-run; the default path performs zero git operations.
- Runs that failed constraint/test gates or showed no positive proxy
  improvement are refused — a PR must never be a way around the gates.
- ``pr_dry_run`` renders the redacted preview with no git/GitHub side effects
  and takes precedence over ``create_pr``.
- ``create_pr`` delegates to PRBuilder, which enforces clean-worktree,
  redaction, and branch-restore semantics.
"""

from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from evolution.core.pr_builder import PRChange, PRMetrics

console = Console()

# file_relpath -> [(baseline_snippet, evolved_snippet), ...]
SourceReplacements = dict[str, list[tuple[str, str]]]


def build_source_replacement_changes(
    hermes_agent_path: Path,
    replacements: SourceReplacements,
    change_type: str,
) -> tuple[list[PRChange], str | None]:
    """Build PRChanges by replacing exact snippets inside hermes-agent files.

    Fails closed: every baseline snippet must occur exactly once in its file
    (accounting for earlier replacements to the same file). A missing or
    ambiguous snippet aborts the whole change set with an error instead of
    guessing — a partially applied PR would silently drop evolved content.

    Returns ``(changes, None)`` on success or ``([], error)`` on refusal.
    """
    changes: list[PRChange] = []
    for relpath, pairs in sorted(replacements.items()):
        source_path = hermes_agent_path / relpath
        if not source_path.is_file():
            return [], f"source file not found: {relpath}"
        try:
            original = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return [], f"cannot read {relpath}: {e}"

        patched = original
        for baseline_snippet, evolved_snippet in pairs:
            if not baseline_snippet.strip():
                return [], f"empty baseline snippet for {relpath}"
            if not evolved_snippet.strip():
                return [], f"empty evolved snippet for {relpath}"
            count = patched.count(baseline_snippet)
            if count == 0:
                return [], (
                    f"baseline text not found verbatim in {relpath}; "
                    "refusing to guess where the evolved text belongs"
                )
            if count > 1:
                return [], (
                    f"baseline text appears {count} times in {relpath}; "
                    "replacement would be ambiguous"
                )
            patched = patched.replace(baseline_snippet, evolved_snippet, 1)

        if patched != original:
            changes.append(
                PRChange(
                    file_path=relpath,
                    original_content=original,
                    evolved_content=patched,
                    change_type=change_type,
                )
            )

    if not changes:
        return [], "no effective source changes to propose"
    return changes, None


def handle_opt_in_pr(
    *,
    create_pr: bool,
    pr_dry_run: bool,
    hermes_agent_path: Path,
    run_metrics: dict,
    build_changes: Callable[[], tuple[list[PRChange], str | None]],
    pr_metrics: PRMetrics,
    title_prefix: str = "evolve",
) -> dict:
    """Gate and execute an opt-in PR request. Returns a JSON-serializable dict.

    ``build_changes`` is only invoked after the deployability/improvement
    gates pass, so refused runs never touch hermes-agent files at all. Its
    ``(changes, error)`` contract matches build_source_replacement_changes.
    """
    info: dict = {
        "requested": True,
        "dry_run": bool(pr_dry_run),
        "created": False,
        "branch_pushed": False,
        "url": None,
        "error": None,
        "skipped_reason": None,
    }

    if not run_metrics.get("deployable"):
        info["skipped_reason"] = "run is not deployable (failed constraint/test gates)"
    elif run_metrics.get("improvement", 0.0) <= 0:
        info["skipped_reason"] = "no positive proxy improvement"
    if info["skipped_reason"]:
        console.print(f"[yellow]⚠ Skipping PR: {info['skipped_reason']}[/yellow]")
        return info

    changes, error = build_changes()
    if error:
        info["error"] = f"could not build PR changes: {error}"
        console.print(f"[red]✗ {info['error']}[/red]")
        return info

    # Imported here (not at module top) so tests can monkeypatch
    # evolution.core.pr_builder.PRBuilder, same as the Phase 1 opt-in step.
    from evolution.core.pr_builder import PRBuilder

    builder = PRBuilder(hermes_agent_path=hermes_agent_path)

    if pr_dry_run:
        console.print("\n[bold]PR dry run — no branch, commit, push, or PR created.[/bold]")
        info["preview"] = builder.preview_pr(changes, pr_metrics, title_prefix=title_prefix)
        return info

    result = builder.create_pr(changes, pr_metrics, title_prefix=title_prefix)
    info["created"] = result.pr_created
    info["branch_pushed"] = result.branch_pushed
    info["branch_name"] = result.branch_name
    info["url"] = result.pr_url
    info["error"] = result.error
    if result.pr_created:
        console.print(f"[bold green]✓ PR created: {result.pr_url}[/bold green]")
    elif result.branch_pushed:
        console.print(
            f"[yellow]⚠ Branch {result.branch_name} pushed but no PR created: "
            f"{result.error}[/yellow]"
        )
    else:
        console.print(f"[red]✗ PR creation failed: {result.error}[/red]")
    return info
