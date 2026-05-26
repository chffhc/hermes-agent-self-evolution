"""Tests for OpenEvolve engine integration in evolve_tool_code."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from evolution.code import evolve_tool_code as module
from evolution.code.openevolve_runner import OpenEvolveRunResult


def _fake_hermes_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "hermes-agent"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    (tools / "toy_tools.py").write_text(
        "from tools.registry import registry\n\n"
        "def normalize_tags(raw):\n"
        "    return [part.strip() for part in raw.split(',')]\n\n"
        "registry.register(name='normalize_tags', handler=normalize_tags)\n",
        encoding="utf-8",
    )
    return repo


def test_create_openevolve_tool_scaffold_writes_patch_only_inputs(tmp_path: Path):
    repo = _fake_hermes_repo(tmp_path)
    organism = module.wrap_tool_as_organism("normalize_tags", repo)
    assert organism is not None

    scaffold = module.create_openevolve_tool_scaffold(organism, tmp_path / "scaffold", iterations=2)

    assert scaffold.initial_program.exists()
    assert scaffold.evaluator.exists()
    assert scaffold.config_file.exists()
    initial_text = scaffold.initial_program.read_text(encoding="utf-8")
    evaluator_text = scaffold.evaluator.read_text(encoding="utf-8")
    config_text = scaffold.config_file.read_text(encoding="utf-8")

    assert "# EVOLVE-BLOCK-START" in initial_text
    assert "# EVOLVE-BLOCK-END" in initial_text
    assert "normalize_tags" in initial_text
    assert "combined_score" in evaluator_text
    assert "py_compile" in evaluator_text
    assert "max_iterations: 2" in config_text
    # The production tool is not modified just to create an OpenEvolve scaffold.
    assert "EVOLVE-BLOCK" not in organism.file_path.read_text(encoding="utf-8")


def test_evolve_tool_code_openevolve_outputs_review_artifacts_without_applying(
    tmp_path: Path, monkeypatch
):
    repo = _fake_hermes_repo(tmp_path)
    original_tool = (repo / "tools" / "toy_tools.py").read_text(encoding="utf-8")
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        best = config.output_root / "best_program.py"
        info = config.output_root / "best_program_info.json"
        best.write_text(config.initial_program.read_text(encoding="utf-8") + "\n# evolved\n", encoding="utf-8")
        info.write_text(json.dumps({"metrics": {"combined_score": 0.9}}), encoding="utf-8")
        return OpenEvolveRunResult(
            success=True,
            command=["fake-openevolve"],
            returncode=0,
            work_dir=config.output_root / "work",
            output_dir=config.output_root,
            baseline_program=config.initial_program,
            evaluator=config.evaluator,
            config_file=config.config_file,
            best_program=best,
            best_info=info,
            best_metrics={"combined_score": 0.9},
            patch_text="--- initial_tool.py\n+++ best_program.py\n@@\n+# evolved\n",
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(module, "run_openevolve_isolated", fake_runner)

    result = module.evolve_tool_code(
        tool_name="normalize_tags",
        iterations=1,
        hermes_repo=str(repo),
        engine="openevolve",
        output_root=str(tmp_path / "out"),
        openevolve_cmd="fake-openevolve",
    )

    assert result is not None
    assert result["engine"] == "openevolve"
    assert result["success"] is True
    output_dir = Path(result["output_dir"])
    assert (output_dir / "initial_tool.py").exists()
    assert (output_dir / "evaluator.py").exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "patch.diff").read_text(encoding="utf-8").startswith("--- initial_tool.py")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["engine"] == "openevolve"
    assert metrics["best_metrics"]["combined_score"] == 0.9
    assert "fake-openevolve" in (output_dir / "report.md").read_text(encoding="utf-8")
    assert captured["config"].openevolve_cmd == "fake-openevolve"
    # OpenEvolve path is patch-only: original Hermes tool remains untouched.
    assert (repo / "tools" / "toy_tools.py").read_text(encoding="utf-8") == original_tool


def test_cli_accepts_engine_openevolve_for_dry_run(tmp_path: Path):
    repo = _fake_hermes_repo(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        module.main,
        [
            "--tool",
            "normalize_tags",
            "--engine",
            "openevolve",
            "--hermes-repo",
            str(repo),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine: openevolve" in result.output
    assert "DRY RUN" in result.output
