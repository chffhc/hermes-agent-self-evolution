"""Tests for the isolated OpenEvolve runner."""

from __future__ import annotations

from pathlib import Path

from evolution.code.openevolve_runner import OpenEvolveRunnerConfig, run_openevolve_isolated


def test_run_openevolve_isolated_copies_inputs_and_returns_patch(tmp_path: Path):
    initial = tmp_path / "initial_tool.py"
    evaluator = tmp_path / "evaluator.py"
    config = tmp_path / "config.yaml"
    fake_bin = tmp_path / "openevolve-run"

    initial.write_text(
        "def tool_normalize_tags(raw):\n"
        "    # EVOLVE-BLOCK-START\n"
        "    return [part.strip() for part in raw.split(',')]\n"
        "    # EVOLVE-BLOCK-END\n",
        encoding="utf-8",
    )
    original_initial = initial.read_text(encoding="utf-8")
    evaluator.write_text("def evaluate(path): return {'combined_score': 1.0}\n", encoding="utf-8")
    config.write_text("max_iterations: 1\n", encoding="utf-8")

    fake_bin.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import json, sys\n"
        "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.joinpath('best').mkdir(parents=True, exist_ok=True)\n"
        "baseline = Path(sys.argv[1]).read_text()\n"
        "best = baseline.replace(\"return [part.strip() for part in raw.split(',')]\", \"return sorted({part.strip().lower() for part in raw.split(',') if part.strip()})\")\n"
        "out.joinpath('best', 'best_program.py').write_text(best)\n"
        "out.joinpath('best', 'best_program_info.json').write_text(json.dumps({'metrics': {'combined_score': 1.0, 'pass_rate': 1.0}}))\n",
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)

    result = run_openevolve_isolated(
        OpenEvolveRunnerConfig(
            initial_program=initial,
            evaluator=evaluator,
            config_file=config,
            iterations=2,
            output_root=tmp_path,
            openevolve_cmd=str(fake_bin),
            env={"HTTP_PROXY": "should-not-leak", "CUSTOM_ENV": "ok"},
        )
    )

    assert result.success
    assert result.returncode == 0
    assert result.best_program is not None
    assert result.best_info is not None
    assert result.best_metrics["combined_score"] == 1.0
    assert "best_program.py" in result.patch_text
    assert "return sorted" in result.patch_text
    assert initial.read_text(encoding="utf-8") == original_initial
    assert result.work_dir != tmp_path
    assert result.baseline_program.read_text(encoding="utf-8") == original_initial
    assert result.config_file is not None and result.config_file.exists()


def test_run_openevolve_isolated_reports_missing_command(tmp_path: Path):
    initial = tmp_path / "initial.py"
    evaluator = tmp_path / "evaluator.py"
    initial.write_text("x = 1\n", encoding="utf-8")
    evaluator.write_text("def evaluate(path): return {'combined_score': 0.0}\n", encoding="utf-8")

    result = run_openevolve_isolated(
        OpenEvolveRunnerConfig(
            initial_program=initial,
            evaluator=evaluator,
            output_root=tmp_path,
            openevolve_cmd=str(tmp_path / "does-not-exist"),
            timeout_seconds=5,
        )
    )

    assert not result.success
    assert result.returncode == 127
    assert result.error is not None
    assert "command not found" in result.error
    assert result.patch_text == ""


def test_run_openevolve_isolated_unsets_proxy_env_by_default(tmp_path: Path):
    initial = tmp_path / "initial.py"
    evaluator = tmp_path / "evaluator.py"
    fake_bin = tmp_path / "openevolve-run"
    initial.write_text("x = 1\n", encoding="utf-8")
    evaluator.write_text("def evaluate(path): return {'combined_score': 0.0}\n", encoding="utf-8")

    fake_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.joinpath('best').mkdir(parents=True, exist_ok=True)\n"
        "out.joinpath('best', 'best_program.py').write_text(Path(sys.argv[1]).read_text())\n"
        "out.joinpath('best', 'best_program_info.json').write_text(json.dumps({'metrics': {'combined_score': 0.0}}))\n"
        "print('HTTP_PROXY=' + str(os.environ.get('HTTP_PROXY')))\n",
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)

    result = run_openevolve_isolated(
        OpenEvolveRunnerConfig(
            initial_program=initial,
            evaluator=evaluator,
            output_root=tmp_path,
            openevolve_cmd=str(fake_bin),
            env={"HTTP_PROXY": "http://proxy.example"},
        )
    )

    assert result.success
    assert "HTTP_PROXY=None" in result.stdout
