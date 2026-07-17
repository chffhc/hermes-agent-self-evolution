"""Distribution reproducibility checks for capability benchmark package data."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]


def test_wheel_excludes_fixture_bytecode_and_caches(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("pyproject.toml", "README.md", "MANIFEST.in"):
        shutil.copy2(REPO / name, checkout / name)
    for package in ("evolution", "benchmarks"):
        shutil.copytree(REPO / package, checkout / package)

    poison = (
        checkout
        / "benchmarks/capability/suites/native_v1/tasks/repair-calculator/workspace"
        / "__pycache__/poison.cpython-311.pyc"
    )
    poison.parent.mkdir(parents=True, exist_ok=True)
    poison.write_bytes(b"untrusted local bytecode")

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    with ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    assert not any("/__pycache__/" in name for name in names)
    assert not any(name.endswith((".pyc", ".pyo")) for name in names)
    assert "benchmarks/capability/suite.py" in names
    assert "benchmarks/capability/fixtures/fake_agent.py" in names
    assert "benchmarks/capability/fixtures/hermes_cli_stub.py" in names
    assert "benchmarks/capability/suites/native_v1/suite.json" in names
    assert (
        "benchmarks/capability/suites/native_v1/tasks/repair-calculator/workspace/calculator.py"
        in names
    )
