"""Adversarial tests for the private-snapshot wheel build entrypoint."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

import build_wheel

REPO = Path(__file__).resolve().parents[1]
_VERSION = "0.1.0"
_DIST_INFO = f"{build_wheel.DIST_NAME}-{_VERSION}.dist-info"


def _make_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("pyproject.toml", "README.md", "MANIFEST.in"):
        shutil.copy2(REPO / name, checkout / name)
    for package in ("evolution", "benchmarks"):
        shutil.copytree(
            REPO / package,
            checkout / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    return checkout


def _pollute_staging(checkout: Path) -> None:
    stale_lib = checkout / "build" / "lib"
    (stale_lib / "evolution" / "__pycache__").mkdir(parents=True)
    (stale_lib / "evolution" / "__pycache__" / "stale.cpython-311.pyc").write_bytes(b"stale")
    (stale_lib / "evolution" / "removed_module.py").write_text("REMOVED = True\n")
    (stale_lib / "poisoned_pkg").mkdir()
    (stale_lib / "poisoned_pkg" / "__init__.py").write_text("")
    (checkout / "build" / "bdist.macosx-fake").mkdir()
    egg_info = checkout / f"{build_wheel.DIST_NAME}.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: hermes-agent-self-evolution\n")
    (egg_info / "SOURCES.txt").write_text("evolution/removed_module.py\n")


def _plain_pip_wheel(checkout: Path, wheel_dir: Path) -> set[str]:
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
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as archive:
        return set(archive.namelist())


def _record_bytes(contents: dict[str, bytes], record_path: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in contents.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", str(len(data))))
    writer.writerow((record_path, "", ""))
    return output.getvalue().encode()


def _valid_contents(
    *,
    metadata_name: str = build_wheel.PROJECT_NAME,
    metadata_version: str = _VERSION,
    wheel_tag: str = "py3-none-any",
    wheel_version: str = "1.0",
) -> dict[str, bytes]:
    contents: dict[str, bytes] = dict.fromkeys(build_wheel.REQUIRED_WHEEL_ENTRIES, b"fixture")
    contents["evolution/__init__.py"] = b""
    contents[f"{_DIST_INFO}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: {metadata_version}\n\n"
    ).encode()
    contents[f"{_DIST_INFO}/WHEEL"] = (
        f"Wheel-Version: {wheel_version}\nGenerator: test\nRoot-Is-Purelib: true\n"
        f"Tag: {wheel_tag}\n\n"
    ).encode()
    return contents


def _write_valid_wheel(
    path: Path,
    *,
    metadata_name: str = build_wheel.PROJECT_NAME,
    metadata_version: str = _VERSION,
    wheel_tag: str = "py3-none-any",
    wheel_version: str = "1.0",
    omit: str | None = None,
    corrupt_record_hash: bool = False,
) -> Path:
    contents = _valid_contents(
        metadata_name=metadata_name,
        metadata_version=metadata_version,
        wheel_tag=wheel_tag,
        wheel_version=wheel_version,
    )
    if omit is not None:
        contents.pop(omit)
    record_path = f"{_DIST_INFO}/RECORD"
    record = _record_bytes(contents, record_path)
    if corrupt_record_hash:
        record = record.replace(b"sha256=", b"sha256=wrong", 1)
    contents[record_path] = record
    with ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return path


def test_entrypoint_builds_clean_snapshot_where_plain_rebuild_leaks(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path)
    _pollute_staging(checkout)

    leaked = _plain_pip_wheel(checkout, tmp_path / "dirty-wheel")
    assert "evolution/removed_module.py" in leaked
    assert any(name.split("/", 1)[0] == "poisoned_pkg" for name in leaked)

    wheel_dir = tmp_path / "clean-wheel"
    exit_code = build_wheel.main(["--project-root", str(checkout), "--wheel-dir", str(wheel_dir)])
    assert exit_code == 0

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    build_wheel.verify_wheel(wheels[0])
    with ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    assert "evolution/removed_module.py" not in names
    assert not any(name.split("/", 1)[0] == "poisoned_pkg" for name in names)
    # The repository staging is never deleted or modified by the supported build.
    assert (checkout / "build/lib/evolution/removed_module.py").is_file()
    assert (checkout / f"{build_wheel.DIST_NAME}.egg-info/SOURCES.txt").is_file()


def test_snapshot_excludes_source_bytecode_without_deleting_it(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path)
    cache = checkout / "evolution" / "__pycache__" / "local.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"local")
    snapshot = tmp_path / "snapshot"

    build_wheel.create_source_snapshot(checkout, snapshot)

    assert cache.read_bytes() == b"local"
    assert not (snapshot / "evolution/__pycache__").exists()


def test_snapshot_refuses_symlinked_source_input(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")
    (checkout / "evolution" / "linked.py").symlink_to(outside)

    with pytest.raises(build_wheel.BuildWheelError, match="symlink"):
        build_wheel.create_source_snapshot(checkout, tmp_path / "snapshot")

    assert outside.read_text() == "SECRET = True\n"


def test_snapshot_detects_source_change_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _make_checkout(tmp_path)
    victim = checkout / "evolution" / "__init__.py"
    original_copy = build_wheel.shutil.copyfile
    changed = False

    def racing_copy(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        follow_symlinks: bool = True,
    ) -> str:
        nonlocal changed
        result = original_copy(source, destination, follow_symlinks=follow_symlinks)
        if Path(source) == victim and not changed:
            victim.write_text("changed during snapshot\n")
            changed = True
        return str(result)

    monkeypatch.setattr(build_wheel.shutil, "copyfile", racing_copy)
    with pytest.raises(build_wheel.BuildWheelError, match="changed during snapshot"):
        build_wheel.create_source_snapshot(checkout, tmp_path / "snapshot")


def test_snapshot_detects_new_source_file_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _make_checkout(tmp_path)
    original_copy = build_wheel.shutil.copyfile
    added = False

    def racing_copy(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        follow_symlinks: bool = True,
    ) -> str:
        nonlocal added
        result = original_copy(source, destination, follow_symlinks=follow_symlinks)
        if not added:
            (checkout / "evolution" / "added_during_snapshot.py").write_text("ADDED = True\n")
            added = True
        return str(result)

    monkeypatch.setattr(build_wheel.shutil, "copyfile", racing_copy)
    with pytest.raises(build_wheel.BuildWheelError, match="changed during snapshot"):
        build_wheel.create_source_snapshot(checkout, tmp_path / "snapshot")


def test_verify_wheel_accepts_bound_metadata_and_record(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl")
    build_wheel.verify_wheel(wheel)


@pytest.mark.parametrize(
    ("name", "version", "match"),
    [
        ("totally-foreign", _VERSION, "Name"),
        (build_wheel.PROJECT_NAME, "999", "Version"),
    ],
)
def test_verify_wheel_binds_metadata_to_filename(
    tmp_path: Path, name: str, version: str, match: str
) -> None:
    wheel = _write_valid_wheel(
        tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl",
        metadata_name=name,
        metadata_version=version,
    )
    with pytest.raises(build_wheel.BuildWheelError, match=match):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_duplicate_metadata_name(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(
        tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl",
        metadata_name=f"{build_wheel.PROJECT_NAME}\nName: other-project",
    )
    with pytest.raises(build_wheel.BuildWheelError, match="uniquely match"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_binds_wheel_tag_to_filename(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(
        tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl",
        wheel_tag="cp311-none-any",
    )
    with pytest.raises(build_wheel.BuildWheelError, match="Tag fields"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_invalid_pep440_filename(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(tmp_path / f"{build_wheel.DIST_NAME}-1..2-py3-none-any.whl")
    with pytest.raises(build_wheel.BuildWheelError, match="invalid wheel filename"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_duplicate_wheel_tag(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(
        tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl",
        wheel_tag="py3-none-any\nTag: py3-none-any",
    )
    with pytest.raises(build_wheel.BuildWheelError, match="duplicate Tag"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_invalid_wheel_version(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(
        tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl",
        wheel_version="1.foo",
    )
    with pytest.raises(build_wheel.BuildWheelError, match="exactly Wheel-Version: 1.0"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_record_hash_mismatch(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(
        tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl",
        corrupt_record_hash=True,
    )
    with pytest.raises(build_wheel.BuildWheelError, match="RECORD hash"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_missing_required_entry(tmp_path: Path) -> None:
    missing = "benchmarks/capability/fixtures/fake_agent.py"
    wheel = _write_valid_wheel(
        tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl",
        omit=missing,
    )
    with pytest.raises(build_wheel.BuildWheelError, match="missing required"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_duplicate_entry(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl")
    with ZipFile(wheel, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("evolution/__init__.py", b"replacement")
    with pytest.raises(build_wheel.BuildWheelError, match="duplicate archive"):
        build_wheel.verify_wheel(wheel)


@pytest.mark.parametrize(
    "alias",
    [
        "evolution/Case.py",
        "evolution/café.py",
        "evolution/sub\\..\\escape.py",
        "evolution/CON.txt",
        "evolution/trailing. ",
        "evolution/control\x01.py",
    ],
)
def test_verify_wheel_rejects_cross_platform_path_ambiguity(tmp_path: Path, alias: str) -> None:
    wheel = _write_valid_wheel(tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl")
    with ZipFile(wheel, "a") as archive:
        archive.writestr(alias, b"ambiguous")
        if alias == "evolution/Case.py":
            archive.writestr("evolution/case.py", b"alias")
        if alias == "evolution/café.py":
            archive.writestr("evolution/cafe\u0301.py", b"unicode alias")
    with pytest.raises(
        build_wheel.BuildWheelError,
        match="alias|unsafe|ambiguous|reserved|control|RECORD omits",
    ):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_symlink_member(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl")
    link = ZipInfo("evolution/link.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(wheel, "a") as archive:
        archive.writestr(link, "target.py")
    with pytest.raises(build_wheel.BuildWheelError, match="symlink archive"):
        build_wheel.verify_wheel(wheel)


def test_verify_wheel_rejects_file_directory_collision(tmp_path: Path) -> None:
    wheel = _write_valid_wheel(tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl")
    with ZipFile(wheel, "a") as archive:
        archive.writestr("evolution/collision", b"file")
        archive.writestr("evolution/collision/child.py", b"child")
    with pytest.raises(build_wheel.BuildWheelError, match="file/directory archive path collision"):
        build_wheel.verify_wheel(wheel)


def test_output_refuses_symlink_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside)

    with pytest.raises(build_wheel.BuildWheelError, match="symlink component"):
        build_wheel._validate_output(alias / "wheels")
    with pytest.raises(build_wheel.BuildWheelError, match="securely create"):
        build_wheel._open_output_directory(alias / "wheels")

    assert not (outside / "wheels").exists()


def test_existing_output_refuses_without_touching_source_staging(tmp_path: Path) -> None:
    project = _make_checkout(tmp_path)
    staged = project / "build/lib/precious.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("preserve")
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    existing = wheel_dir / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl"
    existing.write_bytes(b"existing")

    assert build_wheel.main(["--project-root", str(project), "--wheel-dir", str(wheel_dir)]) == 1
    assert staged.read_text() == "preserve"
    assert existing.read_bytes() == b"existing"


def test_post_link_temp_cleanup_failure_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl"
    source.write_bytes(b"verified")
    wheel_dir = tmp_path / "wheels"
    original_unlink = build_wheel.os.unlink

    def fail_hidden_cleanup(path: str, *, dir_fd: int | None = None) -> None:
        if str(path).endswith(".tmp"):
            raise OSError("injected cleanup failure")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(build_wheel.os, "unlink", fail_hidden_cleanup)
    published = build_wheel._atomic_publish(source, wheel_dir)

    assert published.read_bytes() == b"verified"


def test_publish_rolls_back_if_directory_is_replaced_during_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl"
    source.write_bytes(b"verified")
    wheel_dir = tmp_path / "wheels"
    moved_dir = tmp_path / "moved-wheels"
    original_unlink = build_wheel.os.unlink
    replaced = False

    def replace_directory_during_cleanup(path: str, *, dir_fd: int | None = None) -> None:
        nonlocal replaced
        if str(path).endswith(".tmp") and not replaced:
            wheel_dir.rename(moved_dir)
            wheel_dir.mkdir()
            replaced = True
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(build_wheel.os, "unlink", replace_directory_during_cleanup)
    with pytest.raises(build_wheel.BuildWheelError, match="identity changed before commit"):
        build_wheel._atomic_publish(source, wheel_dir)

    assert not (wheel_dir / source.name).exists()
    assert not (moved_dir / source.name).exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup probe")
def test_process_timeout_kills_same_group_descendant(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    with pytest.raises(build_wheel.BuildWheelError, match="process tree was terminated"):
        build_wheel._run_process(
            [sys.executable, "-c", script, str(pid_file)], cwd=tmp_path, timeout=0.3
        )
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"same-group descendant {child_pid} survived timeout cleanup")


@pytest.mark.skipif(os.name != "posix", reason="POSIX detached-descendant cleanup probe")
def test_process_timeout_kills_detached_descendant(tmp_path: Path) -> None:
    pid_file = tmp_path / "detached.pid"
    marker = tmp_path / "late-write.txt"
    child_code = (
        "import pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid())); "
        "time.sleep(0.8); pathlib.Path(sys.argv[2]).write_text('escaped')"
    )
    parent_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]], "
        "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "time.sleep(30)"
    )
    with pytest.raises(build_wheel.BuildWheelError, match="process tree was terminated"):
        build_wheel._run_process(
            [
                sys.executable,
                "-c",
                parent_code,
                child_code,
                str(pid_file),
                str(marker),
            ],
            cwd=tmp_path,
            timeout=0.3,
        )

    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"detached descendant {child_pid} survived timeout cleanup")
    time.sleep(0.9)
    assert not marker.exists()


def test_main_normalizes_filesystem_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _make_checkout(tmp_path)

    def fail_tempdir(*args: object, **kwargs: object) -> None:
        raise OSError("injected filesystem failure")

    monkeypatch.setattr(build_wheel.tempfile, "TemporaryDirectory", fail_tempdir)
    assert (
        build_wheel.main(["--project-root", str(project), "--wheel-dir", str(tmp_path / "wheels")])
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: private wheel workflow failed")
    assert "Traceback" not in captured.err


def test_main_normalizes_project_root_symlink_loop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.symlink_to(second)
    second.symlink_to(first)

    assert build_wheel.main(["--project-root", str(first)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: cannot resolve build paths:")
    assert "Traceback" not in captured.err


def test_direct_script_reports_missing_declared_packaging_dependency() -> None:
    pyproject = (REPO / "pyproject.toml").read_text()
    assert '"packaging>=23.0"' in pyproject

    completed = subprocess.run(
        [sys.executable, "-S", str(REPO / "build_wheel.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == (
        "error: missing required dependency 'packaging>=23'; "
        "install project dependencies first\n"
    )


def test_main_fails_closed_without_pyproject(tmp_path: Path) -> None:
    assert (
        build_wheel.main(["--project-root", str(tmp_path), "--wheel-dir", str(tmp_path / "w")]) == 1
    )
    assert not (tmp_path / "w").exists()
