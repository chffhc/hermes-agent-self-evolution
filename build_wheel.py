"""Build and verify a wheel from a clean, read-only source snapshot.

A normal setuptools rebuild can reuse project-local ``build/lib`` and
``*.egg-info`` staging, leaking removed modules or bytecode into a wheel. This
entrypoint never deletes or reuses repository staging. It snapshots only the
explicit project inputs into a temporary directory, verifies the source did
not change while copied, builds in that private snapshot, validates the wheel
archive and RECORD, and publishes with no-clobber directory-relative syscalls.

Arbitrary third-party ``pip wheel .`` invocations that bypass this entrypoint
remain outside its stale-staging isolation guarantee. This workflow does not
claim byte-for-byte or environment-independent reproducibility.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

try:
    from packaging.tags import Tag, parse_tag
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
except ModuleNotFoundError as exc:  # stable direct-script diagnostic before argparse starts
    raise SystemExit(
        "error: missing required dependency 'packaging>=23'; install project dependencies first"
    ) from exc

PROJECT_NAME = "hermes-agent-self-evolution"
DIST_NAME = "hermes_agent_self_evolution"
_SOURCE_FILES = ("pyproject.toml", "README.md", "MANIFEST.in")
_SOURCE_DIRECTORIES = ("benchmarks", "evolution")
_MAX_WHEEL_ENTRIES = 10_000
_MAX_WHEEL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_BUILD_TIMEOUT_SECONDS = 300.0
_CLEANUP_TIMEOUT_SECONDS = 5.0
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)

REQUIRED_WHEEL_ENTRIES = (
    "benchmarks/capability/suite.py",
    "benchmarks/capability/fixtures/fake_agent.py",
    "benchmarks/capability/fixtures/hermes_cli_stub.py",
    "benchmarks/capability/suites/native_v1/suite.json",
    "benchmarks/capability/suites/native_v1/tasks/repair-calculator/workspace/calculator.py",
    "evolution/core/capability_feedback.py",
)
_ALLOWED_TOP_LEVEL_PACKAGES = frozenset({"benchmarks", "evolution"})


class BuildWheelError(RuntimeError):
    """The snapshot/build/verification workflow aborted without a valid publication."""


@dataclass(frozen=True)
class _SourceRecord:
    relative_path: Path
    size: int
    digest: str
    device: int
    inode: int
    mtime_ns: int


def _is_cache_path(relative: Path) -> bool:
    return any(part == "__pycache__" for part in relative.parts) or relative.name.endswith(
        (".pyc", ".pyo")
    )


def _hash_regular_file(path: Path) -> tuple[str, os.stat_result]:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise BuildWheelError(f"source input is not a regular file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise BuildWheelError(f"cannot read source input: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise BuildWheelError(f"source input changed while hashing: {path}")
    return digest.hexdigest(), after


def _enumerate_source(project_root: Path) -> list[_SourceRecord]:
    candidates: list[tuple[Path, Path]] = []
    for name in _SOURCE_FILES:
        candidates.append((project_root / name, Path(name)))
    for directory_name in _SOURCE_DIRECTORIES:
        directory = project_root / directory_name
        try:
            directory_stat = directory.stat(follow_symlinks=False)
        except OSError as exc:
            raise BuildWheelError(f"missing source directory: {directory}") from exc
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise BuildWheelError(f"source directory is not a regular directory: {directory}")
        try:
            descendants = sorted(directory.rglob("*"))
        except OSError as exc:
            raise BuildWheelError(f"cannot enumerate source directory: {directory}") from exc
        for path in descendants:
            relative = path.relative_to(project_root)
            if _is_cache_path(relative):
                continue
            try:
                item_stat = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise BuildWheelError(f"cannot inspect source input: {path}") from exc
            if stat.S_ISLNK(item_stat.st_mode):
                raise BuildWheelError(f"source snapshot refuses symlink: {path}")
            if stat.S_ISDIR(item_stat.st_mode):
                continue
            if not stat.S_ISREG(item_stat.st_mode):
                raise BuildWheelError(f"source snapshot refuses non-regular input: {path}")
            candidates.append((path, relative))

    records: list[_SourceRecord] = []
    for source, relative in candidates:
        try:
            source_stat = source.stat(follow_symlinks=False)
        except OSError as exc:
            raise BuildWheelError(f"missing source input: {source}") from exc
        if stat.S_ISLNK(source_stat.st_mode):
            raise BuildWheelError(f"source snapshot refuses symlink: {source}")
        digest, stable_stat = _hash_regular_file(source)
        records.append(
            _SourceRecord(
                relative_path=relative,
                size=stable_stat.st_size,
                digest=digest,
                device=stable_stat.st_dev,
                inode=stable_stat.st_ino,
                mtime_ns=stable_stat.st_mtime_ns,
            )
        )
    return records


def create_source_snapshot(project_root: Path, snapshot_root: Path) -> None:
    """Copy explicit source inputs and fail if any input changes during the snapshot."""
    records = _enumerate_source(project_root)
    try:
        snapshot_root.mkdir(parents=True, exist_ok=False)
        for record in records:
            source = project_root / record.relative_path
            destination = snapshot_root / record.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination, follow_symlinks=False)
    except OSError as exc:
        raise BuildWheelError("could not create private source snapshot") from exc

    current_records = _enumerate_source(project_root)
    if current_records != records:
        raise BuildWheelError("source input set or content changed during snapshot")

    for record in records:
        copied_digest, copied = _hash_regular_file(snapshot_root / record.relative_path)
        if copied.st_size != record.size or copied_digest != record.digest:
            raise BuildWheelError(f"private snapshot verification failed: {record.relative_path}")


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - exercised on Windows runners
                process.kill()
        except (ProcessLookupError, PermissionError):
            pass
    elif os.name == "posix":
        # The leader may have exited while same-group descendants retain pipes.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _token_process_ids(token: str) -> set[int]:
    """Find same-user POSIX processes that still inherit this private build token."""
    if os.name != "posix":
        return set()
    marker = f"HERMES_WHEEL_BUILD_PROCESS_TOKEN={token}"
    try:
        completed = subprocess.run(
            ["ps", "eww", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    process_ids: set[int] = set()
    for line in completed.stdout.splitlines():
        pid_text, separator, command = line.strip().partition(" ")
        if separator and marker in command:
            try:
                process_ids.add(int(pid_text))
            except ValueError:
                continue
    return process_ids


def _terminate_build_processes(process: subprocess.Popen[str], token: str) -> None:
    """Kill the build group plus detached descendants that retained the build token."""
    for process_id in _token_process_ids(token):
        if process_id == os.getpid():
            continue
        try:
            os.kill(process_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _terminate_process_group(process)


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _run_process(command: Sequence[str], *, cwd: Path, timeout: float) -> tuple[int, str, str]:
    token = secrets.token_hex(24)
    environment = os.environ.copy()
    environment["HERMES_WHEEL_BUILD_PROCESS_TOKEN"] = token
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        raise BuildWheelError("wheel build process could not start") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_build_processes(process, token)
        try:
            process.communicate(timeout=_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _close_process_pipes(process)
            try:
                process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        raise BuildWheelError("wheel build timed out and its process tree was terminated") from exc
    finally:
        _terminate_build_processes(process, token)
    return process.returncode, stdout, stderr


def build_wheel(snapshot_root: Path, private_wheel_dir: Path) -> Path:
    """Build one wheel inside private snapshot-owned output."""
    try:
        private_wheel_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise BuildWheelError("could not create private wheel output") from exc
    returncode, stdout, stderr = _run_process(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(private_wheel_dir),
        ],
        cwd=snapshot_root,
        timeout=_BUILD_TIMEOUT_SECONDS,
    )
    if returncode != 0:
        raise BuildWheelError(f"wheel build failed:\n{stdout}\n{stderr}")
    wheels = sorted(private_wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise BuildWheelError(f"expected exactly one wheel in private staging, found {len(wheels)}")
    _parse_wheel_identity(wheels[0])
    return wheels[0]


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_wheel_identity(wheel_path: Path) -> tuple[str, frozenset[Tag]]:
    try:
        distribution, version, _build, tags = parse_wheel_filename(wheel_path.name)
    except InvalidWheelFilename as exc:
        raise BuildWheelError(f"invalid wheel filename: {wheel_path.name}") from exc
    if _canonical_distribution(distribution) != _canonical_distribution(PROJECT_NAME):
        raise BuildWheelError(f"unexpected wheel distribution: {distribution}")
    return str(version), tags


def _validate_archive_path(name: str) -> tuple[str, str | None]:
    if not name or name.endswith("/") or name.startswith("/") or "\\" in name:
        return "", f"unsafe archive path: {name!r}"
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return "", f"control character in archive path: {name!r}"
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return "", f"unsafe archive path: {name!r}"
    for part in parts:
        if part.endswith((" ", ".")) or ":" in part:
            return "", f"platform-ambiguous archive path: {name!r}"
        stem = part.split(".", 1)[0].casefold()
        if stem in _WINDOWS_RESERVED:
            return "", f"Windows-reserved archive path: {name!r}"
    canonical = "/".join(unicodedata.normalize("NFC", part).casefold() for part in parts)
    return canonical, None


def _read_member(archive: ZipFile, info: ZipInfo) -> bytes:
    if info.file_size > _MAX_WHEEL_UNCOMPRESSED_BYTES:
        raise BuildWheelError(f"wheel member exceeds size limit: {info.filename}")
    try:
        data = archive.read(info)
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise BuildWheelError(f"wheel member failed CRC/decompression: {info.filename}") from exc
    if len(data) != info.file_size:
        raise BuildWheelError(f"wheel member size mismatch: {info.filename}")
    return data


def _verify_record(record_path: str, record_bytes: bytes, contents: dict[str, bytes]) -> None:
    try:
        rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="strict"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise BuildWheelError("wheel RECORD is not strict UTF-8 CSV") from exc
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3 or row[0] in seen:
            raise BuildWheelError("wheel RECORD contains malformed or duplicate rows")
        path, hash_field, size_field = row
        seen.add(path)
        if path not in contents:
            raise BuildWheelError(f"wheel RECORD references missing member: {path}")
        if path == record_path:
            if hash_field or size_field:
                raise BuildWheelError("wheel RECORD self-row must omit hash and size")
            continue
        expected_hash = base64.urlsafe_b64encode(hashlib.sha256(contents[path]).digest()).rstrip(
            b"="
        )
        if hash_field != f"sha256={expected_hash.decode('ascii')}":
            raise BuildWheelError(f"wheel RECORD hash mismatch: {path}")
        if size_field != str(len(contents[path])):
            raise BuildWheelError(f"wheel RECORD size mismatch: {path}")
    if seen != set(contents):
        missing = sorted(set(contents) - seen)
        raise BuildWheelError(f"wheel RECORD omits archive members: {missing}")


def verify_wheel(wheel_path: Path) -> None:
    """Verify archive identity, cross-platform paths, metadata, CRCs, and RECORD."""
    version, filename_tags = _parse_wheel_identity(wheel_path)
    expected_dist_info = f"{DIST_NAME}-{version}.dist-info"

    try:
        with ZipFile(wheel_path) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_WHEEL_ENTRIES:
                raise BuildWheelError(f"wheel has too many entries: {len(infos)}")
            total_size = sum(info.file_size for info in infos)
            if total_size > _MAX_WHEEL_UNCOMPRESSED_BYTES:
                raise BuildWheelError(f"wheel uncompressed size exceeds limit: {total_size}")
            contents = {info.filename: _read_member(archive, info) for info in infos}
    except BuildWheelError:
        raise
    except (BadZipFile, OSError) as exc:
        raise BuildWheelError(f"wheel verification could not read {wheel_path.name}") from exc

    names = [info.filename for info in infos]
    problems: list[str] = []
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        problems.append(f"duplicate archive entries: {duplicates}")

    canonical_paths: dict[str, str] = {}
    for info in infos:
        name = info.filename
        canonical, path_problem = _validate_archive_path(name)
        if path_problem:
            problems.append(path_problem)
            continue
        prior = canonical_paths.get(canonical)
        if prior is not None and prior != name:
            problems.append(f"cross-platform path alias: {prior!r} and {name!r}")
        canonical_paths[canonical] = name
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            problems.append(f"symlink archive entry must not ship: {name}")
        parts = name.split("/")
        if "__pycache__" in parts:
            problems.append(f"cache entry must not ship: {name}")
        if name.endswith((".pyc", ".pyo")):
            problems.append(f"bytecode entry must not ship: {name}")
        top = parts[0]
        if top not in _ALLOWED_TOP_LEVEL_PACKAGES and top != expected_dist_info:
            problems.append(f"unexpected top-level entry: {name}")

    for canonical, original in canonical_paths.items():
        parts = canonical.split("/")
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index])
            if prefix in canonical_paths:
                problems.append(
                    f"file/directory archive path collision: "
                    f"{canonical_paths[prefix]!r} and {original!r}"
                )

    present = set(names)
    for required in REQUIRED_WHEEL_ENTRIES:
        if required not in present:
            problems.append(f"missing required entry: {required}")
    metadata_path = f"{expected_dist_info}/METADATA"
    wheel_metadata_path = f"{expected_dist_info}/WHEEL"
    record_path = f"{expected_dist_info}/RECORD"
    for required in (metadata_path, wheel_metadata_path, record_path):
        if required not in present:
            problems.append(f"missing required wheel metadata: {required}")
    if problems:
        raise BuildWheelError(
            f"wheel verification failed for {wheel_path.name}:\n" + "\n".join(sorted(problems))
        )

    metadata_bytes = contents[metadata_path]
    wheel_metadata_bytes = contents[wheel_metadata_path]
    if len(metadata_bytes) > _MAX_METADATA_BYTES or len(wheel_metadata_bytes) > _MAX_METADATA_BYTES:
        raise BuildWheelError("wheel metadata exceeds validation limit")
    try:
        metadata_text = metadata_bytes.decode("utf-8", errors="strict")
        wheel_metadata_text = wheel_metadata_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BuildWheelError("wheel metadata is not strict UTF-8") from exc
    metadata = Parser().parsestr(metadata_text, headersonly=True)
    wheel_metadata = Parser().parsestr(wheel_metadata_text, headersonly=True)
    if metadata.defects or wheel_metadata.defects:
        raise BuildWheelError("wheel metadata contains malformed headers")
    names_in_metadata = metadata.get_all("Name", [])
    versions_in_metadata = metadata.get_all("Version", [])
    if len(names_in_metadata) != 1 or _canonical_distribution(
        names_in_metadata[0]
    ) != _canonical_distribution(PROJECT_NAME):
        raise BuildWheelError("wheel METADATA Name does not uniquely match wheel distribution")
    if versions_in_metadata != [version]:
        raise BuildWheelError("wheel METADATA Version does not uniquely match wheel filename")

    wheel_versions = wheel_metadata.get_all("Wheel-Version", [])
    if wheel_versions != ["1.0"]:
        raise BuildWheelError("wheel WHEEL metadata must contain exactly Wheel-Version: 1.0")
    tag_values = wheel_metadata.get_all("Tag", [])
    parsed_wheel_tags: set[Tag] = set()
    for tag_value in tag_values:
        try:
            value_tags = set(parse_tag(tag_value))
        except ValueError as exc:
            raise BuildWheelError("wheel WHEEL metadata contains an invalid Tag") from exc
        if not value_tags or parsed_wheel_tags.intersection(value_tags):
            raise BuildWheelError("wheel WHEEL metadata contains duplicate Tag coverage")
        parsed_wheel_tags.update(value_tags)
    if parsed_wheel_tags != set(filename_tags):
        raise BuildWheelError("wheel WHEEL Tag fields do not match wheel filename")
    _verify_record(record_path, contents[record_path], contents)


def _reject_existing_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            item = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise BuildWheelError(f"cannot inspect output path component: {current}") from exc
        if stat.S_ISLNK(item.st_mode):
            raise BuildWheelError(f"output path contains symlink component: {current}")


def _validate_output(wheel_dir: Path) -> None:
    _reject_existing_symlink_components(wheel_dir)
    try:
        if wheel_dir.exists() and not wheel_dir.is_dir():
            raise BuildWheelError(f"refusing output directory {wheel_dir}: expected a directory")
        existing = sorted(wheel_dir.glob(f"{DIST_NAME}-*.whl")) if wheel_dir.is_dir() else []
    except OSError as exc:
        raise BuildWheelError(f"cannot inspect output directory: {wheel_dir}") from exc
    if existing:
        raise BuildWheelError(
            f"refusing output directory {wheel_dir}: project wheel already exists; "
            "remove it or choose a fresh --wheel-dir"
        )


def _same_directory(path: Path, directory_fd: int) -> bool:
    try:
        path_stat = path.stat(follow_symlinks=False)
        fd_stat = os.fstat(directory_fd)
    except OSError:
        return False
    return stat.S_ISDIR(path_stat.st_mode) and (path_stat.st_dev, path_stat.st_ino) == (
        fd_stat.st_dev,
        fd_stat.st_ino,
    )


def _open_output_directory(wheel_dir: Path) -> int:
    """Create/open an absolute output path component-wise without following symlinks."""
    if os.name != "posix":  # pragma: no cover - fail closed on unsupported platforms
        raise BuildWheelError("secure wheel publication requires POSIX dirfd support")
    absolute = wheel_dir.absolute()
    if any(part in {".", ".."} for part in absolute.parts):
        raise BuildWheelError(f"output path contains ambiguous components: {wheel_dir}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory_fd = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise BuildWheelError(f"cannot open output path root: {absolute.anchor}") from exc
    try:
        for part in absolute.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except OSError as exc:
        os.close(directory_fd)
        raise BuildWheelError(f"cannot securely create output directory: {wheel_dir}") from exc
    return directory_fd


def _atomic_publish(source: Path, wheel_dir: Path) -> Path:
    directory_fd = _open_output_directory(wheel_dir)
    temporary_name = f".{source.name}.{secrets.token_hex(12)}.tmp"
    destination_name = source.name
    linked = False
    committed = False
    failure: BuildWheelError | None = None
    rollback_failed = False
    try:
        if not _same_directory(wheel_dir, directory_fd):
            raise BuildWheelError("output directory identity changed before publication")
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with source.open("rb") as source_stream, os.fdopen(temporary_fd, "wb") as output:
                shutil.copyfileobj(source_stream, output)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
            raise
        os.link(
            temporary_name,
            destination_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        linked = True
        if not _same_directory(wheel_dir, directory_fd):
            raise BuildWheelError("output directory identity changed during publication")
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except OSError:
            # A hidden cleanup residue does not invalidate the verified link, but
            # path identity must still be rechecked after this side effect.
            pass
        if not _same_directory(wheel_dir, directory_fd):
            raise BuildWheelError("output directory identity changed before commit")
        committed = True
    except BuildWheelError as exc:
        failure = exc
    except OSError as exc:
        failure = BuildWheelError(f"failed to publish verified wheel to {wheel_dir}")
        failure.__cause__ = exc
    finally:
        if not committed and linked:
            try:
                os.unlink(destination_name, dir_fd=directory_fd)
            except OSError:
                rollback_failed = True
        if not committed:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)

    if rollback_failed:
        raise BuildWheelError("wheel publication rollback could not be verified") from failure
    if failure is not None:
        raise failure
    return wheel_dir / destination_name


def build_verify_publish(project_root: Path, wheel_dir: Path) -> Path:
    """Snapshot source, privately build/verify, then atomically publish."""
    if not (project_root / "pyproject.toml").is_file():
        raise BuildWheelError(f"{project_root} does not contain pyproject.toml")
    _validate_output(wheel_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-wheel-build-") as temporary_dir:
            temporary_root = Path(temporary_dir)
            snapshot_root = temporary_root / "source"
            private_output = temporary_root / "wheel"
            create_source_snapshot(project_root, snapshot_root)
            wheel_path = build_wheel(snapshot_root, private_output)
            verify_wheel(wheel_path)
            return _atomic_publish(wheel_path, wheel_dir)
    except BuildWheelError:
        raise
    except OSError as exc:
        raise BuildWheelError("private wheel workflow failed") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify a wheel from a clean private source snapshot."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory containing pyproject.toml (default: this repository).",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=None,
        help="Output directory for the verified wheel (default: <project-root>/dist).",
    )
    args = parser.parse_args(argv)

    try:
        project_root = args.project_root.resolve()
        wheel_dir = args.wheel_dir if args.wheel_dir is not None else project_root / "dist"
        if not wheel_dir.is_absolute():
            wheel_dir = Path.cwd() / wheel_dir
        wheel_path = build_verify_publish(project_root, wheel_dir)
    except BuildWheelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as exc:
        print(f"error: cannot resolve build paths: {exc}", file=sys.stderr)
        return 1
    print(f"verified clean wheel: {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
