"""Tests for the unsigned wheel build receipt and its fail-closed verifier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest

import build_wheel

REPO = Path(__file__).resolve().parents[1]
_VERSION = "0.1.0"
_WHEEL_NAME = f"{build_wheel.DIST_NAME}-{_VERSION}-py3-none-any.whl"


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


def _records() -> list[build_wheel._SourceRecord]:
    return [
        build_wheel._SourceRecord(
            relative_path=Path("evolution/__init__.py"),
            size=0,
            digest=hashlib.sha256(b"").hexdigest(),
            device=1,
            inode=2,
            mtime_ns=3,
        ),
        build_wheel._SourceRecord(
            relative_path=Path("pyproject.toml"),
            size=4,
            digest=hashlib.sha256(b"toml").hexdigest(),
            device=1,
            inode=5,
            mtime_ns=6,
        ),
    ]


def _publish_pair(tmp_path: Path) -> tuple[Path, Path]:
    wheel = tmp_path / _WHEEL_NAME
    wheel.write_bytes(b"wheel-bytes")
    receipt_bytes = build_wheel._render_receipt(
        wheel.name, hashlib.sha256(b"wheel-bytes").hexdigest(), len(b"wheel-bytes"), _records()
    )
    receipt = tmp_path / f"{wheel.name}{build_wheel.RECEIPT_SUFFIX}"
    receipt.write_bytes(receipt_bytes)
    return receipt, wheel


def _rewrite(receipt: Path, mutate) -> None:
    document = json.loads(receipt.read_text())
    mutate(document)
    receipt.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def test_receipt_round_trip_verifies(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    document = build_wheel.verify_receipt(receipt, wheel)
    assert document["schema"] == "hermes-wheel-build-receipt-v1"
    assert document["claims"] == {
        "workflow_asserted_stale_staging_isolated": True,
        "workflow_asserted_source_snapshot_unchanged": True,
        "workflow_asserted_wheel_archive_verified": True,
        "receipt_authenticated": False,
        "byte_for_byte_reproducible": False,
        "environment_independent": False,
        "hermetic_build": False,
        "capability_evidence": False,
    }
    snapshot = document["source_snapshot"]
    assert [entry["path"] for entry in snapshot["files"]] == [
        "evolution/__init__.py",
        "pyproject.toml",
    ]
    assert snapshot["file_count"] == 2
    assert snapshot["total_size_bytes"] == 4


def test_verify_receipt_rejects_modified_wheel_bytes(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    wheel.write_bytes(b"tampered-bts")
    with pytest.raises(build_wheel.BuildWheelError, match="do not match the build receipt digest"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_manifest_entry_tamper(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)

    def swap_digest(document: dict) -> None:
        document["source_snapshot"]["files"][0]["sha256"] = hashlib.sha256(b"evil").hexdigest()

    _rewrite(receipt, swap_digest)
    with pytest.raises(build_wheel.BuildWheelError, match="manifest_sha256"):
        build_wheel.verify_receipt(receipt, wheel)


def test_receipt_verifier_does_not_claim_manifest_authentication(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)

    def replace_self_consistent_manifest(document: dict) -> None:
        files = document["source_snapshot"]["files"]
        files[0]["sha256"] = hashlib.sha256(b"different-untrusted-source").hexdigest()
        manifest = [(entry["path"], entry["sha256"], entry["size_bytes"]) for entry in files]
        document["source_snapshot"]["manifest_sha256"] = build_wheel._manifest_digest(manifest)

    _rewrite(receipt, replace_self_consistent_manifest)
    document = build_wheel.verify_receipt(receipt, wheel)
    claims = document["claims"]
    assert isinstance(claims, dict)
    assert claims["receipt_authenticated"] is False


def test_verify_receipt_rejects_reproducibility_overclaim(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)

    def overclaim(document: dict) -> None:
        document["claims"]["byte_for_byte_reproducible"] = True

    _rewrite(receipt, overclaim)
    with pytest.raises(build_wheel.BuildWheelError, match="byte_for_byte_reproducible"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_non_boolean_claim(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)

    def numeric_claim(document: dict) -> None:
        document["claims"]["workflow_asserted_stale_staging_isolated"] = 1

    _rewrite(receipt, numeric_claim)
    with pytest.raises(build_wheel.BuildWheelError, match="workflow_asserted_stale"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    text = receipt.read_text()
    text = text.replace('"schema":', '"schema": "duplicate",\n  "schema":', 1)
    receipt.write_text(text)
    with pytest.raises(build_wheel.BuildWheelError, match="duplicate JSON key"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    _rewrite(receipt, lambda document: document.update(extra=True))
    with pytest.raises(build_wheel.BuildWheelError, match="exactly the fields"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_unsorted_manifest(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)

    def reverse_files(document: dict) -> None:
        document["source_snapshot"]["files"].reverse()

    _rewrite(receipt, reverse_files)
    with pytest.raises(build_wheel.BuildWheelError, match="unique and sorted"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_traversal_manifest_path(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)

    def traverse(document: dict) -> None:
        document["source_snapshot"]["files"][0]["path"] = "../escape.py"

    _rewrite(receipt, traverse)
    with pytest.raises(build_wheel.BuildWheelError, match="unsafe"):
        build_wheel.verify_receipt(receipt, wheel)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-20T00:00:00",
        "2026-07-20T08:00:00+08:00",
        "2026-07-20T00:00:00-00:00",
        "2026-07-20X00:00:00+00:00",
        "20260720T000000+0000",
    ],
)
def test_verify_receipt_rejects_noncanonical_timestamp(tmp_path: Path, timestamp: str) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    _rewrite(receipt, lambda document: document.update(created_utc=timestamp))
    with pytest.raises(build_wheel.BuildWheelError, match="must use YYYY"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_cross_platform_manifest_alias(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)

    def alias_path(document: dict) -> None:
        document["source_snapshot"]["files"][1]["path"] = "EVOLUTION/__init__.py"

    _rewrite(receipt, alias_path)
    with pytest.raises(build_wheel.BuildWheelError, match="manifest path alias"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_bounds_bytes_read(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    receipt.write_bytes(b" " * (build_wheel._MAX_RECEIPT_BYTES + 1))
    with pytest.raises(build_wheel.BuildWheelError, match="size limit"):
        build_wheel.verify_receipt(receipt, wheel)


def test_verify_receipt_rejects_wrong_wheel_filename_binding(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    other = tmp_path / f"{build_wheel.DIST_NAME}-9.9.9-py3-none-any.whl"
    other.write_bytes(wheel.read_bytes())
    with pytest.raises(build_wheel.BuildWheelError, match="filename does not match"):
        build_wheel.verify_receipt(receipt, other)


def test_verify_receipt_rejects_symlinked_receipt(tmp_path: Path) -> None:
    receipt, wheel = _publish_pair(tmp_path)
    alias = tmp_path / "alias.receipt.json"
    alias.symlink_to(receipt)
    with pytest.raises(build_wheel.BuildWheelError, match="not a regular file"):
        build_wheel.verify_receipt(alias, wheel)


def test_output_dir_refuses_preexisting_receipt(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    (wheel_dir / f"{_WHEEL_NAME}{build_wheel.RECEIPT_SUFFIX}").write_text("{}")
    with pytest.raises(build_wheel.BuildWheelError, match="already"):
        build_wheel._validate_output(wheel_dir)


def test_publish_pair_rolls_back_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel_source = tmp_path / _WHEEL_NAME
    wheel_source.write_bytes(b"verified")
    receipt_source = tmp_path / f"{_WHEEL_NAME}{build_wheel.RECEIPT_SUFFIX}"
    receipt_source.write_bytes(b"{}")
    wheel_dir = tmp_path / "wheels"
    original_link = build_wheel.os.link

    def fail_wheel_commit_link(source: str, destination: str, **kwargs: Any) -> None:
        if str(destination).endswith(".whl"):
            raise OSError("injected wheel commit-marker link failure")
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(build_wheel.os, "link", fail_wheel_commit_link)
    with pytest.raises(build_wheel.BuildWheelError, match="failed to publish"):
        build_wheel._atomic_publish_all(
            [
                build_wheel._PublishItem(receipt_source, receipt_source.name),
                build_wheel._PublishItem(wheel_source, wheel_source.name),
            ],
            wheel_dir,
        )

    assert not (wheel_dir / wheel_source.name).exists()
    assert not (wheel_dir / receipt_source.name).exists()
    assert [entry for entry in os.listdir(wheel_dir) if entry.endswith(".tmp")] == []


def test_publish_rolls_back_post_link_error_in_commit_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_source = tmp_path / _WHEEL_NAME
    wheel_source.write_bytes(b"verified")
    receipt_source = tmp_path / f"{_WHEEL_NAME}{build_wheel.RECEIPT_SUFFIX}"
    receipt_source.write_bytes(b"{}")
    wheel_dir = tmp_path / "wheels"
    original_link = build_wheel.os.link
    original_unlink = build_wheel.os.unlink
    rollback_order: list[str] = []

    def link_then_fail(source: str, destination: str, **kwargs: Any) -> None:
        original_link(source, destination, **kwargs)
        if str(destination).endswith(".whl"):
            raise OSError("injected error after real wheel link")

    def record_unlink(path: str, **kwargs: Any) -> None:
        if str(path) in {wheel_source.name, receipt_source.name}:
            rollback_order.append(str(path))
        original_unlink(path, **kwargs)

    monkeypatch.setattr(build_wheel.os, "link", link_then_fail)
    monkeypatch.setattr(build_wheel.os, "unlink", record_unlink)
    with pytest.raises(build_wheel.BuildWheelError, match="failed to publish"):
        build_wheel._atomic_publish_all(
            [
                build_wheel._PublishItem(receipt_source, receipt_source.name),
                build_wheel._PublishItem(wheel_source, wheel_source.name),
            ],
            wheel_dir,
        )

    assert rollback_order == [wheel_source.name, receipt_source.name]
    assert not (wheel_dir / wheel_source.name).exists()
    assert not (wheel_dir / receipt_source.name).exists()
    assert [entry for entry in os.listdir(wheel_dir) if entry.endswith(".tmp")] == []


def test_publish_binds_bytes_to_verified_digest(tmp_path: Path) -> None:
    source = tmp_path / _WHEEL_NAME
    source.write_bytes(b"drifted-after-verification")
    wheel_dir = tmp_path / "wheels"
    with pytest.raises(build_wheel.BuildWheelError, match="do not match verified digest"):
        build_wheel._atomic_publish_all(
            [
                build_wheel._PublishItem(
                    source, source.name, hashlib.sha256(b"verified").hexdigest()
                )
            ],
            wheel_dir,
        )
    assert not (wheel_dir / source.name).exists()


def test_main_requires_paired_verify_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert build_wheel.main(["--verify-receipt", str(tmp_path / "r.json")]) == 1
    assert build_wheel.main(["--wheel", str(tmp_path / "w.whl")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("must be used together") == 2


def test_build_rejects_snapshot_changed_by_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _make_checkout(tmp_path)
    wheel_dir = tmp_path / "wheels"

    def mutating_backend(snapshot_root: Path, private_output: Path) -> Path:
        (snapshot_root / "evolution/__init__.py").write_text("changed during build\n")
        private_output.mkdir()
        wheel = private_output / _WHEEL_NAME
        wheel.write_bytes(b"not-reached")
        return wheel

    monkeypatch.setattr(build_wheel, "build_wheel", mutating_backend)
    with pytest.raises(build_wheel.BuildWheelError, match="snapshot changed during wheel build"):
        build_wheel.build_verify_publish(checkout, wheel_dir)
    assert not wheel_dir.exists()


def test_entrypoint_publishes_verifiable_receipt_pair(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _make_checkout(tmp_path)
    wheel_dir = tmp_path / "wheels"
    original_link = build_wheel.os.link
    linked_destinations: list[str] = []

    def record_link(source: str, destination: str, **kwargs: Any) -> None:
        linked_destinations.append(str(destination))
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(build_wheel.os, "link", record_link)
    assert build_wheel.main(["--project-root", str(checkout), "--wheel-dir", str(wheel_dir)]) == 0
    assert linked_destinations[-2].endswith(build_wheel.RECEIPT_SUFFIX)
    assert linked_destinations[-1].endswith(".whl")

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    receipt = wheel_dir / f"{wheel.name}{build_wheel.RECEIPT_SUFFIX}"
    assert receipt.is_file()
    build_output = capsys.readouterr()
    assert str(receipt) in build_output.out

    document = build_wheel.verify_receipt(receipt, wheel)
    assert document["wheel"] == {
        "filename": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "size_bytes": wheel.stat().st_size,
    }
    manifest_paths = {entry["path"] for entry in document["source_snapshot"]["files"]}
    assert "pyproject.toml" in manifest_paths
    assert "evolution/__init__.py" in manifest_paths
    assert not any("__pycache__" in path for path in manifest_paths)

    # The receipt binds sources actually shipped: every packaged module is in the manifest.
    with ZipFile(wheel) as archive:
        packaged = {
            name
            for name in archive.namelist()
            if name.split("/", 1)[0] in {"benchmarks", "evolution"} and name.endswith(".py")
        }
    assert packaged <= manifest_paths

    assert build_wheel.main(["--verify-receipt", str(receipt), "--wheel", str(wheel)]) == 0
    verify_output = capsys.readouterr()
    assert verify_output.out.startswith("verified receipt/wheel digest consistency:")

    wheel.write_bytes(wheel.read_bytes() + b"#tamper")
    assert build_wheel.main(["--verify-receipt", str(receipt), "--wheel", str(wheel)]) == 1
    tampered = capsys.readouterr()
    assert "do not match the build receipt digest" in tampered.err
