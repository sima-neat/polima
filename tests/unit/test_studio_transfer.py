from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from polima.studio import transfer
from polima.studio.runtime import Conflict, StudioRuntime

ELF = b"\x7fELF compiled graph"


def bundle_files(bundle_id: str = "act-demo-1234") -> dict[str, bytes]:
    manifest = {
        "bundle_id": bundle_id, "policy": "act", "format": "polima-bundle-v1",
        "tasks": ["Pick up the block.", "Stack the blocks."], "default_task": 1,
        "graphs": [{"name": "backbone", "elf": "models/backbone.elf",
                    "elf_bytes": len(ELF), "sha256": hashlib.sha256(ELF).hexdigest()}],
    }
    return {
        f"{bundle_id}/bundle.json": json.dumps(manifest).encode(),
        f"{bundle_id}/models/backbone.elf": ELF,
        f"{bundle_id}/robot_client/start.sh": b"#!/usr/bin/env bash\n",
        f"{bundle_id}/robot_client/calibration/so-arm101.json": b'{"uploaded": true}',
    }


def upload(uploads: transfer.Uploads, files: dict[str, bytes], chunk: int = 7) -> str:
    """Send each file in small chunks, the way the browser does."""
    state = uploads.create([{"path": p, "size": len(d)} for p, d in files.items()], 0)
    for path, data in files.items():
        for offset in range(0, len(data), chunk):
            piece = data[offset:offset + chunk]
            uploads.receive(state["id"], path, offset, io.BytesIO(piece), len(piece))
    return state["id"]


def progress(*_: object) -> None:
    pass


def test_paths_cannot_leave_their_root(tmp_path: Path) -> None:
    for bad in ("../x", "/etc/passwd", "a/../../x", "", "a\\..\\x"):
        with pytest.raises(ValueError):
            transfer.safe_relpath(bad)
    (tmp_path / "inside").mkdir()
    (tmp_path / "link").symlink_to("/")
    assert transfer.confined(tmp_path, "inside") == tmp_path / "inside"
    with pytest.raises(ValueError):
        transfer.confined(tmp_path, "link/etc")


def test_interrupted_upload_resumes_from_what_is_on_disk(tmp_path: Path) -> None:
    uploads = transfer.Uploads(tmp_path)
    state = uploads.create([{"path": "a.bin", "size": 10}], 0)
    uploads.receive(state["id"], "a.bin", 0, io.BytesIO(b"01234"), 5)
    # A retry of the first chunk is told where to resume instead of duplicating data.
    assert uploads.receive(state["id"], "a.bin", 0, io.BytesIO(b"01234"), 5) == {
        "ok": False, "path": "a.bin", "received": 5}
    uploads.receive(state["id"], "a.bin", 5, io.BytesIO(b"56789"), 5)
    assert uploads.status(state["id"])["complete"]


def test_install_verifies_and_keeps_the_boards_calibration(tmp_path: Path) -> None:
    models, uploads = tmp_path / "models", transfer.Uploads(tmp_path / "uploads")
    installed = models / "act-demo-1234/robot_client/calibration/so-arm101.json"
    installed.parent.mkdir(parents=True)
    installed.write_text('{"board": true}')

    upload_id = upload(uploads, bundle_files())
    result = transfer.install_bundle(uploads.tree(upload_id, progress), models,
                                     uploads.path(upload_id), progress, lambda _: None)

    assert result["replaced"] and result["kept_calibration"] == ["so-arm101.json"]
    assert json.loads(installed.read_text()) == {"board": True}
    assert (models / "act-demo-1234/robot_client/start.sh").stat().st_mode & 0o111


def test_install_refuses_a_corrupted_elf(tmp_path: Path) -> None:
    files = bundle_files()
    files["act-demo-1234/models/backbone.elf"] = b"\x7fELF truncated....."
    uploads = transfer.Uploads(tmp_path / "uploads")
    upload_id = upload(uploads, files)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        transfer.install_bundle(uploads.tree(upload_id, progress), tmp_path / "models",
                                uploads.path(upload_id), progress, lambda _: None)
    assert not (tmp_path / "models/act-demo-1234").exists()


def test_archive_members_must_be_plain_files_inside_the_bundle(tmp_path: Path) -> None:
    for member in (tarfile.TarInfo("../escape"), tarfile.TarInfo("bundle/link")):
        archive = tmp_path / "bad.tar"
        with tarfile.open(archive, "w") as tar:
            if member.name.endswith("link"):
                member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(member, io.BytesIO(b""))
        with pytest.raises(ValueError):
            transfer.extract_archive(archive, tmp_path / "out", progress)
        assert not (tmp_path / "escape").exists()


def test_folder_download_is_a_valid_tar_without_symlinks(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    (logs / "run").mkdir(parents=True)
    (logs / "run/robot.log").write_text("step 1\n")
    (logs / "secret").symlink_to("/etc/passwd")

    with tarfile.open(fileobj=io.BytesIO(b"".join(transfer.stream_tar(logs)))) as tar:
        assert tar.getnames() == ["logs/run/robot.log"]
        assert tar.extractfile("logs/run/robot.log").read() == b"step 1\n"


def test_studio_never_deletes_the_active_bundle(tmp_path: Path) -> None:
    runtime = StudioRuntime(tmp_path)
    for name, data in bundle_files().items():
        (tmp_path / "models" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / name).write_bytes(data)
    (tmp_path / "current").symlink_to(tmp_path / "models/act-demo-1234")

    with pytest.raises(Conflict):
        runtime.delete_bundle("act-demo-1234")
    (tmp_path / "current").unlink()
    assert runtime.delete_bundle("act-demo-1234")["calibration_backup"]
    assert not (tmp_path / "models/act-demo-1234").exists()


def test_default_task_index_selects_from_tasks(tmp_path: Path) -> None:
    manifest = json.loads(bundle_files()["act-demo-1234/bundle.json"])
    assert StudioRuntime._bundle_tasks(tmp_path, manifest)[0] == "Stack the blocks."


@pytest.fixture
def arm_calibration():
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    return {name: {"id": index, "drive_mode": 0, "homing_offset": -120,
                   "range_min": 100, "range_max": 3900}
            for index, name in enumerate(names, 1)}


def test_calibration_upload_accepts_so101_registers(arm_calibration):
    arm_calibration["wrist_roll"].update(range_min=0, range_max=4095, drive_mode=1)
    assert transfer.validate_calibration(arm_calibration, None) == arm_calibration


@pytest.mark.parametrize("field,value", [
    ("id", 2), ("id", True), ("drive_mode", 2), ("drive_mode", False),
    ("range_min", -1), ("range_min", 3900), ("range_min", 4000),
    ("range_max", 4096), ("homing_offset", 0.5),
])
def test_calibration_upload_rejects_invalid_registers(arm_calibration, field, value):
    arm_calibration["shoulder_pan"][field] = value
    with pytest.raises(ValueError):
        transfer.validate_calibration(arm_calibration, None)


@pytest.mark.parametrize("replace", [False, True])
def test_calibration_upload_requires_all_six_joints(arm_calibration, replace):
    joint = arm_calibration.pop("gripper")
    if replace:
        arm_calibration["another_joint"] = joint
    with pytest.raises(ValueError, match="six SO-101"):
        transfer.validate_calibration(arm_calibration, None)
