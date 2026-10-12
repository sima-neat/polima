"""Dataset contract validation, driven by DatasetContract."""

from __future__ import annotations

import json

import pytest

from polima.data.contract import canonical_task, validate, validate_all
from polima.policies.act import ACT_SPEC

CONTRACT = ACT_SPEC.dataset

JOINTS = list(CONTRACT.state_names)


def write_dataset(root, *, features=None, fps=30, episodes=10, frames=1000,
                  version="v3.0", with_tasks=True):
    """Build a minimal LeRobot-shaped tree: meta/info.json + data/."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    if features is None:
        features = {
            "observation.state": {"shape": [6], "names": JOINTS},
            "action": {"shape": [6], "names": JOINTS},
            "observation.images.overhead": {"shape": [480, 640, 3]},
            "observation.images.wrist": {"shape": [480, 640, 3]},
        }
    info = {
        "features": features, "fps": fps, "total_episodes": episodes,
        "total_frames": frames, "codebase_version": version,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    if with_tasks:
        tasks_path = root / "meta" / "tasks.parquet"
        try:
            import pyarrow as arrow
            from pyarrow import parquet
        except ImportError:
            # PyArrow is optional for the base package; validation records that
            # the task check was skipped when it is not installed.
            tasks_path.touch()
        else:
            parquet.write_table(arrow.table({"task": ["pick up the red cube"]}), tasks_path)
    return root


def test_canonical_task():
    assert canonical_task("Pick up the RED cube.") == "pick up the red cube"
    assert canonical_task("  multiple   spaces  ") == "multiple spaces"
    assert canonical_task("Trailing!!!") == "trailing"
    assert canonical_task("Question?") == "question"
    assert canonical_task("Keep. inner. dots") == "keep. inner. dots"


def test_valid_dataset_passes(tmp_path):
    report = validate(write_dataset(tmp_path / "ok"), CONTRACT)
    assert report.ok, report.violations
    assert report.episodes == 10 and report.frames == 1000 and report.fps == 30


def test_missing_dataset_is_reported_not_raised(tmp_path):
    report = validate(tmp_path / "nope", CONTRACT)
    assert not report.ok
    assert "incomplete LeRobot dataset" in report.violations[0]


def test_wrong_fps(tmp_path):
    report = validate(write_dataset(tmp_path / "fps", fps=60), CONTRACT)
    assert not report.ok
    assert any("30 fps" in v for v in report.violations)


def test_wrong_state_dim(tmp_path):
    features = {
        "observation.state": {"shape": [7], "names": JOINTS + ["extra.pos"]},
        "action": {"shape": [6], "names": JOINTS},
        "observation.images.overhead": {"shape": [480, 640, 3]},
        "observation.images.wrist": {"shape": [480, 640, 3]},
    }
    report = validate(write_dataset(tmp_path / "dim", features=features), CONTRACT)
    assert not report.ok
    assert any("shape must be [6]" in v for v in report.violations)


def test_wrong_camera_keys(tmp_path):
    features = {
        "observation.state": {"shape": [6], "names": JOINTS},
        "action": {"shape": [6], "names": JOINTS},
        "observation.images.front": {"shape": [480, 640, 3]},
        "observation.images.wrist": {"shape": [480, 640, 3]},
    }
    report = validate(write_dataset(tmp_path / "cams", features=features), CONTRACT)
    assert not report.ok
    assert any("expected image keys" in v for v in report.violations)


def test_wrong_camera_shape(tmp_path):
    features = {
        "observation.state": {"shape": [6], "names": JOINTS},
        "action": {"shape": [6], "names": JOINTS},
        "observation.images.overhead": {"shape": [240, 320, 3]},
        "observation.images.wrist": {"shape": [480, 640, 3]},
    }
    report = validate(write_dataset(tmp_path / "shape", features=features), CONTRACT)
    assert not report.ok
    assert any("HWC shape [480, 640, 3]" in v for v in report.violations)


def test_all_violations_collected_not_just_the_first(tmp_path):
    """Every problem is reported, not just the first."""
    features = {
        "observation.state": {"shape": [7], "names": ["a"]},
        "action": {"shape": [7], "names": ["a"]},
        "observation.images.front": {"shape": [240, 320, 3]},
    }
    report = validate(write_dataset(tmp_path / "many", features=features, fps=60), CONTRACT)
    assert not report.ok
    assert len(report.violations) >= 4


def test_wrong_codebase_version(tmp_path):
    report = validate(write_dataset(tmp_path / "v21", version="v2.1"), CONTRACT)
    assert not report.ok
    assert any("codebase_version" in v for v in report.violations)


def test_validate_all_cross_checks_tasks(tmp_path, monkeypatch):
    a = write_dataset(tmp_path / "a")
    b = write_dataset(tmp_path / "b")

    import polima.data.contract as contract_module

    tasks = {str(a.resolve()): ["pick red"], str(b.resolve()): ["pick blue"]}

    def fake_check(report, root, tasks_path, contract, allow_mixed):
        report.tasks = tasks[report.root]
        report.task_labels = report.tasks

    monkeypatch.setattr(contract_module, "_check_tasks", fake_check)
    reports, found = validate_all([a, b], CONTRACT)
    assert found == ["pick blue", "pick red"]
    assert all(not r.ok for r in reports)
    assert any("different tasks" in v for r in reports for v in r.violations)

    reports, _ = validate_all([a, b], CONTRACT, allow_mixed_tasks=True)
    assert all(r.ok for r in reports)


@pytest.mark.parametrize("missing", ["meta/info.json", "data"])
def test_incomplete_layouts(tmp_path, missing):
    root = write_dataset(tmp_path / "partial")
    target = root / missing
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()
    assert not validate(root, CONTRACT).ok
