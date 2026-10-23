from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from polima.robot import evidence

META = {"bundle_id": "act-x", "policy": "ACT", "robot": {"type": "SO-101 follower"},
        "task": {"id": 0, "instruction": "Pick the block"}, "cameras": {"overhead": "/dev/a"},
        "calibration": {"file": "so-arm101.json", "sha256": "ab"}, "rest_pose": {"method": "median"},
        "settings": {"fps": 30}}


def run(tmp_path: Path, **meta) -> evidence.RunRecorder:
    return evidence.RunRecorder.create(tmp_path, **{**META, **meta})


def test_a_completed_run_summarizes_what_happened(tmp_path):
    recorder = run(tmp_path)
    for ms in (20.0, 30.0, 40.0):
        recorder.event("inference_completed", step=0, actions=16, model_ms=ms, queue_ms=ms + 90)
    for _ in range(5):
        recorder.event("action_sent", step=1, action={"gripper.pos": 2.0})
    recorder.event("grasp_detected", gripper=1.0)
    recorder.event("cycle_completed", cycle=1)
    recorder.event("warning", message="slow camera")

    summary = recorder.finish("completed")

    events = evidence.read_events(recorder.folder)
    assert [events[0]["event"], events[-1]["event"]] == ["run_started", "run_completed"]
    assert summary["status"] == "completed" and summary["bundle_id"] == "act-x"
    assert summary["task"]["instruction"] == "Pick the block"
    assert (summary["inferences"], summary["actions"], summary["cycles"], summary["grasps"]) == (3, 5, 1, 1)
    assert summary["latency_ms"]["model"] == {"p50": 30.0, "p95": 40.0, "max": 40.0}
    assert summary["warnings"] == ["slow camera"] and summary["outcome"] == "unknown"
    assert (recorder.folder / "summary.json").is_file()
    assert recorder.finish("failed")["status"] == "completed"  # only the first ending counts


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_failed_and_interrupted_runs_keep_their_status(tmp_path, status):
    summary = run(tmp_path).finish(status, "policy request failed" if status == "failed" else None)
    assert summary["status"] == status
    assert summary["errors"] == (["policy request failed"] if status == "failed" else [])


def test_a_killed_run_reads_as_running_then_interrupted(tmp_path):
    """An emergency halt ends the process before any final event is written."""
    recorder = run(tmp_path)
    recorder.event("action_sent", step=0, action={})
    recorder.handle.flush()
    with (recorder.folder / "events.jsonl").open("a") as handle:
        handle.write('{"t": 1, "event": "action_')  # cut short mid-line
    modified = (recorder.folder / "events.jsonl").stat().st_mtime
    assert evidence.summarize(recorder.folder, now=modified + 1)["status"] == "running"
    stale = evidence.summarize(recorder.folder, now=modified + evidence.STALE_S + 1)
    assert stale["status"] == "interrupted" and stale["actions"] == 1


REVIEW = {"outcome": "failed", "notes": "closed beside the block", "reviewed_by": "operator",
          "failure_tags": ["missed_grasp", "hesitation"],
          "selected_ranges": [{"unit": "time", "start": 4.5, "end": 7.0, "label": "approach"},
                              {"unit": "event", "start": 120, "end": 180}]}


def finished(tmp_path, status="completed", **meta):
    recorder = run(tmp_path, **meta)
    recorder.finish(status)
    return recorder.folder


def test_a_review_records_outcome_tags_notes_and_ranges(tmp_path):
    folder = finished(tmp_path)
    assert evidence.summarize(folder)["needs_review"]
    summary = evidence.annotate(folder, REVIEW)
    review = summary["annotation"]
    assert review["outcome"] == "failed" and review["failure_tags"] == ["missed_grasp", "hesitation"]
    assert review["selected_ranges"][0] == {"unit": "time", "start": 4.5, "end": 7.0, "label": "approach"}
    assert review["reviewed_by"] == "operator" and review["reviewed_at"]
    assert not summary["needs_review"] and summary["failure_tags"] == review["failure_tags"]
    assert evidence.annotate(folder, {"outcome": "unknown"})["needs_review"]  # unknown is allowed


@pytest.mark.parametrize(("change", "message"), [
    ({"outcome": "great"}, "outcome"),
    ({"failure_tags": ["missed_grasp", "bad_luck"]}, "failure tags"),
    ({"failure_tags": "jitter"}, "failure tags"),
    ({"selected_ranges": [{"start": 9, "end": 3}]}, "start <= end"),
    ({"selected_ranges": [{"start": -1, "end": 3}]}, "start <= end"),
    ({"selected_ranges": [{"unit": "event", "start": 1.5, "end": 3}]}, "whole numbers"),
    ({"selected_ranges": [{"unit": "frames", "start": 1, "end": 3}]}, "unit"),
    ({"notes": "x" * (evidence.NOTES_MAX + 1)}, "notes"),
])
def test_a_bad_review_is_refused_and_changes_nothing(tmp_path, change, message):
    folder = finished(tmp_path)
    evidence.annotate(folder, REVIEW)
    with pytest.raises(ValueError, match=message):
        evidence.annotate(folder, {**REVIEW, **change})
    assert evidence.read_annotation(folder)["outcome"] == "failed"
    assert evidence.annotate(folder, {**REVIEW, "notes": "x" * evidence.NOTES_MAX})["notes"]


def test_a_review_overwrites_cleanly_and_repeating_it_changes_nothing(tmp_path):
    folder = finished(tmp_path)
    first = evidence.annotate(folder, REVIEW)["annotation"]
    assert evidence.annotate(folder, REVIEW)["annotation"] == first  # same review, same record
    second = evidence.annotate(folder, {"outcome": "success"})["annotation"]
    assert second["failure_tags"] == [] and second["selected_ranges"] == [] and second["notes"] == ""


def test_export_holds_reviewed_failures_unless_asked_for_all(tmp_path):
    reviewed = {}
    for stamp, outcome in (("090000", "failed"), ("090100", "partial"), ("090200", "success"),
                           ("090300", None)):
        folder = evidence.runs_root(tmp_path) / f"20261002-{stamp}-abcdef"
        evidence.RunRecorder(folder, META).finish("completed")
        if outcome:
            evidence.annotate(folder, {**REVIEW, "outcome": outcome})
            reviewed[folder.name] = outcome
    exported = evidence.export_reviews(tmp_path)
    assert sorted(item["outcome"] for item in exported) == ["failed", "partial"]
    item = exported[0]
    assert item["bundle_id"] == "act-x" and item["task"]["instruction"] == "Pick the block"
    assert item["failure_tags"] == REVIEW["failure_tags"] and item["metrics"]["status"] == "completed"
    assert item["events"].endswith(item["run_id"] + "/events.jsonl")
    assert sorted(i["run_id"] for i in evidence.export_reviews(tmp_path, everything=True)) == sorted(reviewed)


def test_an_older_annotation_still_reads(tmp_path):
    folder = finished(tmp_path)
    (folder / "annotation.json").write_text('{"outcome": "partial", "notes": "n", "annotated_at": 5}')
    review = evidence.summarize(folder)["annotation"]
    assert (review["outcome"], review["failure_tags"], review["reviewed_at"]) == ("partial", [], 5)


def test_runs_list_newest_first_and_ids_are_checked(tmp_path):
    folders = []
    for stamp in ("20261001-090000", "20261002-090000"):
        folder = evidence.runs_root(tmp_path) / f"{stamp}-abcdef"
        evidence.RunRecorder(folder, META).finish("completed")
        folders.append(folder.name)
    assert [r["run_id"] for r in evidence.list_runs(tmp_path)] == folders[::-1]
    assert len(evidence.list_runs(tmp_path, limit=1)) == 1
    for bad in ("../x", "20261001-090000-zzzzzz", "20261003-090000-abcdef"):
        with pytest.raises(FileNotFoundError):
            evidence.run_dir(tmp_path, bad)


def test_studio_serves_robot_runs(tmp_path):
    pytest.importorskip("flask")
    from polima.studio.app import create_app
    from polima.studio.runtime import StudioRuntime

    recorder = run(tmp_path)
    recorder.event("cycle_completed", cycle=1)
    recorder.finish("completed")
    client = create_app(tmp_path, StudioRuntime(tmp_path)).test_client()

    listing = client.get("/api/v1/robot-runs").get_json()
    assert listing["failure_tags"] == list(evidence.FAILURE_TAGS) and "unknown" in listing["outcomes"]
    items = listing["items"]
    assert [(r["run_id"], r["status"], r["cycles"]) for r in items] == [(recorder.folder.name, "completed", 1)]
    one = client.get(f"/api/v1/robot-runs/{recorder.folder.name}").get_json()
    assert one["summary"]["bundle_id"] == "act-x" and one["events"][0]["event"] == "run_started"
    assert client.get("/api/v1/robot-runs/20990101-000000-000000").status_code == 404

    path = f"/api/v1/robot-runs/{recorder.folder.name}/annotation"
    assert client.post(path, json={"outcome": "success"}).status_code == 403  # no CSRF token
    token = client.get("/api/v1/session").get_json()["csrf_token"]
    answer = client.post(path, json={"outcome": "success", "notes": "clean"},
                         headers={"X-CSRF-Token": token})
    assert answer.status_code == 200 and answer.get_json()["outcome"] == "success"
    assert client.post(path, json={"outcome": "great"}, headers={"X-CSRF-Token": token}).status_code == 400
    assert client.post(path, json={**REVIEW, "failure_tags": ["nope"]},
                       headers={"X-CSRF-Token": token}).status_code == 400
    assert client.post(path, json=REVIEW, headers={"X-CSRF-Token": token}).status_code == 200
    detail = client.get(f"/api/v1/robot-runs/{recorder.folder.name}").get_json()
    assert detail["annotation"]["failure_tags"] == REVIEW["failure_tags"]
    assert detail["summary"]["outcome"] == "failed" and not detail["summary"]["needs_review"]
    reviews = client.get("/api/v1/robot-runs/reviews").get_json()["items"]
    assert [r["run_id"] for r in reviews] == [recorder.folder.name]


def test_recording_writes_little_and_flushes_promptly(tmp_path):
    recorder = run(tmp_path)
    started = time.perf_counter()
    for step in range(3000):  # ten minutes of actions at 5 Hz, or 100 s at 30 Hz
        recorder.event("action_sent", step=step, action={"gripper.pos": 2.0})
    assert time.perf_counter() - started < 1.0
    recorder.finish("completed")
    assert os.path.getsize(recorder.folder / "events.jsonl") < 3000 * 120
    assert json.loads((recorder.folder / "summary.json").read_text())["actions"] == 3000


def reviewed_run(tmp_path, stamp, outcome, tags=(), latency=True, finish=True):
    folder = evidence.runs_root(tmp_path) / f"20261002-{stamp}-abcdef"
    recorder = evidence.RunRecorder(folder, META, clock=lambda: float(stamp))  # starts at its stamp
    if latency:
        recorder.event("inference_completed", model_ms=28.0, queue_ms=120.0, actions=100, step=0)
    if finish:
        recorder.finish("completed")
    else:  # killed an hour ago: no final event and no summary.json
        recorder.handle.close()
        stale = time.time() - 3600
        os.utime(folder / "events.jsonl", (stale, stale))
    if outcome:
        evidence.annotate(folder, {**REVIEW, "outcome": outcome, "failure_tags": list(tags)})
    return folder


@pytest.fixture
def reviewed(tmp_path):
    reviewed_run(tmp_path, "090000", "failed", ["missed_grasp", "latency_issue"])
    reviewed_run(tmp_path, "080000", "partial", ["camera_issue"], latency=False, finish=False)
    reviewed_run(tmp_path, "100000", "success")
    reviewed_run(tmp_path, "110000", "unknown")
    reviewed_run(tmp_path, "120000", None)
    return tmp_path


def test_the_manifest_holds_reviewed_failures_oldest_first(reviewed):
    manifest = evidence.improvement_manifest(reviewed, board="devkit", now=1000.0)
    assert manifest["schema_version"] == evidence.MANIFEST_SCHEMA and manifest["source_board"] == "devkit"
    assert manifest["filters"] == {"reviewed_only": True, "outcomes": ["failed", "partial"],
                                   "order": "oldest first"}
    assert manifest["counts"] == {"total_runs_considered": 5, "exported_runs": 2, "failed": 1, "partial": 1}
    assert [r["outcome"] for r in manifest["runs"]] == ["partial", "failed"]  # 08:00, then 09:00


def test_all_adds_reviewed_successes_but_never_unknown(reviewed):
    manifest = evidence.improvement_manifest(reviewed, everything=True, now=1000.0)
    assert [r["outcome"] for r in manifest["runs"]] == ["partial", "failed", "success"]
    assert manifest["counts"] == {"total_runs_considered": 5, "exported_runs": 3, "failed": 1,
                                  "partial": 1, "success": 1}


def test_a_manifest_record_carries_metrics_and_provenance(reviewed):
    runs = evidence.improvement_manifest(reviewed, now=1000.0)["runs"]
    partial, failed = runs
    assert failed["metrics"]["model_latency_ms"] == {"p50": 28.0, "p95": 28.0, "max": 28.0}
    assert failed["metrics"]["observation_to_queue_ms"]["p50"] == 120.0
    assert failed["provenance"]["summary"].endswith("/summary.json")
    assert failed["provenance"]["events"].endswith(failed["run_id"] + "/events.jsonl")
    assert failed["provenance"]["calibration_sha256"] == "ab"
    assert failed["provenance"]["rest_pose"] == {"method": "median"}
    assert failed["provenance"]["cameras"] == {"overhead": "/dev/a"}
    assert failed["selected_ranges"] == [{"label": "", **item} for item in REVIEW["selected_ranges"]]
    assert failed["task"]["instruction"] == "Pick the block"
    # A run killed mid-way has no summary.json and no latency; it still exports.
    assert partial["provenance"]["summary"] is None and partial["metrics"]["model_latency_ms"] is None
    assert partial["status"] == "interrupted"


def test_the_same_reviews_give_the_same_manifest(reviewed):
    first = json.dumps(evidence.improvement_manifest(reviewed, board="b", now=5.0), sort_keys=True)
    assert json.dumps(evidence.improvement_manifest(reviewed, board="b", now=5.0), sort_keys=True) == first


@pytest.mark.parametrize(("tags", "hint", "hints"), [
    (["missed_grasp"], "candidate_demo_or_label_fix", ["candidate_demo_or_label_fix"]),
    (["dropped_object", "wrong_target"], "candidate_demo_or_label_fix", ["candidate_demo_or_label_fix"]),
    (["jitter"], "runtime_or_policy_rate_investigation", ["runtime_or_policy_rate_investigation"]),
    (["latency_issue", "camera_issue"], "setup_data_quality_fix",
     ["setup_data_quality_fix", "runtime_or_policy_rate_investigation"]),
    (["calibration_issue", "collision_risk"], "safety_review", ["safety_review", "setup_data_quality_fix"]),
    (["other"], "manual_review", ["manual_review"]),
    ([], "manual_review", ["manual_review"]),
])
def test_failure_tags_suggest_what_to_do_next(tmp_path, tags, hint, hints):
    reviewed_run(tmp_path, "090000", "failed", tags)
    record = evidence.improvement_manifest(tmp_path)["runs"][0]
    assert (record["improvement_hint"], record["improvement_hints"]) == (hint, hints)


def test_studio_and_the_cli_export_the_same_manifest(reviewed, tmp_path, capsys):
    pytest.importorskip("flask")
    from polima.cli import robot as robot_cli
    from polima.studio.app import create_app
    from polima.studio.runtime import StudioRuntime

    client = create_app(reviewed, StudioRuntime(reviewed)).test_client()
    served = client.get("/api/v1/robot-runs/improvement-manifest").get_json()
    assert served["counts"]["exported_runs"] == 2
    assert client.get("/api/v1/robot-runs/improvement-manifest?all=1").get_json()["counts"]["success"] == 1
    assert len(client.get("/api/v1/robot-runs/reviews").get_json()["items"]) == 2  # unchanged

    assert robot_cli.needs_capability(["export-improvement-manifest"]) is None
    output = tmp_path / "manifest.json"
    assert robot_cli.run(["export-improvement-manifest", "--root", str(reviewed), "--output", str(output)]) == 0
    written = json.loads(output.read_text())
    assert [r["run_id"] for r in written["runs"]] == [r["run_id"] for r in served["runs"]]
    assert "2 of 5 runs" in capsys.readouterr().out


def test_camera_failure_reason_is_saved_in_summary(tmp_path):
    recorder = run(tmp_path)
    recorder.event('camera_fault', camera='wrist', message='wrist camera read thread stopped')
    summary = recorder.finish('failed')
    assert summary['errors'] == ['wrist camera read thread stopped']


def test_summary_consumes_events_once_with_bounded_memory():
    import tracemalloc
    def events():
        yield {"event": "run_started", "t": 1}
        for i in range(50000):
            yield {"event": "inference_completed", "model_ms": i, "queue_ms": i + 1}
            yield {"event": "action_sent", "positions": {"joint": i}}
        yield {"event": "run_failed", "t": 2, "error": "camera lost"}
    tracemalloc.start()
    try:
        summary = evidence.summary_of(events(), "test", "failed")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4_000_000
    assert summary["actions"] == summary["inferences"] == 50000
    assert summary["latency_ms"]["model"]["max"] == 49999
    assert summary["ended_at"] == 2
    assert summary["errors"] == ["camera lost"]


def test_finish_and_live_summary_do_not_materialize_event_log(tmp_path, monkeypatch):
    recorder = run(tmp_path)
    recorder.event("action_sent", positions={"joint": 1})
    recorder.handle.flush()
    monkeypatch.setattr(evidence, "read_events", lambda *a, **k: pytest.fail("whole-log read"))
    assert evidence.summarize(recorder.folder)["actions"] == 1
    assert recorder.finish("completed")["actions"] == 1


def test_detailed_recording_has_a_size_budget(tmp_path):
    from polima.robot.evidence import RunRecorder, read_events
    recorder = RunRecorder(tmp_path, {})
    recorder.max_bytes = 1024
    for _ in range(1000):
        recorder.event('action_sent', positions={'joint.pos': 1.0})
    summary = recorder.finish('completed')
    events = read_events(tmp_path)
    assert (tmp_path / 'events.jsonl').stat().st_size < 2048
    assert any(e.get('recording_truncated') for e in events)
    assert summary['status'] == 'completed'
