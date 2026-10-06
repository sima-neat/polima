from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from polima.workbench.app import (
    Workbench,
    checkpoint,
    latest_evidence,
    latest_robot_runs,
    readiness,
    robot_run_stats,
    serve,
    walk,
)


def save(folder, config):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "model.safetensors").write_bytes(b"")
    return folder


def test_checkpoints_are_named_by_training_run_and_step(tmp_path):
    lerobot = save(tmp_path / "act_eraser/checkpoints/100000/pretrained_model", {"type": "act"})
    groot = save(tmp_path / "groot_eraser/output/checkpoint-5000", {"model_type": "Gr00tN1d6"})
    final = save(tmp_path / "groot_eraser/output", {"model_type": "Gr00tN1d6"})
    assert [(c["policy"], c["run"], c["step"]) for c in map(checkpoint, (lerobot, groot, final))] == [
        ("ACT", "act_eraser", 100000), ("GR00T N1.6", "groot_eraser", 5000),
        ("GR00T N1.6", "groot_eraser", "final")]


def test_walk_does_not_descend_into_what_it_found(tmp_path):
    save(tmp_path / "run/output", {"model_type": "Gr00tN1d6"})
    save(tmp_path / "run/output/checkpoint-1", {"model_type": "Gr00tN1d6"})
    assert walk([tmp_path], "config.json") == [tmp_path / "run/output"]


def test_page_and_snapshot_are_served(tmp_path):
    save(tmp_path / "models/act_eraser/checkpoints/100/pretrained_model", {"type": "act"})
    built = tmp_path / "bundles/act-x"
    built.mkdir(parents=True)
    (built / "bundle.json").write_text(json.dumps(
        {"bundle_id": "act-x", "policy": "act", "graphs": [{"precision": "bf16"}]}))
    server = serve(Workbench([tmp_path / "models"], [tmp_path / "bundles"], []), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        snapshot = json.load(urllib.request.urlopen(base + "/api/v1/snapshot", timeout=5))
        page = urllib.request.urlopen(base + "/", timeout=5).read()
    finally:
        server.shutdown()
    assert [c["run"] for c in snapshot["checkpoints"]] == ["act_eraser"]
    assert snapshot["bundles"][0]["policy"] == "ACT" and snapshot["boards"] == []
    assert b"PoLiMa Workbench" in page


def benchmark(bundle, status, median, ended):
    if status != "completed":
        return {"bundle": bundle, "status": status, "ended_at": ended, "result": None}
    result = {"median": median, "p95": median + 1, "p99": median + 2, "throughput_hz": 1000 / median,
              "iterations": 50, "stages": [], "validation": {"status": "pass", "cosine": 0.99999,
                                                             "mean_abs": 0.001, "max_abs": 0.004}}
    return {"bundle": bundle, "status": status, "ended_at": ended, "result": result}


def test_evidence_is_each_bundles_newest_completed_benchmark():
    records = [benchmark("act-x", "failed", 0, 30), benchmark("act-x", "completed", 28.0, 20),
               benchmark("act-x", "completed", 31.0, 10), benchmark("groot-y", "running", 0, 25)]
    evidence = latest_evidence(records, "http://board:8080")
    assert list(evidence) == ["act-x"]
    assert evidence["act-x"]["median_ms"] == 28.0 and evidence["act-x"]["cosine"] == 0.99999


def fake_studio(replies):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Studio(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            studio.requests.append(self.path)
            reply = replies.get(self.path.split("?")[0])
            if reply is None:
                return self.send_error(404)
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    studio = ThreadingHTTPServer(("127.0.0.1", 0), Studio)
    studio.requests = []
    threading.Thread(target=studio.serve_forever, daemon=True).start()
    return studio


def one_bundle(tmp_path):
    built = tmp_path / "bundles/act-x"
    built.mkdir(parents=True)
    (built / "bundle.json").write_text(json.dumps({"bundle_id": "act-x", "policy": "act", "graphs": []}))
    return tmp_path / "bundles"


BOARD = {"/api/v1/snapshot": {"state": "idle", "bundles": [{"id": "act-x", "active": True}]},
         "/api/v1/history": {"items": []}}


def test_a_boards_benchmarks_and_robot_runs_reach_the_matching_bundle(tmp_path):
    run = {"run_id": "20261002-120000-abcdef", "bundle_id": "act-x", "status": "completed",
           "started_at": 2.0, "ended_at": 9.0, "cycles": 3, "actions": 900, "inferences": 40,
           "outcome": "success", "notes": "", "latency_ms": {"model": {"p50": 28.1, "p95": 29, "max": 31}}}
    older = {**run, "run_id": "20261001-120000-abcdef", "started_at": 1.0, "status": "failed"}
    studio = fake_studio({**BOARD, "/api/v1/benchmarks": {"items": [benchmark("act-x", "completed", 28.0, 20)]},
                          "/api/v1/robot-runs": {"items": [run, older]}})
    try:
        snapshot = Workbench([], [one_bundle(tmp_path)], [f"127.0.0.1:{studio.server_address[1]}"]).snapshot()
    finally:
        studio.shutdown()
    board = snapshot["boards"][0]
    assert board["online"] and "evidence" not in board and "robot_runs" not in board
    bundle = snapshot["bundles"][0]
    assert bundle["evidence"]["median_ms"] == 28.0 and bundle["evidence"]["check"] == "pass"
    assert bundle["robot_run"]["run_id"] == run["run_id"] and bundle["robot_run"]["model_ms_p50"] == 28.1


def test_the_latest_run_carries_its_review_state(tmp_path):
    reviewed = {"run_id": "20261002-130000-abcdef", "bundle_id": "act-x", "status": "failed",
                "started_at": 3.0, "outcome": "failed", "notes": "slipped", "needs_review": False,
                "failure_tags": ["dropped_object"], "latency_ms": {"model": None}}
    pending = {**reviewed, "bundle_id": "act-y", "outcome": "unknown", "notes": "", "needs_review": True,
               "failure_tags": []}
    latest = latest_robot_runs([reviewed, pending])
    assert latest["act-x"]["failure_tags"] == ["dropped_object"] and not latest["act-x"]["needs_review"]
    assert latest["act-y"]["needs_review"] and latest["act-y"]["model_ms_p50"] is None


def test_a_board_with_an_older_studio_still_shows(tmp_path):
    studio = fake_studio(BOARD)  # no benchmark or robot-run endpoints
    try:
        snapshot = Workbench([], [one_bundle(tmp_path)], [f"127.0.0.1:{studio.server_address[1]}"]).snapshot()
    finally:
        studio.shutdown()
    assert snapshot["boards"][0]["online"] and snapshot["boards"][0]["manifest"] is None
    assert snapshot["bundles"][0]["evidence"] is None and snapshot["bundles"][0]["robot_run"] is None


def robot(bundle, outcome="unknown", start=1.0, tags=(), p50=None, p95=None, policy="ACT", **extra):
    model = {"p50": p50, "p95": p95} if p50 is not None or p95 is not None else None
    return {"run_id": f"r{start}", "bundle_id": bundle, "status": "completed", "started_at": start,
            "outcome": outcome, "failure_tags": list(tags), "latency_ms": {"model": model},
            "cycles": 2, "actions": 100, "policy": policy, **extra}


def test_reviewed_runs_decide_the_rates_and_unknown_ones_stay_apart():
    runs = [robot("act-x", "success", 1), robot("act-x", "success", 2), robot("act-x", "partial", 3),
            robot("act-x", "failed", 4, tags=["missed_grasp"]), robot("act-x", "unknown", 5),
            robot("act-x", "unknown", 6)]
    stat = robot_run_stats(runs)["act-x"]
    assert (stat["runs"], stat["reviewed"], stat["unreviewed"]) == (6, 4, 2)
    assert (stat["success"], stat["partial"], stat["failed"]) == (2, 1, 1)
    assert stat["success_rate"] == 0.5 and stat["partial_plus_rate"] == 0.75  # of 4 reviewed, not 6
    assert (stat["cycles"], stat["actions"]) == (12, 600)
    labels = stat["labels"]
    assert labels["reviewed"] == "4 of 6 runs · 2 need review"
    assert (labels["success"], labels["partial_plus"]) == ("2/4 success", "3/4 partial+")


def test_failure_tags_are_counted_most_common_first():
    runs = [robot("b", "failed", 1, tags=["latency_issue", "missed_grasp"]),
            robot("b", "partial", 2, tags=["missed_grasp", "jitter"]),
            robot("b", "failed", 3, tags=["missed_grasp", "jitter"]),
            robot("b", "unknown", 4, tags=["camera_issue"])]  # not reviewed, not counted
    stat = robot_run_stats(runs)["b"]
    assert stat["failure_tags"] == [("missed_grasp", 3), ("jitter", 2), ("latency_issue", 1)]
    assert stat["labels"]["failures"] == "missed_grasp x3, jitter x2, latency_issue x1"


def test_robot_latency_ignores_runs_without_it():
    runs = [robot("b", "success", 1, p50=400, p95=600), robot("b", "success", 2),
            robot("b", "failed", 3, p50=420, p95=610), robot("b", "unknown", 4, p50=410, p95=None)]
    stat = robot_run_stats(runs)["b"]
    assert stat["latency_ms"] == {"p50": 410, "p95": 610}
    assert stat["labels"]["latency"] == "p50 410 ms / p95 610 ms"
    assert robot_run_stats([robot("c", "success")])["c"]["labels"]["latency"] == "—"


def test_the_latest_reviewed_outcome_is_the_newest_reviewed_run():
    runs = [robot("b", "failed", 5), robot("b", "success", 9), robot("b", "unknown", 12),
            robot("b", "partial", 7)]
    assert robot_run_stats(runs)["b"]["latest_reviewed"] == {"run_id": "r9", "outcome": "success",
                                                             "started_at": 9}


def test_policy_is_reported_only_when_the_runs_agree():
    assert robot_run_stats([robot("b", policy="GR00T N1.6")])["b"]["policy"] == "GR00T N1.6"
    assert robot_run_stats([robot("b", policy="ACT", start=1), robot("b", policy="SmolVLA", start=2)])["b"]["policy"] is None
    assert robot_run_stats([robot("b", policy=None)])["b"]["policy"] is None


def test_compare_says_needs_review_or_no_reviewed_runs(tmp_path):
    folder = tmp_path / "bundles"
    for bundle_id in ("act-x", "groot-y", "smolvla-z"):
        (folder / bundle_id).mkdir(parents=True)
        (folder / bundle_id / "bundle.json").write_text(json.dumps(
            {"bundle_id": bundle_id, "policy": bundle_id.split("-")[0], "graphs": []}))
    runs = [robot("act-x", "success", 3, p50=28, p95=31), robot("act-x", "failed", 2, tags=["jitter"]),
            robot("groot-y", "unknown", 1)]
    studio = fake_studio({**BOARD, "/api/v1/robot-runs": {"items": runs}})
    try:
        snapshot = Workbench([], [folder], [f"127.0.0.1:{studio.server_address[1]}"]).snapshot()
    finally:
        studio.shutdown()
    labels = {b["id"]: b["robot_labels"] for b in snapshot["bundles"]}
    assert labels["act-x"]["success"] == "1/2 success" and labels["act-x"]["failures"] == "jitter x1"
    assert labels["groot-y"]["reviewed"] == "Needs review" and labels["groot-y"]["success"] == "—"
    assert labels["smolvla-z"]["reviewed"] == "No reviewed runs"
    by_id = {b["id"]: b for b in snapshot["bundles"]}
    assert by_id["act-x"]["robot_run"]["run_id"] == "r3"  # the latest run is still shown
    assert by_id["smolvla-z"]["robot_stats"] is None


MANIFEST = {"schema_version": "polima-improvement-manifest-v1", "created_at": 1790956800,
            "source_board": "modalix", "filters": {"outcomes": ["failed", "partial"]},
            "counts": {"total_runs_considered": 3, "exported_runs": 2, "failed": 1, "partial": 1}, "runs": []}


def test_a_board_reports_what_it_can_export(tmp_path):
    runs = [robot("act-x", "failed", 1), robot("act-x", "unknown", 2, needs_review=True)]
    studio = fake_studio({**BOARD, "/api/v1/robot-runs": {"items": runs},
                          "/api/v1/robot-runs/improvement-manifest": MANIFEST})
    try:
        board = Workbench([], [one_bundle(tmp_path)], [f"127.0.0.1:{studio.server_address[1]}"]).snapshot()["boards"][0]
    finally:
        studio.shutdown()
    assert board["manifest"]["exported_runs"] == 2 and board["runs_needing_review"] == 1


def test_export_failures_downloads_a_configured_boards_manifest(tmp_path):
    studio = fake_studio({**BOARD, "/api/v1/robot-runs/improvement-manifest": MANIFEST})
    workbench = Workbench([], [], [f"127.0.0.1:{studio.server_address[1]}"])
    server = serve(workbench, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/v1/improvement-manifest"
    try:
        with urllib.request.urlopen(base + "?board=0&all=1", timeout=5) as response:
            disposition = response.headers["Content-Disposition"]
            body = json.loads(response.read())
        assert 'filename="improvement-manifest-modalix-20261002-160000.json"' in disposition
        assert body["counts"]["exported_runs"] == 2
        assert studio.requests[-1] == "/api/v1/robot-runs/improvement-manifest?all=1"
        for bad in ("?board=1", "?board=http://elsewhere", ""):  # only configured boards
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(base + bad, timeout=5)
            assert refused.value.code == 404
    finally:
        server.shutdown()
        studio.shutdown()


def test_export_from_an_unreachable_board_is_reported_not_crashed():
    server = serve(Workbench([], [], ["127.0.0.1:9"]), port=0)  # nothing listens on port 9
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as failed:
            urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/v1/improvement-manifest?board=0",
                                   timeout=15)
        assert failed.value.code == 502
    finally:
        server.shutdown()


ONLINE = [{"online": True, "bundles": ["act-x"], "manifest": {"exported_runs": 0}}]


def stages_of(build, boards=ONLINE, elsewhere=False):
    result = readiness({"id": "act-x", "robot_client": True, **build}, boards, elsewhere)
    return {s["key"]: s["status"] for s in result["stages"]}, result["next_action"], result


def stats_for(*outcomes):
    return robot_run_stats([robot("act-x", o, i + 1) for i, o in enumerate(outcomes)])["act-x"]


def test_an_installed_bundle_without_a_benchmark_needs_one():
    stages, action, result = stages_of({})
    assert stages["installed"] == "done" and stages["benchmark"] == "needs_action"
    assert action == "Run benchmark" and result["summary"]["benchmark"] == "No benchmark"
    assert stages["reviewed"] == stages["compared"] == stages["export"] == "unavailable"


def test_a_benchmarked_bundle_is_next_run_on_the_robot():
    evidence = {"median_ms": 28.1, "check": "pass"}
    stages, action, result = stages_of({"evidence": evidence})
    assert stages["benchmark"] == "done" and stages["robot_run"] == "needs_action"
    assert action == "Run on robot" and result["summary"]["benchmark"] == "28.1 ms median · check pass"


def test_an_unreviewed_run_asks_for_a_review():
    stages, action, result = stages_of({"evidence": {"median_ms": 28.0}, "robot_stats": stats_for("unknown"),
                                        "robot_run": {"status": "completed", "needs_review": True}})
    assert stages["robot_run"] == "done" and stages["reviewed"] == "needs_action"
    assert action == "Review latest run" and result["summary"]["robot_run"] == "completed, needs review"


def test_reviewed_failures_make_the_export_available():
    stat = stats_for("failed", "partial")
    stages, action, result = stages_of({"evidence": {"median_ms": 28.0}, "robot_stats": stat,
                                        "robot_run": {"status": "failed", "outcome": "failed"},
                                        "robot_labels": stat["labels"]}, elsewhere=True)
    assert stages["reviewed"] == stages["compared"] == "done"
    assert stages["export"] == "needs_action" and action == "Export failures"
    assert result["summary"]["reviews"] == "0 success · 1 partial · 1 failed"


def test_reviewed_successes_need_no_export_but_need_another_policy_to_compare():
    stat = stats_for("success", "success")
    stages, action, result = stages_of({"evidence": {"median_ms": 28.0}, "robot_stats": stat,
                                        "robot_run": {"status": "completed", "outcome": "success"}})
    assert stages["export"] == "done"
    assert next(s for s in result["stages"] if s["key"] == "export")["note"] == "No export needed"
    assert stages["compared"] == "needs_action" and action == "Compare policies"
    _, action, _ = stages_of({"evidence": {"median_ms": 28.0}, "robot_stats": stat,
                              "robot_run": {"status": "completed", "outcome": "success"}}, elsewhere=True)
    assert action is None  # nothing left to do


def test_without_a_board_or_an_exporting_studio_stages_are_unavailable_not_failed():
    stages, action, result = stages_of({}, boards=[])
    assert stages["installed"] == "unavailable" and action == "Add a DevKit" and not result["complete"]
    assert result["stages"][0]["note"] == "No board online"
    old = [{"online": True, "bundles": ["act-x"], "manifest": None}]
    stages, _, result = stages_of({"evidence": {"median_ms": 28.0}, "robot_stats": stats_for("failed"),
                                   "robot_run": {"status": "failed", "outcome": "failed"}}, boards=old)
    assert stages["export"] == "unavailable"
    assert next(s for s in result["stages"] if s["key"] == "export")["note"] == "The board's Studio cannot export yet"


def test_readiness_renders_for_a_board_with_an_older_studio(tmp_path):
    studio = fake_studio({"/api/v1/snapshot": {"state": "idle", "bundles": [{"id": "act-x"}]},
                          "/api/v1/history": {"items": []}})  # no benchmarks, robot runs or manifest
    try:
        bundle = Workbench([], [one_bundle(tmp_path)], [f"127.0.0.1:{studio.server_address[1]}"]).snapshot()["bundles"][0]
    finally:
        studio.shutdown()
    stages = {s["key"]: s["status"] for s in bundle["readiness"]["stages"]}
    assert stages["installed"] == "done" and bundle["readiness"]["next_action"] == "Run benchmark"
    assert bundle["readiness"]["summary"]["robot_run"] == "No robot run"
