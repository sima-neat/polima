from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from polima.studio.models import RunConfig, StudioState
from polima.studio.runtime import (
    Conflict,
    StudioRuntime,
    benchmark_failure,
    parse_benchmark_output,
    process_failure,
)
from polima.studio.store import StudioStore


def config(root: Path) -> RunConfig:
    bundle = root / "models/demo"
    bundle.mkdir(parents=True)
    calibration = bundle / "robot_client/calibration/so-arm101.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}")
    devices = root / "devices"
    devices.mkdir()
    for name in ("arm", "overhead", "wrist"):
        (devices / name).touch()
    return RunConfig(
        bundle="demo",
        task="Pick up the block.",
        robot_port=str(devices / "arm"),
        overhead_camera=str(devices / "overhead"),
        wrist_camera=str(devices / "wrist"),
    )


def test_run_config_validation() -> None:
    with pytest.raises(ValueError, match="missing run configuration"):
        RunConfig.from_json({})
    with pytest.raises(ValueError, match="fps"):
        RunConfig.from_json({
            "bundle": "x", "task": "x", "robot_port": "x",
            "overhead_camera": "x", "wrist_camera": "x", "fps": 100,
        })


def test_only_one_controller_holds_the_baton_at_a_time(tmp_path: Path) -> None:
    """One controller at a time is a safety property -- two tabs driving one
    arm is genuinely unsafe -- but it is enforced by handing the baton on,
    not by refusing the second browser. Renewal by the holder is unchanged."""
    runtime = StudioRuntime(tmp_path)
    first = runtime.claim_lease()
    renewed = runtime.claim_lease(first["token"])
    assert renewed["token"] == first["token"]
    runtime.require_lease(first["token"])

    stolen = runtime.claim_lease()
    assert stolen["token"] != first["token"]
    runtime.require_lease(stolen["token"])
    with pytest.raises(Conflict):
        runtime.require_lease(first["token"])


def test_same_browser_can_recover_lease_without_in_memory_token(tmp_path: Path) -> None:
    runtime = StudioRuntime(tmp_path)
    first = runtime.claim_lease(owner="browser-a")
    recovered = runtime.claim_lease(owner="browser-a")
    assert recovered["token"] == first["token"]
    # A different browser now takes over rather than being locked out until
    # the 30s lease lapses; the displaced token stops validating.
    other = runtime.claim_lease(owner="browser-b")
    assert other["token"] != first["token"]
    with pytest.raises(Conflict):
        runtime.require_lease(first["token"])


def test_arming_token_is_bound_to_exact_config(tmp_path: Path) -> None:
    runtime = StudioRuntime(tmp_path, command=tmp_path / "polima")
    runtime.command.touch()
    runtime.command.chmod(0o755)
    run = config(tmp_path)
    armed = runtime.arm(run)
    assert runtime.state == StudioState.ARMING
    changed = RunConfig(**{**run.to_json(), "task": "A different task"})
    with pytest.raises(Conflict, match="does not match"):
        runtime.start_robot(changed, armed["arming_token"])
    assert runtime.state == StudioState.IDLE


def test_missing_follower_calibration_blocks_arming(tmp_path: Path) -> None:
    runtime = StudioRuntime(tmp_path, command=tmp_path / "polima")
    runtime.command.touch()
    runtime.command.chmod(0o755)
    run = config(tmp_path)
    (tmp_path / "models/demo/robot_client/calibration/so-arm101.json").unlink()

    with pytest.raises(ValueError, match="follower-arm calibration"):
        runtime.arm(run)


def test_studio_calibration_uses_polima_robot_command(tmp_path: Path, monkeypatch) -> None:
    run = config(tmp_path)
    (tmp_path / "models/demo/bundle.json").write_text('{"policy":"act"}')
    command = tmp_path / "polima"
    command.touch()
    command.chmod(0o755)
    runtime = StudioRuntime(tmp_path, command=command)
    captured: list[object] = []
    monkeypatch.setattr(runtime, "_spawn", lambda *args, **kwargs: captured.extend(args))

    runtime.start_calibration("demo", run.robot_port)

    assert captured[0:2] == ["calibration", [
        str(command), "robot", "--policy", "act", "calibrate", "--yes",
        "--robot-port", run.robot_port, "--calibration-dir",
        str(tmp_path / "models/demo/robot_client/calibration"),
    ]]


def test_store_records_and_prunes_runs(tmp_path: Path) -> None:
    store = StudioStore(tmp_path / "studio.sqlite3", max_runs=2)
    for index in range(3):
        run_id = store.begin_run({"bundle": "demo", "task": str(index)}, f"{index}.log")
        store.finish_run(run_id, "completed")
    runs = store.list_runs()
    assert [item["task"] for item in runs] == ["2", "1"]


def test_preview_process_is_supervised(tmp_path: Path) -> None:
    run = config(tmp_path)
    command = tmp_path / "polima"
    command.write_text("#!/bin/sh\necho 'Camera preview server: http://127.0.0.1:5001/token/'\n")
    command.chmod(0o755)
    runtime = StudioRuntime(tmp_path, command=command)
    runtime.start_preview("demo", run.overhead_camera, run.wrist_camera)
    deadline = time.time() + 2
    while runtime.state == StudioState.PREVIEWING and time.time() < deadline:
        time.sleep(0.01)
    assert runtime.state == StudioState.IDLE
    assert "Camera preview server" in (tmp_path / "var/log/studio/preview.log").read_text()


def test_preview_readiness_is_emitted_after_delayed_startup(tmp_path: Path) -> None:
    run = config(tmp_path)
    command = tmp_path / "polima"
    command.write_text(
        "#!/bin/sh\n"
        "sleep 0.05\n"
        "echo 'Camera preview server: http://127.0.0.1:5001/token/'\n"
        "sleep 0.2\n"
    )
    command.chmod(0o755)
    runtime = StudioRuntime(tmp_path, command=command)
    emitted = []
    runtime.emit = lambda kind, **data: emitted.append({"type": kind, **data})

    runtime.start_preview("demo", run.overhead_camera, run.wrist_camera)
    # The snapshot timing this protects: the child is running, but has not
    # announced its stream URL yet.
    assert runtime.snapshot()["preview_url"] is None
    deadline = time.time() + 2
    while not any(event["type"] == "preview" for event in emitted) and time.time() < deadline:
        time.sleep(0.01)

    event = next(event for event in emitted if event["type"] == "preview")
    assert event["url"] == "http://127.0.0.1:5001/token/"
    runtime.close()


def test_control_stop_waits_for_client_then_stops_server(tmp_path: Path) -> None:
    run = config(tmp_path)
    command = tmp_path / "polima"
    marker = tmp_path / "server-stopped"
    command.write_text(
        "#!/bin/sh\n"
        f"marker='{marker}'\n"
        "if [ \"$1\" = server ]; then echo \"$2\" > \"$marker\"; exit 0; fi\n"
        "trap 'exit 0' INT\n"
        "echo 'Camera preview server: http://127.0.0.1:5001/token/'\n"
        "while :; do sleep 0.05; done\n"
    )
    command.chmod(0o755)
    runtime = StudioRuntime(tmp_path, command=command)
    runtime.start_preview("demo", run.overhead_camera, run.wrist_camera)

    stopped = runtime.stop_control()
    assert stopped["server_stopping"] is True
    deadline = time.time() + 2
    while (
        (not marker.exists() or runtime.state != StudioState.IDLE)
        and time.time() < deadline
    ):
        time.sleep(0.01)

    assert marker.read_text().strip() == "stop"
    assert runtime.state == StudioState.IDLE


def test_bundle_listing_exposes_launcher_task_and_manifest_tasks(tmp_path: Path) -> None:
    models = tmp_path / "models"
    older = models / "older"
    older.joinpath("robot_client").mkdir(parents=True)
    older.joinpath("bundle.json").write_text('{"policy":"act"}')
    older.joinpath("robot_client/start.sh").write_text(
        'TASK="${POLIMA_TASK:-Place the grey eraser in white basket.}"\n'
    )
    modern = models / "modern"
    modern.mkdir()
    modern.joinpath("bundle.json").write_text(
        '{"policy":"groot","default_task":"Pick red cube.",'
        '"tasks":["Pick blue cube."]}'
    )
    fixed = models / "fixed"
    fixed.joinpath("robot_client").mkdir(parents=True)
    fixed.joinpath("bundle.json").write_text('{"policy":"act"}')
    fixed.joinpath("robot_client/start.sh").write_text(
        'exec client --task="Place the yellow cube in the tray." --policy_type=act\n'
    )

    bundles = {item["id"]: item for item in StudioRuntime(tmp_path).bundles()}
    assert bundles["older"]["default_task"] == "Place the grey eraser in white basket."
    assert bundles["older"]["tasks"] == ["Place the grey eraser in white basket."]
    assert bundles["modern"]["tasks"] == ["Pick red cube.", "Pick blue cube."]
    assert bundles["fixed"]["default_task"] == "Place the yellow cube in the tray."


def test_api_requires_csrf_and_controller_lease(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from polima.studio.app import create_app

    runtime = StudioRuntime(tmp_path)
    client = create_app(tmp_path, runtime).test_client()
    assert client.post("/api/v1/robot/stop").status_code == 403
    session = client.get("/api/v1/session").get_json()
    headers = {"X-CSRF-Token": session["csrf_token"]}
    lease = client.post("/api/v1/lease", json={}, headers=headers).get_json()
    headers["X-Controller-Lease"] = lease["token"]
    assert client.post("/api/v1/robot/stop", json={}, headers=headers).status_code == 200
    # Emergency halt deliberately does not require the controller lease.
    assert client.post(
        "/api/v1/robot/halt", json={}, headers={"X-CSRF-Token": session["csrf_token"]}
    ).status_code == 200


def test_parse_benchmark_output() -> None:
    result = parse_benchmark_output(
        "  min 20.1 / mean 21.0 / median 20.8 / p95 22.4 / p99 23.0 / max 23.0 "
        "ms over 50 runs (3 warm-up)\n"
        "  total 21.2 ms\n"
        "    run_elf:vision=5.100000\n"
        "    run_elf:vision=5.200000\n"
        "  PASS  cosine=0.999990 mean_abs=0.000100 max_abs=0.001000\n"
    )
    assert result is not None
    assert result["p95"] == 22.4
    assert result["throughput_hz"] == pytest.approx(47.619)
    assert result["stages"] == [
        {"name": "run_elf:vision", "ms": 5.1},
        {"name": "run_elf:vision", "ms": 5.2},
    ]
    assert result["validation"]["status"] == "pass"


def test_benchmark_failure_preserves_native_diagnostic() -> None:
    output = (
        "  failed to load demo: Connecting to server failed\n"
        "  nothing is loaded now\n"
        "polima>   no model loaded -- `use <n>` first\n"
    )
    assert benchmark_failure(output) == "failed to load demo: Connecting to server failed"


def test_process_failure_exposes_actionable_recent_output() -> None:
    assert process_failure(
        "robot", 1, ["opening devices", "Error: wrist camera permission denied"]
    ) == "robot: Error: wrist camera permission denied"
    assert process_failure("preview", 2, ["starting"]) == (
        "preview exited with code 2; check Runtime Events for details"
    )


def test_runtime_calibration_mismatch_stops_robot_and_guides_operator(tmp_path: Path) -> None:
    command = tmp_path / "robot"
    command.write_text(
        "#!/bin/sh\n"
        "echo 'Mismatch between calibration values in the motor and the calibration file or no calibration file found'\n"
        "sleep 5\n"
    )
    command.chmod(0o755)
    runtime = StudioRuntime(tmp_path)
    runtime._spawn("robot", [str(command)], StudioState.RUNNING, tmp_path / "robot.log", {})

    deadline = time.time() + 2
    while runtime.state != StudioState.FAULT and time.time() < deadline:
        time.sleep(0.01)

    assert runtime.state == StudioState.FAULT
    assert runtime.fault is not None
    assert "Manual Calibration (c)" in runtime.fault
    assert any(event.get("type") == "calibration" and event.get("action") == "required"
               for event in runtime._events)


def test_hardware_free_benchmark_is_supervised_and_persisted(tmp_path: Path) -> None:
    run = config(tmp_path)
    (tmp_path / "models/demo/fixtures/inputs").mkdir(parents=True)
    command = tmp_path / "polima"
    command.write_text(
        "#!/bin/sh\n"
        "read a; read b; read c; read d\n"
        "echo '  min 9.0 / mean 10.0 / median 10.0 / p95 11.0 / p99 12.0 / max 12.0 ms over 20 runs (2 warm-up)'\n"
        "echo '    run_elf:vision=4.000000'\n"
        "echo '  PASS  cosine=1.000000 mean_abs=0.000000 max_abs=0.000000'\n"
    )
    command.chmod(0o755)
    runtime = StudioRuntime(tmp_path, command=command)
    started = runtime.start_benchmark("demo", 20, 2)
    assert started["state"] == "benchmarking"
    deadline = time.time() + 2
    while (
        runtime.store.list_benchmarks()[0]["status"] == "running"
        and time.time() < deadline
    ):
        time.sleep(0.01)
    benchmarks = runtime.store.list_benchmarks()
    assert benchmarks[0]["status"] == "completed"
    assert benchmarks[0]["result"]["mean"] == 10.0
    assert benchmarks[0]["result"]["validation"]["status"] == "pass"
    assert run.bundle == "demo"


# ------------------------------------------------- autocomplete / reset / lease


def _run_config(**overrides):
    from polima.studio.models import RunConfig

    base = {"bundle": "b", "task": "t", "robot_port": "/dev/ttyACM0",
            "overhead_camera": "cam0", "wrist_camera": "cam1"}
    return RunConfig.from_json({**base, **overrides})


def test_autocomplete_defaults_to_policy_and_round_trips():
    assert _run_config().autocomplete is None
    assert _run_config(autocomplete=None).autocomplete is None
    assert _run_config(autocomplete=False).autocomplete is False
    assert _run_config(autocomplete=False).to_json()["autocomplete"] is False


def test_speed_defaults_fast_and_validates():
    assert _run_config().speed == "fast"
    assert _run_config(speed="slow").to_json()["speed"] == "slow"
    with pytest.raises(ValueError, match="speed"):
        _run_config(speed="unknown")


def _started_env(tmp_path, monkeypatch, config, policy="groot"):
    from polima.studio.runtime import StudioRuntime

    engine = StudioRuntime(tmp_path)
    (tmp_path / "models" / "b").mkdir(parents=True)
    (tmp_path / "models" / "b" / "bundle.json").write_text(json.dumps({"policy": policy}))
    captured: dict = {}
    monkeypatch.setattr(engine, "_spawn", lambda *a, **k: captured.update(args=a[1], env=a[4] if len(a) > 4
                                                                          else k.get("env")))
    monkeypatch.setattr(engine.store, "begin_run", lambda *a, **k: 1)
    import json as _json
    import time as _time
    engine._arm_tokens["t"] = (_time.time() + 60, _json.dumps(config.to_json(), sort_keys=True))
    engine.start_robot(config, "t")
    return captured


def test_studio_speed_choices_reach_the_robot_run(tmp_path, monkeypatch):
    started = _started_env(tmp_path, monkeypatch, _run_config(bundle="b"))
    assert started["env"]["POLIMA_FAST"] == "1" and "--fast" in started["args"]
    started = _started_env(tmp_path / "2", monkeypatch, _run_config(bundle="b", speed="slow"))
    assert started["env"]["POLIMA_FAST"] == "0" and "--slow" in started["args"]
    assert "POLIMA_CYCLE_RESET" not in started["env"]


def test_autocomplete_reaches_the_launcher_under_the_policys_own_name(tmp_path, monkeypatch):
    """The launcher hardcodes `${<POLICY>_AUTO_COMPLETE:-1}` with no POLIMA_
    spelling, so a generic variable alone is silently ignored and the arm keeps
    running the grasp-release detector."""
    from polima.studio.runtime import StudioRuntime

    engine = StudioRuntime(tmp_path)
    bundle = tmp_path / "models" / "b"
    bundle.mkdir(parents=True)
    (bundle / "bundle.json").write_text('{"policy": "groot"}')

    captured: dict = {}
    monkeypatch.setattr(engine, "_spawn",
                        lambda *a, **k: captured.update(env=a[4] if len(a) > 4 else k.get("env")))
    monkeypatch.setattr(engine.store, "begin_run", lambda *a, **k: 1)
    token = "t"
    import json as _json
    import time as _time
    engine._arm_tokens[token] = (
        _time.time() + 60, _json.dumps(_run_config(bundle="b", autocomplete=False).to_json(),
                                       sort_keys=True))

    engine.start_robot(_run_config(bundle="b", autocomplete=False), token)
    assert captured["env"]["GROOT_AUTO_COMPLETE"] == "0"
    assert captured["env"]["POLIMA_AUTO_COMPLETE"] == "0"


def test_lease_is_taken_over_rather_than_refused():
    """A stale tab must not lock everyone out for the rest of the 30s lease."""
    import tempfile
    from pathlib import Path

    from polima.studio.runtime import Conflict, StudioRuntime

    engine = StudioRuntime(Path(tempfile.mkdtemp()))
    first = engine.claim_lease(owner="browser-A")
    second = engine.claim_lease(owner="browser-B")

    assert second["token"] != first["token"], "the newcomer gets a fresh baton"
    engine.require_lease(second["token"])
    with pytest.raises(Conflict):
        engine.require_lease(first["token"])


def test_lease_takeover_is_announced():
    import tempfile
    from pathlib import Path

    from polima.studio.runtime import StudioRuntime

    engine = StudioRuntime(Path(tempfile.mkdtemp()))
    engine.claim_lease(owner="browser-A")
    engine.claim_lease(owner="browser-B")
    assert any(e.get("type") == "lease" and e.get("event") == "taken_over"
               for e in engine._events)


def test_reset_clears_the_lease_so_recovery_is_not_blocked_by_it(tmp_path, monkeypatch):
    from polima.deploy import mla
    from polima.studio.runtime import StudioRuntime

    engine = StudioRuntime(tmp_path)
    engine.claim_lease(owner="a-dead-tab")
    monkeypatch.setattr(mla, "reset", lambda *a, **k: mla.ResetReport(True, "recovery-script"))
    monkeypatch.setattr(engine, "server_status", lambda: {"running": False})

    result = engine.reset()
    assert result["ok"] and engine._lease_token is None
    # Whoever is recovering can take control immediately.
    assert engine.claim_lease(owner="the-operator")["token"]


def test_reset_route_needs_no_controller_lease():
    """A recovery path that the wedge itself can block is not a recovery path."""
    import inspect

    from polima.studio import app as studio_app

    source = inspect.getsource(studio_app.create_app)
    reset = source.split('"/api/v1/system/reset"', 1)[1].split("@app.", 1)[0]
    assert "lease=False" in reset


def test_studio_serve_passes_its_options_to_studio(monkeypatch):
    from polima.cli import studio as studio_cli
    from polima.studio import main as studio_main

    seen = []
    monkeypatch.setattr(studio_main, "main", lambda argv: seen.append(argv) or 0)
    assert studio_cli.run(["serve", "--port", "8081", "--host", "127.0.0.1"]) == 0
    assert seen == [["--port", "8081", "--host", "127.0.0.1"]]


def test_opening_another_tab_preserves_the_session(tmp_path):
    pytest.importorskip("flask")
    from polima.studio.app import create_app

    client = create_app(tmp_path).test_client()
    first = client.get("/api/v1/session").get_json()["csrf_token"]
    second = client.get("/api/v1/session").get_json()["csrf_token"]
    assert second == first
    response = client.post("/api/v1/lease", json={}, headers={"X-CSRF-Token": first})
    assert response.status_code == 200


def test_failed_accelerator_reset_is_reported(tmp_path, monkeypatch):
    from polima.deploy import mla

    engine = StudioRuntime(tmp_path)
    monkeypatch.setattr(engine, "server_status", lambda: {"running": False})
    monkeypatch.setattr(mla, "reset", lambda *a, **k: mla.ResetReport(False, "recovery-script", detail="reset: rc=1"))
    with pytest.raises(RuntimeError, match="Accelerator reset failed"):
        engine.reset()
    assert engine.fault and "Accelerator reset failed" in engine.fault


@pytest.mark.parametrize("policy,setting,expected", [
    ("groot", None, "0"), ("groot", True, "1"), ("groot", False, "0"),
    ("act", None, "1"), ("act", False, "0"),
])
def test_completion_policy_default_reaches_robot(tmp_path, monkeypatch, policy, setting, expected):
    run = _run_config(autocomplete=setting)
    started = _started_env(tmp_path, monkeypatch, run, policy)
    assert started["env"]["POLIMA_AUTO_COMPLETE"] == expected
    assert started["env"]["ACT_AUTO_COMPLETE"] == expected


def test_studio_preflight_accepts_cli_calibration(tmp_path, monkeypatch):
    from polima.robot import so101

    run = config(tmp_path)
    runtime = StudioRuntime(tmp_path)
    runtime._calibration_path(run.bundle).unlink()
    devkit = tmp_path / "devkit-calibration"
    devkit.mkdir()
    (devkit / "so-arm101.json").write_text("{}")
    monkeypatch.setattr(so101, "DEVKIT_CALIBRATION", devkit)
    checks = {item["name"]: item["ok"] for item in runtime.preflight(run)}
    assert checks["follower-arm calibration"]
    assert runtime._calibration_path(run.bundle).parent != devkit


def test_reset_under_service_restrictions_does_not_stop_running_process(tmp_path, monkeypatch):
    from polima.studio.runtime import StudioRuntime
    engine = StudioRuntime(tmp_path)
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/status":
            return "Name:\tpython\nNoNewPrivs:\t1\n"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(engine, "server_status", lambda: pytest.fail("must reject before touching server"))
    with pytest.raises(RuntimeError, match="authorized terminal"):
        engine.reset()


def test_child_logs_use_writable_studio_root(tmp_path, monkeypatch):
    import sys

    from polima.studio.runtime import StudioRuntime, StudioState
    engine = StudioRuntime(tmp_path)
    monkeypatch.setattr(engine, "_monitor", lambda *args: None)
    engine._spawn("preview", [sys.executable, "-c", "from pathlib import Path; print(Path.cwd())"],
                  StudioState.PREVIEWING, tmp_path / "output.log", {})
    output, _ = engine._process.communicate(timeout=5)
    assert output.strip() == str(tmp_path)


def test_process_failure_reports_the_cli_message_not_the_detail_hint():
    recent = ["polima robot: error: unknown policy 'polima'; available: ['act', 'groot']",
              "  (for more detail, set POLIMA_LOG_LEVEL=DEBUG and run the command again)"]
    assert process_failure("calibration", 1, recent) == f"calibration: {recent[0]}"


@pytest.mark.parametrize("value, expected", [(12, 12), (12.0, 12), ("30", 30)])
def test_run_config_sends_whole_degree_motion_limits(value, expected):
    config = RunConfig.from_json({"bundle": "b", "task": "t", "robot_port": "/dev/arm",
                                  "overhead_camera": "/dev/o", "wrist_camera": "/dev/w",
                                  "max_relative_target": value})
    assert config.max_relative_target == expected


@pytest.mark.parametrize("value", [0, 12.5, 31, 90])
def test_run_config_refuses_motion_limits_the_robot_would_refuse(value):
    with pytest.raises(ValueError, match="max_relative_target"):
        RunConfig.from_json({"bundle": "b", "task": "t", "robot_port": "/dev/arm",
                             "overhead_camera": "/dev/o", "wrist_camera": "/dev/w",
                             "max_relative_target": value})


def test_the_chosen_task_reaches_the_robot_run_as_its_catalog_index(tmp_path, monkeypatch):
    """Studio used to pass the task only as POLIMA_TASK, which nothing reads, so every
    run used task 0. The robot client resolves --task against bundle.json `tasks`."""
    tasks = ["Place the grey eraser in the white basket", "Place the red cube in the white basket"]
    bundle = tmp_path / "models" / "b"
    bundle.mkdir(parents=True)
    (bundle / "bundle.json").write_text(json.dumps({"policy": "groot", "tasks": tasks, "default_task": 0}))
    engine = StudioRuntime(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(engine, "_spawn", lambda *a, **k: captured.update(args=a[1]))
    monkeypatch.setattr(engine.store, "begin_run", lambda *a, **k: 1)
    for index, task in enumerate(tasks):
        run = _run_config(bundle="b", task=task)
        engine._arm_tokens["t"] = (time.time() + 60, json.dumps(run.to_json(), sort_keys=True))
        engine.start_robot(run, "t")
        args = captured["args"]
        assert args[args.index("--task") + 1] == str(index)


def test_a_task_the_bundle_was_not_compiled_for_fails_preflight(tmp_path: Path) -> None:
    runtime = StudioRuntime(tmp_path, command=tmp_path / "polima")
    runtime.command.touch()
    runtime.command.chmod(0o755)
    run = config(tmp_path)  # task "Pick up the block."
    (tmp_path / "models/demo/bundle.json").write_text(json.dumps({"policy": "groot", "tasks": ["Something else"]}))
    checks = {item["name"]: item["ok"] for item in runtime.preflight(run)}
    assert checks["task is one the bundle was compiled for"] is False
    with pytest.raises(ValueError, match="task is one the bundle was compiled for"):
        runtime.arm(run)


def test_a_bundle_without_a_task_catalog_runs_task_0_whatever_the_label(tmp_path, monkeypatch):
    """ACT has no task catalog: the Studio text is only a label, and --task must stay 0."""
    started = _started_env(tmp_path, monkeypatch, _run_config(bundle="b", task="Place the object in the basket."),
                           policy="act")
    assert "--task" not in started["args"]
