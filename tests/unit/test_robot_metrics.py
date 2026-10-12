import json
import queue
import threading
from types import SimpleNamespace

import pytest

from polima.robot.metrics import TrialMetrics, instrument_loop, run_instrumented


def test_records_queue_starvation_timing_and_preserves_methods(tmp_path):
    class Client:
        def __init__(self):
            self.action_queue = queue.Queue()
            self.action_queue_lock = threading.Lock()
        def actions_available(self):
            return not self.action_queue.empty()
        def control_loop_action(self):
            self.latest_action = self.action_queue.get_nowait().get_timestep()
            return {"joint": 1}
        def control_loop_observation(self):
            return {"camera": 1}
        def _aggregate_action_queues(self, incoming):
            self.action_queue.put(incoming[0])
            return "unchanged"
    original = Client.control_loop_action
    metrics = TrialMetrics(tmp_path / "trial", {"fps": 30})
    restore = instrument_loop(Client, metrics)
    client = Client()
    assert client.actions_available() is False
    import time
    item = SimpleNamespace(get_timestamp=lambda: time.time() - 0.2, get_timestep=lambda: 7)
    assert client._aggregate_action_queues([item]) == "unchanged"
    assert client.actions_available() is True
    assert client.control_loop_action() == {"joint": 1}
    assert client.control_loop_observation() == {"camera": 1}
    metrics.record("prediction", server_compute_ms=100, prediction_ms=120)
    metrics.record("prediction", server_compute_ms=200, prediction_ms=240)
    restore()
    assert Client.control_loop_action is original
    metrics.close()
    summary = json.loads((metrics.root / "summary.json").read_text())
    assert summary["queue_check"]["empty_fraction"] == 0.5
    assert summary["prediction"]["server_compute_ms"]["mean"] == 150
    assert summary["action"]["count"] == 1
    events = [json.loads(line) for line in (metrics.root / "events.jsonl").read_text().splitlines()]
    assert next(e for e in events if e["event"] == "queue")["queued_steps"] == [7]
    assert next(e for e in events if e["event"] == "action")["timestep"] == 7
    assert summary["queue"]["observation_to_queue_ms"]["mean"] >= 190
    assert len((metrics.root / "events.jsonl").read_text().splitlines()) == 9
    with pytest.raises(FileExistsError):
        TrialMetrics(metrics.root, {})


def test_bootstrap_installs_after_script_patches_and_saves_on_interrupt(tmp_path, monkeypatch):
    import sys
    import types
    module = types.ModuleType("lerobot.async_inference.robot_client")
    class Client:
        def actions_available(self): return False
        def control_loop_action(self): return {}
        def control_loop_observation(self): return {}
        def _aggregate_action_queues(self, incoming): return None
    module.RobotClient = Client
    def original():
        Client().control_loop_action()
        raise KeyboardInterrupt
    module.async_client = original
    root = types.ModuleType("lerobot")
    package = types.ModuleType("lerobot.async_inference")
    root.async_inference = package
    package.robot_client = module
    for name, value in [("lerobot", root), ("lerobot.async_inference", package),
                        ("lerobot.async_inference.robot_client", module)]:
        monkeypatch.setitem(sys.modules, name, value)
    metrics = TrialMetrics(tmp_path / "trial", {})
    called = []
    def patched(self):
        called.append(True)
        return {"joint": 2}
    def script():
        entry = module.async_client
        Client.control_loop_action = patched
        entry()
    with pytest.raises(KeyboardInterrupt):
        run_instrumented(script, metrics)
    assert called == [True]
    assert Client.control_loop_action is patched
    assert module.async_client is original
    assert json.loads((metrics.root / "summary.json").read_text())["action"]["count"] == 1
