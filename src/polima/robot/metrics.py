"""Opt-in timing of the existing SO101 loop; no changes to robot operations."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np


class TrialMetrics:
    def __init__(self, directory, config):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.events = []
        self.stream = (self.root / "events.jsonl").open("w")
        (self.root / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def record(self, kind, **values):
        with self.lock:
            event = dict(event=kind, elapsed_s=time.perf_counter() - self.started, **values)
            self.events.append(event)
            self.stream.write(json.dumps(event) + "\n")
            self.stream.flush()

    def close(self):
        with self.lock:
            if self.stream.closed:
                return
            summary = {"config": self.config, "duration_s": time.perf_counter() - self.started}
            for kind in ("prediction", "action", "observation", "queue", "queue_check"):
                rows = [e for e in self.events if e["event"] == kind]
                data = {"count": len(rows)}
                times = [r["elapsed_s"] for r in rows]
                if len(times) > 1:
                    data["rate_hz"] = (len(times) - 1) / (times[-1] - times[0])
                    gaps = np.diff(times) * 1000
                    data["interval_ms_mean"] = float(np.mean(gaps))
                    data["interval_ms_p95"] = float(np.percentile(gaps, 95))
                    data["interval_ms_max"] = float(np.max(gaps))
                for key in sorted({k for r in rows for k in r if k.endswith("_ms")}):
                    values = [r[key] for r in rows if key in r]
                    data[key] = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}
                if kind == "queue_check" and rows:
                    data["empty_fraction"] = sum(not r["available"] for r in rows) / len(rows)
                summary[kind] = data
            (self.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self.stream.close()
        print("Trial metrics:", self.root / "summary.json", flush=True)


def instrument_loop(cls, metrics):
    """Wrap the final ACT-patched methods, retaining their return values/lifecycle."""
    originals = {}
    def install(name, wrapper):
        originals[name] = getattr(cls, name)
        setattr(cls, name, wrapper(originals[name]))

    def timed(kind):
        def wrap(original):
            def call(self, *args, **kwargs):
                started = time.perf_counter()
                metrics.record(kind + "_started")
                result = original(self, *args, **kwargs)
                if result is not None:
                    metrics.record(kind, timestep=getattr(self, "latest_action", None), duration_ms=(time.perf_counter() - started) * 1000)
                return result
            return call
        return wrap

    def availability(original):
        def call(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            metrics.record("queue_check", available=bool(result))
            return result
        return call

    def aggregate(original):
        def call(self, incoming, *args, **kwargs):
            age = (time.time() - incoming[0].get_timestamp()) * 1000 if incoming else None
            result = original(self, incoming, *args, **kwargs)
            with self.action_queue_lock:
                depth = self.action_queue.qsize()
                with self.action_queue.mutex:
                    queued_steps = [a.get_timestep() for a in self.action_queue.queue]
            if age is not None:
                from polima.robot.action_trace import request_client
                request_id = getattr(request_client(getattr(self, "som_client", None)), "request_id", None)
                metrics.record("queue", observation_to_queue_ms=age, depth=depth,
                               request_id=request_id,
                               incoming_steps=[a.get_timestep() for a in incoming], queued_steps=queued_steps)
            return result
        return call

    install("control_loop_action", timed("action"))
    install("control_loop_observation", timed("observation"))
    install("actions_available", availability)
    install("_aggregate_action_queues", aggregate)
    def restore():
        for name, method in originals.items():
            setattr(cls, name, method)
    return restore


def run_instrumented(run, metrics):
    # The packaged script imports this entry point before patching RobotClient,
    # then calls it after installing its existing lifecycle wrappers.
    import lerobot.async_inference.robot_client as module
    original = module.async_client
    def entry(*args, **kwargs):
        restore = instrument_loop(module.RobotClient, metrics)
        restore_trace = None
        try:
            if os.environ.get("POLIMA_ACTION_TRACE") == "1":
                from polima.robot.action_trace import ActionTrace
                restore_trace = ActionTrace(metrics).install(module.RobotClient)
            return original(*args, **kwargs)
        finally:
            if restore_trace is not None:
                restore_trace()
            restore()
    module.async_client = entry
    try:
        return run()
    finally:
        module.async_client = original
        metrics.close()
