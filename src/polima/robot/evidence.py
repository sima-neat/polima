"""Robot run evidence: one durable record for every run of a policy on the arm.

A run is a folder on the board, <POLIMA_ROOT>/var/runs/<run_id>/:

    events.jsonl     what happened, one JSON object per line: {"t": epoch, "event": kind, ...}
    summary.json     written when the run ends: status, counts, latency, warnings
    annotation.json  the operator's review: outcome, failure tags, notes, selected ranges

The robot client writes it, so terminal runs are recorded as well as Studio
ones. A run that ends without a final event -- an emergency halt kills the
process -- reads as interrupted once its events stop changing.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = "polima-robot-run-v1"
ENDINGS = {"run_completed": "completed", "run_failed": "failed", "run_interrupted": "interrupted"}
OUTCOMES = ("success", "partial", "failed", "unknown")
FAILURE_TAGS = ("missed_grasp", "dropped_object", "wrong_target", "hesitation", "jitter",
                "collision_risk", "camera_issue", "calibration_issue", "latency_issue",
                "operator_stop", "other")
RANGE_UNITS = ("time", "event")  # seconds since the run started, or event indices
NOTES_MAX, LABEL_MAX, RANGES_MAX = 4000, 200, 50
ENDED = ("completed", "failed", "interrupted")
RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
STALE_S = 10.0  # a run whose events have not changed for this long is not running
METADATA = ("bundle_id", "policy", "robot", "task", "cameras", "calibration", "rest_pose", "settings")


def runs_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("POLIMA_ROOT", "/media/nvme/polima")) / "var/runs"


def run_dir(root: str | Path | None, run_id: str) -> Path:
    if not RUN_ID.match(str(run_id)):
        raise FileNotFoundError(f"no robot run {run_id!r}")
    folder = runs_root(root) / run_id
    if not (folder / "events.jsonl").is_file():
        raise FileNotFoundError(f"no robot run {run_id!r}")
    return folder


class RunRecorder:
    """Appends events from the control and inference threads; flushes about once a second."""

    def __init__(self, folder: Path, metadata: dict[str, Any], clock=time.time) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.folder, self.clock = folder, clock
        self.lock = threading.Lock()
        self.handle = (folder / "events.jsonl").open("a", encoding="utf-8")
        self.flushed, self.status = 0.0, None
        self.event("run_started", schema=SCHEMA, run_id=folder.name,
                   **{key: metadata.get(key) for key in METADATA})

    @classmethod
    def create(cls, root: str | Path | None = None, **metadata: Any) -> RunRecorder:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        return cls(runs_root(root) / run_id, metadata)

    def event(self, kind: str, **fields: Any) -> None:
        line = json.dumps({"t": round(self.clock(), 3), "event": kind, **fields},
                          separators=(",", ":"), default=str)
        with self.lock:
            if self.handle.closed:
                return
            self.handle.write(line + "\n")
            now = time.monotonic()
            if now - self.flushed >= 1.0 or kind in ("run_started", "warning", "error"):
                self.handle.flush()
                self.flushed = now

    def finish(self, status: str, error: str | None = None) -> dict[str, Any]:
        """Record how the run ended and write its summary. Only the first call counts."""
        ending = {value: key for key, value in ENDINGS.items()}[status]
        with self.lock:
            if self.status is not None:
                return summarize(self.folder)
            self.status = status
        self.event(ending, **({"error": error} if error else {}))
        with self.lock:
            self.handle.close()
        summary = summary_of(read_events(self.folder), self.folder.name, status)
        (self.folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summarize(self.folder)


def read_events(folder: Path, limit: int | None = None) -> list[dict[str, Any]]:
    events = []
    with (folder / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue  # a line cut short when the process was killed
            if limit is not None and len(events) >= limit:
                break
    return events


def summary_of(events: list[dict[str, Any]], run_id: str, status: str) -> dict[str, Any]:
    started = next((e for e in events if e["event"] == "run_started"), {})
    inferences = [e for e in events if e["event"] == "inference_completed"]
    ending = next((e for e in reversed(events) if e["event"] in ENDINGS), None)
    count = lambda kind: sum(1 for e in events if e["event"] == kind)
    return {
        "schema": SCHEMA, "run_id": run_id, "status": status,
        "started_at": started.get("t"), "ended_at": (ending or (events[-1] if events else {})).get("t"),
        **{key: started.get(key) for key in METADATA},
        "inferences": len(inferences), "actions": count("action_sent"),
        "cycles": count("cycle_completed"), "grasps": count("grasp_detected"),
        "latency_ms": {
            "model": percentiles([e.get("model_ms") for e in inferences]),
            "observation_to_queue": percentiles([e.get("queue_ms") for e in inferences]),
        },
        "warnings": [e.get("message") for e in events if e["event"] == "warning"][:20],
        "errors": [e.get("message") or e.get("error") for e in events
                   if e["event"] == "error" or (e["event"] == "run_failed" and e.get("error"))][:20],
    }


def percentiles(values: list[Any]) -> dict[str, float] | None:
    values = sorted(float(v) for v in values if isinstance(v, int | float))
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, round(q * (len(values) - 1)))]
    return {"p50": round(pick(0.5), 2), "p95": round(pick(0.95), 2), "max": round(values[-1], 2)}


def summarize(folder: Path, now: float | None = None) -> dict[str, Any]:
    """The run's summary plus the operator's annotation, for a finished or live run."""
    stored = folder / "summary.json"
    if stored.is_file():
        summary = json.loads(stored.read_text())
    else:
        events = read_events(folder)
        ending = next((e for e in reversed(events) if e["event"] in ENDINGS), None)
        age = (now or time.time()) - (folder / "events.jsonl").stat().st_mtime
        status = ENDINGS[ending["event"]] if ending else ("running" if age < STALE_S else "interrupted")
        summary = summary_of(events, folder.name, status)
    review = read_annotation(folder)
    return {**summary, "annotation": review, "outcome": review["outcome"], "notes": review["notes"],
            "failure_tags": review["failure_tags"],
            "needs_review": review["outcome"] == "unknown" and summary["status"] in ENDED}


def read_annotation(folder: Path) -> dict[str, Any]:
    path = folder / "annotation.json"
    stored = json.loads(path.read_text()) if path.is_file() else {}
    return {"outcome": stored.get("outcome", "unknown"), "notes": stored.get("notes", ""),
            "failure_tags": stored.get("failure_tags", []),
            "selected_ranges": stored.get("selected_ranges", []),
            "reviewed_at": stored.get("reviewed_at", stored.get("annotated_at")),
            "reviewed_by": stored.get("reviewed_by")}


def validate_annotation(body: dict[str, Any]) -> dict[str, Any]:
    """The operator's review, checked; anything wrong raises ValueError (HTTP 400)."""
    if not isinstance(body, dict):
        raise ValueError("the review must be a JSON object")  # noqa: TRY004 - HTTP 400
    outcome = body.get("outcome", "unknown")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
    notes = body.get("notes") or ""
    if not isinstance(notes, str) or len(notes) > NOTES_MAX:
        raise ValueError(f"notes must be text of at most {NOTES_MAX} characters")
    tags = body.get("failure_tags") or []
    unknown = [tag for tag in tags if tag not in FAILURE_TAGS] if isinstance(tags, list) else [tags]
    if unknown:
        raise ValueError(f"unknown failure tags {unknown}; use {', '.join(FAILURE_TAGS)}")
    ranges = body.get("selected_ranges") or []
    if not isinstance(ranges, list) or len(ranges) > RANGES_MAX:
        raise ValueError(f"selected_ranges must be a list of at most {RANGES_MAX} ranges")
    reviewer = body.get("reviewed_by")
    if reviewer is not None and (not isinstance(reviewer, str) or len(reviewer) > LABEL_MAX):
        raise ValueError(f"reviewed_by must be text of at most {LABEL_MAX} characters")
    return {"outcome": outcome, "notes": notes, "failure_tags": list(dict.fromkeys(tags)),
            "selected_ranges": [checked_range(item) for item in ranges],
            "reviewed_at": time.time(), "reviewed_by": reviewer or None}


def checked_range(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("each selected range is an object with start and end")  # noqa: TRY004
    unit, start, end = item.get("unit", "time"), item.get("start"), item.get("end")
    numbers = (int,) if unit == "event" else (int, float)
    if unit not in RANGE_UNITS:
        raise ValueError(f"range unit must be one of {', '.join(RANGE_UNITS)}")
    if not all(isinstance(v, numbers) and not isinstance(v, bool) for v in (start, end)):
        raise ValueError(f"{unit} range start and end must be {'whole ' if unit == 'event' else ''}numbers")
    if not 0 <= start <= end:
        raise ValueError(f"a range needs 0 <= start <= end, got {start} to {end}")
    label = item.get("label") or ""
    if not isinstance(label, str) or len(label) > LABEL_MAX:
        raise ValueError(f"a range label is text of at most {LABEL_MAX} characters")
    return {"unit": unit, "start": start, "end": end, "label": label}


def annotate(folder: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Replace the run's review with `body`; the same body twice gives the same review."""
    review = validate_annotation(body)
    previous = read_annotation(folder)
    if {**previous, "reviewed_at": None} == {**review, "reviewed_at": None}:
        review["reviewed_at"] = previous["reviewed_at"]  # unchanged: keep when it was reviewed
    temporary = folder / "annotation.json.tmp"
    temporary.write_text(json.dumps(review, indent=2) + "\n")
    temporary.replace(folder / "annotation.json")
    return summarize(folder)


def reviewed_runs(root: str | Path | None = None,
                  everything: bool = False) -> tuple[list[dict[str, Any]], int]:
    """Reviewed runs with the wanted outcomes, oldest first, and how many runs were considered."""
    wanted = ("success", "partial", "failed") if everything else ("partial", "failed")
    runs = list_runs(root, limit=100000)
    chosen = [r for r in runs if r["annotation"]["reviewed_at"] is not None
              and r["annotation"]["outcome"] in wanted]
    return sorted(chosen, key=lambda r: (r.get("started_at") or 0, r["run_id"])), len(runs)


def export_reviews(root: str | Path | None = None, everything: bool = False) -> list[dict[str, Any]]:
    """Reviewed runs to learn from: partial and failed ones, or every reviewed run.

    The seed of the improvement loop -- each item points at its events.jsonl.
    """
    exported = []
    for summary in reviewed_runs(root, everything)[0]:
        review = summary["annotation"]
        exported.append({
            "run_id": summary["run_id"], "bundle_id": summary["bundle_id"],
            "policy": summary["policy"], "task": summary["task"], "outcome": review["outcome"],
            "failure_tags": review["failure_tags"], "selected_ranges": review["selected_ranges"],
            "notes": review["notes"], "reviewed_at": review["reviewed_at"],
            "reviewed_by": review["reviewed_by"],
            "metrics": {key: summary[key] for key in (
                "status", "started_at", "ended_at", "inferences", "actions", "cycles", "grasps",
                "latency_ms")},
            "events": str(runs_root(root) / summary["run_id"] / "events.jsonl"),
        })
    return exported


MANIFEST_SCHEMA = "polima-improvement-manifest-v1"
#: What each failure suggests doing next, most urgent first.
HINTS = {
    "safety_review": ("collision_risk",),
    "setup_data_quality_fix": ("camera_issue", "calibration_issue"),
    "candidate_demo_or_label_fix": ("missed_grasp", "dropped_object", "wrong_target"),
    "runtime_or_policy_rate_investigation": ("latency_issue", "jitter", "hesitation"),
    "manual_review": ("operator_stop", "other"),
}


def improvement_hints(tags: list[str]) -> list[str]:
    """Next steps suggested by a run's failure tags, most urgent first; untagged runs need a person."""
    hints = [hint for hint, members in HINTS.items() if any(tag in members for tag in tags)]
    return hints or ["manual_review"]


def improvement_manifest(root: str | Path | None = None, everything: bool = False,
                         board: str | None = None, now: float | None = None) -> dict[str, Any]:
    """Reviewed failed and partial runs (all reviewed runs with `everything`) as one manifest.

    Runs are listed oldest first, so a manifest only grows at the end as runs are added.
    This is what a later dataset-repair or retraining step consumes.
    """
    chosen, considered = reviewed_runs(root, everything)
    outcomes = ("success", "partial", "failed") if everything else ("partial", "failed")
    counts = {"total_runs_considered": considered, "exported_runs": len(chosen)}
    counts.update({outcome: sum(1 for r in chosen if r["outcome"] == outcome)
                   for outcome in sorted(outcomes)})
    return {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": time.time() if now is None else now,
        "source_board": board,
        "filters": {"reviewed_only": True, "outcomes": sorted(outcomes), "order": "oldest first"},
        "counts": counts,
        "runs": [manifest_record(root, summary) for summary in chosen],
    }


def manifest_record(root: str | Path | None, summary: dict[str, Any]) -> dict[str, Any]:
    review, folder = summary["annotation"], runs_root(root) / summary["run_id"]
    latency = summary.get("latency_ms") or {}
    hints = improvement_hints(review["failure_tags"])
    return {
        "run_id": summary["run_id"], "bundle_id": summary.get("bundle_id"),
        "policy": summary.get("policy"), "task": summary.get("task"),
        "outcome": review["outcome"], "failure_tags": review["failure_tags"],
        "notes": review["notes"], "selected_ranges": review["selected_ranges"],
        "reviewed_at": review["reviewed_at"], "reviewed_by": review["reviewed_by"],
        "status": summary.get("status"),
        "started_at": summary.get("started_at"), "ended_at": summary.get("ended_at"),
        "metrics": {
            "model_latency_ms": latency.get("model"),
            "observation_to_queue_ms": latency.get("observation_to_queue"),
            "inferences": summary.get("inferences"), "cycles": summary.get("cycles"),
            "actions": summary.get("actions"),
        },
        "provenance": {
            "summary": str(folder / "summary.json") if (folder / "summary.json").is_file() else None,
            "events": str(folder / "events.jsonl"),
            "calibration_sha256": (summary.get("calibration") or {}).get("sha256"),
            "rest_pose": summary.get("rest_pose"),
            "cameras": summary.get("cameras"),
        },
        "improvement_hint": hints[0],
        "improvement_hints": hints,
    }


def list_runs(root: str | Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Newest first; run ids start with their start time."""
    base = runs_root(root)
    folders = sorted((p for p in base.iterdir() if RUN_ID.match(p.name)
                      and (p / "events.jsonl").is_file()), reverse=True) if base.is_dir() else []
    return [summarize(folder) for folder in folders[:limit]]
