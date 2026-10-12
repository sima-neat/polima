"""PoLiMa Workbench: checkpoints, builds and boards in one page, on port 8686.

It lists the trained checkpoints and compiled bundles on this computer and the
Modalix boards it knows: boards given with --board, read through their Studio,
and DevKits added on the page (polima.config.devkits), checked over ssh as well.
Standard library only, so the build host needs nothing beyond PoLiMa itself.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from polima.config import devkits
from polima.config.base import BoardConfig

STATIC = Path(__file__).with_name("static")
LOGO = Path(__file__).resolve().parents[1] / "studio/static/assets/neat-mark.svg"
SKIP = {".git", ".venv", "node_modules", "__pycache__", "datasets", "wandb", "models", "fixtures"}
POLICIES = {"act": "ACT", "smolvla": "SmolVLA", "groot": "GR00T N1.6", "groot17": "GR00T N1.7",
            "Gr00tN1d6": "GR00T N1.6", "Gr00tN1d7": "GR00T N1.7"}


def walk(roots: list[Path], marker: str, depth: int = 8, limit: int = 400) -> list[Path]:
    """Directories under `roots` that contain `marker`, without entering their insides."""
    found: list[Path] = []
    for root in roots:
        stack = [(Path(root).expanduser(), 0)]
        while stack and len(found) < limit:
            folder, level = stack.pop()
            try:
                if (folder / marker).is_file():
                    found.append(folder)
                    continue
                if level < depth:
                    stack += [(child, level + 1) for child in folder.iterdir()
                              if child.is_dir() and child.name not in SKIP]
            except OSError:
                continue
    return found


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


GENERIC = {"pretrained_model", "checkpoints", "output", "outputs", "runs", "last"}


def checkpoint(folder: Path) -> dict[str, Any] | None:
    """A LeRobot (…/checkpoints/<step>/pretrained_model) or GR00T (…/checkpoint-<step>) save."""
    config = read_json(folder / "config.json")
    kind = config.get("type") or config.get("model_type")
    if kind not in POLICIES or not any(folder.glob("*.safetensors")):
        return None
    names = list(reversed(folder.parts))
    run = next((n for n in names if n not in GENERIC and not re.fullmatch(r"(checkpoint-)?\d+", n)),
               folder.name)
    step = next((int(m.group(1)) for n in names[:3]
                 if (m := re.fullmatch(r"(?:checkpoint-)?(\d+)", n))), None)
    return {"policy": POLICIES[kind], "run": run, "step": step if step is not None else "final",
            "path": str(folder), "modified": folder.stat().st_mtime}


def bundle(folder: Path) -> dict[str, Any]:
    manifest, plan = read_json(folder / "bundle.json"), read_json(folder / "plan.json")
    graphs = manifest.get("graphs", [])
    robot = plan.get("robot") or {}
    wire = plan.get("wire") or {}
    return {
        "id": manifest.get("bundle_id", folder.name), "path": str(folder),
        "policy": "GR00T N1.7" if "-v1.7-" in folder.name else
                  POLICIES.get(manifest.get("policy"), manifest.get("policy", "?")),
        "graphs": len(graphs), "bytes": manifest.get("total_elf_bytes", 0),
        "precision": sorted({str(g.get("precision", "bf16")) for g in graphs}),
        "robot": robot.get("calibration_id", ""), "actions": wire.get("response_shape", []),
        "tasks": plan.get("tasks") or manifest.get("tasks") or [],
        "robot_client": (folder / "robot_client/start.sh").is_file(),
        "modified": folder.stat().st_mtime,
    }


def latest_evidence(records: list[dict], board: str) -> dict[str, dict[str, Any]]:
    """The newest completed benchmark of each bundle, from a board's Studio records.

    Studio lists benchmarks newest first; failed or unfinished ones are not evidence.
    """
    evidence: dict[str, dict[str, Any]] = {}
    for record in records:
        result = record.get("result")
        if record.get("status") != "completed" or not result or record.get("bundle") in evidence:
            continue
        check = result.get("validation") or {}
        evidence[record["bundle"]] = {
            "board": board, "measured_at": record.get("ended_at") or record.get("started_at"),
            "median_ms": result.get("median"), "p95_ms": result.get("p95"),
            "p99_ms": result.get("p99"), "rate_hz": result.get("throughput_hz"),
            "iterations": result.get("iterations"), "stages": result.get("stages", []),
            "check": check.get("status"), "cosine": check.get("cosine"), "max_abs": check.get("max_abs"),
        }
    return evidence


def latest_robot_runs(runs: list[dict]) -> dict[str, dict[str, Any]]:
    """The newest robot run of each bundle; Studio lists robot runs newest first."""
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        bundle_id = run.get("bundle_id")
        if bundle_id and bundle_id not in latest:
            latency = (run.get("latency_ms") or {}).get("model") or {}
            latest[bundle_id] = {key: run.get(key) for key in (
                "run_id", "status", "started_at", "ended_at", "cycles", "actions", "inferences",
                "outcome", "notes", "failure_tags", "needs_review", "board")} | {
                    "model_ms_p50": latency.get("p50")}
    return latest


REVIEWED = ("success", "partial", "failed")


def nearest_rank(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def robot_run_stats(runs: list[dict]) -> dict[str, dict[str, Any]]:
    """Real robot performance per bundle, from Studio's robot-run summaries.

    Only reviewed runs (success, partial, failed) enter the rates; unknown
    outcomes are counted apart. Summaries keep each run's latency percentiles,
    not every inference, so latency is the median of the runs' p50 and the
    95th percentile of their p95.
    """
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        if run.get("bundle_id"):
            grouped.setdefault(run["bundle_id"], []).append(run)
    stats = {}
    for bundle_id, group in grouped.items():
        reviewed = [r for r in group if r.get("outcome") in REVIEWED]
        count = {outcome: sum(1 for r in reviewed if r["outcome"] == outcome) for outcome in REVIEWED}
        tags: dict[str, int] = {}
        for run in reviewed:
            for tag in run.get("failure_tags") or []:
                tags[tag] = tags.get(tag, 0) + 1
        latency = [(r.get("latency_ms") or {}).get("model") or {} for r in group]
        p50 = [v["p50"] for v in latency if isinstance(v.get("p50"), int | float)]
        p95 = [v["p95"] for v in latency if isinstance(v.get("p95"), int | float)]
        latest = max(reviewed, key=lambda r: (r.get("started_at") or 0,
                                              (r.get("annotation") or {}).get("reviewed_at") or 0),
                     default=None)
        policies = {r["policy"] for r in group if r.get("policy")}
        stats[bundle_id] = {
            "runs": len(group), "reviewed": len(reviewed), "unreviewed": len(group) - len(reviewed),
            **count,
            "success_rate": count["success"] / len(reviewed) if reviewed else None,
            "partial_plus_rate": (count["success"] + count["partial"]) / len(reviewed) if reviewed else None,
            "failure_tags": sorted(tags.items(), key=lambda item: (-item[1], item[0])),
            "latency_ms": {"p50": nearest_rank(p50, 0.5) if p50 else None,
                           "p95": nearest_rank(p95, 0.95) if p95 else None},
            "cycles": sum(r.get("cycles") or 0 for r in group),
            "actions": sum(r.get("actions") or 0 for r in group),
            "latest_reviewed": latest and {"run_id": latest.get("run_id"), "outcome": latest["outcome"],
                                           "started_at": latest.get("started_at")},
            "policy": policies.pop() if len(policies) == 1 else None,  # never guessed
        }
        stats[bundle_id]["labels"] = stat_labels(stats[bundle_id])
    return stats


def stat_labels(stat: dict[str, Any] | None) -> dict[str, str]:
    """The short text the Compare table shows for one bundle."""
    if not stat or not stat["reviewed"]:
        state = "Needs review" if stat and stat["runs"] else "No reviewed runs"
        return {"reviewed": state, "success": "—", "partial_plus": "—", "failures": "—",
                "latency": latency_label(stat)}
    reviewed = stat["reviewed"]
    pending = f" · {stat['unreviewed']} need review" if stat["unreviewed"] else ""
    return {
        "reviewed": f"{reviewed} of {stat['runs']} runs{pending}",
        "success": f"{stat['success']}/{reviewed} success",
        "partial_plus": f"{stat['success'] + stat['partial']}/{reviewed} partial+",
        "failures": ", ".join(f"{tag} x{n}" for tag, n in stat["failure_tags"][:3]) or "—",
        "latency": latency_label(stat),
    }


def latency_label(stat: dict[str, Any] | None) -> str:
    latency = (stat or {}).get("latency_ms") or {}
    parts = [f"{key} {latency[key]:.0f} ms" for key in ("p50", "p95") if latency.get(key) is not None]
    return " / ".join(parts) or "—"


STAGES = (("installed", "Installed"), ("benchmark", "Benchmark"), ("robot_run", "Robot run"),
          ("reviewed", "Reviewed"), ("compared", "Compared"), ("export", "Export"))


def readiness(build: dict[str, Any], boards: list[dict[str, Any]],
              reviewed_elsewhere: bool) -> dict[str, Any]:
    """Where one bundle stands, checkpoint to measured skill, from evidence only.

    Each stage is done, needs action (with the next step), or unavailable until
    an earlier stage is done. `reviewed_elsewhere`: another bundle has reviewed
    robot runs, so this one can be compared against something real.
    """
    online = [b for b in boards if b.get("online")]
    installed = any(build["id"] in (b.get("bundles") or []) for b in online)
    stats, latest = build.get("robot_stats") or {}, build.get("robot_run") or {}
    reviewed, runs = stats.get("reviewed", 0), stats.get("runs", 0)
    to_fix = stats.get("failed", 0) + stats.get("partial", 0)
    exportable = any(b.get("manifest") is not None for b in online)

    def stage(done: bool, action: str | None, available: bool, note: str = "") -> dict[str, Any]:
        status = "done" if done else "needs_action" if available else "unavailable"
        return {"status": status, "action": action if status == "needs_action" else None, "note": note}

    stages = {
        "installed": stage(installed, "Install on a board", bool(online),
                           "" if online else "No board online"),
        "benchmark": stage(bool(build.get("evidence")), "Run benchmark", installed),
        "robot_run": stage(bool(latest), "Run on robot", installed and build.get("robot_client", True),
                           "" if build.get("robot_client", True) else "Benchmark-only bundle"),
        "reviewed": stage(reviewed > 0 and not latest.get("needs_review"), "Review latest run", runs > 0),
        "compared": stage(reviewed > 0 and reviewed_elsewhere, "Compare policies", reviewed > 0,
                          "" if reviewed_elsewhere else "Review runs of another policy to compare"),
        "export": stage(reviewed > 0 and to_fix == 0, "Export failures", to_fix > 0 and exportable,
                        "No export needed" if reviewed and not to_fix
                        else "" if exportable or not to_fix else "The board's Studio cannot export yet"),
    }
    next_action = next((stages[key]["action"] for key, _ in STAGES if stages[key]["action"]), None)
    complete = all(stage["status"] == "done" or stage["note"] == "No export needed" for stage in stages.values())
    if next_action is None and not online:
        next_action = "Add a DevKit"
    return {"stages": [{"key": key, "label": label, **stages[key]} for key, label in STAGES],
            "next_action": next_action, "complete": complete, "summary": evidence_summary(build)}


def evidence_summary(build: dict[str, Any]) -> dict[str, str]:
    """One short line per kind of evidence the readiness rests on."""
    benchmark, latest = build.get("evidence") or {}, build.get("robot_run") or {}
    labels = build.get("robot_labels") or stat_labels(None)
    stats = build.get("robot_stats") or {}
    check = benchmark.get("check")
    run = latest.get("status", "")
    if latest.get("outcome") not in (None, "unknown"):
        run += f", reviewed {latest['outcome']}"
    elif latest.get("needs_review"):
        run += ", needs review"
    return {
        "benchmark": f"{benchmark['median_ms']:.1f} ms median · check {check or 'unavailable'}"
                     if benchmark.get("median_ms") is not None else "No benchmark",
        "robot_run": run or "No robot run",
        "reviews": f"{stats['success']} success · {stats['partial']} partial · {stats['failed']} failed"
                   if stats.get("reviewed") else labels["reviewed"],
        "failures": labels["failures"] if labels["failures"] != "—" else "",
    }


def fetch(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def board_url(address: str) -> str:
    """A board's Studio address; plain host names get http:// and Studio's port, 8080."""
    url = address if "://" in address else f"http://{address}"
    return url if url.rsplit(":", 1)[-1].isdigit() else f"{url}:8080"


def board(address: str, timeout: float = 2.0) -> dict[str, Any]:
    """One board's state from its Studio; offline when Studio does not answer."""
    url = board_url(address)
    status: dict[str, Any] = {"address": url, "online": False}
    try:
        snapshot = fetch(f"{url}/api/v1/snapshot", timeout)
        history = fetch(f"{url}/api/v1/history", timeout)
    except (OSError, ValueError, urllib.error.URLError):
        return status
    evidence: dict[str, list] = {}
    for key, path in (("benchmarks", "benchmarks"), ("robot_runs", "robot-runs?limit=200")):
        try:  # a board with an older Studio still reports its state
            evidence[key] = fetch(f"{url}/api/v1/{path}", timeout).get("items", [])
        except (OSError, ValueError, urllib.error.URLError):
            evidence[key] = []
    try:  # counts only; the manifest itself is fetched when it is downloaded
        manifest = fetch(f"{url}/api/v1/robot-runs/improvement-manifest", timeout).get("counts")
    except (OSError, ValueError, urllib.error.URLError):
        manifest = None  # this board's Studio cannot export yet
    bundles = snapshot.get("bundles", [])
    status.update(online=True, state=snapshot.get("state"),
                  active=next((b["id"] for b in bundles if b.get("active")), None),
                  bundles=[b["id"] for b in bundles], server=snapshot.get("server", {}),
                  storage=snapshot.get("storage", {}),
                  runs=history.get("runs", history.get("items", []))[:20],
                  evidence=latest_evidence(evidence["benchmarks"], url),
                  robot_runs=[{**run, "board": url} for run in evidence["robot_runs"]],
                  manifest=manifest,
                  runs_needing_review=sum(1 for run in evidence["robot_runs"] if run.get("needs_review")))
    return status


def devkit(entry: dict[str, Any]) -> dict[str, Any]:
    """A saved DevKit: its Studio's state (when Studio runs) plus what ssh finds."""
    return {**board(studio_address(entry)), "name": entry["name"], "host": entry["host"],
            "ssh": devkits.probe(entry, BoardConfig().ssh_options)}


def studio_address(entry: dict[str, Any]) -> str:
    return entry["host"].split("@")[-1]


def board_manifest(workbench: Workbench, index: str, everything: bool) -> tuple[bytes, str]:
    """A configured board's improvement manifest, read server-side like its other evidence.

    Only boards the Workbench knows are reachable, by position, so the download
    cannot be pointed at any other address.
    """
    addresses = workbench.addresses()
    if not index.isdigit() or int(index) >= len(addresses):
        raise LookupError("no such board")
    url = board_url(addresses[int(index)])
    manifest = fetch(f"{url}/api/v1/robot-runs/improvement-manifest{'?all=1' if everything else ''}", 10.0)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(manifest.get("created_at") or 0))
    board_name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(manifest.get("source_board") or f"board{index}"))
    return json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n", \
        f"improvement-manifest-{board_name}-{stamp}.json"


class Workbench:
    def __init__(self, models: list[Path], bundles: list[Path], boards: list[str]) -> None:
        self.models, self.bundles, self.boards = models, bundles, boards
        self.lock, self.scanned, self.found = threading.Lock(), 0.0, ([], [])

    def scan(self, force: bool = False) -> tuple[list, list]:
        """Folder contents, re-read at most once a minute: large trees are slow to walk."""
        with self.lock:
            if force or time.time() - self.scanned > 60:
                checkpoints = [c for c in map(checkpoint, walk(self.models, "config.json")) if c]
                builds = [bundle(folder) for folder in walk(self.bundles, "bundle.json")]
                self.found, self.scanned = (checkpoints, builds), time.time()
            return self.found

    def addresses(self) -> list[str]:
        """Studio addresses in the order the snapshot lists boards."""
        return self.boards + [studio_address(entry) for entry in devkits.load()]

    def snapshot(self, refresh: bool = False) -> dict[str, Any]:
        from polima.compile.toolchain import find_compiler_bin

        checkpoints, builds = self.scan(refresh)
        compiler = find_compiler_bin()
        with ThreadPoolExecutor(max_workers=8) as pool:  # an offline board costs its timeout once
            boards = list(pool.map(board, self.boards)) + list(pool.map(devkit, devkits.load()))
        measured: dict[str, dict[str, Any]] = {}
        for found in boards:
            for bundle_id, item in found.pop("evidence", {}).items():
                previous = measured.get(bundle_id)
                if previous is None or (item["measured_at"] or 0) > (previous["measured_at"] or 0):
                    measured[bundle_id] = item  # the newest measurement across boards
        runs = sorted((run for found in boards for run in found.pop("robot_runs", [])),
                      key=lambda run: run.get("started_at") or 0, reverse=True)
        latest, stats = latest_robot_runs(runs), robot_run_stats(runs)
        builds = [{**b, "evidence": measured.get(b["id"]), "robot_run": latest.get(b["id"]),
                   "robot_stats": stats.get(b["id"]),
                   "robot_labels": stats[b["id"]]["labels"] if b["id"] in stats else stat_labels(None)}
                  for b in builds]
        reviewed_ids = {bundle_id for bundle_id, stat in stats.items() if stat["reviewed"]}
        builds = [{**b, "readiness": readiness(b, boards, bool(reviewed_ids - {b["id"]}))} for b in builds]
        return {
            "computer": {"host": socket.gethostname(), "compiler": str(compiler or ""),
                         "model_folders": [str(p) for p in self.models],
                         "bundle_folders": [str(p) for p in self.bundles]},
            "checkpoints": sorted(checkpoints, key=lambda c: -c["modified"]),
            "bundles": sorted(builds, key=lambda b: -b["modified"]),
            "boards": boards,
            "generated": self.scanned,
        }


def serve(workbench: Workbench, host: str = "127.0.0.1", port: int = 8686) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(32)  # writes need it as both cookie and header, as in Studio

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body: bytes, kind: str, status: int = 200, filename: str | None = None,
                 cookie: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            if cookie:
                self.send_header("Set-Cookie", cookie)
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self.send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/logo.svg":
                return self.send(LOGO.read_bytes(), "image/svg+xml")
            if path == "/api/v1/session":
                return self.send(json.dumps({"csrf_token": token}).encode(), "application/json",
                                 cookie=f"polima_csrf={token}; HttpOnly; SameSite=Strict; Path=/")
            if path == "/api/v1/snapshot":
                refresh = self.path.endswith("?refresh=1")
                return self.send(json.dumps(workbench.snapshot(refresh)).encode(), "application/json")
            if path == "/api/v1/improvement-manifest":
                query = dict(item.split("=", 1) for item in self.path.partition("?")[2].split("&") if "=" in item)
                try:
                    body, name = board_manifest(workbench, query.get("board", ""), query.get("all") == "1")
                except LookupError:
                    return self.send(b"no such board", "text/plain", 404)
                except (OSError, ValueError, urllib.error.URLError) as error:
                    return self.send(f"the board did not provide a manifest: {error}".encode(),
                                     "text/plain", 502)
                return self.send(body, "application/json", filename=name)
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            """Add or remove a saved DevKit. Only the saved list changes; ssh runs a fixed probe."""
            path = self.path.split("?", 1)[0]
            cookie = SimpleCookie(self.headers.get("Cookie") or "").get("polima_csrf")
            header = self.headers.get("X-CSRF-Token") or ""
            if not (cookie and hmac.compare_digest(cookie.value.encode(), token.encode())
                    and hmac.compare_digest(header.encode(), token.encode())):
                return self.send(json.dumps({"error": "invalid CSRF token"}).encode(), "application/json", 403)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(length, 4096)) or b"{}")
                if path == "/api/v1/devkits":
                    entry = devkits.add(str(body.get("host", "")), str(body.get("key", "")),
                                        str(body.get("name", "")))
                    reply = {**entry, "ssh": devkits.probe(entry, BoardConfig().ssh_options)}
                elif path == "/api/v1/devkits/remove":
                    devkits.remove(str(body.get("name", "")))
                    reply = {"removed": body["name"]}
                else:
                    return self.send(b"not found", "text/plain", 404)
            except (ValueError, LookupError, AttributeError) as error:
                return self.send(json.dumps({"error": str(error)}).encode(), "application/json", 400)
            self.send(json.dumps(reply).encode(), "application/json")

    for candidate in range(port, port + 10):
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
            break
        except OSError:
            continue
    else:
        raise OSError(f"ports {port}-{port + 9} are all in use")
    server.daemon_threads = True
    return server


def folders(values: list[str] | None, environment: str, default: list[Path]) -> list[Path]:
    given = values or [item for item in os.environ.get(environment, "").split(os.pathsep) if item]
    return [Path(item).expanduser() for item in given] if given else default
