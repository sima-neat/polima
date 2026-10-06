from __future__ import annotations

import json
import secrets
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from polima.robot import evidence

from . import transfer
from .models import RunConfig
from .runtime import Conflict, StudioRuntime


def create_app(root: Path | str = "/media/nvme/polima", runtime: StudioRuntime | None = None):
    from flask import (
        Flask,
        Response,
        jsonify,
        make_response,
        request,
        send_file,
        send_from_directory,
    )

    static = Path(__file__).with_name("static")
    app = Flask(__name__, static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=1_000_000, JSON_SORT_KEYS=False)
    engine = runtime or StudioRuntime(Path(root))
    app.extensions["polima_runtime"] = engine
    csrf_tokens: set[str] = set()

    def mutation(fn: Callable[[], Any], lease: bool = True):
        cookie = request.cookies.get("polima_csrf")
        header = request.headers.get("X-CSRF-Token")
        if not cookie or cookie != header or cookie not in csrf_tokens:
            return jsonify(error="invalid CSRF token"), 403
        if lease:
            engine.require_lease(request.headers.get("X-Controller-Lease"))
        return jsonify(fn())

    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(Conflict)
    def conflict(error):
        return jsonify(error=str(error)), 409

    @app.errorhandler(FileNotFoundError)
    def missing(error):
        return jsonify(error=str(error)), 404

    @app.errorhandler(RuntimeError)
    def failed(error):
        return jsonify(error=str(error)), 500

    @app.get("/")
    def index():
        return send_from_directory(static, "index.html")

    @app.get("/assets/<path:name>")
    def asset(name: str):
        return send_from_directory(static / "assets", name)

    @app.get("/api/v1/session")
    def session():
        token = secrets.token_urlsafe(24)
        csrf_tokens.add(token)
        response = make_response(jsonify(csrf_token=token))
        response.set_cookie(
            "polima_csrf", token, httponly=True, samesite="Strict", secure=False, max_age=86400,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/v1/snapshot")
    def snapshot():
        return jsonify(engine.snapshot())

    @app.get("/api/v1/bundles")
    def bundles():
        return jsonify(items=engine.bundles())

    @app.get("/api/v1/hardware")
    def hardware():
        return jsonify(engine.hardware())

    @app.post("/api/v1/lease")
    def lease():
        return mutation(
            lambda: engine.claim_lease(
                (request.get_json(silent=True) or {}).get("token"),
                request.headers.get("X-Browser-ID"),
            ),
            lease=False,
        )

    @app.post("/api/v1/bundles/<bundle>/activate")
    def activate(bundle: str):
        return mutation(lambda: engine.activate(bundle))

    @app.post("/api/v1/server/<action>")
    def server(action: str):
        body = request.get_json(silent=True) or {}
        if action == "stop":
            return mutation(engine.stop_control)
        return mutation(lambda: engine.server(action, body.get("bundle")))

    @app.post("/api/v1/preview/start")
    def preview_start():
        body = request.get_json(force=True)

        def start():
            engine.start_preview(
                str(body["bundle"]), str(body["overhead_camera"]), str(body["wrist_camera"]),
            )
            return {"state": engine.state.value}

        return mutation(start)

    @app.post("/api/v1/preview/stop")
    def preview_stop():
        return mutation(lambda: (engine.stop() or {"state": engine.state.value}))

    @app.post("/api/v1/robot/arm")
    def robot_arm():
        config = RunConfig.from_json(request.get_json(force=True))
        return mutation(lambda: engine.arm(config))

    @app.post("/api/v1/robot/start")
    def robot_start():
        body = request.get_json(force=True)
        config = RunConfig.from_json(body.get("config") or {})
        return mutation(lambda: engine.start_robot(config, str(body.get("arming_token", ""))))

    @app.post("/api/v1/robot/stop")
    def robot_stop():
        return mutation(engine.stop_control)

    @app.post("/api/v1/robot/halt")
    def robot_halt():
        return mutation(lambda: (engine.halt() or {"state": engine.state.value}), lease=False)

    @app.post("/api/v1/system/reset")
    def system_reset():
        # lease=False on purpose: this is the way out of a wedged studio, and
        # a stale lease is one of the things it clears. CSRF still applies.
        return mutation(engine.reset, lease=False)

    @app.post("/api/v1/calibration/start")
    def calibration_start():
        body = request.get_json(force=True)
        return mutation(lambda: engine.start_calibration(str(body["bundle"]), str(body["robot_port"])))

    @app.post("/api/v1/calibration/input")
    def calibration_input():
        body = request.get_json(silent=True) or {}
        return mutation(lambda: (engine.calibration_input(str(body.get("value", ""))) or {"ok": True}))

    @app.post("/api/v1/calibration/stop")
    def calibration_stop():
        return mutation(lambda: (engine.stop() or {"state": engine.state.value}))

    @app.post("/api/v1/calibration/restore")
    def calibration_restore():
        return mutation(engine.restore_calibration)

    @app.post("/api/v1/benchmarks/start")
    def benchmark_start():
        body = request.get_json(force=True)
        return mutation(
            lambda: engine.start_benchmark(
                str(body["bundle"]), int(body.get("iterations", 50)), int(body.get("warmup", 3))
            )
        )

    @app.post("/api/v1/benchmarks/stop")
    def benchmark_stop():
        return mutation(lambda: (engine.stop() or {"state": engine.state.value}))

    @app.get("/api/v1/benchmarks")
    def benchmarks():
        return jsonify(items=engine.store.list_benchmarks())

    @app.get("/api/v1/logs/benchmarks/<int:benchmark_id>")
    def benchmark_log(benchmark_id: int):
        return Response(engine.read_benchmark_log(benchmark_id), mimetype="text/plain")

    @app.get("/api/v1/history")
    def history():
        return jsonify(items=engine.store.list_runs())

    # Robot run evidence, written by the robot client for every run (polima.robot.evidence).
    @app.get("/api/v1/robot-runs")
    def robot_runs():
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
        return jsonify(items=evidence.list_runs(engine.root, limit), outcomes=evidence.OUTCOMES,
                       failure_tags=evidence.FAILURE_TAGS)

    @app.get("/api/v1/robot-runs/reviews")
    def robot_run_reviews():
        """Reviewed partial and failed runs (?all=1: every reviewed run), for improvement."""
        everything = request.args.get("all") in ("1", "true")
        return jsonify(items=evidence.export_reviews(engine.root, everything))

    @app.get("/api/v1/robot-runs/improvement-manifest")
    def robot_run_improvement_manifest():
        """Reviewed failed and partial runs (?all=1: every reviewed run) as one manifest."""
        everything = request.args.get("all") in ("1", "true")
        return jsonify(evidence.improvement_manifest(engine.root, everything, socket.gethostname()))

    @app.get("/api/v1/robot-runs/<run_id>")
    def robot_run(run_id: str):
        folder = evidence.run_dir(engine.root, run_id)
        limit = min(max(int(request.args.get("events", 5000)), 0), 100000)
        summary = evidence.summarize(folder)
        return jsonify(summary=summary, annotation=summary["annotation"],
                       events=evidence.read_events(folder, limit))

    @app.post("/api/v1/robot-runs/<run_id>/annotation")
    def robot_run_annotation(run_id: str):
        body = request.get_json(force=True)
        folder = evidence.run_dir(engine.root, run_id)
        return mutation(lambda: evidence.annotate(folder, body), lease=False)

    @app.get("/api/v1/logs/server")
    def server_log():
        return Response(engine.read_log(), mimetype="text/plain")

    @app.get("/api/v1/logs/runs/<int:run_id>")
    def run_log(run_id: int):
        return Response(engine.read_log(run_id), mimetype="text/plain")

    @app.get("/api/v1/events")
    def events():
        def stream():
            for event in engine.events():
                yield "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"

        return Response(stream(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    @app.get("/api/v1/cameras/<role>.mjpg")
    def camera(role: str):
        if role not in {"overhead", "wrist"} or not engine.preview_url:
            return jsonify(error="camera preview is not running"), 404
        candidates = ("overhead", "perspective", "camera1") if role == "overhead" else (
            "wrist", "camera2"
        )
        upstream = None
        for candidate in candidates:
            url = engine.preview_url.rstrip("/") + f"/stream/{candidate}"
            try:
                upstream = urllib.request.urlopen(url, timeout=5)
                break
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise RuntimeError(f"camera proxy failed: {error}") from error
        if upstream is None:
            return jsonify(error=f"{role} stream is unavailable"), 404

        def relay():
            with upstream:
                while chunk := upstream.read(64 * 1024):
                    yield chunk

        return Response(
            relay(), mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # -------------------------------------------- bundles and files over HTTP

    @app.get("/api/v1/storage")
    def storage():
        return jsonify(engine.storage())

    @app.get("/api/v1/uploads")
    def uploads():
        return jsonify(items=engine.uploads.pending())

    @app.post("/api/v1/uploads")
    def upload_create():
        files = request.get_json(force=True).get("files")
        return mutation(lambda: engine.create_upload(files))

    @app.get("/api/v1/uploads/<upload_id>")
    def upload_status(upload_id: str):
        return jsonify(engine.uploads.status(upload_id))

    @app.put("/api/v1/uploads/<upload_id>/chunk")
    def upload_chunk(upload_id: str):
        # The raw body, so MAX_CONTENT_LENGTH stays at 1 MB for the JSON routes.
        args, length = request.args, int(request.headers.get("Content-Length") or 0)
        return mutation(lambda: engine.uploads.receive(
            upload_id, args.get("path", ""), int(args.get("offset", 0)),
            request.environ["wsgi.input"], length,
        ))

    @app.post("/api/v1/uploads/<upload_id>/install")
    def upload_install(upload_id: str):
        activate = bool((request.get_json(silent=True) or {}).get("activate"))
        return mutation(lambda: engine.install_upload(upload_id, activate))

    @app.post("/api/v1/uploads/<upload_id>/cancel")
    def upload_cancel(upload_id: str):
        return mutation(lambda: engine.uploads.discard(upload_id) or {"ok": True})

    @app.get("/api/v1/bundles/<bundle>/details")
    def bundle_details(bundle: str):
        return jsonify(engine.bundle_details(bundle))

    @app.post("/api/v1/bundles/<bundle>/verify")
    def bundle_verify(bundle: str):
        return mutation(lambda: engine.verify_bundle(bundle))

    @app.post("/api/v1/bundles/<bundle>/delete")
    def bundle_delete(bundle: str):
        return mutation(lambda: engine.delete_bundle(bundle))

    @app.get("/api/v1/bundles/<bundle>/calibration")
    def calibration_download(bundle: str):
        path = engine.read_calibration(engine._model_dir(bundle).name)
        return send_file(path, as_attachment=True, download_name=f"{bundle}-so-arm101.json")

    @app.post("/api/v1/bundles/<bundle>/calibration")
    def calibration_upload(bundle: str):
        body = request.get_json(force=True)
        name = engine._model_dir(bundle).name
        return mutation(lambda: engine.set_calibration(
            name, body.get("calibration"), body.get("from_bundle"),
        ))

    def download_root(key: str) -> Path:
        roots = transfer.download_roots(engine.root)
        if key not in roots:
            raise FileNotFoundError(f"unknown download location {key!r}")
        return roots[key]

    @app.get("/api/v1/files")
    def file_roots():
        return jsonify(items=[{"key": key, "path": str(path), "exists": path.is_dir()}
                              for key, path in transfer.download_roots(engine.root).items()])

    @app.get("/api/v1/files/<key>/list")
    def file_list(key: str):
        base = download_root(key)
        entries = transfer.list_directory(base, request.args.get("path", "")) if (
            base.is_dir()
        ) else []
        return jsonify(entries=entries)

    @app.get("/api/v1/files/<key>/download")
    def file_download(key: str):
        target = transfer.confined(download_root(key), request.args.get("path", ""))
        if target.is_file():
            return send_file(target, as_attachment=True)
        if not target.is_dir():
            raise FileNotFoundError("no such file")
        return Response(transfer.stream_tar(target), mimetype="application/x-tar", headers={
            "Content-Disposition": f'attachment; filename="{target.name}.tar"',
        })

    @app.get("/api/v1/health")
    def health():
        return jsonify(ok=True, state=engine.state.value)

    return app
