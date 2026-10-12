"""Stream the overhead and wrist cameras to a browser, without moving the arm.

`polima robot preview` runs this through a bundle's
robot_client/preview_robot_cameras.py. It prints `Camera preview server: <url>`
(Studio proxies `<url>/stream/overhead` and `<url>/stream/wrist`) and serves
until it is signalled.
"""

from __future__ import annotations

import argparse
import json
import secrets
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from polima.util.net import lan_address

PAGE = """<!doctype html><meta charset="utf-8"><title>PoLiMa cameras</title>
<style>body{margin:0;padding:16px;background:#111;color:#ccc;font:14px system-ui}
img{width:min(100%,640px);display:block;margin-bottom:12px;background:#000}</style>
<p>Overhead</p><img src="stream/overhead"><p>Wrist</p><img src="stream/wrist">"""


class Feed:
    """One camera, read continuously; the latest frame is kept as a small JPEG."""

    def __init__(self, role: str, source: str, fourcc: str, focus: dict) -> None:
        import cv2

        self.role, self.frame, self.error = role, None, None
        self.ready = threading.Condition()
        self.camera = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.camera.isOpened():
            raise RuntimeError(f"cannot open the {role} camera at {source}")
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        autofocus = bool(focus.get("autofocus", True))
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if not autofocus:
            self.camera.set(cv2.CAP_PROP_FOCUS, int(focus.get("focus", 0)))
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self) -> None:
        import cv2

        next_encode = 0.0
        while not self.stopped.is_set():
            ok, frame = self.camera.read()
            if not ok:
                self.error = f"the {self.role} camera stopped sending frames"
                break
            if time.monotonic() < next_encode:
                continue
            next_encode = time.monotonic() + 0.1
            ok, jpeg = cv2.imencode(".jpg", cv2.resize(frame, (480, 360)),
                                    [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                with self.ready:
                    self.frame = jpeg.tobytes()
                    self.ready.notify_all()
        with self.ready:
            self.ready.notify_all()

    def frames(self):
        last = None
        while not self.stopped.is_set() and self.error is None:
            with self.ready:
                self.ready.wait_for(lambda seen=last: self.frame is not seen or self.error, timeout=2)
                frame = self.frame
            if frame is not None and frame is not last:
                last = frame
                yield frame

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=2)
        self.camera.release()


def serve(feeds: dict[str, Feed], host: str, port: int, token: str) -> ThreadingHTTPServer:
    aliases = {"overhead": "overhead", "perspective": "overhead", "wrist": "wrist"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if not parts or not secrets.compare_digest(parts[0], token):
                return self.send_error(404)
            if len(parts) == 1:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            feed = feeds.get(aliases.get(parts[-1], "")) if parts[1:2] == ["stream"] else None
            if feed is None:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for frame in feed.frames():
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--perspective", required=True, help="overhead camera device")
    parser.add_argument("--wrist", required=True, help="wrist camera device")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--focus-config", type=Path, help="focus saved for each camera")
    parser.add_argument("--view-only", action="store_true", help="accepted; always view-only")
    args = parser.parse_args(argv)

    try:
        focus = json.loads(args.focus_config.read_text()) if args.focus_config else {}
    except (OSError, ValueError):
        focus = {}
    feeds: dict[str, Feed] = {}
    try:
        for role, source, key in (("overhead", args.perspective, "perspective"),
                                  ("wrist", args.wrist, "wrist")):
            feeds[role] = Feed(role, source, args.fourcc, focus.get(key, {}))
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        for feed in feeds.values():
            feed.close()
        return 1

    token = secrets.token_urlsafe(18)
    server = serve(feeds, args.host, args.port, token)
    print(f"Camera preview server: http://{lan_address(args.host)}:{args.port}/{token}/",
          flush=True)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda *_: stop.set())
    try:
        while not stop.wait(0.25):
            failed = next((feed.error for feed in feeds.values() if feed.error), None)
            if failed:
                print(f"Error: {failed}", file=sys.stderr)
                return 1
        return 0
    finally:
        server.shutdown()
        for feed in feeds.values():
            feed.close()


if __name__ == "__main__":
    raise SystemExit(main())
