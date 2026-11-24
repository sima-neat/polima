#!/usr/bin/env python3
"""Start the installed web applications and verify their public health endpoints."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def wait(url: str, process: subprocess.Popen[str]) -> bytes:
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f"process exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return response.read()
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(url)


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
    if process.poll() is None:
        process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="polima-web-smoke-", dir=args.output.parent) as temporary:
        root = Path(temporary)
        studio_port, workbench_port = free_ports(2)
        commands = [
            ([args.python, "-m", "polima", "studio", "serve", "--host", "127.0.0.1",
              "--port", str(studio_port), "--root", str(root / "studio")],
             f"http://127.0.0.1:{studio_port}/api/v1/health"),
            ([args.python, "-m", "polima", "workbench", "serve", "--host", "127.0.0.1",
              "--port", str(workbench_port)], f"http://127.0.0.1:{workbench_port}/"),
        ]
        environment = {**os.environ, "POLIMA_WORKBENCH_HOME": str(root / "workbench")}
        results = {}
        for command, url in commands:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, env=environment)
            try:
                body = wait(url, process)
                if url.endswith("/health"):
                    json.loads(body)
                results[url] = "passed"
            finally:
                stop(process)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"web smoke passed with {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
