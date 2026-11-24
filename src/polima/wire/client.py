"""One client for every policy's DevKit server.

Where normalization happens is `WireSpec.normalization_side`: ACT normalizes on
the client from normalization_stats.npz, while GR00T normalizes on the server.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polima.policies.base import PolicySpec
from polima.wire.protocol import DTYPE, Protocol, ProtocolError, coerce_frame, coerce_image


@dataclass
class Normalization:
    """Per-checkpoint mean/std, read from normalization_stats.npz.

    Written at export from the checkpoint's policy_preprocessor.json /
    policy_postprocessor.json.
    """

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    camera_mean: list[np.ndarray]
    camera_std: list[np.ndarray]

    @classmethod
    def load(cls, path: str | Path, cameras: int = 2) -> Normalization:
        values = np.load(path)
        return cls(
            state_mean=values["state_mean"].astype(np.float32),
            state_std=values["state_std"].astype(np.float32),
            action_mean=values["action_mean"].astype(np.float32),
            action_std=values["action_std"].astype(np.float32),
            # Reshaped to (1, 1, 3) so they broadcast over an HWC image.
            camera_mean=[
                values[f"camera{i}_mean"].reshape(1, 1, 3).astype(np.float32)
                for i in range(cameras)
            ],
            camera_std=[
                values[f"camera{i}_std"].reshape(1, 1, 3).astype(np.float32)
                for i in range(cameras)
            ],
        )


class PolimaSOMClient:
    """Talk to a polima_server."""

    def __init__(
        self,
        spec: PolicySpec,
        address: str,
        *,
        stats: str | Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.spec = spec
        self.protocol = Protocol(spec.wire)
        self.host, _, port = address.rpartition(":")
        if not self.host:
            self.host, port = address, str(spec.wire.default_port)
        self.port = int(port)
        self.timeout = timeout
        self.normalization: Normalization | None = None
        if stats is not None and spec.wire.normalization_side == "client":
            self.normalization = Normalization.load(stats, len(spec.dataset.camera_keys))
        # Seed from the clock so concurrent clients are unlikely to collide on
        # request ids.
        self.request_id = int(time.time()) & 0xFFFFFFFF

    # ------------------------------------------------------------- transport

    def _send(self, payload: bytes) -> tuple[np.ndarray, float]:
        trace = getattr(self, "trace_event", None)
        if trace is not None:
            trace("request_started", request_id=self.request_id)
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as stream:
            stream.sendall(payload)
            if trace is not None:
                trace("request_sent", request_id=self.request_id)
            header = self.protocol.parse_response_header(
                _recv_exact(stream, self.protocol.response_size)
            )
            self.protocol.check_response(header, self.request_id)
            raw = _recv_exact(stream, header.count * 4)
            if trace is not None:
                trace("reply_received", request_id=self.request_id, server_compute_ms=float(header.latency_ms))
        return self.protocol.parse_result(raw), header.latency_ms

    # ---------------------------------------------------------------- public

    def predict_raw(
        self, tensors: dict[str, np.ndarray], *, task_id: int = 0
    ) -> tuple[np.ndarray, float]:
        """Send already-prepared tensors. No normalization, no image coercion.

        This is what `polima run --fixture` uses: the golden .f32 files are
        already normalized, so touching them would invalidate the comparison.
        """
        self.request_id = (self.request_id + 1) & 0xFFFFFFFF
        return self._send(
            self.protocol.pack_request(self.request_id, tensors, flags=int(task_id))
        )

    def predict(
        self, images: list[np.ndarray], state: np.ndarray, *, task_id: int = 0
    ) -> tuple[np.ndarray, float]:
        """Full path from raw camera frames and joint positions to actions.

        Images may be HWC or CHW, uint8 or float; state is the raw joint vector.
        Returns de-normalized actions in joint units.
        """
        tensors = self.prepare_inputs(images, state)
        actions, latency_ms = self.predict_raw(tensors, task_id=task_id)
        return self.decode_actions(actions), latency_ms

    def prepare_inputs(
        self, images: list[np.ndarray], state: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Prepare exactly the tensors that :meth:`predict` sends.

        This boundary lets diagnostics persist a replayable request without
        duplicating normalization or image coercion.
        """

        tensors: dict[str, np.ndarray] = {}
        image_specs = [t for t in self.spec.wire.request_tensors if t.name.startswith("image")]
        if len(images) != len(image_specs):
            raise ProtocolError(
                f"expected {len(image_specs)} images, got {len(images)}"
            )

        for index, (tensor, image) in enumerate(zip(image_specs, images)):
            if tensor.dtype == "uint8":
                # Raw frame: the server normalizes it (WireSpec decode in plan.json).
                tensors[tensor.name] = coerce_frame(image, tensor.shape)  # type: ignore[arg-type]
                continue
            prepared = coerce_image(image, tensor.shape)  # type: ignore[arg-type]
            if self.normalization is not None:
                prepared = (
                    prepared - self.normalization.camera_mean[index]
                ) / self.normalization.camera_std[index]
            tensors[tensor.name] = np.ascontiguousarray(prepared, dtype=DTYPE)

        flat_state = np.asarray(state, dtype=np.float32).reshape(-1)
        if flat_state.size != self.spec.dataset.state_dim:
            raise ProtocolError(
                f"expected {self.spec.dataset.state_dim} joints, got {flat_state.size}"
            )
        if self.normalization is not None:
            flat_state = (
                flat_state - self.normalization.state_mean
            ) / self.normalization.state_std
        tensors["state"] = np.ascontiguousarray(flat_state, dtype=DTYPE)
        return tensors

    def decode_actions(self, actions: np.ndarray) -> np.ndarray:
        """Convert a raw server response to robot-unit actions."""

        if self.normalization is not None:
            actions = actions * self.normalization.action_std + self.normalization.action_mean
        return actions.astype(np.float32)

    def ping(self, timeout: float = 2.0) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout):
                return True
        except OSError:
            return False

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


def _recv_exact(stream: socket.socket, count: int) -> bytes:
    """Read exactly `count` bytes; a short read would silently truncate an
    action chunk."""
    chunks = bytearray()
    while len(chunks) < count:
        part = stream.recv(count - len(chunks))
        if not part:
            raise ProtocolError(
                f"server closed the connection after {len(chunks)} of {count} bytes"
            )
        chunks.extend(part)
    return bytes(chunks)


def wait_for_port(host: str, port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """Poll until the server accepts connections.

    A real socket connect can distinguish "refused" from "no route" and needs no
    subshell per probe.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=interval):
                return True
        except OSError:
            time.sleep(interval)
    return False
