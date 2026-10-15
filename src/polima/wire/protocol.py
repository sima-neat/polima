"""The DevKit wire protocol, derived from WireSpec instead of hardcoded.

The packers come from the spec, so one Protocol serves every policy and the
C++ side reads the same description out of bundle.json. Byte-exactness is
pinned by tests/unit/test_wire.py against captured bytes.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from polima.policies.base import WireSpec

#: Little-endian float32 -- what both the C++ server and the .f32 fixtures use.
DTYPE = "<f4"


@dataclass(frozen=True)
class ResponseHeader:
    magic: int
    version: int
    request_id: int
    status: int
    latency_ms: float
    count: int

    @property
    def ok(self) -> bool:
        return self.status == 0


class ProtocolError(RuntimeError):
    pass


class Protocol:
    """Pack requests and parse responses for one policy's WireSpec."""

    def __init__(self, spec: WireSpec) -> None:
        self.spec = spec
        self.request_header = struct.Struct(spec.request_header)
        self.response_header = struct.Struct(spec.response_header)

    @property
    def response_size(self) -> int:
        return self.response_header.size

    @property
    def expected_elements(self) -> int:
        return self.spec.response_elements

    def pack_request(
        self,
        request_id: int,
        tensors: Mapping[str, np.ndarray],
        *,
        flags: int = 0,
    ) -> bytes:
        """Header followed by each declared tensor, in spec order.

        Tensors must already be normalized when
        `WireSpec.normalization_side == "client"` -- PolimaSOMClient does that;
        this function only frames bytes.
        """
        missing = [t.name for t in self.spec.request_tensors if t.name not in tensors]
        if missing:
            raise ProtocolError(f"missing request tensor(s): {missing}")

        payload = bytearray(
            self.request_header.pack(
                self.spec.magic, self.spec.version, request_id & 0xFFFFFFFF, flags
            )
        )
        for tensor in self.spec.request_tensors:
            array = np.asarray(tensors[tensor.name], dtype=np.float32)
            if array.size != tensor.elements:
                raise ProtocolError(
                    f"{tensor.name}: expected {tensor.elements} elements "
                    f"{tensor.shape}, got {array.size} {array.shape}"
                )
            payload += np.ascontiguousarray(array.reshape(-1), dtype=DTYPE).tobytes()
        return bytes(payload)

    def parse_response_header(self, raw: bytes) -> ResponseHeader:
        if len(raw) != self.response_header.size:
            raise ProtocolError(
                f"response header is {len(raw)} bytes, expected {self.response_header.size}"
            )
        magic, version, request_id, status, latency_ms, count = self.response_header.unpack(raw)
        return ResponseHeader(magic, version, request_id, status, float(latency_ms), count)

    def check_response(self, header: ResponseHeader, request_id: int) -> None:
        """Check a response header: magic, version, request id, status and count."""
        if (header.magic, header.version) != (self.spec.magic, self.spec.version):
            raise ProtocolError(
                f"protocol mismatch: got magic={header.magic:#x} version={header.version}, "
                f"expected {self.spec.magic:#x}/{self.spec.version}"
            )
        if header.request_id != (request_id & 0xFFFFFFFF):
            raise ProtocolError(
                f"response is for request {header.request_id}, expected {request_id}"
            )
        if not header.ok:
            raise ProtocolError(f"server reported status={header.status}")
        if self.expected_elements and header.count != self.expected_elements:
            raise ProtocolError(
                f"server returned {header.count} elements, expected {self.expected_elements}"
            )

    def parse_result(self, raw: bytes) -> np.ndarray:
        array = np.frombuffer(raw, dtype=DTYPE)
        if self.spec.response_shape:
            array = array.reshape(self.spec.response_shape)
        return array.astype(np.float32)

    # Used by the pure-python reference server and by tests.
    def pack_response(
        self, request_id: int, result: np.ndarray, *, latency_ms: float = 0.0, status: int = 0
    ) -> bytes:
        flat = np.ascontiguousarray(np.asarray(result, dtype=np.float32).reshape(-1), dtype=DTYPE)
        header = self.response_header.pack(
            self.spec.magic, self.spec.version, request_id & 0xFFFFFFFF,
            status, float(latency_ms), flat.size,
        )
        return header + flat.tobytes()

    def unpack_request(self, raw: bytes) -> tuple[int, dict[str, np.ndarray]]:
        """Inverse of pack_request, for the reference server and round-trip tests."""
        header = self.request_header.unpack(raw[: self.request_header.size])
        magic, version, request_id, _flags = header
        if (magic, version) != (self.spec.magic, self.spec.version):
            raise ProtocolError(f"protocol mismatch: {magic:#x}/{version}")
        cursor = self.request_header.size
        tensors: dict[str, np.ndarray] = {}
        for tensor in self.spec.request_tensors:
            width = tensor.elements * 4
            chunk = raw[cursor: cursor + width]
            if len(chunk) != width:
                raise ProtocolError(f"{tensor.name}: truncated payload")
            tensors[tensor.name] = np.frombuffer(chunk, dtype=DTYPE).reshape(tensor.shape)
            cursor += width
        return request_id, tensors

    def request_size(self) -> int:
        return self.request_header.size + sum(t.elements * 4 for t in self.spec.request_tensors)


def coerce_image(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Accept CHW or HWC, return HWC float32 in [0, 1].

    An image whose maximum exceeds 1.0 is taken as 0..255 and divided by 255.
    """
    array = np.asarray(image)
    height, width, channels = shape
    if array.shape == (channels, height, width):
        array = array.transpose(1, 2, 0)
    if array.shape != (height, width, channels):
        raise ProtocolError(f"expected a {height}x{width} image with {channels} channels, "
                            f"got {array.shape}")
    array = array.astype(np.float32)
    if array.max(initial=0) > 1.0:
        array = array / 255.0
    return array
