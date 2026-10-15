"""Wire protocol.

The critical test is byte-exactness: the C++ server on the DevKit is compiled
against this framing, so the spec-driven packer must produce identical bytes.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from polima.policies.act import ACT_SPEC
from polima.wire.protocol import Protocol, ProtocolError, coerce_image

# The framing the C++ server expects.
WIRE_MAGIC = 0x4D544341
WIRE_VERSION = 1
WIRE_HEADER = struct.Struct("<IIII")
WIRE_RESPONSE = struct.Struct("<IIIIfI")


@pytest.fixture
def protocol():
    return Protocol(ACT_SPEC.wire)


def test_spec_constants_match_the_wire_format():
    assert ACT_SPEC.wire.magic == WIRE_MAGIC
    assert ACT_SPEC.wire.version == WIRE_VERSION
    assert ACT_SPEC.wire.magic_ascii == "ACTM"


def test_header_layouts_match(protocol):
    assert protocol.request_header.size == WIRE_HEADER.size == 16
    assert protocol.response_header.size == WIRE_RESPONSE.size == 24


def test_packed_request_is_byte_identical_to_act_som_client(protocol):
    """Header, then each tensor in spec order, as little-endian float32."""
    rng = np.random.default_rng(0)
    image0 = rng.standard_normal((480, 640, 3)).astype(np.float32)
    image1 = rng.standard_normal((480, 640, 3)).astype(np.float32)
    state = rng.standard_normal(6).astype(np.float32)
    request_id = 12345

    expected = WIRE_HEADER.pack(WIRE_MAGIC, WIRE_VERSION, request_id, 0)
    expected += (
        np.ascontiguousarray(image0, dtype="<f4").tobytes()
        + np.ascontiguousarray(image1, dtype="<f4").tobytes()
        + np.ascontiguousarray(state, dtype="<f4").tobytes()
    )

    ours = protocol.pack_request(
        request_id, {"image0": image0, "image1": image1, "state": state}
    )
    assert ours == expected
    # 16-byte header + 2 images + 6 joints
    assert len(ours) == 16 + 2 * 480 * 640 * 3 * 4 + 6 * 4 == protocol.request_size()


def test_response_parsing_matches_the_wire_format(protocol):
    actions = np.arange(600, dtype=np.float32).reshape(100, 6)
    raw = protocol.pack_response(7, actions, latency_ms=12.5)

    magic, version, request_id, status, latency, count = WIRE_RESPONSE.unpack(raw[:24])
    assert (magic, version, request_id, status, count) == (
        WIRE_MAGIC, WIRE_VERSION, 7, 0, 600
    )
    assert latency == pytest.approx(12.5)

    header = protocol.parse_response_header(raw[:24])
    protocol.check_response(header, 7)
    result = protocol.parse_result(raw[24:])
    assert result.shape == (100, 6)
    np.testing.assert_array_equal(result, actions)


def test_round_trip(protocol):
    rng = np.random.default_rng(1)
    tensors = {
        "image0": rng.standard_normal((480, 640, 3)).astype(np.float32),
        "image1": rng.standard_normal((480, 640, 3)).astype(np.float32),
        "state": rng.standard_normal(6).astype(np.float32),
    }
    request_id, recovered = protocol.unpack_request(protocol.pack_request(99, tensors))
    assert request_id == 99
    for name, array in tensors.items():
        np.testing.assert_array_equal(recovered[name], array)


def test_wrong_element_count_is_rejected(protocol):
    with pytest.raises(ProtocolError, match="expected 6 elements"):
        protocol.pack_request(1, {
            "image0": np.zeros((480, 640, 3), np.float32),
            "image1": np.zeros((480, 640, 3), np.float32),
            "state": np.zeros(7, np.float32),
        })


def test_missing_tensor_is_rejected(protocol):
    with pytest.raises(ProtocolError, match="missing request tensor"):
        protocol.pack_request(1, {"image0": np.zeros((480, 640, 3), np.float32)})


@pytest.mark.parametrize("bad_id", [8, 0])
def test_mismatched_request_id_is_rejected(protocol, bad_id):
    raw = protocol.pack_response(7, np.zeros((100, 6), np.float32))
    header = protocol.parse_response_header(raw[:24])
    with pytest.raises(ProtocolError, match="response is for request"):
        protocol.check_response(header, bad_id)


def test_nonzero_status_is_rejected(protocol):
    raw = protocol.pack_response(7, np.zeros((100, 6), np.float32), status=1)
    header = protocol.parse_response_header(raw[:24])
    with pytest.raises(ProtocolError, match="status=1"):
        protocol.check_response(header, 7)


def test_wrong_element_count_from_server_is_rejected(protocol):
    raw = protocol.pack_response(7, np.zeros((50, 6), np.float32))
    header = protocol.parse_response_header(raw[:24])
    with pytest.raises(ProtocolError, match="returned 300 elements, expected 600"):
        protocol.check_response(header, 7)


def test_coerce_image_accepts_chw_and_hwc():
    chw = np.zeros((3, 480, 640), np.float32)
    assert coerce_image(chw, (480, 640, 3)).shape == (480, 640, 3)
    hwc = np.zeros((480, 640, 3), np.float32)
    assert coerce_image(hwc, (480, 640, 3)).shape == (480, 640, 3)


def test_coerce_image_scales_uint8_range():
    """The `>1.0 => /255` heuristic."""
    image = np.full((480, 640, 3), 255, np.uint8)
    assert coerce_image(image, (480, 640, 3)).max() == pytest.approx(1.0)
    small = np.full((480, 640, 3), 0.5, np.float32)
    assert coerce_image(small, (480, 640, 3)).max() == pytest.approx(0.5)
