from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from polima.policies.groot import GROOT_SPEC
from polima.robot import policy
from polima.wire.protocol import Protocol

JOINTS = list(GROOT_SPEC.robot.joint_names)
CONTEXT = {
    "image_shape": [252, 336],
    "image_preprocess": {
        "format": "groot-eagle-image-v1", "color_space": "RGB",
        "eagle": {"image_mean": [0.5] * 3, "image_std": [0.5] * 3,
                  "rescale_factor": 1 / 255, "resize_resample": "bicubic",
                  "resize_to_compiled_shape": True},
        "gr00t_eval": {"use_albumentations": True, "shortest_image_edge": 256,
                       "crop_fraction": 0.95, "letter_box_transform": False},
    },
}
PLAN = {
    "wire": {"magic": 1414877767, "version": 2, "default_port": 8093,
             "request_tensors": [{"name": "patches0", "shape": [432, 588]},
                                 {"name": "patches1", "shape": [432, 588]},
                                 {"name": "state", "shape": [6]},
                                 {"name": "noise", "shape": [50, 128]}],
             "response_shape": [16, 6]},
    "robot": {"camera_roles": [["wrist", "Wrist"], ["overhead", "Overhead"]],
              "joint_names": JOINTS},
    "runtime_context": CONTEXT,
    "tasks": ["Place the eraser in the basket", "Stack the blocks"], "default_task": 0,
}


@pytest.fixture
def groot_bundle(tmp_path):
    (tmp_path / "bundle.json").write_text(json.dumps(
        {"bundle_id": "groot-x", "policy": "groot", "graphs": []}))
    (tmp_path / "plan.json").write_text(json.dumps(PLAN))
    return tmp_path


def test_task_from_studio_text_index_or_default():
    tasks = ["pick", "place"]
    assert policy.task_index("place", tasks) == 1
    assert policy.task_index("1", tasks) == 1
    assert policy.task_index("", tasks, default=1) == 1
    with pytest.raises(ValueError, match="knows"):
        policy.task_index("dance", tasks)


def test_groot_follows_the_bundle_not_the_registry(groot_bundle):
    bundle = policy.BundlePolicy(groot_bundle, "127.0.0.1", "Stack the blocks")
    assert bundle.camera_keys == ["wrist", "overhead"]  # the registry lists overhead first
    assert bundle.spec.wire.version == 2 and bundle.spec.wire.response_shape == (16, 6)
    assert bundle.client.port == 8093 and bundle.task_id == 1


def test_groot_request_carries_patches_state_noise_and_task(groot_bundle, monkeypatch):
    bundle = policy.BundlePolicy(groot_bundle, "127.0.0.1", "Stack the blocks",
                                 rng=np.random.default_rng(0))
    sent = {}
    monkeypatch.setattr(bundle.client, "_send", lambda payload: (
        sent.setdefault("payload", payload), (np.zeros((16, 6), np.float32), 9.0))[1])
    frame = np.full((480, 640, 3), 128, np.uint8)
    observation = {**dict(zip(JOINTS, range(6))), "wrist": frame, "overhead": frame}

    actions, model_ms = bundle.predict(observation)

    payload = sent["payload"]
    magic, version, _request, task = struct.unpack_from("<IIII", payload)
    assert (magic, version, task) == (1414877767, 2, 1)
    protocol = Protocol(bundle.spec.wire)
    assert len(payload) == protocol.request_header.size + 4 * sum(
        t.elements for t in bundle.spec.wire.request_tensors)
    state = np.frombuffer(payload, "<f4", 6, protocol.request_header.size + 4 * 2 * 432 * 588)
    assert state.tolist() == list(range(6))  # raw joints: GR00T normalizes on the board
    assert actions.shape == (16, 6) and model_ms == 9.0


def test_groot_patches_are_normalized_14px_tiles():
    patches = policy.groot_patches(np.full((480, 640, 3), 255, np.uint8), CONTEXT)
    assert patches.shape == (18 * 24, 14 * 14 * 3) and patches.dtype == np.float32
    assert np.allclose(patches, 1.0)  # white, rescaled to 1, then (1 - 0.5) / 0.5


def test_an_eagle_contract_is_required():
    with pytest.raises(ValueError, match="Eagle"):
        policy.groot_patches(np.zeros((480, 640, 3), np.uint8), {"image_preprocess": {}})
