from __future__ import annotations

import json

import pytest

from polima.policies.groot import GROOT_SPEC
from polima.robot import policy

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
