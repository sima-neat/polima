"""SmolVLA through the same SO101 client path as GR00T (transport and packaging only)."""

import json
import socket
import struct
import threading
from argparse import Namespace

import numpy as np
import pytest

from polima.bundle.import_legacy import collect_robot_files
from polima.bundle.layout import Bundle
from polima.bundle.pack import _plan_dict
from polima.policies.registry import get_policy
from polima.robot.so101 import JOINTS, SO101PolicyClient, client_configuration, load_contract
from polima.util.jsonio import write_json

TASK = "Place the grey eraser in the white basket"


def profile(camera_roles=("overhead", "wrist"), task=TASK):
    selected = {
        "state_names": list(JOINTS),
        "state_dim": 6,
        "action_dim": 6,
        "camera_keys": [f"observation.images.{role}" for role in camera_roles],
        "fps": 30,
        "calibration_id": "so-arm101",
    }
    if task:
        selected["runtime_context"] = {
            "task_catalog": [{"id": 0, "dataset_id": 0, "instruction": task}]
        }
    return selected


def make_bundle(tmp_path, camera_roles=("overhead", "wrist"), task=TASK):
    spec = get_policy("smolvla").specialize_profile(profile=profile(camera_roles, task))
    bundle = Bundle(
        root=tmp_path,
        policy="smolvla",
        bundle_id="smolvla-so101-test",
        tasks=[TASK] if task else [],
    )
    write_json(tmp_path / "plan.json", _plan_dict(spec.build_runtime_plan(), wire=spec))
    write_json(tmp_path / "bundle.json", bundle.to_dict())
    write_json(
        tmp_path / "robot_client/robot_rest_position.json", {"positions": dict.fromkeys(JOINTS, 0)}
    )
    write_json(tmp_path / "robot_client/calibration/so-arm101.json", {})
    return bundle, spec


def test_plan_records_model_camera_order_and_task(tmp_path):
    bundle, _ = make_bundle(tmp_path)
    plan = json.loads(bundle.plan_path.read_text())
    assert plan["profile"]["camera_keys"] == [
        "observation.images.overhead", "observation.images.wrist"
    ]
    assert plan["tasks"] == [TASK]
    loaded_bundle, loaded, task_id = load_contract(tmp_path, "0")
    assert task_id == 0 and loaded_bundle.tasks == [TASK]
    assert loaded.robot.camera_keys == ("overhead", "wrist")
    assert loaded.wire.response_shape == (50, 6)
    assert [t.name for t in loaded.wire.request_tensors] == ["image0", "image1", "state", "noise"]


def test_contract_refuses_wrist_first_or_taskless_bundles(tmp_path):
    make_bundle(tmp_path / "wrist_first", camera_roles=("wrist", "overhead"))
    with pytest.raises(ValueError, match="overhead as image0"):
        load_contract(tmp_path / "wrist_first")
    make_bundle(tmp_path / "taskless", task=None)
    with pytest.raises(ValueError, match="names no task"):
        load_contract(tmp_path / "taskless")
    with pytest.raises(ValueError, match="catalog"):
        load_contract(make_bundle(tmp_path / "ok")[0].root, "1")


def test_transport_sends_overhead_as_image0_and_does_not_renormalize(tmp_path, monkeypatch):
    _bundle, spec = make_bundle(tmp_path)
    monkeypatch.setenv("POLIMA_SO101_BUNDLE", str(tmp_path))
    monkeypatch.setenv("POLIMA_TASK_ID", "0")
    monkeypatch.setenv("POLIMA_GROOT_NOISE", "zero")
    monkeypatch.setattr(
        "polima.robot.policy.prepare_smolvla_image",
        lambda image: np.full((512, 512, 3), float(image[0, 0, 0]), dtype=np.float32),
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    captured = {}
    expected = np.arange(300, dtype="<f4").reshape(50, 6)

    def receive_exact(stream, count):
        result = bytearray()
        while len(result) < count:
            part = stream.recv(count - len(result))
            if not part:
                raise RuntimeError("incomplete request")
            result.extend(part)
        return bytes(result)

    def serve():
        try:
            stream, _ = listener.accept()
            with stream:
                stream.settimeout(5)
                magic, version, request_id, task = struct.unpack("<IIII", receive_exact(stream, 16))
                captured.update(magic=magic, task=task)
                for tensor in spec.wire.request_tensors:
                    captured[tensor.name] = np.frombuffer(
                        receive_exact(stream, tensor.elements * 4), dtype="<f4"
                    ).reshape(tensor.shape)
                stream.sendall(
                    struct.pack("<IIIIfI", magic, version, request_id, 0, 300.0, expected.size)
                    + expected.tobytes()
                )
        except (OSError, RuntimeError, ValueError) as error:
            captured["error"] = error
        finally:
            listener.close()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client = SO101PolicyClient(f"127.0.0.1:{listener.getsockname()[1]}", timeout=5)
    state = np.arange(6, dtype=np.float32)
    overhead = np.full((480, 640, 3), 7, dtype=np.uint8)
    wrist = np.full((480, 640, 3), 3, dtype=np.uint8)
    actions, latency = client.predict(overhead, wrist, state)
    worker.join(6)
    assert "error" not in captured, captured
    assert captured["magic"] == spec.wire.magic and captured["task"] == 0
    assert np.all(captured["image0"] == 7)      # overhead -> the model's camera1
    assert np.all(captured["image1"] == 3)      # wrist -> camera2
    np.testing.assert_array_equal(captured["state"], state)  # raw degrees; server normalizes
    assert captured["noise"].shape == (50, 32) and not captured["noise"].any()
    np.testing.assert_array_equal(actions, expected)  # server already denormalized
    assert latency == 300.0


def test_launcher_uses_the_same_lerobot_configuration_as_groot(tmp_path):
    bundle, spec = make_bundle(tmp_path)
    args = Namespace(
        actions_per_chunk=None, fps=15, robot_port="/dev/arm", front_camera="/dev/c920",
        wrist_camera="/dev/sonix", server_address="127.0.0.1:8081", max_relative_target=12,
    )
    env, argv = client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    assert env["ACT_SOM_DIRECT"] == "1"
    assert f"--task={TASK}" in argv
    assert "--actions_per_chunk=50" in argv and "--fps=15" in argv
    assert "--robot.use_degrees=true" in argv
    cameras = json.loads(next(v.split("=", 1)[1] for v in argv if v.startswith("--robot.cameras=")))
    assert cameras["overhead"]["index_or_path"] == "/dev/c920"
    assert cameras["wrist"]["index_or_path"] == "/dev/sonix"
    assert cameras["overhead"]["fps"] == 30   # capture mode independent of control rate


def test_packaging_reuses_the_groot_so101_client(tmp_path, monkeypatch):
    bundle, spec = make_bundle(tmp_path / "export")
    act = tmp_path / "stacks" / "ACT"
    for file in [
        "run_robot_client_with_live_view.py",
        "preview_robot_cameras.py",
        ".camera_focus_config.json",
        "calibration/robots/so_follower/so-arm101.json",
    ]:
        dest = act / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("existing source")
    monkeypatch.setenv("POLIMA_REPO_ROOT", str(tmp_path / "stacks"))
    groot = collect_robot_files(
        get_policy("groot").specialize_profile(profile={
            **profile(), "actions_per_chunk": 16, "embodiment_tag": "new_embodiment",
            "embodiment_id": 10, "relative_action_dim": 5,
            "image_preprocess": {"format": "groot-eagle-image-v1"},
        }),
        bundle.root,
    )
    smolvla = collect_robot_files(spec, bundle.root)
    assert smolvla == groot
    assert "scripts/smolvla_som_client.py" not in smolvla
    assert "SO101PolicyClient" in smolvla["scripts/act_som_client.py"].read_text()
    with pytest.raises(FileNotFoundError, match="dataset-derived"):
        collect_robot_files(spec)


def write_checkpoint(root, rename, features):
    root.mkdir(parents=True)
    write_json(root / "config.json", {
        "input_features": {feature: {"type": "VISUAL"} for feature in features},
        "output_features": {"action": {"shape": [6]}},
        "max_state_dim": 32,
        "max_action_dim": 32,
    })
    write_json(root / "policy_preprocessor.json", {"steps": [
        {"registry_name": "rename_observations_processor", "config": {"rename_map": rename}},
    ]})


def write_dataset(root, cameras, tasks):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    joint = {"shape": [6], "names": list(JOINTS)}
    write_json(root / "meta/info.json", {
        "fps": 30,
        "features": {
            "action": joint, "observation.state": joint,
            **{key: {"dtype": "video", "shape": [480, 640, 3]} for key in cameras},
        },
    })
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        root / "meta/tasks.parquet"
    )


def test_checkpoint_profile_orders_cameras_like_the_model_not_the_dataset(tmp_path):
    """grey_eraser_961 lists wrist first; its checkpoint maps overhead->camera1."""
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint(
        checkpoint,
        {"observation.images.overhead": "observation.images.camera1",
         "observation.images.wrist": "observation.images.camera2"},
        ["observation.state", "observation.images.camera1", "observation.images.camera2",
         "observation.images.camera3", "observation.images.empty_camera_0"],
    )
    dataset = tmp_path / "dataset"
    write_dataset(dataset, ["observation.images.wrist", "observation.images.overhead"], [TASK])
    spec = get_policy("smolvla").specialize_profile(checkpoint=checkpoint, dataset_root=dataset)
    assert spec.dataset.camera_keys == ("observation.images.overhead", "observation.images.wrist")
    assert spec.robot.camera_keys == ("overhead", "wrist")
    assert spec.runtime_context["task_catalog"][0]["instruction"] == TASK

    several = tmp_path / "several"
    write_dataset(several, ["observation.images.wrist", "observation.images.overhead"],
                  [TASK, "another task"])
    spec = get_policy("smolvla").specialize_profile(checkpoint=checkpoint, dataset_root=several)
    # Which task the exporter's first sample embeds is not knowable here: name none.
    assert "task_catalog" not in spec.runtime_context
