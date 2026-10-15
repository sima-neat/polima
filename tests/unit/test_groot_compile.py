"""`polima compile --policy groot`: robot profile for any layout, the bound-chain DiT plan
rewrite, compile flags, CLI routing. No GR00T environment or compiler needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polima.cli.compile import _groot_build
from polima.compile import groot
from polima.policies.groot import dit
from polima.policies.groot.profile import checkpoint_metadata


def _checkpoint(root: Path, keys, widths, reps, video=("front", "wrist"), model_type="Gr00tN1d7") -> Path:
    ckpt = root / "checkpoint-30000"
    ckpt.mkdir(parents=True)
    modality = {"video": {"delta_indices": [0], "modality_keys": list(video)},
                "state": {"delta_indices": [0], "modality_keys": list(keys)},
                "action": {"delta_indices": list(range(16)), "modality_keys": list(keys),
                           "action_configs": [{"rep": r} for r in reps]}}
    (ckpt / "processor_config.json").write_text(json.dumps(
        {"processor_kwargs": {"modality_configs": {"new_embodiment": modality}}}))
    stats = {s: {k: {"mean": [0.0] * w} for k, w in zip(keys, widths)} for s in ("state", "action")}
    (ckpt / "statistics.json").write_text(json.dumps({"new_embodiment": stats}))
    (ckpt / "embodiment_id.json").write_text(json.dumps({"new_embodiment": 10}))
    (ckpt / "config.json").write_text(json.dumps({"model_type": model_type, "action_horizon": 40}))
    return ckpt


def _dataset(root: Path, names, ranges, cameras, fps=15, robot="franka") -> Path:
    ds = root / "dataset"
    (ds / "meta").mkdir(parents=True)
    features = {"observation.state": {"shape": [len(names)], "names": list(names)}}
    features.update({f"observation.images.{c}": {"shape": [480, 640, 3]} for c in sorted(cameras.values())})
    (ds / "meta/info.json").write_text(json.dumps(
        {"codebase_version": "v2.1", "fps": fps, "robot_type": robot, "features": features}))
    modality = {s: {k: {"start": a, "end": b} for k, (a, b) in ranges.items()} for s in ("state", "action")}
    modality["video"] = {k: {"original_key": f"observation.images.{v}"} for k, v in cameras.items()}
    (ds / "meta/modality.json").write_text(json.dumps(modality))
    return ds


def test_any_robot_profile_comes_from_checkpoint_and_dataset(tmp_path):
    names = [f"joint_{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
    ckpt = _checkpoint(tmp_path, ("arm", "gripper"), (7, 1), ("RELATIVE", "ABSOLUTE"))
    ds = _dataset(tmp_path, names, {"arm": (0, 7), "gripper": (7, 8)}, {"front": "scene", "wrist": "hand"})
    meta = checkpoint_metadata(ckpt, ds, image_preprocess=False)
    assert meta["state_names"] == names and meta["state_dim"] == 8
    assert meta["calibration_id"] == "franka" and meta["fps"] == 15
    # Camera order is the model's video-key order, not the dataset's feature order.
    assert meta["camera_keys"] == ["observation.images.scene", "observation.images.hand"]
    assert meta["relative_action_dim"] == 7 and meta["action_reps"] == ["relative", "absolute"]
    assert meta["group_widths"] == [7, 1] and meta["actions_per_chunk"] == 16


def test_any_robot_needs_its_dataset(tmp_path):
    ckpt = _checkpoint(tmp_path, ("arm", "gripper"), (7, 1), ("RELATIVE", "ABSOLUTE"))
    with pytest.raises(ValueError, match="derived from its dataset"):
        checkpoint_metadata(ckpt, image_preprocess=False)


def test_so101_profile_is_unchanged(tmp_path):
    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
             "gripper.pos"]
    ckpt = _checkpoint(tmp_path, ("single_arm", "gripper"), (5, 1), ("RELATIVE", "ABSOLUTE"))
    ds = _dataset(tmp_path, names, {"single_arm": (0, 5), "gripper": (5, 6)},
                  {"front": "overhead", "wrist": "wrist"}, fps=30, robot="so_follower")
    meta = checkpoint_metadata(ckpt, ds, image_preprocess=False)
    assert meta["calibration_id"] == "so-arm101" and meta["state_names"] == names
    assert meta["camera_keys"] == ["observation.images.overhead", "observation.images.wrist"]


def _n16_plan():
    steps, buffers = [], {"backbone_features": 1, "text_mask": 1, "image_mask": 1}
    for k in range(4):
        hidden = f"hidden_{k}"
        for j, name in enumerate(dit.names("n1.6")["pairs"]):
            steps.append({"op": "run_elf", "out": f"h_{k}_{j}", "args": {"graph": name, "in": [
                hidden if j == 0 else f"h_{k}_{j - 1}", f"temb_{k}", "backbone_features",
                "text_mask" if j % 2 == 0 else "image_mask"]}})
        steps.append({"op": "run_elf", "out": f"v_{k}", "args": {"graph": "action_tail", "in": [f"h_{k}_15", f"temb_{k}"]}})
    return {"buffers": buffers, "steps": steps, "sidecars": ["temb_0"], "result": "v_3"}


def _manifest(family):
    n = dit.names(family)
    return {"graphs": [{"name": g, "kind": "kv"} for g in n["kv"]] + [
        {"name": g, "kind": "blocks", "ada_input": f"ada_{4 * i:02d}_{4 * i + 3:02d}", "ada_offset": 4 * i * 3072,
         "ada_count": 4 * 3072} for i, g in enumerate(n["blocks"])]}


def test_n16_plan_runs_one_bound_chain_per_step_with_host_adaln():
    plan = dit.bound_plan(_n16_plan(), "n1.6", _manifest("n1.6"))
    chains = [s for s in plan["steps"] if s["op"] == "run_bound_chain"]
    assert len(chains) == 4 and not any(s["args"].get("graph", "").startswith("dit_blocks") for s in plan["steps"])
    first = chains[0]["args"]
    assert first["prelude"] == ["dit_kv_00_14", "dit_kv_16_30"] and len(first["graphs"]) == 8
    assert first["bind"][:4] == ["hidden", "backbone_features", "text_additive_mask", "image_additive_mask"]
    assert first["in"][4:] == [f"ada_{4 * i:02d}_{4 * i + 3:02d}_s0" for i in range(8)]
    assert chains[3]["out"] == "h_3_15" and plan["sidecars"][-4:] == [f"ada_embedding_{k}" for k in range(4)]
    packs = [s for s in plan["steps"] if s["out"].endswith("_s2")]
    assert packs[1]["args"]["parts"][0] == {"count": 12288, "dst_offset": 0, "src_offset": 12288,
                                            "sidecar": True, "src": "ada_embedding_2"}


def test_n17_plan_keeps_the_two_mask_packs_before_the_adaln_packs():
    steps = []
    for k in range(4):
        for j, name in enumerate(dit.names("n1.7")["pairs"]):
            mask = f"mask_{k}_{name}"
            steps.append({"op": "pack", "out": mask, "args": {"parts": [{"src": mask, "sidecar": True}]}})
            steps.append({"op": "run_elf", "out": f"{name}_{k}", "args": {"graph": name, "in": [
                f"hidden_{k}" if j == 0 else f"{dit.names('n1.7')['pairs'][j - 1]}_{k}", f"temb_{k}", "vl_03",
                mask]}})
    plan = dit.bound_plan({"buffers": {}, "steps": steps, "sidecars": [], "result": "x"}, "n1.7", _manifest("n1.7"))
    ops = [(s["op"], s["out"]) for s in plan["steps"][:11]]
    assert ops[:2] == [("pack", "mask_0_dit_00_01"), ("pack", "mask_0_dit_02_03")]
    assert ops[2][1] == "ada_00_03_s0" and ops[10][0] == "run_bound_chain"
    assert plan["steps"][10]["args"]["bind"][:4] == ["hidden", "backbone", "mask_even", "mask_odd"]


def test_compile_flags_follow_the_established_builds(tmp_path):
    build = tmp_path / "b"
    (build / "export").mkdir(parents=True)
    (build / "dit").mkdir()
    names = ["eagle_vision_patch", "eagle_vision_00_01", "eagle_vision_connector", "eagle_qwen_00_01",
             "eagle_output_norm", "dit_blocks_00_01", "action_tail"]
    (build / "export/input_contract.json").write_text(json.dumps({"graphs": [f"{n}.onnx" for n in names]}))
    (build / "dit/manifest.json").write_text(json.dumps({"graphs": [
        {"name": "dit_kv_00_14", "layout": "NHWC", "outputs": ["k_00"]},
        {"name": "ditkv_blocks_00_03", "layout": "NCHW", "outputs": ["hidden_out"]}]}))
    (build / "build.json").write_text(json.dumps({"family": "n1.6", "precision": "mixed"}))
    jobs = {j["name"]: j for j in groot.graph_jobs(groot.Build(build))}
    assert "dit_blocks_00_01" not in jobs  # replaced by the K/V + 4-block graphs
    assert jobs["eagle_vision_00_01"]["flags"] == ["--external-dram-layout", "HWC", "--promote-rank3-hwc"]
    assert jobs["eagle_vision_00_01"]["weights"] == "int8" and jobs["eagle_vision_patch"]["weights"] == "bf16"
    assert jobs["eagle_vision_connector"]["flags"] == ["--mla-tessellation"]
    assert jobs["eagle_qwen_00_01"]["flags"] == ["--external-dram-layout", "HWC"]
    assert jobs["ditkv_blocks_00_03"]["persistent"] and jobs["ditkv_blocks_00_03"]["layout"] == "NCHW"
    assert jobs["dit_kv_00_14"]["calibration"].name == "dit_kv_00_14.npz"


def test_mla_output_names_peel_only_when_asked():
    from polima.compile.tensor import _mla_output_names

    class Node:
        def __init__(self, name, inputs):
            self.name, self.input_node_names = name, inputs

    nodes = {"out/Reshape": Node("out/Reshape", ["mm"]), "mm": Node("mm", ["x"])}
    mla = type("M", (), {"ir": type("I", (), {"nodes": nodes, "output_node_name": "out/Reshape"})()})()
    assert _mla_output_names(mla, lambda n: False) == ["out/Reshape"]
    assert _mla_output_names(mla, lambda n: False, peel_terminal_reshape=True) == ["mm"]


def test_cli_routes_groot_builds_but_not_the_build_dir_paths():
    assert _groot_build(["--policy", "groot", "--checkpoint", "c", "--dataset", "d", "--output", "o"])
    assert not _groot_build(["--policy", "groot", "--build-dir", "x"])
    assert not _groot_build(["--policy", "act", "--output", "o"])


def test_family_comes_from_the_checkpoint(tmp_path):
    assert groot.detect_family(_checkpoint(tmp_path, ("a",), (1,), ("ABSOLUTE",))) == "n1.7"
    (tmp_path / "x").mkdir()
    (tmp_path / "x/config.json").write_text(json.dumps({"model_type": "Other"}))
    with pytest.raises(groot.StageError):
        groot.detect_family(tmp_path / "x")


def test_so101_camera_order_follows_the_model_even_when_the_dataset_lists_wrist_first(tmp_path):
    """The SO-101 dataset lists the wrist camera first; the model's order is front, then wrist."""
    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
             "gripper.pos"]
    ckpt = _checkpoint(tmp_path, ("single_arm", "gripper"), (5, 1), ("RELATIVE", "ABSOLUTE"))
    ds = _dataset(tmp_path, names, {"single_arm": (0, 5), "gripper": (5, 6)},
                  {"front": "overhead", "wrist": "wrist"}, fps=30, robot="so_follower")
    info = json.loads((ds / "meta/info.json").read_text())
    features = info["features"]
    info["features"] = {"observation.images.wrist": features.pop("observation.images.wrist"), **features}
    (ds / "meta/info.json").write_text(json.dumps(info))
    assert next(k for k in info["features"] if k.startswith("observation.images.")).endswith("wrist")
    meta = checkpoint_metadata(ckpt, ds, image_preprocess=False)
    assert meta["camera_keys"] == ["observation.images.overhead", "observation.images.wrist"]


def _stub_build(tmp_path, monkeypatch, check_board, calls):
    """A finished build whose stages only record that they ran."""
    root = tmp_path / "build"
    (root / "stages").mkdir(parents=True)
    for stage in ("export", "dit", "compile", "bundle"):
        (root / "stages" / f"{stage}.done").write_text("")
    settings = {"check_board": check_board, "bundle": str(tmp_path / "bundle"), "precision": "mixed"}
    (root / "build.json").write_text(json.dumps(settings))
    monkeypatch.setattr(groot, "_prepare", lambda args, config: groot.Build(root))
    monkeypatch.setitem(groot.RUNNERS, "check", lambda build: calls.append(build.settings["check_board"]))
    return root


def test_a_skipped_check_stays_to_do_and_a_devkit_given_later_is_checked(tmp_path, monkeypatch):
    calls = []
    root = _stub_build(tmp_path, monkeypatch, None, calls)
    argv = ["--checkpoint", "c", "--dataset", "d", "--output", str(root), "--resume"]
    assert groot.main(argv, config=object()) == 0
    assert not (root / "stages/check.done").exists()
    settings = json.loads((root / "build.json").read_text())
    (root / "build.json").write_text(json.dumps({**settings, "check_board": "bench"}))
    assert groot.main(argv, config=object()) == 0
    assert groot.main(argv, config=object()) == 0  # with a DevKit, every run checks again
    assert calls == [None, "bench", "bench"] and (root / "stages/check.done").exists()


def test_devkit_check_uses_the_saved_devkit_and_the_installed_runtime(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("")
    saved = tmp_path / "devkits.json"
    saved.write_text(json.dumps([{"name": "bench", "host": "sima@192.0.2.7", "key": str(key)}]))
    monkeypatch.setenv("POLIMA_DEVKITS", str(saved))
    (tmp_path / "build.json").write_text(json.dumps({"bundle": str(tmp_path / "b"), "precision": "mixed"}))
    build = groot.Build(tmp_path)
    seen = []

    class Done:
        returncode = 1

    monkeypatch.setattr(groot.subprocess, "run", lambda argv, check: seen.append(argv) or Done())
    with pytest.raises(groot.StageError, match="could not reach sima@192.0.2.7"):
        groot.board_check(build, "bench")
    assert "sima@192.0.2.7" in seen[0] and str(key) in seen[0]
    assert "mkdir -p /media/nvme/polima/compile-check/bundles" in seen[0][-1]

    class Ok:
        returncode = 0

    runs = []
    monkeypatch.setattr(groot.subprocess, "run",
                        lambda argv, check: runs.append(argv) or (Done() if len(runs) == 3 else Ok()))
    with pytest.raises(groot.StageError, match="polima server stop"):
        groot.board_check(build, "bench")
    assert "/media/nvme/polima/bin/polima_cli --bundle" in runs[2][-1]


@pytest.mark.parametrize("local_alias", [False, True])
def test_resume_rejects_changed_backbone(tmp_path, monkeypatch, local_alias):
    from types import SimpleNamespace

    from polima.config.loader import load

    checkpoint = _checkpoint(tmp_path, ('single_arm',), (6,), ('ABSOLUTE',))
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / 'meta/info.json').write_text('{"codebase_version":"v2.1"}')
    args = SimpleNamespace(output=tmp_path / 'build', checkpoint=checkpoint, dataset=dataset,
                           precision='mixed', embodiment_tag=None, jobs=1, check_board=None,
                           label=None, resume=False)
    monkeypatch.setattr(groot, 'interpreters', lambda *a: {'python':'python', 'source':'source', 'sdk':'sdk'})
    backbone = 'org/backbone-a'
    if local_alias:
        assets = tmp_path / 'snapshot-hash'
        assets.mkdir()
        alias = tmp_path / 'nvidia/Cosmos-Reason2-2B'
        alias.parent.mkdir()
        alias.symlink_to(assets, target_is_directory=True)
        backbone = str(alias)
    monkeypatch.setenv('POLIMA_GROOT17_BACKBONE', backbone)
    build = groot._prepare(args, load())
    assert build.settings['backbone'] == backbone
    args.resume = True
    assert groot._prepare(args, load()).settings['backbone'] == backbone
    monkeypatch.setenv('POLIMA_GROOT17_BACKBONE', 'org/backbone-b')
    with pytest.raises(groot.StageError, match='different backbone'):
        groot._prepare(args, load())
    assert groot.Build(args.output).settings['backbone'] == backbone


@pytest.mark.parametrize('recorded', [None, 'org/recorded-backbone'])
def test_worker_backbone_comes_from_recorded_settings(monkeypatch, recorded):
    monkeypatch.setenv('POLIMA_GROOT17_BACKBONE', 'org/ambient-backbone')
    env = groot.worker_env({'backbone':recorded})
    assert env.get('POLIMA_GROOT17_BACKBONE') == recorded


def test_build_records_configured_backbone_over_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from polima.config.loader import load

    checkpoint = _checkpoint(tmp_path, ("single_arm",), (6,), ("ABSOLUTE",))
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{"codebase_version":"v2.1"}')
    args = SimpleNamespace(output=tmp_path / "build", checkpoint=checkpoint, dataset=dataset,
                           precision="mixed", embodiment_tag=None, jobs=1, check_board=None,
                           label=None, resume=False)
    monkeypatch.setattr(groot, "interpreters", lambda *a: {"python":"python", "source":"source", "sdk":"sdk"})
    monkeypatch.setenv("POLIMA_GROOT17_BACKBONE", "ambient/model")
    config = load(overrides={"paths.groot17_backbone": "customer/model"})
    assert groot._prepare(args, config).settings["backbone"] == "customer/model"


@pytest.mark.parametrize("family,tag", [("n1.6", "16"), ("n1.7", "17")])
def test_interpreters_honor_resolved_config_over_environment(tmp_path, monkeypatch, family, tag):
    from polima.config.loader import load

    python = tmp_path / "python"
    python.touch()
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv(f"POLIMA_GROOT{tag}_PYTHON", str(tmp_path / "ambient-python"))
    monkeypatch.setenv(f"POLIMA_GROOT{tag}_SOURCE", str(tmp_path / "ambient-source"))
    monkeypatch.setattr("polima.compile.toolchain.require_compiler_python", lambda config: python)
    config = load(overrides={f"paths.groot{tag}_python": str(python),
                             f"paths.groot{tag}_source": str(source)})
    assert groot.interpreters(config, family) == {
        "python": str(python), "source": str(source), "sdk": str(python)}
