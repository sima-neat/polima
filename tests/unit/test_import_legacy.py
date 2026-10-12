"""Importing a legacy compiler build tree.

The decoy tests are the point. The real ACT build carries 16 ELFs across 11
directories in `retained/`, six of them abandoned experiments:

    vision_backbone/                    <- shipped
    vision_backbone_rank3/              <- byte-identical to the above
    vision_backbone_rejected_rank4/     <- different, and wrong
    decoder_action_tail/                <- shipped
    decoder_action_tail_v2/             <- byte-identical to the above
    decoder_action_tail_rejected_apu/   <- different, and wrong

Picking the wrong one produces a bundle that deploys cleanly and returns
garbage, which is the worst possible failure mode.
"""

from __future__ import annotations

import json

import pytest

from polima.bundle import retained
from polima.bundle.import_legacy import (
    _declared_graphs,
    LegacyBuild,
    collect_fixtures,
    collect_robot_files,
    collect_sidecars,
    detect,
    hydrate_robot_client,
    hydrate_runtime_metadata,
    resolve_elfs,
)
from polima.bundle.layout import Bundle, GraphArtifact
from polima.policies.act import ACT_SPEC

GRAPHS = list(ACT_SPEC.compile.names)


def make_elf(path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def make_retained(root, layout: dict[str, bytes]):
    """layout: variant_dir -> elf payload. The graph name is the variant's prefix."""
    for variant, payload in layout.items():
        graph = variant
        for candidate in sorted(GRAPHS, key=len, reverse=True):
            if variant.startswith(candidate):
                graph = candidate
                break
        make_elf(root / "retained" / variant / f"{graph}_stage1_mla.elf", payload)
    return root


def test_exact_directory_wins_over_decoys(tmp_path):
    make_retained(tmp_path, {
        "vision_backbone": b"SHIPPED",
        "vision_backbone_rank3": b"SHIPPED",              # identical twin
        "vision_backbone_rejected_rank4": b"WRONG",
    })
    chosen = retained.select(tmp_path / "retained", "vision_backbone")
    assert chosen.variant == "vision_backbone"
    assert chosen.path.read_bytes() == b"SHIPPED"


def test_rejected_variants_are_flagged(tmp_path):
    make_retained(tmp_path, {
        "decoder_action_tail_rejected_apu": b"WRONG",
        "decoder_action_tail_v2": b"SHIPPED",
    })
    variants = retained.find_variants(tmp_path / "retained", "decoder_action_tail")
    by_name = {c.variant: c for c in variants}
    assert by_name["decoder_action_tail_rejected_apu"].rejected
    assert not by_name["decoder_action_tail_v2"].rejected


def test_single_viable_variant_is_accepted(tmp_path):
    make_retained(tmp_path, {
        "decoder_action_tail_rejected_apu": b"WRONG",
        "decoder_action_tail_v2": b"SHIPPED",
    })
    chosen = retained.select(tmp_path / "retained", "decoder_action_tail")
    assert chosen.variant == "decoder_action_tail_v2"


def test_ambiguity_raises_rather_than_guessing(tmp_path):
    """Two plausible variants and no exact name: refuse, don't pick one."""
    make_retained(tmp_path, {
        "decoder_action_tail_v2": b"ONE",
        "decoder_action_tail_v3": b"TWO",
    })
    with pytest.raises(ValueError, match="ambiguous"):
        retained.select(tmp_path / "retained", "decoder_action_tail")


def test_prefix_collision_does_not_match(tmp_path):
    """`encoder_layer_01` must not be offered as a variant of `encoder_layer_0`."""
    make_elf(
        tmp_path / "retained" / "encoder_layer_01" / "encoder_layer_01_stage1_mla.elf", b"X"
    )
    assert retained.find_variants(tmp_path / "retained", "encoder_layer_0") == []


def test_missing_graph_raises(tmp_path):
    (tmp_path / "retained").mkdir()
    with pytest.raises(FileNotFoundError):
        retained.select(tmp_path / "retained", "vision_backbone")


def test_declared_graphs_drops_tensor_prepared_intermediates():
    """input_contract.json lists 12 entries for a 6-graph policy."""
    contract = {"graphs": sorted(
        [f"{g}.onnx" for g in GRAPHS] + [f"{g}_tensor_prepared.onnx" for g in GRAPHS]
    )}
    assert _declared_graphs(contract) == sorted(GRAPHS)


def test_deployed_tree_is_preferred_over_retained(tmp_path):
    """models_uncompressed/ is what the legacy deploy actually shipped."""
    make_retained(tmp_path, {"vision_backbone": b"FROM_RETAINED"})
    (tmp_path / "input_contract.json").write_text(json.dumps({
        "format": "act-modalix-v1",
        "graphs": [f"{g}.onnx" for g in GRAPHS],
        "checkpoint": "/x/checkpoints/100000/pretrained_model",
        "dataset_root": "/ml_datasets/rcwb_f_t",
    }))
    for graph in GRAPHS:
        make_elf(
            tmp_path / "models_uncompressed" / graph / "share" / f"{graph}_stage1_mla.elf",
            b"FROM_DEPLOYED",
        )

    build = detect(tmp_path)
    elfs = resolve_elfs(build, ACT_SPEC)
    assert elfs["vision_backbone"].path.read_bytes() == b"FROM_DEPLOYED"
    assert elfs["vision_backbone"].variant == "deployed"


def test_detect_reads_contract_metadata(tmp_path):
    (tmp_path / "input_contract.json").write_text(json.dumps({
        "format": "act-modalix-v1",
        "checkpoint": "/a/b/checkpoints/100000/pretrained_model",
        "dataset_root": "/ml_datasets/rcwb_f_t",
        "graphs": [f"{g}.onnx" for g in GRAPHS],
    }))
    build = detect(tmp_path)
    assert build.format == "act-modalix-v1"
    assert build.dataset_name == "rcwb_f_t"
    assert build.steps == 100000
    assert set(build.graphs) == set(GRAPHS)


def test_detect_rejects_unrecognised_tree(tmp_path):
    with pytest.raises(ValueError, match="no recognisable manifest"):
        detect(tmp_path)


def test_detect_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect(tmp_path / "absent")


def test_smolvla_npz_becomes_wire_fixtures(tmp_path):
    import numpy as np

    from polima.policies.smolvla import SMOLVLA_SPEC

    arrays = {
        tensor.name: np.arange(tensor.elements, dtype=np.float32).reshape(tensor.shape)
        for tensor in SMOLVLA_SPEC.wire.request_tensors
    }
    # Export fixtures store the vision graph input as NCHW even though the
    # live wire protocol sends HWC.
    for name in ("image0", "image1"):
        arrays[name] = arrays[name].transpose(2, 0, 1)[None]
    arrays["normalized_action"] = np.arange(
        SMOLVLA_SPEC.wire.response_elements, dtype=np.float32
    ).reshape(SMOLVLA_SPEC.wire.response_shape)
    arrays["action"] = arrays["normalized_action"] + 1000
    np.savez(tmp_path / SMOLVLA_SPEC.compile.fixture_file, **arrays)
    np.savez(
        tmp_path / "normalization_stats.npz",
        state_mean=np.arange(6, dtype=np.float32) + 10,
        state_std=np.arange(6, dtype=np.float32) + 1,
        action_mean=np.zeros(6, dtype=np.float32),
        action_std=np.ones(6, dtype=np.float32),
    )

    fixtures = collect_fixtures(
        LegacyBuild(tmp_path, "polima-export-v1"), SMOLVLA_SPEC
    )

    for tensor in SMOLVLA_SPEC.wire.request_tensors:
        path = fixtures[f"inputs/{tensor.name}.f32"]
        actual = np.fromfile(path, dtype="<f4")
        assert actual.size == tensor.elements
        if tensor.name.startswith("image"):
            expected_image = arrays[tensor.name][0].transpose(1, 2, 0).reshape(-1)
            np.testing.assert_array_equal(actual, expected_image)
        elif tensor.name == "state":
            np.testing.assert_array_equal(
                actual, arrays["state"] * np.arange(1, 7) + np.arange(10, 16)
            )
    expected = fixtures["expected/normalized_actions.f32"]
    np.testing.assert_array_equal(
        np.fromfile(expected, dtype="<f4"), arrays["action"].reshape(-1)
    )


# --------------------------------------------- adopting hand-built trees


def _deployed(root, directory, elf_name):
    share = root / "models_uncompressed" / directory / "share"
    share.mkdir(parents=True)
    (share / elf_name).write_bytes(b"\x7fELF" * 16)
    return share / elf_name


def test_a_graph_is_found_under_its_legacy_directory_name(tmp_path):
    """PoLiMa names graphs for what they are (`vision`); the SmolVLA scripts
    encode precision and compiler in the directory (`vision_llima_bf16`)."""
    from polima.bundle.retained import from_deployed_tree

    _deployed(tmp_path, "vision_llima_bf16", "vision_llima_bf16_stage1_mla.elf")
    found = from_deployed_tree(tmp_path / "models_uncompressed", "vision_llima_bf16")
    assert found.path.name.endswith(".elf")


def test_a_uniquely_named_elf_is_accepted(tmp_path):
    """LLiMa names the ELF for the ONNX it came from, so requiring
    `<graph>_stage1_mla.elf` would reject an unambiguous tree."""
    from polima.bundle.retained import from_deployed_tree

    _deployed(tmp_path, "vision_llima_bf16",
              "7b375e1b73b11138ff12fe22c8f2822d8fe03467_vision_with_output_stage1_mla.elf")
    found = from_deployed_tree(tmp_path / "models_uncompressed", "vision_llima_bf16")
    assert found.path.name.startswith("7b375e1b")


def test_two_elfs_in_one_share_is_refused(tmp_path):
    """Several means the directory holds more than one graph; picking is a guess."""
    import pytest

    from polima.bundle.retained import from_deployed_tree

    share = tmp_path / "models_uncompressed" / "g" / "share"
    share.mkdir(parents=True)
    for name in ("a_stage1_mla.elf", "b_stage1_mla.elf"):
        (share / name).write_bytes(b"\x7fELF")
    with pytest.raises(FileNotFoundError, match="cannot choose"):
        from_deployed_tree(tmp_path / "models_uncompressed", "g")


def test_a_tree_with_models_but_no_manifest_is_importable(tmp_path):
    """compile_deploy_smolvla_som.sh writes no manifest at all."""
    from polima.bundle.import_legacy import detect

    _deployed(tmp_path, "vision_llima_bf16", "vision_llima_bf16_stage1_mla.elf")
    assert detect(tmp_path).format == "unlabelled"


def test_a_tree_with_neither_is_still_rejected(tmp_path):
    import pytest

    from polima.bundle.import_legacy import detect

    (tmp_path / "notes.txt").write_text("nothing here")
    with pytest.raises(ValueError, match="no recognisable manifest"):
        detect(tmp_path)


def test_smolvla_declares_its_legacy_names():
    from polima.policies.registry import get_policy

    aliases = {g.name: g.legacy_names for g in get_policy("smolvla").compile.graphs}
    assert "vision_llima_bf16" in aliases["vision"]
    assert "denoise_single_bf16" in aliases["denoise"]


def test_smolvla_legacy_constants_become_named_runtime_sidecars(tmp_path):
    import numpy as np

    from polima.bundle.import_legacy import LegacyBuild
    from polima.policies.registry import get_policy

    constants = tmp_path / "constants"
    constants.mkdir()
    direct = (
        "empty_image_embedding",
        "language_embedding",
        "state_project_weight",
        "state_project_bias",
    )
    for name in direct:
        (constants / f"{name}.f32").write_bytes(name.encode())
    np.arange(24, dtype="<f4").tofile(constants / "normalization_stats.f32")

    sidecars = collect_sidecars(LegacyBuild(tmp_path, "unlabelled"), get_policy("smolvla"))

    assert set(sidecars) == {
        *direct, "state_mean", "state_std", "action_mean", "action_std",
    }
    assert np.fromfile(sidecars["state_mean"], dtype="<f4").tolist() == list(range(6))
    assert np.fromfile(sidecars["action_std"], dtype="<f4").tolist() == list(range(18, 24))


def test_act_robot_client_ships_with_polima(tmp_path):
    """No legacy stack, no arm calibration: the launchers are PoLiMa's own."""
    files = collect_robot_files(ACT_SPEC, LegacyBuild(tmp_path, "polima-export-v1"))
    assert set(files) == {"start.sh", "preview_robot_cameras.py"}
    assert all(path.is_file() for path in files.values())
    assert "polima.robot.so101" in files["start.sh"].read_text()


def test_act_rest_pose_comes_from_the_training_dataset(tmp_path):
    pd = pytest.importorskip("pandas")
    dataset = tmp_path / "pick_block"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps(
        {"features": {"observation.state": {"names": list(ACT_SPEC.robot.joint_names)}}}))
    (dataset / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame({"frame_index": [0, 1, 0, 0],
                  "observation.state": [[float(i)] * 6 for i in (1, 50, 3, 2)]},
                 ).to_parquet(dataset / "data/chunk-000/episode.parquet")

    files = collect_robot_files(ACT_SPEC, LegacyBuild(tmp_path, "x", dataset_root=str(dataset)))
    rest = json.loads(files["robot_rest_position.json"].read_text())
    assert rest["positions"]["gripper.pos"] == 2.0  # median of episode starts 1, 3, 2
    assert rest["source_dataset"] == "pick_block"  # a name, never a host path


def test_old_bundle_is_upgraded_to_the_built_in_client(tmp_path):
    bundle = Bundle(root=tmp_path, policy="act", bundle_id="act-test",
                    sidecars=["normalization_stats.npz",
                              "robot_client/run_robot_client_with_live_view.py"])
    client = tmp_path / "robot_client"
    (client / "calibration").mkdir(parents=True)
    for name in ("run_robot_client_with_live_view.py", "robot_rest_position.json",
                 "calibration/so-arm101.json"):
        (client / name).write_text("kept" if "run_" not in name else "old")
    (tmp_path / "normalization_stats.npz").write_bytes(b"stats")

    assert hydrate_robot_client(bundle, ACT_SPEC)
    assert (client / "start.sh").is_file() and (client / "preview_robot_cameras.py").is_file()
    assert not (client / "run_robot_client_with_live_view.py").exists()
    assert not (tmp_path / "normalization_stats.npz").exists()
    assert (client / "robot_rest_position.json").read_text() == "kept"
    assert (client / "calibration/so-arm101.json").read_text() == "kept"
    manifest = json.loads((tmp_path / "bundle.json").read_text())
    assert manifest["sidecars"] == ["robot_client/preview_robot_cameras.py", "robot_client/start.sh"]
    assert not hydrate_robot_client(bundle, ACT_SPEC)


def test_old_smolvla_bundle_is_hydrated_with_declared_output_layouts(tmp_path):
    from polima.policies.registry import get_policy

    bundle = Bundle(
        root=tmp_path,
        policy="smolvla",
        bundle_id="smolvla-test",
        graphs=[
            GraphArtifact(name, f"models/{name}/share/{name}_stage1_mla.elf", "a" * 64, 1)
            for name in ("vision", "prefix", "suffix", "denoise")
        ],
    )
    # Bundles packed before 2026-09-10 declared denoise as HWC16.
    stale = bundle.graph("denoise")
    stale.dram_layout, stale.logical_width, stale.logical_channels = "hwc16", 50, 32

    assert hydrate_runtime_metadata(bundle, get_policy("smolvla"))
    assert bundle.graph("prefix").dram_layout == "plain"
    assert bundle.graph("prefix").external_dram_layout == "HWC"
    # The denoise ELF downloads plain BF16 (see test_smolvla), so hydration
    # declares it plain with no HWC16 geometry.
    assert bundle.graph("denoise").dram_layout == "plain"
    assert bundle.graph("denoise").logical_width is None
    assert bundle.graph("denoise").logical_channels is None
    assert bundle.graph("vision").dram_layout == "plain"
    assert not hydrate_runtime_metadata(bundle, get_policy("smolvla"))

    restored = Bundle.from_dict(json.loads((tmp_path / "bundle.json").read_text()), tmp_path)
    assert restored.graph("denoise").dram_layout == "plain"
    assert restored.graph("denoise").logical_channels is None
