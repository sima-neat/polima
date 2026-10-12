"""The compile path's pure logic.

The expensive part (afe) cannot run here, and is covered instead by the Phase-1b
reproduction proof: driving these modules against the checkpoint that produced
the deployed bundle regenerates `encoder_layer_01_stage1_mla.elf` byte for byte
(sha256 09609ded..., 17189232 bytes).

What is left is exactly the logic that unifying three divergent copies could get
wrong quietly -- calibration shaping, resume keys, and command construction.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import replace

import numpy as np
import pytest

from polima.compile import calibration as calib
from polima.compile import mpk
from polima.compile.driver import Driver, GraphResult
from polima.compile.tensor import (
    MPK_FAILURE,
    build_parser,
    detect_io,
    mpk_only_failure,
    promote_rank3_hwc,
)
from polima.policies.registry import get_policy

# --------------------------------------------------------------- calibration


def test_npz_yields_one_sample_per_leading_index(tmp_path):
    path = tmp_path / "c.npz"
    np.savez(path, x=np.zeros((5, 2, 3), np.float32))
    samples = calib.from_npz(path, ["x"], [(2, 3)], "NHWC")
    assert len(samples) == 5
    assert samples[0]["x"].shape == (2, 3)


def test_rank4_nchw_is_transposed_to_nhwc(tmp_path):
    """afe's quantizer wants NHWC calibration data regardless of model layout."""
    path = tmp_path / "c.npz"
    np.savez(path, image=np.zeros((2, 1, 3, 8, 9), np.float32))
    sample = calib.from_npz(path, ["image"], [(1, 3, 8, 9)], "NCHW")[0]
    assert sample["image"].shape == (1, 8, 9, 3)


def test_rank4_nhwc_is_left_alone(tmp_path):
    path = tmp_path / "c.npz"
    np.savez(path, image=np.zeros((2, 1, 8, 9, 3), np.float32))
    sample = calib.from_npz(path, ["image"], [(1, 8, 9, 3)], "NHWC")[0]
    assert sample["image"].shape == (1, 8, 9, 3)


def test_low_rank_tensors_are_never_transposed(tmp_path):
    """This is the bug SmolVLA's copy existed to avoid: the curated helper
    transposes unconditionally, which corrupts rank-2/3 action-side tensors."""
    path = tmp_path / "c.npz"
    np.savez(path, hidden=np.zeros((2, 601, 512), np.float32))
    sample = calib.from_npz(path, ["hidden"], [(601, 512)], "NCHW")[0]
    assert sample["hidden"].shape == (601, 512)


def test_calibration_samples_are_contiguous(tmp_path):
    path = tmp_path / "c.npz"
    np.savez(path, image=np.zeros((1, 1, 3, 4, 5), np.float32))
    sample = calib.from_npz(path, ["image"], [(1, 3, 4, 5)], "NCHW")[0]
    assert sample["image"].flags["C_CONTIGUOUS"]


def test_missing_array_names_the_file(tmp_path):
    path = tmp_path / "c.npz"
    np.savez(path, x=np.zeros((1, 2), np.float32))
    with pytest.raises(KeyError, match="missing calibration arrays"):
        calib.from_npz(path, ["x", "y"], [(2,), (2,)])


def test_shape_without_a_sample_axis_is_rejected(tmp_path):
    """(2, 3) where (N, 2, 3) is meant would otherwise quantize against a single
    reshaped sample -- which succeeds, and produces a worse model."""
    path = tmp_path / "c.npz"
    np.savez(path, x=np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="expected"):
        calib.from_npz(path, ["x"], [(2, 3)])


def test_inputs_must_agree_on_sample_count(tmp_path):
    path = tmp_path / "c.npz"
    np.savez(path, x=np.zeros((4, 2), np.float32), y=np.zeros((3, 2), np.float32))
    with pytest.raises(ValueError, match="sample count"):
        calib.from_npz(path, ["x", "y"], [(2,), (2,)])


def test_empty_npz_is_rejected(tmp_path):
    path = tmp_path / "c.npz"
    np.savez(path, x=np.zeros((0, 2), np.float32))
    with pytest.raises(ValueError, match="no calibration samples"):
        calib.from_npz(path, ["x"], [(2,)])


def test_raw_f32_splits_into_whole_samples(tmp_path):
    path = tmp_path / "c.f32"
    np.arange(24, dtype=np.float32).tofile(path)
    samples = calib.from_raw_f32(path, ["prefix"], [(2, 3)])
    assert len(samples) == 4
    assert samples[1]["prefix"].tolist() == [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]


def test_raw_f32_rejects_a_partial_sample(tmp_path):
    path = tmp_path / "c.f32"
    np.arange(25, dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="whole number"):
        calib.from_raw_f32(path, ["prefix"], [(2, 3)])


def test_raw_f32_rejects_multi_input_graphs(tmp_path):
    path = tmp_path / "c.f32"
    np.arange(6, dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="exactly one input"):
        calib.from_raw_f32(path, ["a", "b"], [(3,), (3,)])


def test_random_matches_the_requested_shapes():
    sample = calib.random(["x", "y"], [(2, 3), (4,)])[0]
    assert sample["x"].shape == (2, 3) and sample["y"].shape == (4,)
    assert sample["x"].dtype == np.float32


def test_random_honours_int8_inputs():
    sample = calib.random(["tokens"], [(4,)], types={"tokens": "int8"})[0]
    assert sample["tokens"].dtype == np.int8


def test_random_is_reproducible():
    first = calib.random(["x"], [(3,)], seed=7)[0]["x"]
    assert np.array_equal(first, calib.random(["x"], [(3,)], seed=7)[0]["x"])


def test_build_dispatches_and_falls_back_to_random(tmp_path):
    path = tmp_path / "c.npz"
    np.savez(path, x=np.zeros((2, 3), np.float32))
    assert len(calib.build("npz", path, ["x"], [(3,)])) == 2
    assert len(calib.build("npz", None, ["x"], [(3,)])) == 1     # no path -> random
    with pytest.raises(ValueError, match="unknown calibration kind"):
        calib.build("guess", path, ["x"], [(3,)])


# ---------------------------------------------------------------------- mpk


def _archive(path, names):
    with tarfile.open(path, "w:gz") as handle:
        for name, payload in names.items():
            info = tarfile.TarInfo(name)
            data = payload.encode()
            info.size = len(data)
            import io

            handle.addfile(info, io.BytesIO(data))


def test_unpack_routes_files_by_suffix(tmp_path):
    archive = tmp_path / "m_mpk.tar.gz"
    _archive(archive, {"a/m.elf": "ELF", "a/m.so": "SO",
                       "a/m.json": json.dumps({"model_info": {"path": "/build/m.so"}})})
    root = mpk.unpack(archive, tmp_path / "out")
    assert (root / "share" / "m.elf").read_text() == "ELF"
    assert (root / "lib" / "m.so").read_text() == "SO"
    assert (root / "etc" / "m.json").exists()


def test_unpack_repoints_config_at_the_new_location(tmp_path):
    archive = tmp_path / "m_mpk.tar.gz"
    _archive(archive, {
        "m.elf": "ELF",
        "m.json": json.dumps({
            "simaai__params": {"model_path": "/some/build/host/m.elf"},
            "model_info": {"path": "/some/build/host/m.so"},
        }),
    })
    root = mpk.unpack(archive, tmp_path / "out")
    config = json.loads((root / "etc" / "m.json").read_text())
    assert config["simaai__params"]["model_path"] == str(root / "share" / "m.elf")
    assert config["model_info"]["path"] == str(root / "lib" / "m.so")


def test_unpack_ignores_path_traversal(tmp_path):
    archive = tmp_path / "m_mpk.tar.gz"
    _archive(archive, {"../escape.elf": "NO", "m.elf": "ELF", "m.json": "{}"})
    root = mpk.unpack(archive, tmp_path / "out")
    assert not (tmp_path / "escape.elf").exists()
    assert (root / "share" / "m.elf").exists()


def test_unpack_rejects_an_archive_with_no_elf(tmp_path):
    archive = tmp_path / "m_mpk.tar.gz"
    _archive(archive, {"m.json": "{}"})
    with pytest.raises(RuntimeError, match="usable model directory"):
        mpk.unpack(archive, tmp_path / "out")


def test_has_elf_detects_an_apu_fallback(tmp_path):
    """A compile can exit 0 and produce an mpk with no ELF; both legacy
    controllers check for this because the board failure is much later."""
    with_elf = tmp_path / "a_mpk.tar.gz"
    without = tmp_path / "b_mpk.tar.gz"
    _archive(with_elf, {"a.elf": "ELF"})
    _archive(without, {"b.json": "{}"})
    assert mpk.has_elf(with_elf)
    assert not mpk.has_elf(without)
    assert not mpk.has_elf(tmp_path / "nonexistent.tar.gz")


def test_find_locates_by_stem(tmp_path):
    (tmp_path / "deep" / "nest").mkdir(parents=True)
    target = tmp_path / "deep" / "nest" / "vision_backbone_mpk.tar.gz"
    target.write_bytes(b"")
    assert mpk.find(tmp_path, "vision_backbone") == target
    assert mpk.find(tmp_path, "absent") is None


# -------------------------------------------------------------------- driver


def _driver(tmp_path, **kwargs) -> Driver:
    return Driver(spec=get_policy("act"), build_dir=tmp_path,
                  compiler_python="/nonexistent/python", **kwargs)


def _write_inputs(driver: Driver, graph, onnx: bytes = b"onnx", calib_bytes: bytes = b"c"):
    driver.onnx_path(graph.name).parent.mkdir(parents=True, exist_ok=True)
    driver.onnx_path(graph.name).write_bytes(onnx)
    path = driver.calibration_path(graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(calib_bytes)


def test_argv_reproduces_the_legacy_invocation(tmp_path):
    """compile_deploy_act_som.sh: bf16, per-graph layout, npz calibration,
    tessellation, and a retained dir that is where the ELF is then read from."""
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("vision_backbone")
    argv = driver.argv(graph, "bf16")
    assert "--precision" in argv and argv[argv.index("--precision") + 1] == "bf16"
    assert argv[argv.index("--model-layout") + 1] == "NCHW"
    assert "--mla-tessellation" in argv
    assert argv[argv.index("--calibration-npz") + 1].endswith("vision_backbone.npz")
    assert argv[argv.index("--retain-compile-dir") + 1].endswith("retained/vision_backbone")


def test_encoder_layers_compile_as_nhwc(tmp_path):
    """Only the vision backbone is NCHW; the packed-token graphs are NHWC."""
    driver = _driver(tmp_path)
    for name in ("encoder_layer_00_stem", "encoder_layer_01", "decoder_action_tail"):
        graph = get_policy("act").compile.graph(name)
        argv = driver.argv(graph, "bf16")
        assert argv[argv.index("--model-layout") + 1] == "NHWC"


def test_shared_graph_argv_requests_rank4_hwc_boundaries(tmp_path):
    driver = _driver(tmp_path)
    graph = replace(
        get_policy("act").compile.graph("encoder_layer_01"),
        inputs=(replace(
            get_policy("act").compile.graph("encoder_layer_01").inputs[0],
            shape=(1, 602, 512),
        ),),
        outputs=(replace(
            get_policy("act").compile.graph("encoder_layer_01").outputs[0],
            shape=(1, 602, 512),
        ),),
        mla_tessellation=False,
        external_dram_layout="HWC",
        promote_rank3_hwc=True,
    )
    argv = driver.argv(graph, "bf16")
    assert argv[argv.index("--external-dram-layout") + 1] == "HWC"
    assert "--promote-rank3-hwc" in argv
    assert "--mla-tessellation" not in argv


def test_smolvla_vision_keeps_plain_model_io(tmp_path):
    driver = Driver(
        spec=get_policy("smolvla"), build_dir=tmp_path,
        compiler_python="/nonexistent/python",
    )
    argv = driver.argv(get_policy("smolvla").compile.graph("vision"), "bf16")
    assert "--mla-tessellation" not in argv
    assert "--retain-compile-dir" in argv
    assert "--exit-on-stable-elf" in argv
    assert "--no-simplify" in argv


def test_smolvla_denoise_uses_calibrated_split_precision(tmp_path):
    driver = Driver(
        spec=get_policy("smolvla"), build_dir=tmp_path,
        compiler_python="/nonexistent/python",
    )
    graph = get_policy("smolvla").compile.graph("denoise")
    argv = driver.argv(graph, "bf16")
    assert argv[argv.index("--activation-precision") + 1] == "bf16"
    assert argv[argv.index("--weight-precision") + 1] == "int8"
    assert argv[argv.index("--calibration-raw-f32") + 1].endswith(
        "calibration/denoise.f32"
    )


def test_smolvla_prefix_uses_plain_bf16_hwc_boundaries(tmp_path):
    driver = Driver(
        spec=get_policy("smolvla"), build_dir=tmp_path,
        compiler_python="/nonexistent/python",
    )
    graph = get_policy("smolvla").compile.graph("prefix")
    argv = driver.argv(graph, graph.precision)
    assert "--activation-precision" not in argv
    assert "--weight-precision" not in argv
    assert "--mla-tessellation" not in argv
    assert argv[argv.index("--external-dram-layout") + 1] == "HWC"
    assert argv[argv.index("--retain-compile-dir") + 1].endswith("retained/prefix")
    assert "--exit-on-stable-elf" in argv


def test_smolvla_vision_uses_plain_bf16_hwc_boundaries(tmp_path):
    driver = Driver(
        spec=get_policy("smolvla"), build_dir=tmp_path,
        compiler_python="/nonexistent/python",
    )
    graph = get_policy("smolvla").compile.graph("vision")
    argv = driver.argv(graph, graph.precision)
    assert argv[argv.index("--precision") + 1] == "bf16"
    assert "--activation-precision" not in argv
    assert "--weight-precision" not in argv
    assert argv[argv.index("--external-dram-layout") + 1] == "HWC"
    assert "--promote-rank3-hwc" in argv
    assert "--calibration-raw-f32" not in argv
    assert argv[argv.index("--retain-compile-dir") + 1].endswith("retained/vision")
    assert "--exit-on-stable-elf" in argv


def test_stage_key_is_stable_for_unchanged_inputs(tmp_path):
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    assert driver.stage_key(graph) == driver.stage_key(graph)


def test_stage_key_tracks_the_onnx(tmp_path):
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    before = driver.stage_key(graph)
    driver.onnx_path(graph.name).write_bytes(b"re-exported")
    assert driver.stage_key(graph) != before


def test_stage_key_tracks_the_calibration_data(tmp_path):
    """The legacy --resume only checked that an ELF existed, so re-exporting and
    re-running silently kept the stale one."""
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    before = driver.stage_key(graph)
    driver.calibration_path(graph).write_bytes(b"different samples")
    assert driver.stage_key(graph) != before


def test_stage_key_tracks_the_sdk_version(tmp_path):
    """An afe upgrade changes code generation with every input byte-identical."""
    graph = get_policy("act").compile.graph("encoder_layer_01")
    first = _driver(tmp_path, sdk_version="2.1.0")
    _write_inputs(first, graph)
    assert first.stage_key(graph) != _driver(tmp_path, sdk_version="2.2.0").stage_key(graph)


def test_stage_key_is_independent_of_the_build_location(tmp_path):
    """So a relocated or renamed build tree still counts as unchanged."""
    graph = get_policy("act").compile.graph("encoder_layer_01")
    keys = []
    for name in ("build_a", "build_b"):
        driver = _driver(tmp_path / name)
        _write_inputs(driver, graph)
        keys.append(driver.stage_key(graph))
    assert keys[0] == keys[1]


def test_reuse_requires_the_elf_to_still_exist(tmp_path):
    """A matching key with a deleted ELF must fall through to a real compile,
    which under --dry-run reports `skipped` rather than `reused`."""
    driver = _driver(tmp_path, dry_run=True)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    driver._record(GraphResult(graph.name, "compiled", precision="bf16",
                               elf=str(tmp_path / "gone.elf"), key=driver.stage_key(graph)))
    assert driver.compile_graph(graph).status == "skipped"


def test_reuse_hits_when_nothing_changed(tmp_path):
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    elf = tmp_path / "prebuilt.elf"
    elf.write_bytes(b"\x7fELF")
    driver._record(GraphResult(graph.name, "compiled", precision="bf16",
                               elf=str(elf), key=driver.stage_key(graph)))
    result = driver.compile_graph(graph)
    assert result.status == "reused"
    # and it is published where the bundle packer looks
    published = tmp_path / "models_uncompressed" / graph.name / "share" / graph.elf_name
    assert published.read_bytes() == b"\x7fELF"


def test_force_defeats_reuse(tmp_path):
    driver = _driver(tmp_path, force=True, dry_run=True)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    elf = tmp_path / "prebuilt.elf"
    elf.write_bytes(b"\x7fELF")
    driver._record(GraphResult(graph.name, "compiled", precision="bf16",
                               elf=str(elf), key=driver.stage_key(graph)))
    assert driver.compile_graph(graph).status == "skipped"     # dry run, not reused


def test_missing_onnx_points_at_the_export_stage(tmp_path):
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    result = driver.compile_graph(graph)
    assert result.status == "failed" and "export" in result.note


def test_select_rejects_an_unknown_graph(tmp_path):
    from polima.compile.driver import CompileError

    with pytest.raises(CompileError, match="unknown graph"):
        _driver(tmp_path).select(["encoder_layer_99"])


def test_select_preserves_plan_order(tmp_path):
    chosen = _driver(tmp_path).select(["decoder_action_tail", "vision_backbone"])
    assert [g.name for g in chosen] == ["vision_backbone", "decoder_action_tail"]


def test_graph_result_ok_covers_reuse():
    assert GraphResult("g", "compiled").ok
    assert GraphResult("g", "reused").ok
    assert not GraphResult("g", "failed").ok
    assert not GraphResult("g", "skipped").ok


# -------------------------------------------------------------- tensor CLI


def test_tensor_cli_defaults_to_bf16_modalix():
    args = build_parser().parse_args(["--model-path", "m.onnx", "--build-dir", "b"])
    assert args.precision == "bf16"
    assert args.device == "modalix"
    assert args.model_layout == "NCHW"
    assert not args.infer_shapes


def test_tensor_cli_accepts_the_smolvla_surface():
    """The SmolVLA copy's extra knobs must all still be expressible."""
    args = build_parser().parse_args([
        "--model-path", "m.onnx", "--build-dir", "b", "--device", "mlsoc",
        "--calibration-raw-f32", "c.f32", "--input-shapes", "1,601,512",
        "--input-types", "float32", "--no-compile", "--infer-shapes",
        "--calib-method", "min_max", "--requant-mode", "tflite",
        "--activation-precision", "bf16", "--weight-precision", "int8",
    ])
    assert args.device == "mlsoc" and args.no_compile and args.infer_shapes
    assert args.activation_precision == "bf16" and args.weight_precision == "int8"


def test_rank3_hwc_promotion_preserves_names_and_element_counts(tmp_path):
    onnx = pytest.importorskip("onnx")
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 7, 16])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 7, 16])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])], "rank3", [x], [y]
    )
    source = tmp_path / "stage.onnx"
    onnx.save(onnx.helper.make_model(graph), source)

    promoted = promote_rank3_hwc(source)

    names, shapes, outputs = detect_io(promoted)
    assert names == ["x"] and outputs == ["y"]
    assert shapes == [(1, 1, 7, 16)]
    model = onnx.load(promoted)
    assert [node.op_type for node in model.graph.node] == ["Reshape", "Identity", "Reshape"]


def test_rank3_hwc_promotion_can_wrap_only_the_output(tmp_path):
    onnx = pytest.importorskip("onnx")
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64, 3])
    reshape_shape = onnx.helper.make_tensor(
        "shape", onnx.TensorProto.INT64, [3], [1, 64, 3]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Reshape", ["x", "shape"], ["y"])],
        "rank4_to_rank3", [x], [y], [reshape_shape],
    )
    source = tmp_path / "vision.onnx"
    onnx.save(onnx.helper.make_model(graph), source)

    promoted = promote_rank3_hwc(source)

    names, shapes, outputs = detect_io(promoted)
    assert names == ["x"] and shapes == [(1, 3, 8, 8)] and outputs == ["y"]
    model = onnx.load(promoted)
    output_shape = [
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ]
    assert output_shape == [1, 1, 64, 3]


def test_rank3_hwc_promotion_uses_nchw_axes_when_requested(tmp_path):
    onnx = pytest.importorskip("onnx")
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64, 960])
    shape = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [3], [1, 64, 960])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Reshape", ["x", "shape"], ["y"])],
        "vision", [x], [y], [shape],
    )
    source = tmp_path / "vision.onnx"
    onnx.save(onnx.helper.make_model(graph), source)

    promoted = promote_rank3_hwc(source, "NCHW")

    model = onnx.load(promoted)
    output_shape = [
        dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
    ]
    assert output_shape == [1, 960, 1, 64]


def test_tensor_module_imports_without_afe():
    """doctor and these tests import it in environments with no compiler."""
    import polima.compile.tensor as module

    assert not hasattr(module, "afe")


# ------------------------------------------------------------------ cli wiring


def test_checkpoint_without_a_build_dir_is_rejected(capsys):
    from polima.cli import compile as compile_cli

    assert compile_cli.run(["--checkpoint", "/some/ckpt"]) == 2
    assert "--build-dir" in capsys.readouterr().err


def test_import_legacy_and_build_dir_are_mutually_exclusive(capsys):
    from polima.cli import compile as compile_cli

    assert compile_cli.run(["--import-legacy", "/a", "--build-dir", "/b"]) == 2
    assert "not both" in capsys.readouterr().err


def test_no_arguments_lists_the_choices(capsys):
    from polima.cli import compile as compile_cli

    assert compile_cli.run([]) == 2
    error = capsys.readouterr().err
    assert "--checkpoint" in error and "--build-dir" in error


def test_import_legacy_still_needs_no_compiler():
    """Phase 1a depends on this path running where afe does not exist."""
    from polima.cli import compile as compile_cli

    assert compile_cli.needs_capability(["--import-legacy", "/x"]) is None
    assert compile_cli.needs_capability(["--build-dir", "/x"]) == "compile"
    assert compile_cli.needs_capability(["--checkpoint", "/x"]) == "compile"


# ------------------------------------------------- calibration is int8-only


def test_bf16_ignores_supplied_calibration():
    """bf16 has no scales to fit. Verified against hardware: ACT's
    decoder_action_tail compiled with 8 real dataset samples and with 1 random
    sample produces the identical ELF, b1eece6992dbddc6..."""
    kind, source, note = calib.plan("bf16", "bf16", npz="/big/file.npz")
    assert kind == "random" and source is None
    assert "no scales" in note


def test_bf16_without_calibration_says_nothing():
    kind, source, note = calib.plan("bf16", "bf16")
    assert (kind, source, note) == ("random", None, "")


def test_int8_still_uses_calibration():
    assert calib.plan("int8", "int8", npz="/c.npz") == ("npz", "/c.npz", "")
    assert calib.plan("int8", "int8", raw_f32="/c.f32") == ("raw_f32", "/c.f32", "")


def test_mixed_precision_keeps_calibration():
    """int8 weights with bf16 activations still fits weight scales."""
    kind, _, _ = calib.plan("bf16", "int8", npz="/c.npz")
    assert kind == "npz"


def test_int8_without_calibration_warns_about_drift():
    _, _, note = calib.plan("int8", "int8")
    assert "drift" in note


# ----------------------------------------------- palette compiler activation


def test_activation_is_sourced_and_filtered(tmp_path):
    """Palette's activate-model-compiler is what puts the compiler's own shared
    libraries on the loader path; without it `import afe` dies on a missing
    libLLVM, which reads like a broken install rather than a missing step."""
    from polima.compile.toolchain import activation_env, find_activation

    script = tmp_path / "activate-model-compiler"
    script.write_text(
        'export LD_LIBRARY_PATH="/opt/mc/lib:${LD_LIBRARY_PATH}"\n'
        "export MODEL_SDK_ROOT=/opt/mc\n"
        "export UNRELATED=noise\n"
    )
    assert find_activation(tmp_path) == script

    changed = activation_env(script)
    assert changed["MODEL_SDK_ROOT"] == "/opt/mc"
    assert changed["LD_LIBRARY_PATH"].startswith("/opt/mc/lib")
    # Copying the whole environment would drag the subshell's PWD/SHLVL along
    # and quietly undo the caller's own settings.
    assert "UNRELATED" not in changed


def test_absent_activation_is_not_an_error(tmp_path):
    from polima.compile.toolchain import find_activation

    assert find_activation(tmp_path) is None


def test_a_failing_activation_yields_no_changes(tmp_path):
    from polima.compile.toolchain import activation_env

    script = tmp_path / "activate-model-compiler"
    script.write_text("exit 1\n")
    assert activation_env(script) == {}


def test_compiler_env_keeps_its_own_settings_over_the_activation(tmp_path):
    """The activation runs first so the explicit PATH prefix still wins."""
    from polima.compile.toolchain import compiler_env

    (tmp_path / "activate-model-compiler").write_text("export PATH=/only/this\n")
    env = compiler_env(tmp_path)
    assert env["PATH"].startswith(f"{tmp_path}:")
    assert env["CUDA_VISIBLE_DEVICES"] == ""


# -------------------------------------------------------------- bin launcher


def _launcher(name: str = "polima"):
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "bin" / name


def test_launcher_runs_without_an_install():
    """The Palette container is recreated and mounts the workspace at a
    different path, so an editable install done on the host would bake in a
    path that does not exist inside it."""
    import subprocess

    result = subprocess.run([str(_launcher()), "--version"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert result.stdout.strip().startswith("polima ")


def test_stage_symlinks_dispatch_on_their_name():
    """`polima-deploy` must behave as `polima deploy`, matching pip's scripts."""
    import subprocess

    assert _launcher("polima-deploy").is_symlink()
    result = subprocess.run([str(_launcher("polima-deploy")), "--help"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert "polima-deploy" in result.stdout


def test_every_stage_has_a_launcher():
    for stage in ("compile", "deploy", "run", "robot", "doctor"):
        assert _launcher(f"polima-{stage}").exists(), stage


# ------------------------------------------------- resume across mount points


def test_recorded_elf_is_relative_to_the_build_tree(tmp_path):
    """The same tree is routinely seen at two paths -- the Palette container
    mounts it as /workspace/... while the host sees ~/SDK/NEAT/workspace/... --
    so an absolute path recorded on one side does not exist on the other and
    every stage recompiles despite matching keys."""
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    elf = driver.elf_path(graph)
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_bytes(b"\x7fELF")

    driver._record(GraphResult(graph.name, "compiled", precision="bf16",
                               elf=str(elf), key="k"))
    recorded = json.loads((tmp_path / "compile_state.json").read_text())
    assert recorded[graph.name]["elf"] == f"retained/{graph.name}/{graph.elf_name}"


def test_a_tree_compiled_elsewhere_still_reuses(tmp_path):
    """Record as if the compile happened at a container path, then reuse from
    the host path -- which is exactly the round trip that failed."""
    import shutil as _shutil

    container = tmp_path / "workspace" / "build"
    graph = get_policy("act").compile.graph("encoder_layer_01")
    driver = _driver(container)
    _write_inputs(driver, graph)
    elf = driver.elf_path(graph)
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_bytes(b"\x7fELF")
    driver._record(GraphResult(graph.name, "compiled", precision="bf16",
                               elf=str(elf), key=driver.stage_key(graph)))

    host = tmp_path / "home" / "build"
    host.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copytree(container, host)

    assert _driver(host).compile_graph(graph).status == "reused"


def test_an_elf_outside_the_tree_stays_absolute(tmp_path):
    driver = _driver(tmp_path / "build")
    outside = tmp_path / "elsewhere.elf"
    outside.write_bytes(b"\x7fELF")
    assert driver._relative_elf(str(outside)) == str(outside)


def test_a_stale_absolute_path_heals_instead_of_recompiling(tmp_path):
    """State written before paths were relative holds an absolute path from the
    other mount. Falling back to the conventional location avoids repeating ~9
    minutes of identical work, and re-records it relative."""
    driver = _driver(tmp_path)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    elf = driver.elf_path(graph)
    elf.parent.mkdir(parents=True, exist_ok=True)
    elf.write_bytes(b"\x7fELF")

    driver._record(GraphResult(graph.name, "compiled", precision="bf16",
                               elf="/workspace/gone/encoder_layer_01_stage1_mla.elf",
                               key=driver.stage_key(graph)))
    assert driver.compile_graph(graph).status == "reused"

    healed = json.loads((tmp_path / "compile_state.json").read_text())
    assert healed[graph.name]["elf"] == f"retained/{graph.name}/{graph.elf_name}"


# ------------------------------------------------------------- parallel jobs


def test_jobs_defaults_to_sequential(tmp_path):
    """Memory is the limit, not CPU, and it is policy-dependent: SmolVLA's
    compile script runs its vision and prefix stages sequentially because they
    'each require substantial host RAM'. Parallelism is opt-in."""
    assert _driver(tmp_path).jobs == 1


def test_parallel_reports_in_plan_order(tmp_path):
    """Completion order varies run to run; the report must not."""
    driver = _driver(tmp_path, jobs=4, dry_run=True)
    graphs = get_policy("act").compile.graphs
    for graph in graphs:
        _write_inputs(driver, graph)
    results = driver.run()
    assert [r.name for r in results] == [g.name for g in graphs]


def test_compile_reports_elf_core_and_effective_job_counts(
    tmp_path, monkeypatch, capsys,
):
    from polima.compile import driver as driver_module

    driver = _driver(tmp_path, jobs=99, dry_run=True)
    graphs = get_policy("act").compile.graphs
    for graph in graphs:
        _write_inputs(driver, graph)
    monkeypatch.setattr(driver_module.os, "cpu_count", lambda: 20)

    driver.run()

    output = capsys.readouterr().out
    assert f"{len(graphs)} ELFs to compile" in output
    assert "20 logical CPU cores" in output
    assert f"{len(graphs)} parallel jobs" in output
    assert "total compile time" in output


def test_effective_jobs_are_also_capped_by_logical_cores(
    tmp_path, monkeypatch, capsys,
):
    from polima.compile import driver as driver_module

    driver = _driver(tmp_path, jobs=99, dry_run=True)
    for graph in get_policy("act").compile.graphs:
        _write_inputs(driver, graph)
    monkeypatch.setattr(driver_module.os, "cpu_count", lambda: 2)

    driver.run()

    output = capsys.readouterr().out
    assert "2 parallel jobs" in output
    assert "running 2 graph(s) at a time" in output


def test_state_writes_are_serialized(tmp_path):
    """Several graphs finish into one compile_state.json; a lost update means a
    stage recompiles next run for no reason."""
    import threading

    driver = _driver(tmp_path, jobs=8)
    graphs = list(get_policy("act").compile.graphs)

    def record(graph):
        driver._record(GraphResult(graph.name, "compiled", precision="bf16",
                                   elf=str(driver.elf_path(graph)), key="k"))

    threads = [threading.Thread(target=record, args=(g,)) for g in graphs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    state = json.loads((tmp_path / "compile_state.json").read_text())
    assert sorted(state) == sorted(g.name for g in graphs)


def test_one_graph_never_takes_the_parallel_path(tmp_path):
    driver = _driver(tmp_path, jobs=8, dry_run=True)
    graph = get_policy("act").compile.graph("encoder_layer_01")
    _write_inputs(driver, graph)
    assert len(driver.run(only=["encoder_layer_01"])) == 1


def test_a_compile_recompiles_by_default(monkeypatch, tmp_path, capsys):
    """`--reuse` is opt-in. "It said reused and I wanted a build" is a worse
    failure than spending the time, especially when the export step re-runs
    either way and makes it look like work happened."""
    from polima.cli import compile as compile_cli

    seen = {}

    class _Driver:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self, only=None):
            return []

        def write_manifest(self, results):
            return None

    monkeypatch.setattr("polima.compile.driver.Driver", _Driver)
    monkeypatch.setattr("polima.compile.driver.sdk_version", lambda *a, **k: "2.1.0")
    monkeypatch.setattr("polima.compile.toolchain.require_compiler_python",
                        lambda *a, **k: tmp_path / "python")
    monkeypatch.setattr("polima.compile.toolchain.compiler_env", lambda *a, **k: {})

    (tmp_path / "retained").mkdir()
    compile_cli.run(["--build-dir", str(tmp_path), "--stop-after", "compile"])
    assert seen["force"] is True

    seen.clear()
    compile_cli.run(["--build-dir", str(tmp_path), "--stop-after", "compile", "--reuse"])
    assert seen["force"] is False


# ------------------------------------------------------------ mpk tolerance


def test_mpk_failure_with_a_finished_elf_is_tolerated(tmp_path):
    elf = tmp_path / "graph_stage1_mla.elf"
    elf.write_bytes(b"\x7fELF" + bytes(64))
    error = RuntimeError(f"{MPK_FAILURE}.  See the log for details.")
    assert mpk_only_failure(error, elf, settle_seconds=0.0)


def test_mpk_failure_without_an_elf_still_fails(tmp_path):
    error = RuntimeError(MPK_FAILURE)
    assert not mpk_only_failure(error, tmp_path / "missing.elf", settle_seconds=0.0)
    assert not mpk_only_failure(error, None, settle_seconds=0.0)


def test_empty_elf_is_not_a_finished_elf(tmp_path):
    elf = tmp_path / "graph_stage1_mla.elf"
    elf.touch()
    assert not mpk_only_failure(RuntimeError(MPK_FAILURE), elf, settle_seconds=0.0)


def test_other_compile_errors_are_never_swallowed(tmp_path):
    elf = tmp_path / "graph_stage1_mla.elf"
    elf.write_bytes(b"\x7fELF" + bytes(64))
    assert not mpk_only_failure(RuntimeError("TVMError: Assert fail"), elf,
                                settle_seconds=0.0)
