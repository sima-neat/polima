"""Quantize and compile one fixed-shape ONNX graph for a SiMa target.

One tool serves every policy. Each graph uses the subset of knobs it needs:
calibration data (npz or raw f32), precision (including split activation/weight
precision), IO overrides, gen1/gen2, --calib-method, --requant-mode,
--no-compile and shape re-inference. Where graphs need different behaviour, the
one that keeps established ELF bytes is the default and the other is a flag --
see `prepare()`.

## Where this runs

Under the SiMa model-compiler venv (python 3.12 + afe + onnx + onnxsim, and
notably *no* torch), not under the training env. That is why `afe` is imported
inside the functions that need it: `polima doctor` and the unit tests import
this module in environments where afe does not exist, and they only need the
argument surface and the plan, not the compiler.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from polima.compile import calibration as calib

Shape = tuple[int, ...]


# --------------------------------------------------------------------- onnx io


def detect_io(path: str | Path) -> tuple[list[str], list[Shape], list[str]]:
    """Read input names/shapes and output names straight off the graph.

    `load_external_data=False` matters for the vision backbones, whose weights
    live in a sidecar file: this only ever touches graph metadata, and loading
    the tensors to read a shape costs seconds and gigabytes for nothing.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    names: list[str] = []
    shapes: list[Shape] = []
    for value in model.graph.input:
        shape = tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"input {value.name!r} of {path} is not static: {shape}. "
                "Re-export with fixed axes; the MLA cannot take a dynamic shape."
            )
        names.append(value.name)
        shapes.append(shape)
    return names, shapes, [value.name for value in model.graph.output]


def prepare(path: str | Path, names: Sequence[str], shapes: Sequence[Shape],
            infer_shapes: bool = False) -> Path:
    """Simplify to a static graph and write `<stem>_tensor_prepared.onnx`.

    `infer_shapes` runs `onnx.shape_inference` after simplification. Shape
    inference rewrites the proto's `value_info`, so turning it on
    unconditionally would change the bytes handed to afe for graphs that do not
    need it, such as ACT's. It is therefore opt-in, requested per graph by
    `GraphSpec`.

    Re-stamping the input dims is unconditional and is a no-op whenever onnxsim
    honoured `overwrite_input_shapes`. It is cheap insurance against a symbolic
    dim surviving simplification, which makes afe fail much later with a
    message that does not mention the shape.
    """
    import onnx
    from onnxsim import simplify

    path = Path(path)
    shape_map = {name: list(shape) for name, shape in zip(names, shapes, strict=True)}
    proto, ok = simplify(str(path), overwrite_input_shapes=shape_map,
                         dynamic_input_shape=False)
    if not ok:
        raise RuntimeError(f"onnxsim validation failed for {path}")

    for tensor in proto.graph.input:
        if tensor.name not in shape_map:
            continue
        tensor.type.tensor_type.shape.ClearField("dim")
        for size in shape_map[tensor.name]:
            tensor.type.tensor_type.shape.dim.add().dim_value = size

    if infer_shapes:
        proto = onnx.shape_inference.infer_shapes(proto)

    # afe's importer rejects newer IR versions, so clamp to 8.
    proto.ir_version = min(proto.ir_version, 8)
    prepared = path.with_name(f"{path.stem}_tensor_prepared.onnx")
    onnx.save(proto, prepared)
    return prepared


def promote_rank3_hwc(path: str | Path, layout: str = "NHWC") -> Path:
    """Wrap each rank-3 [N,W,C] boundary as a layout-correct rank-4 tensor.

    This satisfies ModelSDK 2.1's four-dimensional external-layout path, which
    is required for directly sharing an HWC MLABuffer between token stages.
    NHWC promotion is a reshape to [N,1,W,C]. NCHW promotion is *not* merely a
    reshape: [N,W,C] must be transposed to [N,C,W] before wrapping it as
    [N,C,1,W]. Otherwise ModelSDK correctly exports NCHW as external HWC but
    the token and channel axes contain reshape-order data, producing a
    deterministic token/channel permutation in the output.
    """
    import onnx

    path = Path(path)
    model = onnx.load(str(path))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("rank-3 HWC promotion requires one input and one output")
    graph_input, graph_output = model.graph.input[0], model.graph.output[0]
    input_shape = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in graph_output.type.tensor_type.shape.dim]
    if len(input_shape) not in (3, 4) or len(output_shape) not in (3, 4):
        raise ValueError(
            f"expected rank-3/rank-4 I/O, got input={input_shape}, output={output_shape}"
        )
    if len(input_shape) == 4 and len(output_shape) == 4:
        return path

    input_name, output_name = graph_input.name, graph_output.name
    rank3_input = input_name + "__rank3" if len(input_shape) == 3 else input_name
    rank3_output = output_name + "__rank3" if len(output_shape) == 3 else output_name
    for node in model.graph.node:
        node.input[:] = [rank3_input if name == input_name else
                         rank3_output if name == output_name else name for name in node.input]
        node.output[:] = [rank3_output if name == output_name else name for name in node.output]

    def rank4(shape: list[int]) -> list[int]:
        if layout.upper() == "NCHW":
            return [shape[0], shape[2], 1, shape[1]]
        return [shape[0], 1, shape[1], shape[2]]

    if len(input_shape) == 3:
        input_shape_name = input_name + "__shape3"
        model.graph.initializer.append(
            onnx.helper.make_tensor(
                input_shape_name, onnx.TensorProto.INT64, [3], input_shape
            )
        )
        if layout.upper() == "NCHW":
            input_nhwc = input_name + "__nhwc"
            model.graph.node.insert(0, onnx.helper.make_node(
                "Reshape", [input_nhwc, input_shape_name], [rank3_input],
                name=input_name + "__unwrap_hwc",
            ))
            model.graph.node.insert(0, onnx.helper.make_node(
                "Transpose", [input_name], [input_nhwc],
                name=input_name + "__nchw_to_hwc", perm=[0, 2, 3, 1],
            ))
        else:
            model.graph.node.insert(0, onnx.helper.make_node(
                "Reshape", [input_name, input_shape_name], [rank3_input],
                name=input_name + "__unwrap_hwc",
            ))
    if len(output_shape) == 3:
        output_shape_name = output_name + "__shape4"
        model.graph.initializer.append(onnx.helper.make_tensor(
            output_shape_name, onnx.TensorProto.INT64, [4],
            rank4(output_shape),
        ))
        if layout.upper() == "NCHW":
            output_nchw = output_name + "__nchw"
            model.graph.node.append(onnx.helper.make_node(
                "Transpose", [rank3_output], [output_nchw],
                name=output_name + "__hwc_to_chw", perm=[0, 2, 1],
            ))
            model.graph.node.append(onnx.helper.make_node(
                "Reshape", [output_nchw, output_shape_name], [output_name],
                name=output_name + "__wrap_hwc",
            ))
        else:
            model.graph.node.append(onnx.helper.make_node(
                "Reshape", [rank3_output, output_shape_name], [output_name],
                name=output_name + "__wrap_hwc",
            ))
    for value, shape in ((graph_input, input_shape), (graph_output, output_shape)):
        if len(shape) != 3:
            continue
        value.type.tensor_type.shape.ClearField("dim")
        for size in rank4(shape):
            value.type.tensor_type.shape.dim.add().dim_value = size
    promoted = path.with_name(f"{path.stem}_rank4_hwc.onnx")
    onnx.save(model, promoted)
    return promoted


# -------------------------------------------------------------- tessellation


def _mla_output_names(mla_node, node_is_tuple, *, peel_terminal_reshape: bool = False) -> list[str]:
    """Compiler-visible MLA outputs.

    With ``peel_terminal_reshape`` (``--groot-boundaries``): AFE 2.1.3 can leave a
    graph-output reshape in the segment IR while lowering the actual MLA boundary
    to that reshape's producer; asking for the reshape then fails later with an
    output list naming only the producer. A one-input terminal reshape changes
    shape, not values, so its producer is the node to detessellate. Off by
    default so other graphs keep their established ELF bytes.
    """
    nodes = mla_node.ir.nodes
    output_node = nodes[mla_node.ir.output_node_name]
    names = list(output_node.input_node_names) if node_is_tuple(output_node) else [output_node.name]
    resolved = []
    for name in names:
        node, seen = nodes.get(name), set()
        while (peel_terminal_reshape and node is not None and name not in seen
               and "reshape" in name.rsplit("/", 1)[-1].lower() and len(node.input_node_names) == 1):
            seen.add(name)
            name = node.input_node_names[0]
            node = nodes.get(name)
        resolved.append(name)
    return resolved


def tessellation_parameters(quantized, *, peel_terminal_reshape: bool = False):
    """Direct HWC in / HWC16 out for the MLA segment.

    This is what lets the DevKit hand the MLA a plain contiguous buffer and read
    a de-tessellated one back, which is the whole reason `plan.cpp` can be a
    generic interpreter instead of per-model glue.

    It reaches into `quant_model._net.nodes["MLA_0"]`, a private attribute, and
    is the most fragile line in the compile path -- an afe upgrade that renames
    that node breaks it. Hence the explicit check: the failure is otherwise a
    KeyError with no context.
    """
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from afe.ir.node import node_is_tuple

    nodes = quantized._net.nodes
    if "MLA_0" not in nodes:
        raise RuntimeError(
            f"MLA_0 not found in the quantized net; nodes={list(nodes)[:30]}. "
            "The graph may have fallen back to the APU, or afe renamed the "
            "segment -- check the quantize log for unsupported operators."
        )
    mla_node = nodes["MLA_0"]

    params = {}
    inbound = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC
    )
    for input_name in mla_node.input_names:
        params[input_name] = dataclasses.replace(inbound)

    output_names = _mla_output_names(mla_node, node_is_tuple,
                                     peel_terminal_reshape=peel_terminal_reshape)
    outbound = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC16
    )
    for output_name in output_names:
        params[f"{output_name}_output"] = dataclasses.replace(outbound)
    return params


def external_layout_parameters(quantized, layout_name: str, *, peel_terminal_reshape: bool = False):
    """Force the same external layout on every MLA input and output."""
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from afe.ir.node import node_is_tuple

    nodes = quantized._net.nodes
    if "MLA_0" not in nodes:
        raise RuntimeError(
            "MLA_0 not found in the quantized net; cannot set an external DRAM layout"
        )
    mla_node = nodes["MLA_0"]
    template = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0), enable_mla=True,
        dram_layout=TensorDRAMLayout[layout_name],
    )
    params = {
        name: dataclasses.replace(template) for name in mla_node.input_names
    }
    output_names = _mla_output_names(mla_node, node_is_tuple,
                                     peel_terminal_reshape=peel_terminal_reshape)
    params.update({
        f"{name}_output": dataclasses.replace(template) for name in output_names
    })
    return params


def aligned_int8_tessellation_parameters(quantized):
    """INT8 MLA boundary with on-MLA padding for short vectors (GR00T W8A8 DiT).

    SDK 2.1.3 cannot pack a 255-byte HWC mask; HWC16 requests channel alignment
    on the MLA while the host boundary stays logical HWC. Batch-one vectors only.
    """
    import math

    from afe.apis.defines import TensorDRAMLayout

    params = external_layout_parameters(quantized, "HWC", peel_terminal_reshape=True)
    segment = quantized._net.nodes["MLA_0"]
    for name in segment.input_names:
        tensor = segment.ir.nodes[name].ir._quant_attrs.type
        if tensor.scalar.name != "int8":
            raise ValueError("aligned INT8 boundary requires INT8 MLA inputs")
        shape = tuple(tensor.shape)
        if math.prod(shape) % 16:
            if not (len(shape) == 2 and shape[0] == 1):
                raise ValueError(f"unsupported unaligned INT8 shape: {shape}")
            params[name] = dataclasses.replace(params[name], dram_layout=TensorDRAMLayout.HWC16)
    return params


# --------------------------------------------------------------------- driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polima-compile-tensor", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("modalix", "mlsoc"), default="modalix")

    parser.add_argument("--precision", choices=("int8", "bf16"), default="bf16",
                        help="sets both sides unless overridden individually")
    parser.add_argument("--activation-precision", choices=("int8", "bf16"))
    parser.add_argument("--weight-precision", choices=("int8", "bf16"))

    parser.add_argument("--calibration-npz", type=Path,
                        help="one array per input, shaped (N, *input_shape)")
    parser.add_argument("--calibration-raw-f32", type=Path,
                        help="flat float32 samples for a single-input graph")
    parser.add_argument("--calib-method", default="mse")
    parser.add_argument(
        "--groot-boundaries", action="store_true",
        help="GR00T graph boundaries: peel terminal output reshapes and HWC16-align "
             "short INT8 vectors (as the GR00T ModelSDK builds were compiled)",
    )
    parser.add_argument("--requant-mode", choices=("sima", "tflite"), default="sima")

    parser.add_argument("--model-layout", choices=("NCHW", "NHWC"), default="NCHW")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mla-tessellation", action="store_true")
    parser.add_argument(
        "--external-dram-layout", choices=("compiler", "HWC", "HWC16"),
        default="compiler", help="force one layout on every MLA input and output",
    )
    parser.add_argument(
        "--promote-rank3-hwc", action="store_true",
        help="wrap [N,W,C] graph I/O as [N,1,W,C] for explicit HWC layouts",
    )
    parser.add_argument("--retain-compile-dir", type=Path,
                        help="keep the compiler temp dir; this is where the ELF lands")
    parser.add_argument(
        "--exit-on-stable-elf", action="store_true",
        help="accept a retained ELF after compiler cleanup exceeds five minutes",
    )

    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--infer-shapes", action="store_true",
                        help="run ONNX shape inference after simplification")
    parser.add_argument("--no-compile", action="store_true",
                        help="quantize and save only, skip ELF generation")

    parser.add_argument("--input-names", nargs="+")
    parser.add_argument("--input-shapes", nargs="+", metavar="D0,D1,...")
    parser.add_argument("--input-types", nargs="+", choices=("float32", "int8"))
    parser.add_argument("--output-names", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from afe.apis.defines import (
        CalibrationMethod,
        RequantizationMode,
        bfloat16_scheme,
        default_quantization,
        gen1_target,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.error_handling_variables import enable_verbose_error_messages
    from afe.apis.loaded_net import load_model
    from afe.apis.release_v1 import get_model_sdk_version
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat

    args = build_parser().parse_args(argv)
    enable_verbose_error_messages()

    names, shapes, outputs = detect_io(args.model_path)
    if args.input_names:
        names = args.input_names
    if args.input_shapes:
        shapes = [tuple(int(p) for p in s.split(",") if p) for s in args.input_shapes]
    if args.output_names:
        outputs = args.output_names
    type_names = args.input_types or ["float32"] * len(names)
    if len(type_names) != len(names):
        raise SystemExit("--input-types must give exactly one type per input")
    input_types = [ScalarType[name] for name in type_names]

    stem = args.model_path.stem
    output = args.build_dir / stem
    output.mkdir(parents=True, exist_ok=True)

    sdk_version = get_model_sdk_version()
    print(f"[INFO] Model Compiler Version: {sdk_version}")
    use_known_host_check_fallback()
    print(f"[INFO] Target: {args.device}   Layout: {args.model_layout}")
    print(f"[INFO] Inputs: {list(zip(names, shapes, strict=True))}")
    print(f"[INFO] Outputs: {outputs}")
    print(f"[INFO] Output directory: {output}")

    calibration_shapes = list(shapes)
    original_shapes = dict(zip(names, shapes, strict=True))
    model_path = (
        args.model_path if args.no_simplify
        else prepare(args.model_path, names, shapes, infer_shapes=args.infer_shapes)
    )
    if args.promote_rank3_hwc:
        if args.external_dram_layout == "compiler":
            raise SystemExit("--promote-rank3-hwc requires --external-dram-layout")
        model_path = promote_rank3_hwc(model_path, args.model_layout)
        names, shapes, outputs = detect_io(model_path)

    importer = ImporterParams(
        format=ModelFormat.onnx,
        file_paths=[str(model_path)],
        input_names=list(names),
        input_shapes=[tuple(s) for s in shapes],
        input_types=input_types,
        layout=args.model_layout,
        output_names=list(outputs),
    )
    target = gen2_target if args.device == "modalix" else gen1_target
    loaded = load_model(importer, target=target)

    activation = args.activation_precision or args.precision
    weights = args.weight_precision or args.precision
    print(f"[INFO] Precision: activation={activation}, weights={weights}")

    if args.calibration_npz and args.calibration_raw_f32:
        raise SystemExit("pass at most one of --calibration-npz / --calibration-raw-f32")

    kind, source, note = calib.plan(
        activation, weights, args.calibration_npz, args.calibration_raw_f32
    )
    if note:
        print(f"[{'WARN' if 'drift' in note else 'INFO'}] {note}")

    samples = calib.build(
        kind, source, names, calibration_shapes if args.promote_rank3_hwc else shapes,
        layout=args.model_layout,
        types=dict(zip(names, type_names, strict=True)),
    )
    if args.promote_rank3_hwc:
        for sample in samples:
            for name in names:
                if len(original_shapes[name]) == 3:
                    sample[name] = sample[name].reshape(
                        sample[name].shape[0], 1, *sample[name].shape[1:]
                    )
    print(f"[INFO] Calibration: {kind}, {len(samples)} sample(s)")
    config = (
        default_quantization
        .with_activation_quantization(
            bfloat16_scheme() if activation == "bf16" else quantization_scheme(True, False, 8)
        )
        .with_weight_quantization(
            bfloat16_scheme() if weights == "bf16" else quantization_scheme(False, True, 8)
        )
        .with_requantization_mode(
            RequantizationMode.sima if args.requant_mode == "sima" else RequantizationMode.tflite
        )
        .with_calibration(CalibrationMethod.from_str(args.calib_method))
    )

    quantized = loaded.quantize(
        calibration_data=samples,
        quantization_config=config,
        any_shape_on_mla=True,
        automatic_layout_conversion=True,
        model_name=stem,
        log_level=logging.INFO,
    )
    quantized.save(model_name=stem, output_directory=str(output))
    print("[INFO] Quantized model saved")

    if args.no_compile:
        return 0

    if args.mla_tessellation and args.external_dram_layout != "compiler":
        raise SystemExit("--mla-tessellation and --external-dram-layout are mutually exclusive")
    peel = args.groot_boundaries
    if args.mla_tessellation and peel and activation == "int8":
        compile_layout = aligned_int8_tessellation_parameters(quantized)
    elif args.mla_tessellation:
        compile_layout = tessellation_parameters(
            quantized, peel_terminal_reshape=peel and args.model_layout != "NCHW")
    elif args.external_dram_layout != "compiler":
        compile_layout = external_layout_parameters(
            quantized, args.external_dram_layout, peel_terminal_reshape=peel)
    else:
        compile_layout = None
    watchdog = None
    if args.exit_on_stable_elf:
        if not args.retain_compile_dir:
            raise SystemExit("--exit-on-stable-elf requires --retain-compile-dir")
        watchdog = _start_elf_watchdog(args.retain_compile_dir / f"{stem}_stage1_mla.elf")

    try:
        quantized.compile(
            output_path=str(output),
            batch_size=args.batch_size,
            log_level=logging.INFO,
            tessellate_parameters=compile_layout,
            retained_temporary_directory_name=(
                str(args.retain_compile_dir) if args.retain_compile_dir else None
            ),
        )
    except Exception as error:  # re-raised below unless it is the MPK packager's
        retry_on_host_check_failure(error)
        elf = (args.retain_compile_dir / f"{stem}_stage1_mla.elf"
               if args.retain_compile_dir else None)
        if not mpk_only_failure(error, elf):
            raise
        print(f"[WARN] MPK packaging failed after the ELF was written ({error}); "
              f"keeping {elf}", flush=True)
        return 0
    finally:
        if watchdog is not None:
            watchdog.set()
    print("[INFO] Compilation complete")
    return 0


#: The compiler's host check of the generated code. Where it cannot run on this CPU,
#: the compiler's own switch SIMA_MLA_COMPILE_USE_JAX=0 runs it another way.
HOST_CHECK_FAILURE = "ofm_chk.mlc"


def _fallback_marker() -> Path:
    from afe.apis.release_v1 import get_model_sdk_version

    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "polima" / f"host-check-fallback-{get_model_sdk_version()}"


def use_known_host_check_fallback() -> None:
    """Start with the switch where this compiler version already needed it on this computer."""
    if "SIMA_MLA_COMPILE_USE_JAX" not in os.environ and _fallback_marker().is_file():
        os.environ["SIMA_MLA_COMPILE_USE_JAX"] = "0"


def retry_on_host_check_failure(error: BaseException) -> None:
    """Restart this compile once with SIMA_MLA_COMPILE_USE_JAX=0 if the host check failed."""
    if HOST_CHECK_FAILURE in str(error) and os.environ.get("SIMA_MLA_COMPILE_USE_JAX") != "0":
        marker = _fallback_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        print("[INFO] Compiling again with SIMA_MLA_COMPILE_USE_JAX=0 for this CPU.", flush=True)
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]],
                  {**os.environ, "SIMA_MLA_COMPILE_USE_JAX": "0"})


#: ModelSDK 2.1's MPK packager (mpk_parser/seq.py) indexes a list of input
#: names per buffer and runs off its end for tessellated graphs whose inputs
#: differ in shape -- GR00T N1.7's DiT pairs and action projectors. It runs
#: after code generation and the checker pass, so the retained ELF is complete;
#: only the .tar.gz, which PoLiMa does not load, is missing.
MPK_FAILURE = "Failed to process the MPK JSON file"


def mpk_only_failure(error: BaseException, elf: Path | None,
                     settle_seconds: float = 2.0) -> bool:
    """True when `error` is the MPK packager's and a finished ELF is on disk."""
    if MPK_FAILURE not in str(error) or elf is None:
        return False
    try:
        before = elf.stat()
        time.sleep(settle_seconds)
        after = elf.stat()
    except FileNotFoundError:
        return False
    return (before.st_size > 0 and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns)


def _start_elf_watchdog(path: Path, stable_seconds: float = 300.0) -> threading.Event:
    """Allow compiler cleanup to finish before accepting an unchanged retained ELF."""
    stopped = threading.Event()

    def watch() -> None:
        previous: tuple[int, int] | None = None
        stable_since: float | None = None
        while not stopped.wait(2.0):
            try:
                stat = path.stat()
                with path.open("rb") as stream:
                    valid = stat.st_size >= 64 and stream.read(4) == b"\x7fELF"
                current = (stat.st_size, stat.st_mtime_ns) if valid else None
            except OSError:
                current = None
            now = time.monotonic()
            if current is not None and current == previous:
                if stable_since is not None and now - stable_since >= stable_seconds:
                    if stopped.is_set():
                        return
                    print(f"[INFO] Compiler cleanup exceeded {stable_seconds:.0f}s; "
                          "retained ELF is unchanged", flush=True)
                    os._exit(0)
            else:
                previous = current
                stable_since = now if current is not None else None

    threading.Thread(target=watch, name="polima-elf-watchdog", daemon=True).start()
    return stopped


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------- persistent bindings


def persistent_main(argv: Sequence[str] | None = None) -> int:
    """Recompile a saved quantized graph (``--no-compile`` output) with one named
    persistent BF16 HWC binding per MLA input/output, for ``run_bound_chain``.

    No re-quantization. Writes contract.json (bindings), mpk.json, retained/ and,
    for several outputs, output_map.json (MLA output node -> ONNX output name, read
    from the saved network's output tuple).
    """
    import hashlib
    import json
    import math
    import tarfile

    parser = argparse.ArgumentParser(prog="compile_onnx.py persistent")
    parser.add_argument("--source", type=Path, required=True, help="saved quantized model dir")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx-outputs", nargs="+", default=[])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing existing output {args.output}")
    args.output.mkdir(parents=True)
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from afe.apis.model import Model
    from afe.ir.defines import get_expected_tensor_value
    from afe.ir.node import node_is_tuple

    model = Model.load(args.name, args.source, include_unquantized_net=False)
    mla = model._net.nodes["MLA_0"]
    outputs = _mla_output_names(mla, node_is_tuple, peel_terminal_reshape=True)
    params, contract = {}, {"name": args.name, "inputs": [], "outputs": []}
    for direction, names in (("input", list(mla.input_names)), ("output", outputs)):
        for index, name in enumerate(names):
            tensor = get_expected_tensor_value(mla.ir.nodes[name].get_type().output)
            if tensor.scalar.name != "bfloat16":
                raise ValueError(f"expected a BF16 boundary: {name}: {tensor}")
            binding = f"{direction}_{index}/{name}"
            params[name if direction == "input" else name + "_output"] = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0), enable_mla=True, dram_layout=TensorDRAMLayout.HWC,
                persistent_mem_name=binding)
            contract[direction + "s"].append({
                "name": name, "binding": binding, "shape": list(tensor.shape),
                "elements": math.prod(tensor.shape), "dtype": "bfloat16", "layout": "HWC"})
    contract["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in args.source.glob(args.name + ".sima*") if p.is_file()}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    use_known_host_check_fallback()
    try:
        model.compile(args.output / "compiled", compress=True, preserve=False, tessellate_parameters=params,
                      retained_temporary_directory_name=args.output / "retained", log_level=20,
                      deployable=False)
    except RuntimeError as error:
        retry_on_host_check_failure(error)
        raise
    archives = list((args.output / "compiled").rglob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one MPK, found {archives}")
    with tarfile.open(archives[0]) as archive:
        meta = json.load(archive.extractfile(next(n for n in archive.getnames() if n.endswith("_mpk.json"))))
    (args.output / "mpk.json").write_text(json.dumps(meta, indent=2) + "\n")
    # A persistent ELF covers MLA_0 only: a second MLA segment would be silently lost.
    if sum(1 for p in meta["plugins"] if p["processor"] == "MLA") != 1:
        raise SystemExit(f"{args.name}: compiler split the graph into several MLA segments")
    if len(args.onnx_outputs) > 1:
        net = model._net
        top = net.nodes[net.output_node_name]
        if len(top.input_node_names) != len(args.onnx_outputs):
            raise SystemExit(f"{len(top.input_node_names)} network outputs for {args.onnx_outputs}")
        mapping = {}
        for position, producer in enumerate(top.input_node_names):
            node = net.nodes[producer]
            while type(node.ir.attrs).__name__ != "TupleGetItemAttrs":
                if len(node.input_node_names) != 1:
                    raise SystemExit(f"unexpected multi-input node {producer} after MLA_0")
                node = net.nodes[node.input_node_names[0]]
            if node.input_node_names != ["MLA_0"]:
                raise SystemExit(f"output {position} does not come from MLA_0")
            mapping[outputs[node.ir.attrs.index]] = args.onnx_outputs[position]
        if sorted(mapping) != sorted(o["name"] for o in contract["outputs"]):
            raise SystemExit("persistent outputs differ from the mapped MLA outputs")
        (args.output / "output_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print("[INFO] Persistent compilation complete", flush=True)
    return 0
