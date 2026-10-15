"""A pure-Python implementation of the plan interpreter.

Mirrors runtime/src/plan.cpp opcode for opcode. It exists for three reasons:

  * the host-side opcodes (pack, gather_strided, slice, scale, ...) can be
    verified against the real per-stage golden .f32 files *without an MLA*, so
    plan semantics are proven before anything is built on the board;
  * `polima run --stub` can serve a bundle on a workstation, with graph outputs
    supplied from the goldens, to exercise clients end to end; and
  * when a board result disagrees with expectation, running the same plan here
    localizes the disagreement to a specific step.

It is a reference, not a fast path: no MLA, no bf16 round-trip.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from polima.util.jsonio import read_json


class PlanError(RuntimeError):
    pass


@dataclass
class StubPlan:
    """A plan.json loaded for host-side execution."""

    buffers: dict[str, int]
    steps: list[dict]
    result: str
    wire: dict = field(default_factory=dict)
    constants_dir: Path | None = None
    #: run_device_chain's plan-wide device buffers (BF16 values held as float32).
    device_buffers: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, bundle_root: str | Path) -> StubPlan:
        root = Path(bundle_root)
        data = read_json(root / "plan.json")
        return cls(
            buffers=dict(data["buffers"]),
            steps=list(data["steps"]),
            result=data["result"],
            wire=data.get("wire", {}),
            constants_dir=root / "constants",
            device_buffers=dict(data.get("device_buffers", {})),
        )

    def run(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_fn: Callable[[str, np.ndarray], np.ndarray] | None = None,
        request_flags: int = 0,
        task: int | None = None,
    ) -> np.ndarray:
        """Execute the plan.

        `graph_fn(graph_name, input_array) -> output_array` supplies the MLA.
        Without it, `run_elf` steps raise -- pass one backed by golden .f32 files
        (see `golden_graph_fn`) to run a full pipeline on the host.
        `task` names the request's task id; it is the header flags the native
        server passes to Plan::execute, so it overrides `request_flags`.
        """
        if task is not None:
            request_flags = task
        state: dict[str, np.ndarray] = {
            name: np.zeros(size, dtype=np.float32) for name, size in self.buffers.items()
        }
        for name, array in inputs.items():
            if name not in state:
                raise PlanError(f"input {name!r} is not a declared buffer")
            flat = np.asarray(array, dtype=np.float32).reshape(-1)
            if flat.size != state[name].size:
                raise PlanError(
                    f"input {name!r}: expected {state[name].size} elements, got {flat.size}"
                )
            state[name] = flat.copy()

        # Device memory starts empty for every request; the native runtime
        # proves at load time that no read depends on an earlier request.
        self._device = {name: np.full(size, np.nan, dtype=np.float32)
                        for name, size in self.device_buffers.items()}
        for index, step in enumerate(self.steps):
            try:
                self._run_step(step, state, graph_fn, request_flags)
            except Exception as exc:
                raise PlanError(f"step {index} ({step['op']} -> {step['out']}): {exc}") from exc
        return state[self.result]

    # -- opcodes; keep in lockstep with runtime/src/plan.cpp ------------------

    def _run_step(self, step, state, graph_fn, request_flags: int) -> None:
        op = step["op"]
        args = step.get("args", {})
        out = state[step["out"]]

        def source() -> np.ndarray:
            name = args.get("src") or (args.get("in") or [None])[0]
            if name is None:
                raise PlanError("step names no source buffer")
            return state[name]

        if op in ("run_elf", "run_elf_chain"):
            if graph_fn is None:
                label = args.get("graph") or args.get("graphs")
                raise PlanError(f"no graph_fn supplied for {label!r}")
            joined = np.concatenate([state[n] for n in args["in"]])
            produced = joined
            graphs = [args["graph"]] if op == "run_elf" else args["graphs"]
            for graph in graphs:
                produced = np.asarray(graph_fn(graph, produced), dtype=np.float32).reshape(-1)
            if produced.size != out.size:
                raise PlanError(
                    f"graph {graphs[-1]!r} produced {produced.size} elements, "
                    f"buffer holds {out.size}"
                )
            state[step["out"]] = produced

        elif op == "run_dit_chain":
            # DiTRunnerChain: graph i reads [hidden, temb, backbone, mask_i] with
            # the two masks alternating, and hands its hidden state to graph i+1.
            if graph_fn is None:
                raise PlanError("no graph_fn supplied for run_dit_chain")
            hidden, temb, backbone, even, odd = (state[n] for n in args["in"])
            for index, graph in enumerate(args["graphs"]):
                mask = odd if index % 2 else even
                hidden = np.asarray(graph_fn(graph, np.concatenate([hidden, temb, backbone, mask])),
                                    dtype=np.float32).reshape(-1)
            state[step["out"]] = hidden

        elif op == "run_device_chain":
            if graph_fn is None:
                raise PlanError("no graph_fn supplied for run_device_chain")
            device = self._device

            def span(value):
                name, offset, count = (list(value) + [0, None])[:3]
                if count is None:
                    count = device[name].size - offset
                return name, offset, count

            for item in args.get("uploads", []):
                count, offset = item["count"], item.get("src_offset", 0)
                if "sidecar" in item:
                    values = self._constant(item["sidecar"])
                    if item.get("source_stride"):
                        if not 0 <= request_flags < item["selector_count"]:
                            raise PlanError("request flags outside task catalog")
                        offset += request_flags * item["source_stride"]
                else:
                    values = state[item["src"]]
                start = item.get("dst_offset", 0)
                device[item["dst"]][start:start + count] = _bf16(values[offset:offset + count])
            for call in args["calls"]:
                joined = np.concatenate([device[n][o:o + c] for n, o, c in map(span, call["in"])])
                if np.isnan(joined).any():
                    raise PlanError(f"{call['graph']} reads device memory never written")
                produced = _bf16(np.asarray(graph_fn(call["graph"], joined)).reshape(-1))
                name, offset, count = span(call["out"][0])
                if produced.size != count:
                    raise PlanError(f"{call['graph']} produced {produced.size}, binding holds {count}")
                device[name][offset:offset + count] = produced
            for item in args.get("downloads", []):
                start, count = item.get("src_offset", 0), item["count"]
                values = device[item["src"]][start:start + count]
                if not np.isfinite(values).all():
                    raise PlanError(f"device chain produced a non-finite value in {item['dst']}")
                target = state[item["dst"]].copy()
                at = item.get("dst_offset", 0)
                target[at:at + count] = values
                state[item["dst"]] = target

        elif op == "pack":
            packed = np.zeros(args.get("size", out.size), dtype=np.float32)
            for part in args["parts"]:
                start = part["dst_offset"]
                count = part["count"]
                # A part reads either a live buffer or a constants/ sidecar.
                origin = (self._constant(part["src"]) if part.get("sidecar")
                          else state[part["src"]])
                source_offset = int(part.get("src_offset", 0))
                source_stride = int(part.get("source_stride", 0))
                selector_count = int(part.get("selector_count", 0))
                if source_stride:
                    if selector_count <= 0:
                        raise PlanError("indexed pack part has no selector_count")
                    if not 0 <= request_flags < selector_count:
                        raise PlanError(
                            f"request flags {request_flags} outside task catalog "
                            f"[0, {selector_count})"
                        )
                    source_offset += request_flags * source_stride
                if source_offset < 0 or source_offset + count > origin.size:
                    raise PlanError(
                        f"pack reads [{source_offset}:{source_offset + count}] past "
                        f"source {part['src']!r} ({origin.size})"
                    )
                if start < 0 or start + count > packed.size:
                    raise PlanError(
                        f"pack writes [{start}:{start + count}] past output ({packed.size})"
                    )
                packed[start:start + count] = origin[source_offset:source_offset + count]
            state[step["out"]] = packed

        elif op == "slice":
            offset = args.get("offset", 0)
            count = args.get("count", out.size)
            state[step["out"]] = source()[offset:offset + count].copy()

        elif op == "gather_strided":
            # Scatter as well as gather: dst_stride defaults to `take` (ACT's
            # contiguous case) and a source stride of 0 broadcasts one span into
            # every block, which is how one shared span (e.g. a time embedding)
            # reaches every action token.
            stride, take, count = args["stride"], args["take"], args["count"]
            dst_stride = args.get("dst_stride") or take
            dst_offset = args.get("dst_offset", 0)
            values = source()
            target = (np.zeros(out.size, dtype=np.float32) if args.get("clear")
                      else out.copy())
            for index in range(count):
                read_at = index * stride
                write_at = index * dst_stride + dst_offset
                target[write_at:write_at + take] = values[read_at:read_at + take]
            state[step["out"]] = target

        elif op == "pixel_unshuffle":
            # Eagle's channel fold, between the vision chain and the connector:
            # a grid x grid map of C channels becomes (grid/f)^2 tokens of C*f^2.
            # Written as the composition torch.pixel_unshuffle performs, because
            # the equivalent index arithmetic is easy to transpose by accident
            # and a transposed fold still produces finite, wrong features.
            height = args.get("height", args.get("grid"))
            width = args.get("width", height)
            channels, factor = args["channels"], args["factor"]
            folded = source().reshape(height, width, channels)
            folded = folded.reshape(
                height // factor, factor, width // factor, factor, channels
            )
            # (row, col, dy, dx, c) -> flatten the trailing three into C*f^2,
            # matching pixel_unshuffle's (c*f^2 + dy*f + dx) channel order.
            folded = folded.transpose(0, 2, 1, 3, 4)
            folded = folded.transpose(0, 1, 4, 2, 3)
            state[step["out"]] = folded.reshape(-1).astype(np.float32)

        elif op == "scale":
            state[step["out"]] = source() * np.float32(args["scalar"])

        elif op == "sincos_state":
            values = source()
            widths = args["group_widths"]
            if (not widths or any(w <= 0 for w in widths)
                    or sum(widths) != values.size or 2 * values.size > out.size):
                raise PlanError("sincos_state shape mismatch")
            encoded = np.zeros_like(out)
            start = target = 0
            for width in widths:
                group = values[start:start + width]
                encoded[target:target + width] = np.sin(group)
                encoded[target + width:target + 2 * width] = np.cos(group)
                start += width
                target += 2 * width
            state[step["out"]] = encoded

        elif op == "clip_unit":
            state[step["out"]] = np.clip(source(), -1.0, 1.0).astype(np.float32)

        elif op == "matvec":
            x = source()
            weights = self._constant(args["weights"])
            rows = args.get("rows", out.size)
            cols = args.get("cols", x.size)
            result = weights.reshape(rows, cols) @ x[:cols]
            if args.get("bias"):
                result = result + self._constant(args["bias"])[:rows]
            state[step["out"]] = result.astype(np.float32)

        elif op == "sincos_time":
            # period = min * (max/min)^(i/(half-1)); angle = t * 2pi/period.
            # The exponent divides by half-1 so the last entry lands exactly on
            # max_period, and the 2pi belongs in the angle -- both are easy to
            # get subtly wrong and neither raises when wrong.
            timestep = np.float32(args["scalar"])
            half = out.size // 2
            min_period = np.float32(args.get("min_period") or 1.0)
            max_period = np.float32(args.get("max_period") or min_period)
            fraction = np.arange(half, dtype=np.float32) / np.float32(max(half - 1, 1))
            period = min_period * np.power(max_period / min_period, fraction)
            angle = timestep * (np.float32(2.0 * np.pi) / period)
            state[step["out"]] = np.concatenate(
                [np.sin(angle), np.cos(angle)]
            ).astype(np.float32)

        elif op == "euler":
            state[step["out"]] = out - np.float32(args["scalar"]) * source()

        elif op in ("normalize", "denormalize"):
            values = source()
            mean = self._constant(args["mean"])
            deviation = self._constant(args["std"])
            tiled_mean = np.resize(mean, values.size)
            tiled_std = np.resize(deviation, values.size)
            # Optional clamp, mirroring plan.cpp: normalize clamps its result,
            # denormalize its input -- where GR00T's min/max scaling clips.
            clip = "clip_min" in args or "clip_max" in args
            low = np.float32(args.get("clip_min", -np.inf))
            high = np.float32(args.get("clip_max", np.inf))
            if op == "normalize":
                result = (values - tiled_mean) / tiled_std
                if clip:
                    result = np.clip(result, low, high)
            else:
                if clip:
                    values = np.clip(values, low, high)
                result = values * tiled_std + tiled_mean
            state[step["out"]] = result.astype(np.float32)

        else:
            raise PlanError(f"unknown opcode {op!r}")

    def _constant(self, name: str) -> np.ndarray:
        if self.constants_dir is None:
            raise PlanError(f"plan needs constant {name!r} but no constants dir is set")
        path = self.constants_dir / name
        if not path.is_file():
            raise PlanError(f"missing sidecar {path}")
        return np.fromfile(path, dtype="<f4")


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 (nearest even), as polima::to_bf16 does on upload."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return (rounded & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


def golden_graph_fn(stages_dir: str | Path) -> Callable[[str, np.ndarray], np.ndarray]:
    """Back `run_elf` with recorded per-graph outputs.

    The ACT build ships `<graph>_output.f32` for every stage plus
    `vision_output_{0,1}.f32`, so the whole pipeline can be replayed on the host
    with no accelerator. Vision is called twice with different images, so calls
    to it are served in order.
    """
    stages = Path(stages_dir)
    calls: dict[str, int] = {}

    def resolve(graph: str, _input: np.ndarray) -> np.ndarray:
        occurrence = calls.get(graph, 0)
        calls[graph] = occurrence + 1
        candidates = [
            stages / f"{graph}_output_{occurrence}.f32",
            stages / f"vision_output_{occurrence}.f32" if graph == "vision_backbone" else None,
            stages / f"{graph}_output.f32",
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return np.fromfile(candidate, dtype="<f4")
        raise PlanError(f"no golden output for graph {graph!r} (call {occurrence}) in {stages}")

    return resolve


def euler_reference(x: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    """Kept beside the opcode so the Euler integration can be unit-tested."""
    return (x - np.float32(dt) * velocity).astype(np.float32)


__all__ = ["PlanError", "StubPlan", "euler_reference", "golden_graph_fn", "math"]
