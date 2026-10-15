"""Calibration sample construction.

This is the part of compilation that differs most between policies, and the part
most likely to be silently wrong: a bad calibration set does not raise, it just
produces a quantized model whose outputs drift. So it lives here, depends on
numpy only, and is covered by unit tests.

The three sources are a union, not a choice of one:

  npz      ACT and GR00T -- one array per model input, leading axis = sample.
  raw_f32  a flat .f32 file for a single-input graph, samples concatenated.
           Used where the calibration data is an activation captured from a
           previous stage rather than a dataset sample.
  random   the fallback when no data is supplied.

## The layout rule

`ImporterParams(layout=...)` tells afe how to read the *model*. It does not
describe the calibration data: afe's quantizer wants NHWC regardless, which is
why the curated `quantize_compile` helper transposes unconditionally. That
unconditional transpose corrupts rank-2/rank-3 action-side tensors that have no
spatial axes at all.

The correct rule is narrower: transpose only a rank-4 tensor, and only when the
model is NCHW. Everything else passes through untouched. It is applied uniformly to all
three sources here so the source cannot change the meaning of the layout flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

Shape = Sequence[int]
Sample = dict[str, np.ndarray]


def to_calibration_layout(value: np.ndarray, layout: str) -> np.ndarray:
    """NCHW rank-4 -> NHWC. Any other rank or layout is returned unchanged."""
    if value.ndim == 4 and layout.upper() == "NCHW":
        value = value.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(value)


def from_npz(path: str | Path, names: Sequence[str], shapes: Sequence[Shape],
             layout: str = "NCHW") -> list[Sample]:
    """One array per input, shaped (N, *input_shape).

    The shape check is worth keeping strict. An npz whose leading axis is the
    sample count is indistinguishable at the API level from one that is missing
    it, and getting that wrong quantizes against a single reshaped sample rather
    than N of them -- which succeeds, and produces a worse model.
    """
    path = Path(path)
    data = np.load(path)
    missing = sorted(set(names) - set(data.files))
    if missing:
        raise KeyError(f"{path} is missing calibration arrays: {missing}")

    count: int | None = None
    for name, shape in zip(names, shapes, strict=True):
        array = data[name]
        if array.ndim != len(shape) + 1 or tuple(array.shape[1:]) != tuple(shape):
            raise ValueError(
                f"{path}:{name} has shape {array.shape}, expected (N, {tuple(shape)})"
            )
        if count is None:
            count = array.shape[0]
        elif array.shape[0] != count:
            raise ValueError(
                f"{path}: inputs disagree on sample count "
                f"({name} has {array.shape[0]}, expected {count})"
            )
    if not count:
        raise ValueError(f"{path} contains no calibration samples")

    return [
        {
            name: to_calibration_layout(data[name][index].astype(np.float32), layout)
            for name, shape in zip(names, shapes, strict=True)
        }
        for index in range(int(count))
    ]


def from_raw_f32(path: str | Path, names: Sequence[str], shapes: Sequence[Shape],
                 layout: str = "NCHW") -> list[Sample]:
    """A flat float32 file for a single-input graph, samples concatenated.

    For a stage whose calibration input is an activation dumped by the
    preceding stage rather than anything that came from a dataset. There is no shape metadata in the file, so the
    only available check is that the size divides evenly into whole samples.
    """
    path = Path(path)
    if len(names) != 1:
        raise ValueError(
            f"raw f32 calibration needs exactly one input, got {list(names)}; "
            "use an npz for multi-input graphs"
        )
    shape = tuple(shapes[0])
    per_sample = int(np.prod(shape))
    flat = np.fromfile(path, dtype=np.float32)
    if flat.size == 0 or flat.size % per_sample:
        raise ValueError(
            f"{path} holds {flat.size} float32 values, which is not a whole "
            f"number of {per_sample}-element samples for shape {shape}"
        )
    stacked = flat.reshape((-1, *shape))
    return [{names[0]: to_calibration_layout(sample, layout)} for sample in stacked]


def random(names: Sequence[str], shapes: Sequence[Shape], layout: str = "NCHW",
           samples: int = 1, seed: int = 123,
           types: Mapping[str, str] | None = None) -> list[Sample]:
    """Fallback when no calibration data exists.

    Fine for a bf16 graph, where "calibration" only shapes the graph and the
    weights keep full range. For int8 it will produce a numerically poor model,
    so callers that request random data for an int8 compile deserve the warning
    the driver emits.
    """
    generator = np.random.default_rng(seed)
    out: list[Sample] = []
    for _ in range(max(1, samples)):
        sample: Sample = {}
        for name, shape in zip(names, shapes, strict=True):
            if (types or {}).get(name) == "int8":
                value = generator.integers(-16, 17, size=tuple(shape)).astype(np.int8)
            else:
                value = generator.standard_normal(tuple(shape)).astype(np.float32)
            sample[name] = to_calibration_layout(value, layout)
        out.append(sample)
    return out


def plan(activation: str, weights: str, npz: str | Path | None = None,
         raw_f32: str | Path | None = None) -> tuple[str, str | Path | None, str]:
    """Decide which calibration source a compile actually needs.

    bf16 is a float format: there are no scales to fit, so calibration data
    cannot change the generated code. afe still wants one sample to trace
    shapes, but the values are ignored: compiling ACT's decoder_action_tail
    with 8 real dataset samples or with 1 random sample gives the identical ELF.

    This matters beyond tidiness: it is what lets a pure-bf16 build skip the
    dataset entirely, and skip reading calibration files that run to hundreds of
    megabytes.
    """
    if activation == "bf16" and weights == "bf16":
        note = "bf16 has no scales to fit" if (npz or raw_f32) else ""
        return "random", None, note
    if npz:
        return "npz", npz, ""
    if raw_f32:
        return "raw_f32", raw_f32, ""
    return "random", None, "int8 with random calibration data -- expect drift"


def build(kind: str, path: str | Path | None, names: Sequence[str],
          shapes: Sequence[Shape], layout: str = "NCHW", samples: int = 8,
          types: Mapping[str, str] | None = None) -> list[Sample]:
    """Dispatch on `CalibrationSource.kind`, falling back to random with no path."""
    if kind == "random" or path is None:
        return random(names, shapes, layout, samples=1, types=types)
    if kind == "npz":
        return from_npz(path, names, shapes, layout)
    if kind == "raw_f32":
        return from_raw_f32(path, names, shapes, layout)
    raise ValueError(f"unknown calibration kind {kind!r}; expected npz/raw_f32/random")
