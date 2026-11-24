"""Keep GR00T activations on the MLA between graphs (`run_device_chain`).

Rewrites the plan of an existing, validated GR00T N1.6 or N1.7 bundle so that
graph-to-graph activations stay in device memory instead of round-tripping
through float32 on the CPU at every graph boundary. No ELF is recompiled or
changed: a producer's output binding is pointed at the hidden-state part of
the consumer's packed input buffer, whose remaining inputs (attention masks,
rotary cos/sin) are uploaded once and stay resident behind it.

What stays on the host, because the host does arithmetic there:

* N1.6: the pixel-unshuffle between the vision chain and the connector.
* N1.7: the deepstack additions after language layers 0, 1 and 2 (float32
  adds, then a BF16 rounding when the next graph is fed). Layers 1 and 2 keep
  their ordinary per-graph path; layer 0 runs on the device and its output is
  downloaded for the add.
* The denoising loop's Euler steps and the state/action projections.

Levels (any subset):

* ``lm``     -- N1.6: 8 Qwen graphs + output norm (with ``connector``: also the
  two connector graphs, writing straight into the image-token slots of the
  first Qwen graph's input). N1.7: language 03..15, vlln and the 4 VL layers.
* ``vision`` -- N1.7 only: both cameras' 17-graph vision stacks (deepstack
  side outputs downloaded for the host adds), vision_merge writing into the
  image slots of language layer 0's input, and language layer 0.
* ``dit``    -- the four persistent-DiT passes as device chains that read the
  backbone output where the LM chain left it (no download, no re-upload).

The equivalence argument is byte-level: BF16 -> float32 -> BF16 is the
identity for finite values, and every value a graph now reads from device
memory is a value the host would have decoded and re-encoded unchanged.
`verify_equivalence` checks this on the real plans with the reference
executor and BF16-rounding stand-in graphs; the native runtime additionally
checks request isolation at load time.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from polima.bundle.layout import compute_bundle_id, parse_bundle_id

N16_DIT = [f"dit_blocks_{i:02d}_{i + 1:02d}" for i in range(0, 32, 2)]
N16_QWEN = [f"eagle_qwen_{i:02d}_{i + 1:02d}" for i in range(0, 16, 2)]
N17_LANGUAGE = ["language_00_00", "language_01_01", "language_02_02", "language_03_04",
                "language_05_06", "language_07_08", "language_09_10", "language_11_12",
                "language_13_14", "language_15_15"]
N17_VL = ["vlln", "vl_00", "vl_01", "vl_02", "vl_03"]
N17_VISION = ["vision_patch", "vision_00_01", "vision_02_03", "vision_04_05", "deepstack_merge_0",
              "vision_06_07", "vision_08_09", "vision_10_11", "deepstack_merge_1",
              "vision_12_13", "vision_14_15", "vision_16_17", "deepstack_merge_2",
              "vision_18_19", "vision_20_21", "vision_22_23", "vision_merge"]
ACT_VISION = "vision_backbone"
ACT_STEM = "encoder_layer_00_stem"
ACT_LAYERS = ["encoder_layer_01", "encoder_layer_02", "encoder_layer_03"]
ACT_DECODER = "decoder_action_tail"
LEVELS = ("vision", "lm", "connector", "dit", "denoise", "act")


class RewriteError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RewriteError(message)


def family(graph_names: Iterable[str]) -> str:
    names = set(graph_names)
    if set(N16_QWEN) <= names and "eagle_output_norm" in names:
        return "n1.6"
    if set(N17_LANGUAGE) <= names and set(N17_VL) <= names:
        return "n1.7"
    if {ACT_VISION, ACT_STEM, ACT_DECODER, *ACT_LAYERS} == names:
        return "act"
    raise RewriteError("bundle is neither a GR00T N1.6, N1.7 nor an ACT plan")


class _Program:
    """One run_device_chain step under construction."""

    def __init__(self, label: str, out: str):
        self.label, self.out = label, out
        self.uploads: list[dict] = []
        self.calls: list[dict] = []
        self.downloads: list[dict] = []

    def upload_host(self, dst, dst_offset, count, src, src_offset=0, cache=False):
        self.uploads.append({"dst": dst, "dst_offset": dst_offset, "count": count,
                             "src": src, "src_offset": src_offset, "cache": cache})

    def upload_sidecar(self, dst, dst_offset, count, part: dict, extra_offset=0, cache=True):
        """Upload from a constants/ sidecar, honouring a pack part's task selection."""
        item = {"dst": dst, "dst_offset": dst_offset, "count": count, "sidecar": part["src"],
                "src_offset": int(part.get("src_offset", 0)) + extra_offset, "cache": cache}
        if part.get("source_stride"):
            item["source_stride"] = int(part["source_stride"])
            item["selector_count"] = int(part["selector_count"])
        self.uploads.append(item)

    def call(self, graph, inputs, output):
        self.calls.append({"graph": graph, "in": [list(x) for x in inputs], "out": [list(output)]})

    def download(self, src, src_offset, count, dst, dst_offset=0):
        self.downloads.append({"src": src, "src_offset": src_offset, "count": count,
                               "dst": dst, "dst_offset": dst_offset})

    def step(self) -> dict:
        args = {"label": self.label, "calls": self.calls}
        if self.uploads:
            args["uploads"] = self.uploads
        if self.downloads:
            args["downloads"] = self.downloads
        return {"op": "run_device_chain", "out": self.out, "args": args}


def _sidecar_pack(step: dict) -> dict:
    parts = step.get("args", {}).get("parts", [])
    _require(step["op"] == "pack" and len(parts) == 1 and parts[0].get("sidecar")
             and parts[0]["dst_offset"] == 0, f"expected a one-part sidecar pack: {step}")
    return parts[0]


def _reads(step: dict) -> set[str]:
    args = step.get("args", {})
    names = set(args.get("in", []))
    if "src" in args:
        names.add(args["src"])
    for part in args.get("parts", []):
        if not part.get("sidecar"):
            names.add(part["src"])
    for upload in args.get("uploads", []):
        if "src" in upload:
            names.add(upload["src"])
    if step["op"] == "euler":  # out -= dt * src reads its output too
        names.add(step["out"])
    if step["op"] == "gather_strided" and not args.get("clear"):
        names.add(step["out"])
    return names


def _writes(step: dict) -> set[str]:
    names = {step["out"]}
    for download in step.get("args", {}).get("downloads", []):
        names.add(download["dst"])
    return names


def _splice(plan: dict, removed: set[int], program_steps: list[tuple[int, dict]]) -> dict:
    """Drop `removed` step indices and insert programs before original index i."""
    steps = plan["steps"]
    inserts: dict[int, list[dict]] = {}
    for index, step in program_steps:
        inserts.setdefault(index, []).append(step)
    result = []
    for index, step in enumerate(steps):
        result.extend(inserts.pop(index, []))
        if index not in removed:
            result.append(step)
    _require(not inserts, "program inserted past the end of the plan")
    plan = copy.deepcopy(plan)
    plan["steps"] = result
    return plan


def _drop_dead_sidecar_packs(plan: dict, candidates: set[str]) -> dict:
    """Remove one-part sidecar packs whose buffer nothing reads any more."""
    read = set()
    for step in plan["steps"]:
        read |= _reads(step)
    read |= {plan["result"]}
    read |= {t["name"] for t in plan.get("wire", {}).get("request_tensors", [])}
    steps = [s for s in plan["steps"]
             if not (s["op"] == "pack" and s["out"] in candidates and s["out"] not in read)]
    plan = copy.deepcopy(plan)
    plan["steps"] = steps
    return plan


def _find(steps: list[dict], op: str, **match) -> list[int]:
    hits = []
    for index, step in enumerate(steps):
        if step["op"] != op:
            continue
        args = step.get("args", {})
        if all((step["out"] if key == "out" else args.get(key)) == value
               for key, value in match.items()):
            hits.append(index)
    return hits


def _one(steps, op, **match) -> int:
    hits = _find(steps, op, **match)
    _require(len(hits) == 1, f"expected exactly one {op} {match}, found {hits}")
    return hits[0]


def _complement(total: int, slots: list[tuple[int, int]]) -> list[tuple[int, int]]:
    regions, cursor = [], 0
    for start, count in sorted(slots):
        _require(start >= cursor, "overlapping image slots")
        if start > cursor:
            regions.append((cursor, start - cursor))
        cursor = start + count
    if cursor < total:
        regions.append((cursor, total - cursor))
    return regions


# ------------------------------------------------------------------- DiT


def _last_writer(steps: list[dict], before: int, name: str) -> int:
    for index in range(before - 1, -1, -1):
        if steps[index]["out"] == name:
            return index
    raise RewriteError(f"nothing writes {name!r} before step {before}")


def _only_readers(steps: list[dict], name: str, allowed: set[int]) -> None:
    readers = {i for i, s in enumerate(steps) if name in _reads(s)}
    _require(readers <= allowed, f"{name!r} is also read by steps {sorted(readers - allowed)}")


def _dit_programs(plan: dict, graphs: dict, backbone_device: str | None,
                  device: dict[str, int], fuse: bool = False
                  ) -> tuple[set[int], list[tuple[int, dict]]]:
    """Express every run_dit_chain step as a device chain reading `backbone_device`.

    With `fuse`, each denoising pass becomes one program: state_project (first
    pass) and action_project write the two parts of the DiT input in place,
    and the last DiT block writes into action_tail's packed input, so only the
    velocity comes back to the host for the Euler step.
    """
    steps = plan["steps"]
    sizes = plan["buffers"]
    chains = _find(steps, "run_dit_chain")
    _require(len(chains) == 4, f"expected four persistent DiT passes, found {len(chains)}")
    bindings = graphs[N16_DIT[0]].get("persistent_input_elements")
    _require(bindings is not None and len(bindings) == 4, "DiT graphs lack persistent bindings")
    for name in N16_DIT:
        _require(graphs[name].get("persistent_input_elements") == bindings,
                 f"{name}: persistent bindings differ")
    hidden, temb, backbone, mask = bindings
    device.update({"dit_in": hidden, "dit_a": hidden, "dit_b": hidden, "dit_temb": temb,
                   "dit_mask_even": mask, "dit_mask_odd": mask})
    if backbone_device is None:
        device["dit_backbone"] = backbone
        backbone_device = "dit_backbone"
    _require(device[backbone_device] == backbone, "backbone device buffer size mismatch")
    removed: set[int] = set(chains)
    programs = []
    state_done = False
    for number, index in enumerate(chains):
        step = steps[index]
        _require(step["args"]["graphs"] == N16_DIT, "unexpected DiT graph order")
        h, t, b, m_even, m_odd = step["args"]["in"]
        _require([sizes[h], sizes[t], sizes[b], sizes[m_even], sizes[m_odd], sizes[step["out"]]]
                 == [hidden, temb, backbone, mask, mask, hidden], "DiT buffer sizes mismatch")
        program = _Program(f"dit_{number}", step["out"])
        # Every host input of the pass (action inputs, temb, masks) is written
        # before the DiT step and the outputs are read only after action_tail.
        insert_at = index
        tail = None
        if fuse:
            # hidden = pack[state_features @0, action_features @state]
            pack = _last_writer(steps, index, h)
            parts = steps[pack]["args"].get("parts", [])
            _require(steps[pack]["op"] == "pack" and len(parts) == 2
                     and not any(p.get("sidecar") or p.get("src_offset") for p in parts)
                     and parts[0]["dst_offset"] == 0
                     and parts[1]["dst_offset"] == parts[0]["count"]
                     and parts[0]["count"] + parts[1]["count"] == hidden,
                     f"pass {number}: DiT input is not [state | action] features")
            state_name, action_name = parts[0]["src"], parts[1]["src"]
            project = _last_writer(steps, pack, action_name)
            _require(steps[project]["op"] == "run_elf"
                     and steps[project]["args"]["graph"] == "action_project",
                     f"pass {number}: action features not from action_project")
            ap = graphs["action_project"]
            _require(ap["output_elements"] == parts[1]["count"], "action_project output size")
            buffer = f"action_in_{number}"
            device[buffer] = ap["input_elements"]
            offset = 0
            for name in steps[project]["args"]["in"]:
                program.upload_host(buffer, offset, sizes[name], name)
                offset += sizes[name]
            _require(offset == ap["input_elements"], "action_project input size")
            if not state_done:
                sp_index = _last_writer(steps, project, state_name)
                sp = steps[sp_index]
                _require(sp["op"] == "run_elf" and sp["args"]["graph"] == "state_project"
                         and len(sp["args"]["in"]) == 1
                         and graphs["state_project"]["output_elements"] == parts[0]["count"],
                         "state features not from a one-input state_project")
                device["state_in"] = graphs["state_project"]["input_elements"]
                program.upload_host("state_in", 0, device["state_in"], sp["args"]["in"][0])
                program.call("state_project", [["state_in"]], ["dit_in", 0, parts[0]["count"]])
                removed.add(sp_index)
                state_done = True
                state_writes = {i for i, s in enumerate(steps) if s["out"] == state_name}
                _require(state_writes == {sp_index}, "state features written more than once")
            program.call("action_project", [[buffer]], ["dit_in", parts[0]["count"], parts[1]["count"]])
            removed |= {pack, project}
            # action_tail(dit_out, temb) right after the chain
            after = [i for i in range(index + 1, len(steps))
                     if steps[i]["op"] == "run_elf" and steps[i]["args"]["graph"] == "action_tail"]
            _require(bool(after), f"pass {number}: no action_tail after the DiT chain")
            tail = after[0]
            tail_in = steps[tail]["args"]["in"]
            at = graphs["action_tail"]
            _require(tail_in[0] == step["out"] and len(tail_in) == 2
                     and hidden + sizes[tail_in[1]] == at["input_elements"],
                     f"pass {number}: action_tail does not read [DiT output, temb]")
            for i in range(pack + 1, tail):
                _require(i in (index, project) or not (_writes(steps[i]) & {h, step["out"]}),
                         f"pass {number}: step {i} touches the DiT input/output")
            device["tail_in"] = at["input_elements"]
            device["velocity"] = at["output_elements"]
            removed.add(tail)
        program.upload_host("dit_temb", 0, temb, t)
        program.upload_host("dit_mask_even", 0, mask, m_even, cache=True)
        program.upload_host("dit_mask_odd", 0, mask, m_odd, cache=True)
        if not fuse:
            program.upload_host("dit_in", 0, hidden, h)
        if backbone_device == "dit_backbone":
            program.upload_host("dit_backbone", 0, backbone, b, cache=True)
        for i, name in enumerate(N16_DIT):
            # Identical to DiTRunnerChain: stage 0 reads the uploaded hidden
            # state, then the two buffers ping-pong; masks alternate by stage.
            source = "dit_in" if i == 0 else ("dit_a" if i % 2 else "dit_b")
            target = "dit_b" if i % 2 else "dit_a"
            if fuse and i == len(N16_DIT) - 1:
                target = ["tail_in", 0, hidden]
            program.call(name, [[source], ["dit_temb"], [backbone_device],
                                ["dit_mask_odd" if i % 2 else "dit_mask_even"]],
                         target if isinstance(target, list) else [target])
        if fuse:
            temb_name = steps[tail]["args"]["in"][1]
            program.upload_host("tail_in", hidden, sizes[temb_name], temb_name)
            program.call("action_tail", [["tail_in"]], ["velocity"])
            program.download("velocity", 0, device["velocity"], steps[tail]["out"])
            program.out = steps[tail]["out"]
        else:
            final = "dit_b" if len(N16_DIT) % 2 == 0 else "dit_a"
            program.download(final, 0, hidden, step["out"])
        programs.append((insert_at, program.step()))
    if fuse:
        for number, index in enumerate(chains):
            for name in steps[index]["args"]["in"][:1] + [steps[index]["out"]]:
                _only_readers(steps, name, removed)
    return removed, programs


# ------------------------------------------------------------------ N1.6


def rewrite_n16(plan: dict, graphs: dict, levels: set[str]) -> dict:
    steps = plan["steps"]
    sizes = plan["buffers"]
    device: dict[str, int] = {}
    removed: set[int] = set()
    programs: list[tuple[int, dict]] = []
    backbone_device = None

    if "lm" in levels:
        qwen = [_one(steps, "run_elf", graph=name) for name in N16_QWEN]
        norm = _one(steps, "run_elf", graph="eagle_output_norm")
        first_in = steps[qwen[0]]["args"]["in"]
        _require(len(first_in) == 2, "Qwen graphs take [hidden, mask]")
        hidden_name, mask_name = first_in
        hidden = sizes[hidden_name]
        mask = sizes[mask_name]
        packed = graphs[N16_QWEN[0]]["input_elements"]
        _require(packed == hidden + mask, "Qwen packed input is not [hidden | mask]")
        previous = hidden_name
        for name, index in zip(N16_QWEN, qwen, strict=True):
            step = steps[index]
            _require(step["args"]["in"] == [previous, mask_name], f"unexpected input to {name}")
            _require(graphs[name]["input_elements"] == packed
                     and graphs[name]["output_elements"] == hidden, f"{name} geometry")
            previous = step["out"]
        _require(steps[norm]["args"]["in"] == [previous], "output norm does not follow the LM")
        _require(qwen == sorted(qwen) and norm > qwen[-1], "LM steps out of order")
        backbone = steps[norm]["out"]
        _require(sizes[backbone] == hidden == graphs["eagle_output_norm"]["output_elements"],
                 "output norm geometry")
        mask_part = _sidecar_pack(steps[_one(steps, "pack", out=mask_name)])

        device.update({"lm_a": packed, "lm_b": packed, "lm_out": hidden})
        program = _Program("lm", backbone)
        insert_at = qwen[0]
        if "connector" in levels:
            embed = _one(steps, "pack", out=hidden_name)
            parts = steps[embed]["args"]["parts"]
            prompt, images = parts[0], parts[1:]
            _require(prompt.get("sidecar") and prompt["dst_offset"] == 0
                     and prompt["count"] == hidden, "language pack must start with the prompt")
            for number, part in enumerate(images):
                _require(not part.get("sidecar") and not part.get("src_offset"),
                         "image token parts must be live buffers")
                conn = _one(steps, "run_elf", out=part["src"])
                _require(steps[conn]["args"]["graph"] == "eagle_vision_connector",
                         "image tokens do not come from the connector")
                conn_in = steps[conn]["args"]["in"]
                _require(len(conn_in) == 1, "connector takes one input")
                _require(sizes[part["src"]] == part["count"]
                         == graphs["eagle_vision_connector"]["output_elements"], "connector size")
                buffer = f"connector_in_{number}"
                device[buffer] = sizes[conn_in[0]]
                program.upload_host(buffer, 0, sizes[conn_in[0]], conn_in[0])
                program.call("eagle_vision_connector", [[buffer]],
                             ["lm_a", part["dst_offset"], part["count"]])
                removed.add(conn)
                readers = [i for i, s in enumerate(steps) if part["src"] in _reads(s)]
                _require(readers == [embed], f"{part['src']} has readers besides the pack")
            for start, count in _complement(hidden, [(p["dst_offset"], p["count"]) for p in images]):
                program.upload_sidecar("lm_a", start, count, prompt, extra_offset=start, cache=False)
            removed.add(embed)
            insert_at = embed
        else:
            program.upload_host("lm_a", 0, hidden, hidden_name)
        for buffer in ("lm_a", "lm_b"):
            program.upload_sidecar(buffer, hidden, mask, mask_part)
        for i, name in enumerate(N16_QWEN):
            source, target = ("lm_a", "lm_b") if i % 2 == 0 else ("lm_b", "lm_a")
            program.call(name, [[source]], [target, 0, hidden])
        last = "lm_b" if len(N16_QWEN) % 2 else "lm_a"
        program.call("eagle_output_norm", [[last, 0, hidden]], ["lm_out"])
        if "dit" not in levels:
            program.download("lm_out", 0, hidden, backbone)
        removed |= set(qwen) | {norm}
        programs.append((insert_at, program.step()))
        backbone_device = "lm_out"

    if "dit" in levels:
        chains, dit = _dit_programs(plan, graphs, backbone_device, device,
                                    fuse="denoise" in levels)
        if backbone_device:
            # The LM result now lives only on the device: nothing else may read it.
            dit_inputs = {steps[i]["args"]["in"][2] for i in _find(steps, "run_dit_chain")}
            _require(len(dit_inputs) == 1, "DiT passes read different backbones")
            for index, step in enumerate(steps):
                if index not in removed and index not in chains:
                    _require(not (_reads(step) & dit_inputs), f"step {index} reads the backbone")
        removed |= chains
        programs += dit

    _require(programs, "no level applies to this plan")
    result = _splice(plan, removed, programs)
    result["device_buffers"] = device
    fed = set().union(*(_reads(steps[i]) for i in removed))
    return _drop_dead_sidecar_packs(result, {s["out"] for s in steps if s["op"] == "pack"} & fed)


# ------------------------------------------------------------------- ACT


def rewrite_act(plan: dict, graphs: dict, levels: set[str]) -> dict:
    """ACT's whole network as one device program.

    Both cameras' vision outputs are written straight into their token slots of
    the stem's packed input, and the stem, three encoder layers and decoder
    ping-pong between two hidden buffers. The host still packs token 0 (the
    state, zero-padded to the hidden width), uploads it with the two images,
    and strides the padded action head after the single download.
    """
    _require(levels == {"act"}, "an ACT plan takes only the act level")
    steps, sizes = plan["steps"], plan["buffers"]
    vision = _find(steps, "run_elf", graph=ACT_VISION)
    _require(len(vision) == 2, "ACT runs the vision backbone once per camera")
    pack = _one(steps, "pack", out=steps[_one(steps, "run_elf", graph=ACT_STEM)]["args"]["in"][0])
    stem = _one(steps, "run_elf", graph=ACT_STEM)
    layers = [_one(steps, "run_elf", graph=name) for name in ACT_LAYERS]
    decoder = _one(steps, "run_elf", graph=ACT_DECODER)
    order = [*vision, pack, stem, *layers, decoder]
    _require(order == sorted(order) and order == list(range(order[0], order[0] + len(order))),
             "ACT network steps are not one contiguous run")

    packed = steps[pack]["out"]
    stem_in = graphs[ACT_STEM]["input_elements"]
    hidden = graphs[ACT_STEM]["output_elements"]
    _require(sizes[packed] == stem_in and steps[pack]["args"]["size"] == stem_in, "stem input size")
    camera = {steps[i]["out"]: steps[i]["args"]["in"] for i in vision}
    parts = steps[pack]["args"]["parts"]
    cameras = [p for p in parts if p["src"] in camera]
    host = [p for p in parts if p["src"] not in camera]
    _require(len(cameras) == 2 and all(not p.get("sidecar") and not p.get("src_offset")
                                       and p["count"] == graphs[ACT_VISION]["output_elements"]
                                       for p in cameras), "camera slots are not whole vision outputs")
    for name in camera:
        readers = [i for i, s in enumerate(steps) if name in _reads(s)]
        _require(readers == [pack], f"{name} has readers besides the stem pack")
    # The host keeps everything ahead of the first camera slot: the state token.
    head = min(p["dst_offset"] for p in cameras)
    _require(all(not p.get("sidecar") and p["dst_offset"] + p["count"] <= head for p in host),
             "host-packed parts must precede the camera slots")
    covered = sorted([(0, head)] + [(p["dst_offset"], p["count"]) for p in cameras])
    _require(not _complement(stem_in, [(o, c) for o, c in covered if c]),
             "stem input is not fully covered by the state token and camera slots")

    previous = steps[stem]["out"]
    for index in layers:
        _require(steps[index]["args"]["in"] == [previous], "encoder layers are not chained")
        _require(graphs[steps[index]["args"]["graph"]]["input_elements"] == hidden
                 == graphs[steps[index]["args"]["graph"]]["output_elements"], "encoder geometry")
        previous = steps[index]["out"]
    _require(steps[decoder]["args"]["in"] == [previous], "decoder does not follow the encoder")
    _require(graphs[ACT_DECODER]["input_elements"] == hidden, "decoder geometry")
    result = steps[decoder]["out"]
    actions = graphs[ACT_DECODER]["output_elements"]
    _require(sizes[result] == actions, "decoder output size")

    token = f"{packed}_head"
    device = {"act_image_0": graphs[ACT_VISION]["input_elements"],
              "act_image_1": graphs[ACT_VISION]["input_elements"],
              "act_stem": stem_in, "act_hidden_a": hidden, "act_hidden_b": hidden,
              "act_actions": actions}
    program = _Program("act", result)
    for number, part in enumerate(sorted(cameras, key=lambda p: p["dst_offset"])):
        (image,) = camera[part["src"]]
        buffer = f"act_image_{number}"
        _require(sizes[image] == device[buffer], f"{image} is not one whole camera image")
        program.upload_host(buffer, 0, device[buffer], image)
        program.call(ACT_VISION, [[buffer]], ["act_stem", part["dst_offset"], part["count"]])
    program.upload_host("act_stem", 0, head, token)
    program.call(ACT_STEM, [["act_stem"]], ["act_hidden_a"])
    source, target = "act_hidden_a", "act_hidden_b"
    for name in ACT_LAYERS:
        program.call(name, [[source]], [target])
        source, target = target, source
    program.call(ACT_DECODER, [[source]], ["act_actions"])
    program.download("act_actions", 0, actions, result)

    new = copy.deepcopy(plan)
    replaced = set(order)
    head_pack = {"op": "pack", "out": token, "args": {"size": head, "parts": copy.deepcopy(host)}}
    new["steps"] = (steps[:order[0]] + [head_pack, program.step()]
                    + [s for i, s in enumerate(steps) if i > order[-1] and i not in replaced])
    dead = set(camera) | {packed} | {steps[i]["out"] for i in (stem, *layers)}
    live = set().union(*(_reads(s) | _writes(s) | {s["out"]} for s in new["steps"]))
    new["buffers"] = {k: v for k, v in sizes.items() if k not in dead or k in live}
    new["buffers"][token] = head
    new["device_buffers"] = device
    return new


# ------------------------------------------------------------------ N1.7


def _n17_language_parts(steps: list[dict], sizes: dict, graphs: dict, name: str, hidden: int):
    index = _one(steps, "run_elf", graph=name)
    ins = steps[index]["args"]["in"]
    _require(len(ins) == 4, f"{name} takes [hidden, cos, sin, mask]")
    _require(sizes[ins[0]] == hidden, f"{name} hidden size")
    tails, offset = [], hidden
    for aux in ins[1:]:
        part = _sidecar_pack(steps[_one(steps, "pack", out=aux)])
        _require(part["count"] == sizes[aux], f"{aux} pack size")
        tails.append((offset, sizes[aux], part))
        offset += sizes[aux]
    _require(offset == graphs[name]["input_elements"], f"{name} packed input size")
    _require(graphs[name]["output_elements"] == hidden, f"{name} output size")
    return index, ins[0], tails


def rewrite_n17(plan: dict, graphs: dict, levels: set[str]) -> dict:
    steps = plan["steps"]
    sizes = plan["buffers"]
    device: dict[str, int] = {}
    removed: set[int] = set()
    programs: list[tuple[int, dict]] = []
    backbone_device = None
    hidden = graphs["vlln"]["input_elements"]

    def load_tails(program, buffer, tails):
        for offset, count, part in tails:
            program.upload_sidecar(buffer, offset, count, part)

    if "vision" in levels:
        layer0, language_input, tails0 = _n17_language_parts(steps, sizes, graphs,
                                                             "language_00_00", hidden)
        pack = _one(steps, "pack", out=language_input)
        parts = steps[pack]["args"]["parts"]
        text, images = parts[0], parts[1:]
        _require(not text.get("sidecar") and text["dst_offset"] == 0 and text["count"] == hidden,
                 "language input must start from the text embeddings")
        text_pack = _sidecar_pack(steps[_one(steps, "pack", out=text["src"])])
        _require(text_pack["count"] == hidden, "text embedding sidecar size")
        device["lang_0"] = graphs["language_00_00"]["input_elements"]
        device["lang_0_out"] = hidden
        program = _Program("vision", steps[layer0]["out"])
        patch_size = graphs["vision_patch"]["input_elements"]
        width = graphs["vision_00_01"]["input_elements"]
        side = graphs["vision_merge"]["output_elements"]
        device.update({"vision_a": width, "vision_b": width})
        # Follow each vision_patch forward along the plan's own data flow.
        stacks: dict[str, list[int]] = {}
        for patch in _find(steps, "run_elf", graph="vision_patch"):
            chain, current = [patch], steps[patch]["out"]
            for name in N17_VISION[1:]:
                hits = [i for i in _find(steps, "run_elf", graph=name)
                        if steps[i]["args"]["in"] == [current]]
                _require(len(hits) == 1, f"vision stack from step {patch}: {name} not unique")
                chain.append(hits[0])
                if not name.startswith("deepstack"):
                    current = steps[hits[0]]["out"]
            _require(chain == sorted(chain), f"vision stack from step {patch} out of order")
            stacks[current] = chain
        first_vision = None
        for camera, part in enumerate(images):
            _require(not part.get("sidecar") and part["count"] == side, "image slot size")
            chain = stacks.get(part["src"])
            _require(chain is not None, f"camera {camera}: slot not fed by a vision stack")
            first_vision = chain[0] if first_vision is None else first_vision
            patches = steps[chain[0]]["args"]["in"]
            _require(len(patches) == 1 and sizes[patches[0]] == patch_size, "patch input")
            buffer = f"patches_{camera}"
            device[buffer] = patch_size
            program.upload_host(buffer, 0, patch_size, patches[0])
            current, spare = "vision_a", "vision_b"
            program.call("vision_patch", [[buffer]], [current])
            for index in chain[1:-1]:
                step = steps[index]
                name = step["args"]["graph"]
                if name.startswith("deepstack"):
                    target = f"deep_{name[-1]}_{camera}"
                    device[target] = side
                    program.call(name, [[current]], [target])
                    readers = [i for i, s in enumerate(steps) if step["out"] in _reads(s)]
                    _require(all(steps[i]["op"] == "pack" for i in readers),
                             f"{step['out']} must only feed deepstack packs")
                    program.download(target, 0, side, step["out"])
                else:
                    program.call(name, [[current]], [spare])
                    current, spare = spare, current
            program.call("vision_merge", [[current]], ["lang_0", part["dst_offset"], side])
            removed |= set(chain)
            readers = [i for i, s in enumerate(steps) if part["src"] in _reads(s)]
            _require(readers == [pack], f"{part['src']} has readers besides the language pack")
        slots = [(p["dst_offset"], p["count"]) for p in images]
        for start, count in _complement(hidden, slots):
            program.upload_sidecar("lang_0", start, count, text_pack, extra_offset=start)
        load_tails(program, "lang_0", tails0)
        program.call("language_00_00", [["lang_0"]], ["lang_0_out"])
        program.download("lang_0_out", 0, hidden, steps[layer0]["out"])
        removed |= {pack, layer0}
        _require(first_vision is not None and first_vision < pack < layer0, "vision order")
        programs.append((layer0, program.step()))

    if "lm" in levels:
        names = N17_LANGUAGE[3:]
        parsed = [_n17_language_parts(steps, sizes, graphs, name, hidden) for name in names]
        vl = [_one(steps, "run_elf", graph=name) for name in N17_VL]
        previous = parsed[0][1]
        entry = previous
        for (index, source, _), name in zip(parsed, names, strict=True):
            _require(source == previous, f"unexpected input to {name}")
            previous = steps[index]["out"]
        for index, name in zip(vl, N17_VL, strict=True):
            _require(steps[index]["args"]["in"] == [previous], f"unexpected input to {name}")
            _require(graphs[name]["input_elements"] == graphs[name]["output_elements"] == hidden,
                     f"{name} geometry")
            previous = steps[index]["out"]
        order = [p[0] for p in parsed] + vl
        _require(order == sorted(order), "LM steps out of order")
        backbone = previous
        program = _Program("lm", backbone)
        buffers = [f"lang_{name[9:]}" for name in names]
        for buffer, name in zip(buffers, names, strict=True):
            device[buffer] = graphs[name]["input_elements"]
        device.update({"vl_a": hidden, "vl_b": hidden, "lm_out": hidden})
        program.upload_host(buffers[0], 0, hidden, entry)
        for buffer, (_, _, tails) in zip(buffers, parsed, strict=True):
            load_tails(program, buffer, tails)
        for i, name in enumerate(names):
            target = [buffers[i + 1], 0, hidden] if i + 1 < len(names) else ["vl_a"]
            program.call(name, [[buffers[i]]], target)
        current, spare = "vl_a", "vl_b"
        for i, name in enumerate(N17_VL):
            target = "lm_out" if i == len(N17_VL) - 1 else spare
            program.call(name, [[current]], [target])
            current, spare = spare, current
        if "dit" not in levels:
            program.download("lm_out", 0, hidden, backbone)
        removed |= set(order)
        programs.append((order[0], program.step()))
        backbone_device = "lm_out"

    if "dit" in levels:
        chains, dit = _dit_programs(plan, graphs, backbone_device, device,
                                    fuse="denoise" in levels)
        if backbone_device:
            dit_inputs = {steps[i]["args"]["in"][2] for i in _find(steps, "run_dit_chain")}
            for index, step in enumerate(steps):
                if index not in removed and index not in chains:
                    _require(not (_reads(step) & dit_inputs), f"step {index} reads the backbone")
        removed |= chains
        programs += dit

    _require(programs, "no level applies to this plan")
    result = _splice(plan, removed, programs)
    result["device_buffers"] = device
    fed = set().union(*(_reads(steps[i]) for i in removed))
    return _drop_dead_sidecar_packs(result, {s["out"] for s in steps if s["op"] == "pack"} & fed)


def rewrite(plan: dict, graphs: list[dict], levels: Iterable[str]) -> dict:
    levels = set(levels)
    unknown = levels - set(LEVELS)
    _require(not unknown, f"unknown levels {sorted(unknown)}")
    _require("denoise" not in levels or "dit" in levels, "denoise needs the dit level")
    _require("device_buffers" not in plan, "plan already uses device chains")
    by_name = {g["name"]: g for g in graphs}
    for graph in graphs:
        _require("int8_inputs" not in graph or graph["name"] not in _chained_names(levels),
                 f"{graph['name']} is INT8")
    kind = family(by_name)
    if kind == "act":
        new = rewrite_act(plan, by_name, levels)
    elif kind == "n1.6":
        _require("vision" not in levels, "N1.6 vision is already a run_elf_chain")
        new = rewrite_n16(plan, by_name, levels)
    else:
        _require("connector" not in levels, "N1.7 has no connector graph")
        new = rewrite_n17(plan, by_name, levels)
    for step in new["steps"]:
        if step["op"] == "run_device_chain":
            for call in step["args"]["calls"]:
                graph = by_name[call["graph"]]
                _require("int8_inputs" not in graph, f"{graph['name']}: INT8 graph in a chain")
                _require(graph.get("dram_layout", "plain") == "plain",
                         f"{graph['name']}: tessellated output cannot be chained")
    return new


def _chained_names(levels: set[str]) -> set[str]:
    names = set()
    if "lm" in levels:
        names |= set(N16_QWEN) | {"eagle_output_norm"} | set(N17_LANGUAGE[3:]) | set(N17_VL)
    if "vision" in levels:
        names |= set(N17_VISION) | {"language_00_00"}
    if "connector" in levels:
        names.add("eagle_vision_connector")
    if "dit" in levels:
        names |= set(N16_DIT)
    if "denoise" in levels:
        names |= {"state_project", "action_project", "action_tail"}
    if "act" in levels:
        names |= {ACT_VISION, ACT_STEM, ACT_DECODER, *ACT_LAYERS}
    return names


# ----------------------------------------------------------- equivalence


def bf16_round(values):
    """float32 -> nearest-even BF16 -> float32, bit-identical to polima::to_bf16."""
    import numpy as np

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return (rounded & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


def stand_in_graphs(graphs: dict[str, dict]):
    """Deterministic BF16-in/BF16-out stand-ins with the bundle's geometry."""
    import numpy as np

    def run(name: str, values):
        x = bf16_round(np.asarray(values, dtype=np.float32))
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        n = graphs[name]["output_elements"]
        j = np.arange(n, dtype=np.int64)
        a = x[(j * 7 + seed) % x.size]
        b = x[(j * 131 + 3) % x.size]
        weights = (np.arange(x.size) % 97 + 1).astype(np.float64)
        total = float(np.fmod(np.dot(x.astype(np.float64), weights), 3.0))
        y = np.tanh(0.7 * a - 0.3 * b + 0.05 * total + (seed % 1000) / 1000.0) * 2.0
        return bf16_round(y.astype(np.float32))

    return run


def verify_equivalence(bundle: Path, new_plan: dict, *, trials: int = 3, seed: int = 0) -> dict:
    """Original vs rewritten plan on the reference executor, random + fixture inputs."""
    import numpy as np

    from polima.wire.server_stub import StubPlan

    manifest = json.loads((bundle / "bundle.json").read_text())
    graphs = {g["name"]: g for g in manifest["graphs"]}
    original = StubPlan.load(bundle)
    candidate = StubPlan.load(bundle)
    candidate.buffers = dict(new_plan["buffers"])
    candidate.steps = list(new_plan["steps"])
    candidate.device_buffers = dict(new_plan.get("device_buffers", {}))
    fn = stand_in_graphs(graphs)
    rng = np.random.default_rng(seed)
    cases = []
    fixtures = bundle / "fixtures" / "inputs"
    tensors = original.wire.get("request_tensors", [])
    if fixtures.is_dir():
        cases.append({t["name"]: np.fromfile(fixtures / f"{t['name']}.f32", dtype=np.float32)
                      for t in tensors})
    for _ in range(trials):
        cases.append({t["name"]: rng.standard_normal(t["elements"]).astype(np.float32)
                      for t in tensors})
    results = []
    for inputs in cases:
        a = original.run(inputs, graph_fn=fn)
        b = candidate.run(inputs, graph_fn=fn)
        results.append(bool(a.tobytes() == b.tobytes()))
    return {"cases": len(cases), "byte_identical": results}


# ----------------------------------------------------------------- bundle


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def derive_bundle(source: Path, output_root: Path, levels: Iterable[str], *,
                  label_suffix: str = "chain", verify: bool = True) -> Path:
    """Write a new bundle beside nothing it came from; models/constants are hard links."""
    levels = sorted(set(levels))
    manifest = json.loads((source / "bundle.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    for graph in manifest["graphs"]:
        if _digest(source / graph["elf"]) != graph["sha256"]:
            raise RewriteError(f"source ELF hash mismatch: {graph['name']}")
    new_plan = rewrite(plan, manifest["graphs"], levels)
    report = verify_equivalence(source, new_plan) if verify else None
    if report and not all(report["byte_identical"]):
        raise RewriteError(f"reference-executor equivalence failed: {report}")

    new = copy.deepcopy(manifest)
    identity = parse_bundle_id(manifest["bundle_id"])
    digests = {g["name"]: g["sha256"] for g in new["graphs"]}
    if identity.get("robot") and identity.get("precision"):
        new["bundle_id"] = compute_bundle_id(
            policy=manifest["policy"], dataset=identity["dataset"], steps=identity["steps"],
            graph_digests=digests, plan=new_plan, robot=identity["robot"],
            version=identity.get("version"), precision=identity["precision"] + label_suffix)
    else:
        new["bundle_id"] = compute_bundle_id(
            policy=manifest["policy"], dataset="-".join(filter(None, (identity["dataset"], label_suffix))),
            steps=identity["steps"], graph_digests=digests, plan=new_plan)
    new["derived_from"] = {  # what this bundle is, not how it was reached
        "bundle_id": manifest["bundle_id"],
        "optimization": "on-device activation chaining (run_device_chain); ELFs unchanged",
    }
    out = output_root / new["bundle_id"]
    if out.exists():
        raise RewriteError(f"refusing existing output: {out}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = out / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.name not in ("bundle.json", "plan.json") or len(relative.parts) > 1:
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative.parts[0] in ("models", "constants"):
                os.link(path, target)
            else:
                shutil.copy2(path, target)
    (out / "plan.json").write_text(json.dumps(new_plan, indent=2) + "\n")
    (out / "bundle.json").write_text(json.dumps(new, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--levels", default="lm,dit",
                        help=f"comma-separated subset of {','.join(LEVELS)}")
    parser.add_argument("--label-suffix", default="chain")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    out = derive_bundle(args.bundle, args.output_root, args.levels.split(","),
                        label_suffix=args.label_suffix, verify=not args.no_verify)
    print(out)


if __name__ == "__main__":
    main()
