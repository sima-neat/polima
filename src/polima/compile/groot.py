"""`polima compile --policy groot`: a GR00T N1.6/N1.7 checkpoint + LeRobot dataset ->
the optimised, robot-ready Modalix bundle.

    polima compile --policy groot --checkpoint CKPT --dataset LEROBOT_V2_DIR --output DIR
                   [--precision mixed|bf16|w8a8] [--jobs 1..3] [--resume]
                   [--stop-after STAGE] [--check-devkit ADDRESS]

Stages (each in its own interpreter, with a marker in DIR/stages/ and a log in DIR/logs/):

  export     GR00T env. Checkpoint -> ONNX partitions + calibration rows. N1.6: PoLiMa's
             exporter (96 dataset observations, GPU reference pass). N1.7: official-policy
             reference capture, static partitions, 32-observation calibration capture,
             SmoothQuant (alpha 0.6, vision + language; INT8-weight builds) and the plan.
  dit        GR00T env, CPU. The DiT rewrite (policies/groot/dit.py): two K/V graphs per
             request plus eight 4-block graphs in NCHW form whose AdaLN modulation is
             precomputed on the host per denoising step.
  compile    ModelSDK. Every graph the bundle runs; the DiT graphs as persistent-binding
             ELFs. <= --jobs at once and <= the host-wide compiler limit (shared lock).
  bundle     PoLiMa venv. One bundle: run_bound_chain DiT with host AdaLN, device chains
             (N1.6 lm+connector, N1.7 vision+lm), camera order from the model's video keys,
             robot client, rest pose and calibration from the dataset.
  check      Optional (--check-devkit): the bundle's fixture on a Modalix DevKit vs the
             PyTorch reference, release metric; non-zero exit outside the known envelope.

`--resume` keeps finished stages and finished graphs; without it DIR must not exist.
Interpreters come from the site config (`.polima.yaml` paths.*) or the environment:
POLIMA_GROOT16_PYTHON / POLIMA_GROOT17_PYTHON / POLIMA_GROOT16_SOURCE / POLIMA_GROOT17_SOURCE
and MODEL_COMPILER_BIN, as for every other compile.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STAGES = ("export", "dit", "compile", "bundle", "check")
PRECISIONS = ("mixed", "bf16", "w8a8")
SRC = Path(__file__).resolve().parents[2]
#: Fixture check vs the PyTorch BF16 reference over the whole chunk (normalized actions).
#: Known SO-101 results: BF16 cos 0.99997 / nMAE 0.002; mixed cos 0.9987-0.9995 / nMAE
#: 0.009-0.010, its moves 17-32 legitimately 0.011-0.014. The mixed bar is ~3x that.
ENVELOPE = {"bf16": {"cosine": 0.999, "mean_abs": 0.01}, "mixed": {"cosine": 0.995, "mean_abs": 0.03}}


class StageError(RuntimeError):
    pass


# ------------------------------------------------------------------- settings


def detect_family(checkpoint: Path) -> str:
    config = json.loads((checkpoint / "config.json").read_text())
    kind = config.get("model_type", "")
    if kind == "Gr00tN1d6":
        return "n1.6"
    if kind == "Gr00tN1d7":
        return "n1.7"
    raise StageError(f"{checkpoint}/config.json: model_type {kind!r} is neither Gr00tN1d6 nor Gr00tN1d7")


def interpreters(config, family: str) -> dict:
    """GR00T env python + Isaac-GR00T source for this family, and the ModelSDK python."""
    from polima.compile.toolchain import require_compiler_python

    tag = "16" if family == "n1.6" else "17"
    paths = getattr(config, "paths", None)
    found = {}
    for key, env in ((f"groot{tag}_python", f"POLIMA_GROOT{tag}_PYTHON"),
                     (f"groot{tag}_source", f"POLIMA_GROOT{tag}_SOURCE")):
        value = getattr(paths, key, None)
        if not value or not Path(value).exists():
            raise StageError(f"GR00T {family} needs {env} (or paths.{key} in your configuration file) "
                             f"pointing at its {'interpreter' if 'python' in key else 'Isaac-GR00T checkout'}")
        found[key.split("_")[1]] = str(value)
    found["sdk"] = str(require_compiler_python(config))
    return found


def worker_env(settings: dict, *, gpu: bool = False) -> dict:
    env = dict(os.environ)
    env.pop("POLIMA_GROOT17_BACKBONE", None)
    if settings.get("backbone"):
        env["POLIMA_GROOT17_BACKBONE"] = settings["backbone"]
    paths = [str(SRC)] + ([settings["source"]] if settings.get("source") else [])
    env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


# --------------------------------------------------------------------- stages


class Build:
    """DIR/build.json (settings), DIR/stages/<stage>.done, DIR/stages.txt, DIR/logs/."""

    def __init__(self, root: Path):
        self.root = root
        self.settings = json.loads((root / "build.json").read_text())

    def path(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def done(self, stage: str) -> bool:
        return self.path("stages", f"{stage}.done").is_file()

    def mark(self, stage: str, note: str = "") -> None:
        self.path("stages").mkdir(exist_ok=True)
        self.path("stages", f"{stage}.done").write_text(json.dumps({"finished": _now(), "note": note}) + "\n")
        self.log(f"{stage} done {note}".rstrip())

    def log(self, text: str) -> None:
        line = f"{_now()} {text}"
        print(line, flush=True)
        with self.path("stages.txt").open("a") as stream:
            stream.write(line + "\n")

    def run_worker(self, stage: str, python: str, *, gpu: bool = False) -> None:
        """Run `python -m polima.compile.groot --worker STAGE DIR` in another interpreter."""
        log = self.path("logs", f"{stage}.log")
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as stream:
            code = subprocess.run([python, "-u", "-m", "polima.compile.groot", "--worker", stage,
                                   str(self.root)], env=worker_env(self.settings, gpu=gpu),
                                  stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise StageError(f"{stage} failed (exit {code}); see {log}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stage_export(build: Build) -> None:
    family = build.settings["family"]
    build.run_worker("export", build.settings["python"], gpu=family == "n1.6")


def stage_dit(build: Build) -> None:
    build.run_worker("dit", build.settings["python"])


def stage_compile(build: Build) -> None:
    failed = compile_all(build)
    if failed:
        raise StageError(f"{len(failed)} graph(s) failed: {', '.join(failed)}; see "
                         f"{build.path('compiled', 'compile_results.json')}")


def stage_bundle(build: Build) -> None:
    bundle = make_bundle(build)
    build.settings["bundle"] = str(bundle)
    build.path("build.json").write_text(json.dumps(build.settings, indent=2) + "\n")
    build.log(f"bundle {bundle}")


def stage_check(build: Build) -> None:
    board = build.settings.get("check_board")
    if not board:
        build.log("check skipped: pass --check-devkit to check the bundle on a DevKit")
        return
    report = board_check(build, board)
    if not report["ok"]:
        raise StageError(f"fixture check outside the release envelope: {report['summary']}")


RUNNERS = {"export": stage_export, "dit": stage_dit, "compile": stage_compile,
           "bundle": stage_bundle, "check": stage_check}


# -------------------------------------------------------------------- compile

HEAVY = {"n1.6": re.compile(r"(?:eagle_vision|eagle_qwen|dit_blocks)_\d\d_\d\d"),
         "n1.7": re.compile(r"(?:vision|language|dit)_\d\d_\d\d")}
REPLACED_DIT = {"n1.6": re.compile(r"dit_blocks_\d\d_\d\d"), "n1.7": re.compile(r"dit_\d\d_\d\d")}
COMPILER = re.compile(r"python[0-9.]*( -u)? [^ ]*compile_(onnx|persistent)\.py")
LLIMA = re.compile(r"llima-compile|sima_lmm\.host\.compile_lmm|compile_lmm\.py")
DRIVER = re.compile(r"compile_n16\.py|compile_partitions\.py|xargs -P")


def graph_jobs(build: Build) -> list[dict]:
    """Every graph the bundle runs, with the compile flags of the established GR00T builds."""
    s, family = build.settings, build.settings["family"]
    export, precision = build.path("export"), s["precision"]
    if family == "n1.6":
        names = [Path(g).stem for g in json.loads((export / "input_contract.json").read_text())["graphs"]]
        onnx, calibration = export / "onnx", export / "calibration"
    else:
        names = [g["name"] for g in json.loads((export / "partition_manifest.json").read_text())["graphs"]]
        onnx = export / ("smooth/onnx" if precision != "bf16" else "onnx")
        calibration = export / "calibration-multi"
    jobs = []
    for name in names:
        if precision != "w8a8" and REPLACED_DIT[family].fullmatch(name):
            continue  # replaced by the persistent K/V + 4-block DiT graphs
        heavy = HEAVY[family].fullmatch(name) is not None
        act = "int8" if heavy and precision == "w8a8" else "bf16"
        weights = "int8" if heavy and precision != "bf16" else "bf16"
        if family == "n1.6" and name.startswith("eagle_qwen_"):
            flags = ["--external-dram-layout", "HWC"]
        elif family == "n1.6" and (name.startswith("eagle_vision_") and name != "eagle_vision_connector"
                                   or name == "eagle_output_norm"):
            flags = ["--external-dram-layout", "HWC", "--promote-rank3-hwc"]
        elif family == "n1.7" and (name.startswith(("vision_", "language_", "deepstack_")) or name == "vlln"):
            flags = ["--external-dram-layout", "HWC"]
        else:
            flags = ["--mla-tessellation"]
        jobs.append({"name": name, "onnx": onnx / f"{name}.onnx", "activation": act, "weights": weights,
                     "calibration": calibration / f"{name}.npz" if weights == "int8" else None,
                     "layout": "NHWC", "flags": flags, "persistent": False})
    if precision != "w8a8":
        variant = json.loads(build.path("dit", "manifest.json").read_text())
        for g in variant["graphs"]:
            jobs.append({"name": g["name"], "onnx": build.path("dit", "onnx", f"{g['name']}.onnx"),
                         "activation": "bf16", "weights": "int8" if precision == "mixed" else "bf16",
                         "calibration": build.path("dit", "calibration", f"{g['name']}.npz")
                         if precision == "mixed" else None, "layout": g["layout"], "flags": [],
                         "persistent": True, "outputs": g["outputs"]})
    return jobs


def _processes() -> dict[int, tuple[int, str]]:
    procs = {}
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit():
            try:
                cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                ppid = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if cmd:
                procs[int(proc.name)] = (ppid, cmd)
    return procs


def host_compilers() -> int:
    """Compiler processes on the host: one per compiler process tree; another build's graph
    driver is charged max(3, its live compilers); an llima-compile run max(workers, -j)."""
    procs, me = _processes(), os.getpid()
    compilers = {p for p, (_, cmd) in procs.items() if COMPILER.search(cmd)}
    drivers = {p for p, (_, cmd) in procs.items() if DRIVER.search(cmd) and not COMPILER.search(cmd)}
    charged, other = {d: 0 for d in drivers}, 0
    for pid in compilers:
        if procs[pid][0] in compilers:
            continue
        chain, parent = [], procs[pid][0]
        while parent in procs and len(chain) < 64:
            chain.append(parent)
            parent = procs[parent][0]
        owner = None if me in chain else next((p for p in chain if p in drivers), None)
        if owner is None:
            other += 1
        else:
            charged[owner] += 1
    total = other + sum(max(3, n) for n in charged.values())
    for pid, (ppid, cmd) in procs.items():
        if LLIMA.search(cmd) and "grep" not in cmd and not LLIMA.search(procs.get(ppid, (0, ""))[1]):
            workers = sum(1 for _, (pp, c) in procs.items() if pp == pid and "multiprocessing" in c)
            jobs = re.search(r"(?:-j|--jobs)[ =](\d+)", cmd)
            total += max(workers, int(jobs.group(1)) if jobs else 1, 1)
    return total


def start_compiler(cmd: list[str], env: dict, log) -> int:
    """Spawn one compiler once the host total allows it. Count-and-spawn is serialized
    across builds by an exclusive lock on one shared file ($POLIMA_COMPILE_SLOT_LOCK)."""
    import fcntl

    limit = int(os.environ.get("POLIMA_COMPILE_HOST_LIMIT", "10"))
    lock = Path(os.environ.get("POLIMA_COMPILE_SLOT_LOCK", "/tmp/polima-compile-slot.lock"))
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            while host_compilers() + 1 > limit:
                time.sleep(30)
            before = host_compilers()
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.time() + 30
            while host_compilers() <= before and proc.poll() is None and time.time() < deadline:
                time.sleep(0.5)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return proc.wait()


def _sha(path: Path | None) -> str | None:
    from polima.util.hashing import sha256_file

    return sha256_file(path) if path else None


def _flatten_dit_mask(model_path: Path, calibration: Path, work: Path) -> Path:
    """N1.7 INT8 DiT pairs: take the [1,1,Lk,1] mask as [1,Lk] (SDK 2.1.3 cannot pack the
    rank-4 INT8 boundary) and reshape it back in-graph; same math, same bytes."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(model_path))
    mask = next(t for t in model.graph.input if t.name == "attention_mask")
    shape = tuple(d.dim_value for d in mask.type.tensor_type.shape.dim)
    for node in model.graph.node:
        node.input[:] = ["polima_attention_mask_rank4" if i == "attention_mask" else i for i in node.input]
    del mask.type.tensor_type.shape.dim[:]
    for dim in (1, shape[2]):
        mask.type.tensor_type.shape.dim.add().dim_value = dim
    model.graph.initializer.append(numpy_helper.from_array(np.array(shape, np.int64), "polima_attention_mask_shape"))
    model.graph.node.insert(0, helper.make_node("Reshape", ["attention_mask", "polima_attention_mask_shape"],
                                                ["polima_attention_mask_rank4"]))
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))
    with np.load(calibration) as archive:
        values = {k: archive[k] for k in archive.files}
    values["attention_mask"] = values["attention_mask"].reshape(len(values["attention_mask"]), 1, shape[2])
    np.savez(work / "calibration-mask-2d.npz", **values)
    return work / "calibration-mask-2d.npz"


def compile_all(build: Build) -> list[str]:
    import shutil
    import threading

    from polima.compile.toolchain import compiler_env

    sdk = Path(build.settings["sdk"])
    out = build.path("compiled")
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    env = compiler_env(sdk.parent, source_root=SRC)
    env.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(out / "tmp"))
    shim = str(SRC / "polima/compile/compile_onnx.py")
    results_path = out / "compile_results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    lock = threading.Lock()

    def elf_of(job) -> Path:
        work = out / job["name"]
        return (work / "persistent" if job["persistent"] else work) / "retained" / f"{job['name']}_stage1_mla.elf"

    def key(job) -> dict:
        return {"source_sha256": _sha(job["onnx"]), "calibration_sha256": _sha(job["calibration"]),
                "activation": job["activation"], "weights": job["weights"], "layout": job["layout"],
                "flags": job["flags"], "persistent": job["persistent"]}

    def one(job) -> dict:
        name, work = job["name"], out / job["name"]
        wanted = key(job)
        old = results.get(name, {})
        if old.get("returncode") == 0 and {k: old.get(k) for k in wanted} == wanted \
                and elf_of(job).is_file() and _sha(elf_of(job)) == old.get("elf_sha256"):
            return {**old, "resumed": True}
        if work.exists():
            work.rename(out / f"{name}.failed-{int(time.time())}")
        work.mkdir()
        shutil.copy2(job["onnx"], work / f"{name}.onnx")
        if job["activation"] == "int8" and re.fullmatch(r"dit_\d\d_\d\d", name):
            job = {**job, "calibration": _flatten_dit_mask(work / f"{name}.onnx", job["calibration"], work)}
        common = [sdk.as_posix(), "-u", shim, "--model-path", str(work / f"{name}.onnx"),
                  "--build-dir", str(work / "compiled"), "--model-layout", job["layout"],
                  "--activation-precision", job["activation"], "--weight-precision", job["weights"]]
        if job["calibration"]:
            common += ["--calibration-npz", str(job["calibration"])]
        started, record = time.time(), dict(wanted)
        if job["persistent"]:
            with (work / "quantize.log").open("w") as log:
                code = start_compiler(common + ["--no-compile"], env, log)
            if code == 0:
                with (work / "persistent.log").open("w") as log:
                    code = start_compiler([sdk.as_posix(), "-u", shim, "persistent", "--source",
                                           str(work / "compiled" / name), "--name", name, "--output",
                                           str(work / "persistent"), "--onnx-outputs", *job["outputs"]], env, log)
        else:
            with (work / "compile.log").open("w") as log:
                code = start_compiler(common + ["--retain-compile-dir", str(work / "retained"),
                                                "--groot-boundaries", *job["flags"]], env, log)
        elf = elf_of(job)
        ok = code == 0 and elf.is_file() and elf.read_bytes()[:4] == b"\x7fELF"
        record.update(returncode=0 if ok else (code or 1), elf=str(elf), minutes=round((time.time() - started) / 60, 1),
                      elf_sha256=_sha(elf) if ok else None)
        return record

    jobs = graph_jobs(build)
    build.log(f"compile {len(jobs)} graphs, {build.settings['jobs']} at a time")
    failed = []
    with ThreadPoolExecutor(max_workers=build.settings["jobs"]) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for future in as_completed(futures):
            name = futures[future]["name"]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - one failed graph must not stop the others
                record = {"returncode": -1, "error": repr(error)}
            with lock:
                results[name] = record
                results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
            if record["returncode"]:
                failed.append(name)
            build.log(f"  {name}: {'ok' if not record['returncode'] else 'FAILED'}"
                      f"{' (kept)' if record.get('resumed') else ''} {record.get('minutes', '')}")
    return failed


# --------------------------------------------------------------------- bundle

PRECISION_LABELS = {("bf16", "bf16"): "bf16", ("bf16", "int8"): "w8a16", ("int8", "int8"): "a8w8"}
CHAIN_LEVELS = {"n1.6": ("lm", "connector"), "n1.7": ("vision", "lm")}


def _mla_input_order(work: Path, name: str) -> list[str]:
    """Graph inputs in the order the compiled MLA reads them (from its MPK metadata)."""
    meta = _mpk(work, name)
    inputs = {node["name"] for node in meta["input_nodes"]}
    producers = {o["name"]: p for p in meta["plugins"] for o in p.get("output_nodes", [])}

    def origins(node):
        if node in inputs:
            return [node]
        return [o for item in producers[node]["input_nodes"] for o in origins(item["name"])]

    mla = [p for p in meta["plugins"] if p.get("processor") == "MLA"]
    if len(mla) != 1:
        raise StageError(f"{name}: expected one MLA segment, found {len(mla)}")
    return [o for node in mla[0]["input_nodes"] for o in origins(node["name"])]


def int8_boundaries(compiled: Path, spec, jobs: dict):
    """W8A8: per-graph INT8 boundary quantization (from the MPK transforms the bare
    runtime omits) and the plan edits it needs: chains expanded, compiler input order,
    explicit zero tail padding of packed INT8 inputs."""
    def quant(plugin, elements, direction):
        params = plugin["config_params"]["params"]
        if len(params["channel_params"]) != 1 or params[f"{direction}_data_type"] != "int8":
            raise StageError(f"unsupported INT8 boundary in {plugin['name']}")
        multiplier, zero_point = params["channel_params"][0]
        return {"elements": elements, "multiplier": multiplier, "zero_point": zero_point}

    fields, io, padding, orders = {}, {}, {}, {}
    allowed = {"quantization_transform", "pack_transform", "dequantization_transform"}
    for graph in spec.compile.graphs:
        if graph.name not in jobs or jobs[graph.name]["activation"] != "int8":
            continue
        meta = _mpk(compiled / graph.name, graph.name)
        nodes = {p["name"]: p for p in meta["plugins"]}
        if any(p.get("processor") != "MLA" and p["config_params"].get("kernel") not in allowed for p in meta["plugins"]):
            raise StageError(f"{graph.name}: the bare runtime cannot omit this graph's transforms")
        mla = nodes["MLA_0"]
        entry = nodes[mla["input_nodes"][0]["name"]]
        quantizers = [nodes[n["name"]] for n in entry["input_nodes"]] \
            if entry["config_params"].get("kernel") == "pack_transform" else [entry]
        inputs = [quant(q, q["output_nodes"][0]["size"], "output") for q in quantizers]
        logical = sum(t.elements for t in graph.inputs)
        storage = sum(n["size"] for n in mla["input_nodes"])
        if storage != logical:  # pack_transform aligns the tail of the concatenated inputs
            if entry["config_params"].get("kernel") != "pack_transform" or storage != (logical + 15) // 16 * 16:
                raise StageError(f"{graph.name}: unsupported INT8 input storage {storage} for {logical}")
            padding[graph.name] = storage - logical
            inputs.append({"elements": storage - logical, "multiplier": 1.0, "zero_point": 0})
        dequant = [p for p in meta["plugins"] if p["config_params"].get("kernel") == "dequantization_transform"]
        output = quant(dequant[0], sum(t.elements for t in graph.outputs), "input")
        fields[graph.name] = {"precision": "a8w8", "int8_inputs": inputs, "int8_output": output,
                              "input_elements": storage}
        io[graph.name] = {"inputs": inputs, "output": output}
        orders[graph.name] = ([q["input_nodes"][0]["name"] for q in quantizers], [t.name for t in graph.inputs])

    def rewrite(plan: dict) -> dict:
        steps = []
        for step in plan["steps"]:
            if step["op"] != "run_elf_chain":
                steps.append(step)
                continue
            previous = step["args"]["in"]
            for name in step["args"]["graphs"]:  # INT8 graphs cannot share device buffers
                steps.append({"op": "run_elf", "out": step["out"], "args": {"graph": name, "in": list(previous)}})
                previous = [step["out"]]
        for step in steps:
            name = step["args"].get("graph") if step["op"] == "run_elf" else None
            if name in orders:
                order, previous = orders[name]
                feeds = dict(zip(previous, step["args"]["in"], strict=True))
                step["args"]["in"] = [feeds[k] for k in order]
                if name in padding:
                    step["args"]["in"].append(f"int8_tail_padding_{padding[name]}")
        pads = []
        for count in sorted(set(padding.values())):
            plan["buffers"][f"int8_tail_padding_{count}"] = count
            pads.append({"op": "pack", "out": f"int8_tail_padding_{count}", "args": {"parts": []}})
        plan["steps"] = pads + steps
        plan["graph_io"] = io
        return plan

    return fields, rewrite


def _mpk(work: Path, name: str) -> dict:
    import tarfile

    with tarfile.open(work / "compiled" / name / f"{name}_mpk.tar.gz") as tar:
        return json.load(tar.extractfile(next(m for m in tar.getnames() if m.endswith("_mpk.json"))))


def write_rest_pose(dataset: Path, path: Path, joint_names, calibration_id: str) -> None:
    """Robot rest pose: median observation.state at frame 0 over the dataset's episodes."""
    import numpy as np
    import pandas as pd

    starts = []
    for parquet in sorted((dataset / "data").rglob("*.parquet")):
        rows = pd.read_parquet(parquet, columns=["frame_index", "observation.state"])
        starts.extend(rows.loc[rows.frame_index == 0, "observation.state"].tolist())
    states = np.asarray(starts, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != len(joint_names) or not np.isfinite(states).all():
        raise StageError(f"{dataset}: no valid episode-start states for the rest pose")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source_dataset": Path(dataset).name,
        "method": f"median observation.state at frame_index 0 across {len(states)} episodes",
        "calibration_id": calibration_id,
        "positions": dict(zip(joint_names, map(float, np.median(states, axis=0)))),
    }, indent=2, sort_keys=True) + "\n")


def make_bundle(build: Build) -> Path:
    import shutil

    from polima.bundle.device_chain import derive_bundle
    from polima.bundle.importer import (
        collect_fixtures,
        collect_robot_files,
        collect_sidecars,
        detect,
    )
    from polima.bundle.pack import BundleInputs, build_bundle
    from polima.bundle.retained import ElfCandidate
    from polima.policies.groot import dit
    from polima.policies.groot.profile import checkpoint_state_encoding
    from polima.policies.registry import get_policy

    s, family, export = build.settings, build.settings["family"], build.path("export")
    compiled = build.path("compiled")
    results = json.loads((compiled / "compile_results.json").read_text())
    jobs = {job["name"]: job for job in graph_jobs(build)}
    if missing := [n for n in jobs if results.get(n, {}).get("returncode") != 0]:
        raise StageError(f"no successful compile for {missing}")
    elfs = {n: ElfCandidate(n, Path(results[n]["elf"]), n, results[n]["elf_sha256"],
                            Path(results[n]["elf"]).stat().st_size) for n in jobs}
    precisions = {n: PRECISION_LABELS[(j["activation"], j["weights"])] for n, j in jobs.items()}
    if family == "n1.6":
        profile = dict(json.loads((export / "input_contract.json").read_text())["profile"])
        profile["state_encoding"] = checkpoint_state_encoding(s["checkpoint"], profile.get("runtime_context", {})
                                                              .get("embodiment_tag"))
        spec = get_policy("groot").specialize_profile(profile=profile)
        prebuilt = detect(export)
        constants, fixtures = collect_sidecars(prebuilt, spec), collect_fixtures(prebuilt, spec)
        for graph in spec.compile.graphs:
            if graph.name in jobs and not jobs[graph.name]["persistent"] and len(graph.inputs) > 1 \
                    and jobs[graph.name]["activation"] == "bf16" \
                    and _mla_input_order(compiled / graph.name, graph.name) != [t.name for t in graph.inputs]:
                raise StageError(f"{graph.name}: compiled input order differs from the plan's")
        for task in spec.runtime_context.get("task_catalog") or ():
            source = export / f"groot_fixture_task_{int(task['id']):03d}.npz"
            if source.is_file():
                from dataclasses import replace

                task_spec = replace(spec, compile=replace(spec.compile, fixture_file=source.name))
                fixtures.update({f"task_{int(task['id']):03d}/{k}": v
                                 for k, v in collect_fixtures(prebuilt, task_spec).items()})
    else:
        profile = json.loads((export / "runtime/profile.json").read_text())
        context = profile["runtime_context"]
        for graph in context["n17_graphs"]:  # direct ELF inputs follow the MPK pack order
            name = graph["name"]
            if name not in jobs or jobs[name]["persistent"] or len(graph["input_order"]) < 2:
                continue
            order = _mla_input_order(compiled / name, name)
            if sorted(order) != sorted(graph["input_order"]):
                raise StageError(f"{name}: compiler inputs {order} differ from {graph['input_order']}")
            for step in context["n17_plan"]["steps"]:
                if step["op"] == "run_elf" and step["args"]["graph"] == name:
                    feeds = dict(zip(graph["input_order"], step["args"]["in"], strict=True))
                    step["args"]["in"] = [feeds[k] for k in order]
            graph["input_order"] = order
        spec = get_policy("groot").specialize_profile(profile=profile)
        constants = {n: export / "runtime/constants" / f"{n}.f32" for n in spec.build_runtime_plan().sidecars}
        root = export / "runtime/fixtures"
        fixtures = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}
        fixtures["normalization_stats.npz"] = export / "runtime/normalization_stats.npz"
    write_rest_pose(Path(s["dataset"]), export / "robot_client/robot_rest_position.json",
                    spec.dataset.state_names, spec.robot.calibration_id)
    try:
        robot_files = collect_robot_files(spec, export)
    except FileNotFoundError as error:
        build.log(f"no robot client in the bundle ({error}); inference server only")
        robot_files = {}
    if not robot_files:  # any robot: its dataset-derived rest pose still ships
        robot_files = {"robot_rest_position.json": export / "robot_client/robot_rest_position.json"}
    extra, rewrite, label = {}, None, s["precision"]
    if s["precision"] == "w8a8":
        extra, rewrite = int8_boundaries(compiled, spec, jobs)
    else:
        manifest = json.loads(build.path("dit", "manifest.json").read_text())
        for g in manifest["graphs"]:
            d = dit.describe_persistent(g["name"], compiled / g["name"] / "persistent", g["outputs"])
            if sorted(i["name"] for i in d["inputs"]) != sorted(g["inputs"]):
                raise StageError(f"{g['name']}: MLA inputs {d['inputs']} differ from the export {g['inputs']}")
            elfs[g["name"]] = ElfCandidate(g["name"], d["elf"], g["name"], d["sha256"], d["elf"].stat().st_size)
            extra[g["name"]] = {"precision": precisions[g["name"]], "external_dram_layout": "HWC",
                                "input_elements": sum(i["elements"] for i in d["inputs"]),
                                "output_elements": sum(o["elements"] for o in d["outputs"]),
                                "persistent_bindings": {"inputs": d["inputs"], "outputs": d["outputs"]}}
        constants.update({f"ada_embedding_{k}": build.path("dit", "sidecars", f"ada_embedding_{k}") for k in range(4)})
        rewrite, label = (lambda plan: dit.bound_plan(plan, family, manifest)), s["precision"]
    checkpoint = Path(s["checkpoint"])
    steps = int(checkpoint.name.rsplit("-", 1)[-1]) if re.fullmatch(r"checkpoint-\d+", checkpoint.name) else 0
    work = build.path("bundles-work")
    shutil.rmtree(work, ignore_errors=True)
    if bar := ENVELOPE.get(s["precision"]):  # the fixture check's pass mark is this precision's envelope
        from dataclasses import replace as _replace

        spec = _replace(spec, smoke=_replace(spec.smoke, cosine_min=bar["cosine"], mean_abs_max=bar["mean_abs"]))
    bundle = build_bundle(BundleInputs(
        spec=spec, elfs=elfs, dataset=s["label"], steps=steps, checkpoint=str(checkpoint), fixtures=fixtures,
        constants=constants, robot_files=robot_files, precisions=precisions, source="polima-compile-groot",
        model_version=family[1:], task_label=s["label"], plan_rewrite=rewrite, extra_graphs=extra,
        precision_label=label), work)
    plan = json.loads(bundle.plan_path.read_text())
    sizes = {i["name"]: i["elements"] for g in extra.values() for i in g["persistent_bindings"]["inputs"]}
    for step in plan["steps"]:
        if step["op"] == "run_bound_chain":
            for bind, buffer in zip(step["args"]["bind"], step["args"]["in"], strict=True):
                if bind in sizes and plan["buffers"][buffer] != sizes[bind]:
                    raise StageError(f"binding {bind} ({sizes[bind]}) != plan buffer {buffer} ({plan['buffers'][buffer]})")
    final = build.path("bundles")
    shutil.rmtree(final, ignore_errors=True)
    final.mkdir()
    if s["precision"] == "w8a8":
        return Path(shutil.copytree(bundle.root, final / bundle.root.name, copy_function=os.link))
    return derive_bundle(bundle.root, final, CHAIN_LEVELS[family], label_suffix="", verify=False)


# ---------------------------------------------------------------------- check


def board_check(build: Build, board: str) -> dict:
    """Run the bundle fixture with the DevKit's installed polima_cli (address or saved
    DevKit name) and compare with the PyTorch reference in the release metric."""
    import numpy as np

    from polima.config import devkits
    from polima.config.base import BoardConfig
    from polima.deploy.smoke import compare_normalized_actions

    bundle = Path(build.settings["bundle"])
    target = BoardConfig(host=board)
    devkits.apply(target)
    host, ssh = target.host, ["ssh", *target.ssh_options]
    workdir = f"{target.root}/compile-check"
    remote, out = f"{workdir}/bundles/{bundle.name}", f"{workdir}/results/{bundle.name}.f32"
    local = build.path("check", "actions.f32")
    local.parent.mkdir(exist_ok=True)
    steps = (
        ([*ssh, host, f"mkdir -p {workdir}/bundles {workdir}/results"], "could not reach {host}"),
        (["rsync", "-a", "-e", " ".join(ssh), f"{bundle}/", f"{host}:{remote}/"], "could not copy the bundle to {host}"),
        ([*ssh, host, (f"flock /tmp/polima-bench.lock {target.root}/bin/polima_cli --bundle {remote} "
                       f"--input-dir {remote}/fixtures/inputs --output {out}")],
         ("the bundle did not run on {host}; PoLiMa must be installed there and the model server "
          "stopped while checking (`polima server stop`)")),
        (["rsync", "-a", "-e", " ".join(ssh), f"{host}:{out}", str(local)], "could not read the result from {host}"),
    )
    for argv, problem in steps:
        if subprocess.run(argv, check=False).returncode:
            raise StageError(problem.format(host=host))
    plan = json.loads((bundle / "plan.json").read_text())
    horizon, width = plan["wire"]["response_shape"]
    actual = np.fromfile(local, "<f4").reshape(horizon, width)
    expected = np.fromfile(bundle / "fixtures/expected/normalized_actions.f32", "<f4").reshape(horizon, width)
    state = np.fromfile(bundle / "fixtures/inputs/state.f32", "<f4")
    with np.load(bundle / "fixtures/normalization_stats.npz") as stats:
        mean = np.broadcast_to(stats["action_mean"], (horizon, width)).copy()
        std = np.broadcast_to(stats["action_std"], (horizon, width)).copy()
    relative = int(plan.get("runtime_context", {}).get("relative_action_dim", 0))
    bar = ENVELOPE.get(build.settings["precision"], {"cosine": -1.0, "mean_abs": float("inf")})  # w8a8: report
    report = {}
    for name, cut in (("all", slice(None)), ("moves_1_16", slice(0, 16)), (f"moves_17_{horizon}", slice(16, None))):
        report[name] = compare_normalized_actions(actual[cut], expected[cut], mean[cut], std[cut], state=state,
                                                  relative_dim=relative, name=name, cosine_min=bar["cosine"],
                                                  mean_abs_max=bar["mean_abs"]).to_dict()
    ok = report["all"]["ok"]
    summary = "; ".join(f"{k} cos {v['cosine']:.5f} nMAE {v['mean_abs']:.5f}" for k, v in report.items())
    (build.path("check", "report.json")).write_text(json.dumps({"ok": ok, "envelope": bar, **report}, indent=2) + "\n")
    build.log(f"fixture vs PyTorch: {summary} -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "summary": summary}


# ------------------------------------------------------------------- workers


def worker(stage: str, root: Path) -> int:
    """Executed inside the GR00T interpreter of the build's family."""
    build = Build(root)
    s = build.settings
    if stage == "export" and s["family"] == "n1.6":
        _export_n16(build)
    elif stage == "export":
        from polima.policies.groot import n17

        n17.export_build(Path(s["checkpoint"]), Path(s["dataset"]), build.path("export"),
                         source=Path(s["source"]), smooth=s["precision"] != "bf16")
    elif stage == "dit":
        from polima.policies.groot import dit

        dit.export_variant(s["family"], Path(s["checkpoint"]), build.path("export"), build.path("dit"),
                           source=Path(s["source"]))
    else:
        raise StageError(f"no worker for {stage}")
    return 0


def _export_n16(build: Build) -> None:
    from polima.export.driver import export
    from polima.policies.registry import get_policy

    s = build.settings
    spec = get_policy("groot").specialize_profile(
        checkpoint=s["checkpoint"], dataset_root=s["dataset"], embodiment_tag=s.get("embodiment_tag"))
    result = export(spec, s["checkpoint"], build.path("export"), dataset_root=s["dataset"],
                    calibration_samples=96, verify=True)
    report = result.verification or {}
    print(f"export: {len(result.graphs)} graphs, {result.calibration_samples} observations, "
          f"ONNX vs PyTorch ok={report.get('ok')} max_abs={report.get('max_abs')}", flush=True)
    if not report.get("ok"):
        raise StageError("the ONNX chain does not match PyTorch; nothing was compiled")


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polima compile --policy groot",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", default="groot", choices=("groot",))
    p.add_argument("--checkpoint", type=Path, required=True, help="GR00T N1.6 or N1.7 checkpoint dir")
    p.add_argument("--dataset", "--dataset-root", dest="dataset", type=Path, required=True,
                   help="the LeRobot v2.0/v2.1 training dataset (calibration rows, robot profile)")
    p.add_argument("--output", type=Path, required=True, help="build directory (bundle under bundles/)")
    p.add_argument("--precision", choices=PRECISIONS, default="mixed",
                   help="mixed: INT8 weights in the transformer blocks (default); bf16; "
                        "w8a8: INT8 activations, plain runtime, fails the accuracy bar")
    p.add_argument("--embodiment-tag", help="checkpoint embodiment (default: the trained one)")
    p.add_argument("--label", help="task label in the bundle id (default: the --output name)")
    p.add_argument("-j", "--jobs", type=int, default=3, choices=(1, 2, 3),
                   help="graphs compiled at once (each ~1.6 GB; host-wide limit applies too)")
    p.add_argument("--resume", action="store_true", help="continue an existing build directory")
    p.add_argument("--stop-after", choices=STAGES)
    p.add_argument("--check-devkit", "--check-board", dest="check_board", metavar="ADDRESS",
                   help="check the bundle on this DevKit (user@host or saved name) against PyTorch")
    return p


def main(argv: list[str] | None = None, config=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--worker"]:
        return worker(argv[1], Path(argv[2]))
    args = build_parser().parse_args(argv)
    if config is None:
        from polima.config.loader import load

        config = load()
    try:
        build = _prepare(args, config)
        if args.precision == "w8a8":
            build.log("WARNING: w8a8 builds fail the release accuracy bar (INT8 activations); "
                      "they run the plain per-graph plan with no device chains")
        for stage in STAGES:
            checking = stage == "check" and bool(build.settings.get("check_board"))
            if build.done(stage) and not checking:  # a DevKit given now is always checked
                build.log(f"{stage} already done (resume)")
            else:
                build.log(f"{stage} start")
                started = time.monotonic()
                RUNNERS[stage](build)
                if stage != "check" or checking:  # a skipped check stays to do
                    build.mark(stage, f"{time.monotonic() - started:.0f}s")
            if stage == args.stop_after:
                break
    except StageError as error:
        print(f"polima compile: {error}", file=sys.stderr)
        return 1
    print(f"bundle: {build.settings.get('bundle', '(not built yet)')}")
    return 0


def _prepare(args, config) -> Build:
    root = args.output.resolve()
    checkpoint, dataset = args.checkpoint.resolve(), args.dataset.resolve()
    if not (checkpoint / "config.json").is_file():
        raise StageError(f"{checkpoint} is not a GR00T checkpoint folder (it has no config.json)")
    if not (dataset / "meta/info.json").is_file():
        raise StageError(f"{dataset} is not a LeRobot dataset (it has no meta/info.json)")
    family = detect_family(checkpoint)
    info = json.loads((dataset / "meta/info.json").read_text())
    if info.get("codebase_version") not in ("v2.0", "v2.1"):
        raise StageError(f"{dataset} is LeRobot {info.get('codebase_version')}; GR00T needs v2.0/v2.1")
    found = interpreters(config, family)
    backbone = config.paths.groot17_backbone.strip() if family == "n1.7" else ""
    if backbone and Path(backbone).expanduser().is_dir():
        backbone = str(Path(backbone).expanduser().absolute())
    settings = {"backbone": backbone or None, "checkpoint": str(checkpoint), "dataset": str(dataset), "family": family,
                "precision": args.precision, "embodiment_tag": args.embodiment_tag,
                "python": found["python"], "source": found["source"], "sdk": found["sdk"],
                "jobs": args.jobs, "check_board": args.check_board, "label": args.label or root.name}
    if root.exists() and not args.resume:
        raise StageError(f"{root} exists; pass --resume to continue it or choose a new --output")
    if args.resume and (root / "build.json").is_file():
        old = json.loads((root / "build.json").read_text())
        for key in ("checkpoint", "dataset", "family", "precision", "backbone"):
            if old.get(key) != settings[key]:
                raise StageError(f"--resume with a different {key}: {old.get(key)} != {settings[key]}")
        settings = {**old, **{k: v for k, v in settings.items() if k in ("jobs", "check_board", "python",
                                                                         "source", "sdk", "label")}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "build.json").write_text(json.dumps(settings, indent=2) + "\n")
    build = Build(root)
    build.log(f"GR00T {family} {args.precision}: {checkpoint} + {dataset} -> {root}")
    return build


if __name__ == "__main__":
    raise SystemExit(main())
