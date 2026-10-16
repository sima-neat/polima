"""`polima compile --policy pi05`: a LeRobot Pi0.5 checkpoint + a LeRobot dataset ->
a Modalix bundle.

    polima compile --policy pi05 --checkpoint CKPT --dataset LEROBOT_DIR --output DIR
                   [--precision bf16|w8a16|w8a8] [--task TEXT] [--camera-map name=dataset_cam ...]
                   [--calibration-samples N] [--jobs 1..3] [--resume] [--stop-after STAGE]
                   [--check-devkit ADDRESS]

Stages (each with a marker in DIR/stages/ and a log in DIR/logs/):

  export   Pi0.5 env ($POLIMA_PI05_PYTHON / paths.pi05_python: torch, transformers >= 5,
           LeRobot with pi05). Static PyTorch graphs (policies/pi05/modules.py) checked
           against LeRobot's PI05Pytorch; eleven ONNX graphs; the ONNX chain checked
           against LeRobot with onnxruntime; calibration rows from dataset observations;
           the fixture; the robot client's tokenizer, embedding table and statistics.
  compile  ModelSDK, every graph, <= --jobs at once and <= the host-wide compiler limit.
           w8a16: INT8 weights, BF16 activations in vision, Gemma and expert graphs;
           w8a8: INT8 activations too (reported, not expected to pass the accuracy bar).
  bundle   PoLiMa venv. Graph inputs reordered to each ELF's MLA input order.
  check    Optional (--check-devkit): the fixture on a Modalix DevKit vs LeRobot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from polima.compile.groot import Build as _Build
from polima.compile.groot import StageError, _mla_input_order, _sha, int8_boundaries, start_compiler

STAGES = ("export", "llima", "compile", "bundle", "check")
PRECISIONS = ("bf16", "w8a16", "w8a8")
SRC = Path(__file__).resolve().parents[2]
#: Fixture vs LeRobot over the whole normalized chunk. w8a8 is reported, never gated.
ENVELOPE = {"bf16": {"cosine": 0.999, "mean_abs": 0.01}, "w8a16": {"cosine": 0.995, "mean_abs": 0.03}}
HEAVY = ("vision", "gemma_", "expert")
LABELS = {("bf16", "bf16"): "bf16", ("bf16", "int8"): "w8a16", ("int8", "int8"): "a8w8"}


def input_order(work: Path, name: str, declared: list[str], known: list[str] | None = None) -> list[str]:
    """The compiled ELF's MLA input order, from its MPK.

    It is not the ONNX order in general (the expert compiles as x, cond, rope, cache,
    bias), and the runner concatenates the inputs in exactly this order, so it is
    never guessed: without a readable MPK the build must record a verified order
    (build.json "input_orders").
    """
    import tarfile

    try:
        return _mla_input_order(work, name)
    except (EOFError, OSError, tarfile.TarError, StopIteration, KeyError) as error:
        if known:
            return list(known)
        raise StageError(f"{name}: no readable MPK to read its compiled input order from ({error}); "
                         f"recompile it or record a verified order in build.json input_orders") from error


class Build(_Build):
    def run_worker(self, stage: str, python: str, *, gpu: bool = False) -> None:
        log = self.path("logs", f"{stage}.log")
        log.parent.mkdir(exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(SRC), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))])
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["CUDA_VISIBLE_DEVICES"] = "" if not gpu else env.get("CUDA_VISIBLE_DEVICES", "")
        with log.open("a") as stream:
            code = subprocess.run([python, "-u", "-m", "polima.compile.pi05", "--worker", stage, str(self.root)],
                                  env=env, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise StageError(f"{stage} failed (exit {code}); see {log}")


# --------------------------------------------------------------------- stages


def stage_export(build: Build) -> None:
    build.run_worker("export", build.settings["python"])
    report = json.loads(build.path("export", "verification.json").read_text())
    if build.settings.get("backend") == "llima":
        modules = report["modules_vs_lerobot"]
        expert = report["expert_onnx_vs_modules"]
        if modules["cosine"] < 0.9999 or expert["cosine"] < 0.9999:
            raise StageError(f"the exported graphs do not match LeRobot ({report}); nothing compiled")
        build.log(f"export: modules vs LeRobot cos {modules['cosine']:.7f}; expert ONNX cos {expert['cosine']:.7f}")
        return
    onnx = report["onnx_vs_lerobot"]
    if onnx["cosine"] < 0.9999:
        raise StageError(f"the ONNX graphs do not match LeRobot (cosine {onnx['cosine']:.6f}); nothing compiled")
    build.log(f"export: ONNX chain vs LeRobot cos {onnx['cosine']:.7f} max_abs {onnx['max_abs']:.2e}")


def graph_jobs(build: Build) -> list[dict]:
    precision, export = build.settings["precision"], build.path("export")
    from polima.policies.pi05.runtime import block_names

    llima = build.settings.get("backend", "onnx") == "llima"
    names = ["expert"] if llima else ["vision", *block_names(), "expert"]
    jobs = []
    for name in names:
        heavy = name.startswith(HEAVY)
        act = "int8" if heavy and precision == "w8a8" else "bf16"
        weights = "int8" if heavy and precision != "bf16" else "bf16"
        if name == "expert" and build.settings.get("backend") == "llima":
            # ModelSDK 2.1.3's all-BF16 code for the 18-layer expert returns ~1e37 on the
            # Modalix (also recorded for SmolVLA's denoiser); BF16 activations with INT8
            # weights are accurate (cos 0.9997 per step). A "bf16" build keeps BF16 vision
            # and prefill and labels its expert w8a16. --expert-weights int4: half the
            # weight traffic of this bandwidth-bound 50-token step (LLiMa's A_BF16_W_INT4).
            weights = build.settings.get("expert_weights", "int8")
        # Graphs with several inputs keep their MPK (it records the compiled input
        # order), so they never stop early at a stable ELF.
        flags = ["--external-dram-layout", "HWC"] + (["--exit-on-stable-elf"] if name == "vision" else [])
        if llima and name == "expert":
            # The simplifier drops the norms' Mul(ones), which ModelSDK needs to fuse
            # RMSNorm into one kernel; unfused, the MLA's INT8 rsqrt lookup costs ~6%
            # relative error per expert step (blocks._rms_channels).
            flags.append("--no-simplify")
        # The device denoise loop needs the Euler expert with one persistent binding per
        # input: quantize only here, then compile_all runs the shim's persistent mode.
        persistent = llima and name == "expert" and device_denoise(build.settings)
        if persistent:
            flags.append("--no-compile")
        jobs.append({"name": name, "onnx": export / "onnx" / f"{name}.onnx", "activation": act, "weights": weights,
                     "calibration": export / "calibration" / f"{name}.npz" if weights == "int8" else None,
                     "layout": "NCHW" if llima and name == "expert" and build.settings.get("expert_form") == "nchw"
                     else "NHWC", "flags": flags, "persistent": persistent})
    return jobs


def device_denoise(settings: dict) -> bool:
    """Whether the ten Euler steps run as one device program (LLiMa backend, NCHW expert).
    Builds recorded before the option existed ran them on the APU."""
    return (settings.get("backend") == "llima" and settings.get("expert_form") == "nchw"
            and settings.get("denoise", "host") == "device")


def compile_all(build: Build) -> list[str]:
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
        return out / job["name"] / "retained" / f"{job['name']}_stage1_mla.elf"

    def key(job) -> dict:
        return {"source_sha256": _sha(job["onnx"]), "calibration_sha256": _sha(job["calibration"]),
                "activation": job["activation"], "weights": job["weights"], "layout": job["layout"],
                "flags": job["flags"], "persistent": job.get("persistent", False)}

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
        cmd = [sdk.as_posix(), "-u", shim, "--model-path", str(job["onnx"]), "--build-dir", str(work / "compiled"),
               "--model-layout", job["layout"], "--activation-precision", job["activation"],
               "--weight-precision", job["weights"], "--retain-compile-dir", str(work / "retained"), *job["flags"]]
        if job["calibration"]:
            cmd += ["--calibration-npz", str(job["calibration"])]
        started, record = time.time(), dict(wanted)
        with (work / "compile.log").open("w") as log:
            code = start_compiler(cmd, env, log)
            if code == 0 and job.get("persistent"):  # quantized only; now one binding per MLA input
                code = start_compiler([sdk.as_posix(), "-u", shim, "persistent", "--source",
                                       str(work / "compiled" / name), "--name", name,
                                       "--output", str(work / "persistent")], env, log)
        elf = elf_of(job)
        retained = work / ("persistent" if job.get("persistent") else "") / "retained"
        if not elf.is_file():  # the retained tree names the ELF after the graph; find it if not
            found = sorted(retained.rglob(f"{name}*_mla.elf")) if retained.exists() else []
            if found:
                elf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found[0], elf)
        ok = code == 0 and elf.is_file() and elf.read_bytes()[:4] == b"\x7fELF"
        record.update(returncode=0 if ok else (code or 1), elf=str(elf), minutes=round((time.time() - started) / 60, 1),
                      elf_sha256=_sha(elf) if ok else None)
        return record

    jobs = graph_jobs(build)
    build.log(f"compile {len(jobs)} graphs ({build.settings['precision']}), {build.settings['jobs']} at a time")
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


def stage_compile(build: Build) -> None:
    failed = compile_all(build)
    if failed:
        raise StageError(f"{len(failed)} graph(s) failed: {', '.join(failed)}; see "
                         f"{build.path('compiled', 'compile_results.json')}; --resume compiles only those")


def make_bundle(build: Build) -> Path:
    from dataclasses import replace

    from polima.bundle.pack import BundleInputs, build_bundle
    from polima.bundle.retained import ElfCandidate
    from polima.policies.registry import get_policy

    s, export, compiled = build.settings, build.path("export"), build.path("compiled")
    results = json.loads((compiled / "compile_results.json").read_text())
    jobs = {job["name"]: job for job in graph_jobs(build)}
    if missing := [n for n in jobs if results.get(n, {}).get("returncode") != 0]:
        raise StageError(f"no successful compile for {missing}")
    profile = json.loads((export / "profile.json").read_text())
    spec = get_policy("pi05").specialize_profile(profile=profile)
    elfs = {n: ElfCandidate(n, Path(results[n]["elf"]), n, results[n]["elf_sha256"],
                            Path(results[n]["elf"]).stat().st_size) for n in jobs}
    precisions = {n: LABELS[(j["activation"], j["weights"])] for n, j in jobs.items()}
    # Multi-input ELFs read one buffer that is the concatenation of their inputs in
    # the compiler's order, which need not be the ONNX order.
    orders = {}
    for graph in spec.compile.graphs:
        if len(graph.inputs) > 1 and jobs[graph.name]["activation"] == "bf16":
            declared = [t.name for t in graph.inputs]
            known = (s.get("input_orders") or {}).get(graph.name)
            orders[graph.name] = (input_order(compiled / graph.name, graph.name, declared, known), declared)
    extra, int8_rewrite = {}, None
    if s["precision"] == "w8a8":
        extra, int8_rewrite = int8_boundaries(compiled, spec, jobs)

    def rewrite(plan: dict) -> dict:
        for step in plan["steps"]:
            name = step["args"].get("graph") if step["op"] == "run_elf" else None
            if name in orders:
                order, declared = orders[name]
                feeds = dict(zip(declared, step["args"]["in"], strict=True))
                step["args"]["in"] = [feeds[k] for k in order]
        return int8_rewrite(plan) if int8_rewrite else plan

    if bar := ENVELOPE.get(s["precision"]):
        spec = replace(spec, smoke=replace(spec.smoke, cosine_min=bar["cosine"], mean_abs_max=bar["mean_abs"]))
    else:  # w8a8: report the agreement, never fail the check on it
        spec = replace(spec, smoke=replace(spec.smoke, cosine_min=-1.0, mean_abs_max=1e9))
    # Sidecars are read by bare name (constants/<name>), as GR00T's ada_embedding_k.
    constants = {f"cond_{k:02d}": export / "constants" / f"cond_{k:02d}.f32" for k in range(10)}
    root = export / "fixtures"
    fixtures = {str(p.relative_to(root)): p for p in root.rglob("*.f32")}
    robot_files = pi05_robot_files(spec, export, Path(s["dataset"]))
    work = build.path("bundles-work")
    shutil.rmtree(work, ignore_errors=True)
    checkpoint = Path(s["checkpoint"])
    bundle = build_bundle(BundleInputs(
        spec=spec, elfs=elfs, dataset=s["label"], steps=0, checkpoint=str(checkpoint), fixtures=fixtures,
        constants=constants, robot_files=robot_files, precisions=precisions, source="polima-compile-pi05",
        task_label=s["label"], plan_rewrite=rewrite, extra_graphs=extra, precision_label=s["precision"],
        tool_versions={"onnx_vs_lerobot": json.loads((export / "verification.json").read_text())["onnx_vs_lerobot"]},
    ), work)
    final = build.path("bundles")
    shutil.rmtree(final, ignore_errors=True)
    final.mkdir()
    return Path(shutil.copytree(bundle.root, final / bundle.root.name, copy_function=os.link))


def stage_bundle(build: Build) -> None:
    if build.settings.get("backend") == "llima":
        bundle = make_llima_bundle(
            export=build.path("export"), llima_mpk=build.path("llima", "out", "sima_files", "mpk"),
            expert_dir=build.path("compiled", "expert", *(["persistent"] if device_denoise(build.settings) else [])),
            work=build.path("bundles-work-llima"),
            output=build.path("bundles"), precision=build.settings["precision"], label=build.settings["label"],
            checkpoint=build.settings["checkpoint"], known_orders=build.settings.get("input_orders") or {},
            expert_weights=build.settings.get("expert_weights", "int8"), dataset=Path(build.settings["dataset"]))
    else:
        bundle = make_bundle(build)
    build.settings["bundle"] = str(bundle)
    build.path("build.json").write_text(json.dumps(build.settings, indent=2) + "\n")
    build.log(f"bundle {bundle}")


def board_check(build: Build, board: str) -> dict:
    """The bundle's fixtures on a DevKit vs LeRobot (normalized actions). With a fixture set
    (fixtures/set/sNN, one observation and noise each) the envelope applies to their mean
    cosine and mean absolute error; the worst fixture is reported too."""
    import numpy as np

    from polima.config import devkits
    from polima.config.base import BoardConfig

    bundle = Path(build.settings["bundle"])
    target = BoardConfig(host=board)
    devkits.apply(target)
    host, ssh = target.host, ["ssh", *target.ssh_options]
    workdir = f"{target.root}/compile-check"
    remote, results = f"{workdir}/bundles/{bundle.name}", f"{workdir}/results/{bundle.name}"
    local = build.path("check", "actions")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    root = bundle / "fixtures"
    samples = sorted(p.name for p in (root / "set").glob("s[0-9][0-9]")) if (root / "set").is_dir() else []
    inputs = {name: f"{remote}/fixtures/set/{name}/inputs" for name in samples} or {"s00": f"{remote}/fixtures/inputs"}
    expected = {name: root / "set" / name / "expected/normalized_actions.f32" for name in samples} or \
        {"s00": root / "expected/normalized_actions.f32"}
    cli = os.environ.get("POLIMA_CHECK_CLI", target.root + "/bin/polima_cli")
    runs = " && ".join(f"{cli} --bundle {remote} --input-dir {where} --output {results}/{name}.f32"
                       for name, where in inputs.items())
    for argv, problem in (
        ([*ssh, host, f"mkdir -p {workdir}/bundles && rm -rf {results} && mkdir -p {results}"],
         "could not reach {host}"),
        (["rsync", "-a", "--exclude", "robot_client/pi05/embedding.f16", "-e", " ".join(ssh),
          f"{bundle}/", f"{host}:{remote}/"], "could not copy the bundle to {host}"),
        ([*ssh, host, f"flock /tmp/polima-bench.lock sh -c '{runs}'"],
         "the bundle did not run on {host}; PoLiMa must be installed there and the policy server stopped"),
        (["rsync", "-a", "-e", " ".join(ssh), f"{host}:{results}/", f"{local}/"], "could not read the results from {host}"),
    ):
        if subprocess.run(argv, check=False).returncode:
            raise StageError(problem.format(host=host))
    plan = json.loads((bundle / "plan.json").read_text())
    shape = tuple(plan["wire"]["response_shape"])
    per = {}
    for name, path in expected.items():
        actual = np.fromfile(local / f"{name}.f32", "<f4").reshape(shape).astype(np.float64)
        want = np.fromfile(path, "<f4").reshape(shape).astype(np.float64)
        per[name] = {"cosine": float((actual * want).sum() / (np.linalg.norm(actual) * np.linalg.norm(want))),
                     "mean_abs": float(np.abs(actual - want).mean()), "max_abs": float(np.abs(actual - want).max()),
                     "finite": bool(np.isfinite(actual).all())}
    worst = max(per, key=lambda name: per[name]["mean_abs"])
    report = {"fixtures": len(per), "cosine": float(np.mean([r["cosine"] for r in per.values()])),
              "mean_abs": float(np.mean([r["mean_abs"] for r in per.values()])),
              "max_abs": max(r["max_abs"] for r in per.values()),
              "finite": all(r["finite"] for r in per.values()),
              "worst": {"fixture": worst, **per[worst]}, "per_fixture": per}
    bar = ENVELOPE.get(build.settings["precision"])
    report["ok"] = bool(report["finite"] and (bar is None or (report["cosine"] >= bar["cosine"]
                                                             and report["mean_abs"] <= bar["mean_abs"])))
    report["envelope"] = bar
    build.path("check", "report.json").write_text(json.dumps(report, indent=2) + "\n")
    build.log(f"{len(per)} fixture(s) on {host} vs LeRobot: mean cos {report['cosine']:.6f} "
              f"mean nMAE {report['mean_abs']:.5f} (worst {worst}: {per[worst]['mean_abs']:.5f}) "
              f"max {report['max_abs']:.4f} -> {'PASS' if report['ok'] else 'FAIL'}"
              f"{' (w8a8: reported, not gated)' if bar is None else ''}")
    return report


def stage_check(build: Build) -> None:
    board = build.settings.get("check_board")
    if not board:
        build.log("check skipped: pass --check-devkit to run the fixture on a DevKit")
        return
    report = board_check(build, board)
    if not report["ok"] and build.settings["precision"] in ENVELOPE:
        raise StageError(f"fixture check outside the envelope: mean cos {report['cosine']:.5f} "
                         f"mean nMAE {report['mean_abs']:.5f} over {report['fixtures']} fixture(s)")


LLIMA_CONFIG = """# polima compile --policy pi05: LLiMa layer configuration for Pi0.5's PaliGemma prefill.
# Vision and every prefill group graph {precision}; no decode graphs; no post graph
# for the last layer (Pi0.5 uses only its K/V, which pre_layer17 writes).
def get_layer_configuration(model_properties, layer):
    last = model_properties["num_hidden_layers"] - 1
    part = layer["part"]
    if part == "VISION":
        return {{"compile": True, "precision": "{precision}"}}
    if not layer["is_group"]:
        return {{"compile": False}}
    if part == "CACHE":
        return {{"compile": layer["index"] == 0, "precision": "BF16"}}
    if part == "POST" and layer["index"] == last:
        return {{"compile": False}}
    if part in ("PRE", "POST"):
        return {{"compile": True, "precision": "{precision}"}}
    return {{"compile": False}}
"""


def stage_llima(build: Build) -> None:
    """The PaliGemma half with LLiMa: one 1024-token prefill group, BF16 K/V cache.

    `--max_num_tokens 2048` with a 1024-token group: with the cache exactly one group
    long ModelSDK 2.1.3 fails generating the K/V outputs (gen_output NameError).
    SIMA_MLA_COMPILE_USE_JAX=0: the compiler's host check cannot build BF16 code with
    JAX/LLVM on some x86 CPUs (see polima.compile.tensor.HOST_CHECK_FAILURE).
    """
    from polima.compile.toolchain import compiler_env

    if build.settings.get("backend") != "llima":
        build.log("llima skipped (ONNX backend)")
        return
    sdk = Path(build.settings["sdk"])
    work = build.path("llima")
    work.mkdir(exist_ok=True)
    precision = "BF16" if build.settings["precision"] == "bf16" else "A_BF16_W_INT8"
    config = work / "layers.py"
    config.write_text(LLIMA_CONFIG.format(precision=precision))
    model = work / "pi05_paligemma"
    if not model.exists():
        model.symlink_to(build.path("export", "paligemma"))
    env = compiler_env(sdk.parent, source_root=SRC)
    env.update(CUDA_VISIBLE_DEVICES="", SIMA_MLA_COMPILE_USE_JAX="0")
    profile = json.loads(build.path("export", "profile.json").read_text())
    group = int(profile["runtime_context"]["prefix_tokens"])
    cmd = [str(sdk.parent / "llima-compile"), "pi05_paligemma", "-o", "out", "-c", str(config),
           "-j", str(build.settings.get("llima_jobs", 12)), "--max_num_tokens", "2048",
           "--language_group_size", str(group),
           "--no-quantize_kv_cache", "--no-quantize_embeddings", "--log_level", "INFO", "--resume"]
    with build.path("logs", "llima.log").open("a") as log:
        code = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode

    missing = [n for n in llima_elf_names(group).values()
               if not (work / "out/sima_files/mpk" / f"{n}_mpk.tar.gz").is_file()]
    if code or missing:
        raise StageError(f"llima-compile exit {code}; missing {missing[:4]}{'...' if len(missing) > 4 else ''}; "
                         f"see {build.path('logs', 'llima.log')}; --resume keeps finished graphs")
    build.log(f"llima: {len(llima_elf_names(group))} ELFs, {group}-token group ({precision} vision and layers)")


RUNNERS = {"export": stage_export, "llima": stage_llima, "compile": stage_compile, "bundle": stage_bundle,
           "check": stage_check}


# -------------------------------------------------------------------- worker


def worker(stage: str, root: Path) -> int:
    build = Build(root)
    s = build.settings
    if stage != "export":
        raise StageError(f"no worker for {stage}")
    from polima.policies.pi05.graphs import export_build

    mapping = dict(item.split("=", 1) for item in s.get("camera_map") or [])
    report = export_build(Path(s["checkpoint"]), Path(s["dataset"]), build.path("export"),
                          samples=s["calibration_samples"], task=s.get("task"), mapping=mapping or None,
                          calibrate=True, backend=s.get("backend", "onnx"),
                          expert_form=s.get("expert_form", "nhwc"),
                          denoise="device" if device_denoise(s) else "host")
    print(json.dumps(report), flush=True)
    return 0


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polima compile --policy pi05", description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", default="pi05", choices=("pi05",))
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="LeRobot pi05 checkpoint folder (config.json + model.safetensors)")
    p.add_argument("--dataset", "--dataset-root", dest="dataset", type=Path, required=True,
                   help="a LeRobot dataset with the checkpoint's cameras: calibration rows and the fixture")
    p.add_argument("--output", type=Path, required=True, help="build directory (bundle under bundles/)")
    p.add_argument("--backend", choices=("llima", "onnx"), default="llima",
                   help="llima: the PaliGemma half compiled by LLiMa and run device-resident (default); "
                        "onnx: PoLiMa's own two-layer partitions")
    p.add_argument("--expert-form", choices=("nchw", "nhwc"), default="nchw",
                   help="LLiMa backend: the action expert in GR00T N1.7's DiT form (nchw, default) or row form")
    p.add_argument("--denoise", choices=("device", "host"), default="device",
                   help="LLiMa backend, NCHW expert: run the ten Euler steps as one device program "
                        "(default), or step by step with the Euler update on the APU")
    p.add_argument("--expert-weights", choices=("int8", "int4"), default="int8",
                   help="LLiMa backend: the expert's weight precision (activations stay BF16)")
    p.add_argument("--precision", choices=PRECISIONS, default="w8a16",
                   help="w8a16: INT8 weights, BF16 activations (default); bf16; w8a8: INT8 activations too")
    p.add_argument("--task", help="instruction for the fixture and calibration (default: the dataset's)")
    p.add_argument("--camera-map", nargs="+", metavar="CKPT_CAM=DATASET_CAM",
                   help="which dataset camera feeds each checkpoint camera (default: by name)")
    p.add_argument("--calibration-samples", type=int, default=16, help="dataset observations to calibrate on")
    p.add_argument("--label", help="label in the bundle id (default: the --output name)")
    p.add_argument("-j", "--jobs", type=int, default=3, choices=(1, 2, 3), help="graphs compiled at once")
    p.add_argument("--resume", action="store_true", help="continue an existing build directory")
    p.add_argument("--stop-after", choices=STAGES)
    p.add_argument("--check-devkit", "--check-board", dest="check_board", metavar="ADDRESS",
                   help="run the fixture on this DevKit (user@host or saved name) and compare with LeRobot")
    return p


def interpreters(config) -> dict:
    from polima.compile.toolchain import require_compiler_python

    value = getattr(getattr(config, "paths", None), "pi05_python", None)
    if not value or not Path(value).exists():
        raise StageError("Pi0.5 export needs POLIMA_PI05_PYTHON (or paths.pi05_python): an interpreter with "
                         "torch, transformers>=5,<5.6 and LeRobot installed with [pi]")
    return {"python": str(value), "sdk": str(require_compiler_python(config))}


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
            build.log("WARNING: w8a8 (INT8 activations) is reported, not expected to pass the accuracy bar")
        for stage in STAGES:
            checking = stage == "check" and bool(build.settings.get("check_board"))
            if build.done(stage) and not checking:
                build.log(f"{stage} already done (resume)")
            else:
                build.log(f"{stage} start")
                started = time.monotonic()
                RUNNERS[stage](build)
                if stage != "check" or checking:
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
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise StageError(f"{checkpoint} is not a Pi0.5 checkpoint folder (config.json + model.safetensors)")
    if json.loads((checkpoint / "config.json").read_text()).get("type") != "pi05":
        raise StageError(f"{checkpoint}/config.json is not a pi05 policy")
    if not (dataset / "meta/info.json").is_file():
        raise StageError(f"{dataset} is not a LeRobot dataset (it has no meta/info.json)")
    found = interpreters(config)
    settings = {"checkpoint": str(checkpoint), "dataset": str(dataset), "precision": args.precision,
                "task": args.task, "camera_map": args.camera_map, "calibration_samples": args.calibration_samples,
                "python": found["python"], "sdk": found["sdk"], "jobs": args.jobs,
                "check_board": args.check_board, "label": args.label or root.name, "backend": args.backend,
                "expert_form": args.expert_form, "expert_weights": args.expert_weights, "denoise": args.denoise}
    if root.exists() and not args.resume:
        raise StageError(f"{root} exists; pass --resume to continue it or choose a new --output")
    if args.resume and (root / "build.json").is_file():
        old = json.loads((root / "build.json").read_text())
        for key in ("checkpoint", "dataset", "precision", "task", "camera_map", "backend", "expert_form",
                    "expert_weights", "denoise"):
            if old.get(key, "host" if key == "denoise" else None) != settings[key]:
                raise StageError(f"--resume with a different {key}: {old.get(key)} != {settings[key]}")
        settings = {**old, **{k: v for k, v in settings.items() if k in ("jobs", "check_board", "python", "sdk", "label")}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "build.json").write_text(json.dumps(settings, indent=2) + "\n")
    build = Build(root)
    build.log(f"Pi0.5 {args.precision}: {checkpoint} + {dataset} -> {root}")
    return build


if __name__ == "__main__":
    raise SystemExit(main())


# ----------------------------------------------------------- LLiMa-backend bundle

LLIMA_MODEL = "pi05_paligemma"


def llima_elf_names(prefix: int, layers: int = 18) -> dict[str, str]:
    """Bundle graph name -> LLiMa ELF name (sima_lmm.model.language_model._get_part_model)."""
    from polima.policies.pi05 import runtime as rt

    names = {"vision": f"{LLIMA_MODEL}_vision", "attn": f"{LLIMA_MODEL}_language_n{prefix}_cache_token0"}
    for layer in range(layers):
        names[rt.llima_pre(layer)] = f"{LLIMA_MODEL}_language_n{prefix}_pre_layer{layer}"
        if layer + 1 < layers:
            names[rt.llima_post(layer)] = f"{LLIMA_MODEL}_language_n{prefix}_post_layer{layer}"
    return names


def extract_llima(mpk_dir: Path, out: Path, prefix: int) -> dict[str, dict]:
    """ELFs and per-binding element counts (MLA I/O in compiler order) from LLiMa's MPKs."""
    import tarfile

    from polima.util.hashing import sha256_file

    out.mkdir(parents=True, exist_ok=True)
    found = {}
    for graph, name in llima_elf_names(prefix).items():
        archive = mpk_dir / f"{name}_mpk.tar.gz"
        if not archive.is_file():
            raise StageError(f"LLiMa did not produce {archive.name}")
        with tarfile.open(archive) as tar:
            members = tar.getnames()
            meta = json.load(tar.extractfile(next(m for m in members if m.endswith("_mpk.json"))))
            elf = out / f"{graph}_stage1_mla.elf"
            elf.write_bytes(tar.extractfile(next(m for m in members if m.endswith("_stage1_mla.elf"))).read())
        mla = [p for p in meta["plugins"] if p.get("processor") == "MLA"]
        if len(mla) != 1 or any(p["config_params"].get("kernel") not in ("cast_transform", "pass_through")
                                for p in meta["plugins"] if p.get("processor") != "MLA"):
            raise StageError(f"{name}: unexpected MPK plugins; the bare runtime runs only the MLA segment")
        ins = [n["size"] // 2 for n in mla[0]["input_nodes"]]  # BF16 sizes in bytes
        outs = [n["size"] // 2 for n in mla[0]["output_nodes"]]
        found[graph] = {"elf": elf, "sha256": sha256_file(elf), "inputs": ins, "outputs": outs}
    return found


def persistent_expert(expert_dir: Path, prefix: int) -> dict:
    """A persistent-binding (`compile_onnx.py persistent`) Euler expert: its ELF, binding
    sizes and the NCHW input each binding is (by element count, which is unique per input)."""
    from polima.policies.pi05 import runtime as rt

    contract = json.loads((expert_dir / "contract.json").read_text())
    sizes = {count: name for name, (_, count) in rt.expert_layout(prefix, rt.NCHW_EXPERT_INPUTS).items()}
    if len(sizes) != len(rt.NCHW_EXPERT_INPUTS):
        raise StageError("expert inputs are not distinguishable by size")
    inputs = [binding["elements"] for binding in contract["inputs"]]
    order = [sizes.get(count) for count in inputs]
    if None in order or sorted(order) != sorted(rt.NCHW_EXPERT_INPUTS):
        raise StageError(f"persistent expert bindings {inputs} do not match the NCHW expert's inputs")
    outputs = [binding["elements"] for binding in contract["outputs"]]
    if outputs != [rt.CHUNK * rt.ACTION_LANE]:
        raise StageError(f"persistent expert must output x ({rt.CHUNK * rt.ACTION_LANE}), not {outputs}")
    elfs = sorted((expert_dir / "retained").rglob("*_stage1_mla.elf"))
    if len(elfs) != 1:
        raise StageError(f"expected one expert ELF under {expert_dir / 'retained'}, found {len(elfs)}")
    return {"order": order, "elf": elfs[0], "inputs": inputs, "outputs": outputs}


def pi05_robot_files(spec, export: Path, dataset: Path | None) -> dict[str, Path]:
    """The robot client's files: Pi0.5's prompt and normalization assets and, for an SO-101
    checkpoint, the rest pose from its dataset (the SO-101 launcher needs it)."""
    client = export / "client"
    files = {f"pi05/{p.name}": p for p in client.iterdir() if p.is_file()} if client.is_dir() else {}
    if spec.robot.calibration_id == "so-arm101":  # an SO-101 fine-tune can drive the arm
        from polima.bundle.importer import collect_robot_files
        from polima.compile.groot import write_rest_pose

        if dataset is None:
            raise StageError("an SO-101 Pi0.5 bundle needs its dataset for the robot's rest pose")
        write_rest_pose(dataset, export / "robot_client/robot_rest_position.json",
                        spec.dataset.state_names, spec.robot.calibration_id)
        files.update(collect_robot_files(spec, export))
    return files


def make_llima_bundle(*, export: Path, llima_mpk: Path, expert_dir: Path, work: Path, output: Path,
                      precision: str, label: str, checkpoint: str, known_orders: dict | None = None,
                      expert_weights: str = "int8", dataset: Path | None = None) -> Path:
    """One device-resident Pi0.5 bundle: LLiMa's vision + prefill ELFs and PoLiMa's expert ELF."""
    from dataclasses import replace

    from polima.bundle.pack import BundleInputs, build_bundle
    from polima.bundle.retained import ElfCandidate
    from polima.policies.pi05 import graphs as pi05_graphs
    from polima.policies.registry import get_policy
    from polima.util.hashing import sha256_file

    profile = json.loads((export / "profile.json").read_text())
    form = "nchw" if (export / "constants" / "ada_00.f32").is_file() else "nhwc"
    profile["runtime_context"] = {**profile.get("runtime_context", {}), "backend": "llima", "expert_form": form}
    base = get_policy("pi05").specialize_profile(profile=profile)
    cameras = int(base.runtime_context["cameras"])
    prefix = int(base.runtime_context["prefix_tokens"])
    derived = work / "derived"
    pi05_graphs.write_llima_export(export, derived, cameras)
    llima = extract_llima(llima_mpk, work / "llima-elfs", prefix)
    expert_elf = expert_dir / "retained" / "expert_stage1_mla.elf"
    from polima.policies.pi05 import runtime as rt

    declared = list(rt.NCHW_EXPERT_INPUTS if form == "nchw" else rt.EXPERT_INPUTS)
    persistent = persistent_expert(expert_dir, prefix) if (expert_dir / "contract.json").is_file() else None
    if persistent:  # the Euler-folded expert with one binding per input: the loop runs on the device
        order, expert_elf = persistent["order"], persistent["elf"]
        profile["runtime_context"]["denoise"] = "device"
    else:
        order = input_order(expert_dir, "expert", declared, (known_orders or {}).get("expert"))
    profile["runtime_context"]["expert_inputs"] = order
    spec = get_policy("pi05").specialize_profile(profile=profile)
    elfs = {g: ElfCandidate(g, v["elf"], g, v["sha256"], v["elf"].stat().st_size) for g, v in llima.items()}
    elfs["expert"] = ElfCandidate("expert", expert_elf, "expert", sha256_file(expert_elf), expert_elf.stat().st_size)
    extra = {g: {"persistent_input_elements": v["inputs"], "persistent_output_elements": v["outputs"],
                 "input_elements": sum(v["inputs"]), "output_elements": sum(v["outputs"])}
             for g, v in llima.items() if len(v["inputs"]) > 1 or len(v["outputs"]) > 1}
    if persistent:
        extra["expert"] = {"persistent_input_elements": persistent["inputs"],
                           "persistent_output_elements": persistent["outputs"],
                           "input_elements": sum(persistent["inputs"]), "output_elements": sum(persistent["outputs"])}
    layer = "w8a16" if precision != "bf16" else "bf16"
    precisions = {g: (layer if g != "attn" else "bf16") for g in llima}
    precisions["expert"] = "w4a16" if expert_weights == "int4" else \
        {"bf16": "w8a16", "w8a16": "w8a16", "w8a8": "a8w8"}[precision]
    if bar := ENVELOPE.get(precision):
        spec = replace(spec, smoke=replace(spec.smoke, cosine_min=bar["cosine"], mean_abs_max=bar["mean_abs"]))
    root = derived / "fixtures"
    fixtures = {str(p.relative_to(root)): p for p in root.rglob("*.f32")}
    constants = {p.stem: p for p in (derived / "constants").glob("*.f32")}
    robot_files = pi05_robot_files(spec, export, dataset)
    config = robot_files.get("pi05/client.json")
    if config:  # the client builds the LLiMa wire (HWC images, 1024-token group)
        data = json.loads(config.read_text())
        data.update(backend="llima", prefix_tokens=prefix)
        patched = work / "client.json"
        patched.write_text(json.dumps(data, indent=2) + "\n")
        robot_files["pi05/client.json"] = patched
    shutil.rmtree(work / "bundles-work", ignore_errors=True)
    bundle = build_bundle(BundleInputs(
        spec=spec, elfs=elfs, dataset=label, steps=0, checkpoint=checkpoint, fixtures=fixtures,
        constants=constants, robot_files=robot_files, precisions=precisions, source="polima-compile-pi05-llima",
        task_label=label, extra_graphs=extra, precision_label=precision), work / "bundles-work")
    output.mkdir(parents=True, exist_ok=True)
    final = output / bundle.root.name
    shutil.rmtree(final, ignore_errors=True)
    return Path(shutil.copytree(bundle.root, final))
