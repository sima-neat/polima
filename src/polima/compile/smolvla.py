"""`polima compile --policy smolvla`: SmolVLA checkpoint + dataset -> optimized Modalix bundle.

Staged and resumable, under OUTPUT/ (settings in build.json, stages/<stage>.done, logs/):

  export   the generic export (onnx/, calibration/, fixture, constants, normalization), then
           device/ (policies.smolvla.optimize): denoise_step, connector, the SmoothQuant'd SigLIP
           tower as a PaliGemma folder for LLiMa, the step time constants, held-out samples
  llima    llima-compile the SigLIP tower, VISION only, INT8 weights (BF16 activations)
  compile  ModelSDK: prefix (BF16, fused RMSNorm), connector (BF16), denoise_step (INT8 weights,
           one persistent binding per input)
  bundle   one device program per request (policies.smolvla.device.optimized_plan): vision,
           connector, prefix and the ten denoising steps, x ping-ponging between device buffers
  check    optional (--check-devkit): the fixture and the held-out samples on a DevKit vs
           LeRobot's own sample_actions, normalized actions, mean over samples

Measured on SO-101 (24 held-out frames, main runtime): 150.8 ms per 50-action plan against
248 ms for the per-graph BF16 build, cosine 0.9996, nMAE 0.015.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from polima.compile.groot import Build, StageError, _now, start_compiler

STAGES = ("export", "llima", "compile", "bundle", "check")
SRC = Path(__file__).resolve().parents[2]
#: Mean over the fixture and held-out samples, normalized actions (GR00T's mixed bar).
ENVELOPE = {"cosine": 0.995, "mean_abs": 0.03}
LLIMA_MODEL = "llima_vision"
LLIMA_CONFIG = '''# polima: SmolVLA's SigLIP tower only, INT8 weights (BF16 activations).
def get_layer_configuration(model_properties, layer):
    if layer["part"] == "VISION":
        return {"compile": True, "precision": "A_BF16_W_INT8"}
    return {"compile": False}
'''


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_env(build: Build) -> dict:
    from polima.compile.toolchain import compiler_env as base

    env = base(Path(build.settings["sdk"]).parent, source_root=SRC)
    threads = str(build.settings.get("threads", 8))
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               TVM_NUM_THREADS=threads, SIMA_MLA_COMPILE_USE_JAX="0")
    env.setdefault("MODEL_COMPILER_CUDA_VISIBLE_DEVICES", "")
    return env


# --------------------------------------------------------------------- stages


def stage_export(build: Build) -> None:
    from polima.export.driver import export
    from polima.policies.registry import get_policy
    from polima.policies.smolvla.optimize import export_device_graphs

    s = build.settings
    spec = get_policy("smolvla").specialize_profile(checkpoint=s["checkpoint"], dataset_root=s["dataset"])
    result = export(spec, s["checkpoint"], build.path("export"), dataset_root=s["dataset"],
                    calibration_samples=s["calibration"], verify=True)
    report = result.verification or {}
    build.log(f"export: {len(result.graphs)} graphs, ONNX vs PyTorch ok={report.get('ok')} "
              f"max_abs={report.get('max_abs')}")
    if not report.get("ok"):
        raise StageError("the ONNX chain does not match PyTorch; nothing was compiled")
    manifest = export_device_graphs(Path(s["checkpoint"]), Path(s["dataset"]), build.path("export", "device"),
                                    calibration=s["calibration"], evaluation=s["evaluation"])
    build.log(f"device graphs: {json.dumps(manifest)}")


def stage_llima(build: Build) -> None:
    work = build.path("llima")
    work.mkdir(exist_ok=True)
    (work / "layers.py").write_text(LLIMA_CONFIG)
    model = work / LLIMA_MODEL
    if not model.exists():
        model.symlink_to(build.path("export", "device", "llima_vision"))
    llima = Path(build.settings["sdk"]).parent / "llima-compile"
    cmd = [str(llima), LLIMA_MODEL, "-o", "out", "-c", "layers.py", "-j", "1", "--max_num_tokens", "2048",
           "--no-quantize_kv_cache", "--no-quantize_embeddings", "--log_level", "INFO", "--resume"]
    log = build.path("logs", "llima.log")
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as stream:
        code = subprocess.run(cmd, cwd=work, env=compiler_env(build), stdout=stream, stderr=subprocess.STDOUT,
                              check=False).returncode
    archive = work / "out/sima_files/mpk" / f"{LLIMA_MODEL}_vision_mpk.tar.gz"
    if code or not archive.is_file():
        raise StageError(f"llima-compile failed (exit {code}); see {log}")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        meta = json.load(tar.extractfile(next(n for n in names if n.endswith("_mpk.json"))))
        elf = tar.extractfile(next(n for n in names if n.endswith("_stage1_mla.elf"))).read()
    mla = next(p for p in meta["plugins"] if p.get("processor") == "MLA")
    sizes = ([n["size"] // 2 for n in mla["input_nodes"]], [n["size"] // 2 for n in mla["output_nodes"]])
    from polima.policies.smolvla.device import VISION_TOKENS
    from polima.policies.smolvla.runtime import IMAGE_ELEMENTS

    if sizes != ([IMAGE_ELEMENTS], [VISION_TOKENS]):
        raise StageError(f"LLiMa vision graph has bindings {sizes}, expected ([{IMAGE_ELEMENTS}], [{VISION_TOKENS}])")
    (work / "vision_stage1_mla.elf").write_bytes(elf)


def graph_jobs(build: Build) -> list[dict]:
    from polima.policies.registry import get_policy

    device = build.path("export", "device")
    prefix = get_policy("smolvla").graph("prefix")
    flags = ["--external-dram-layout", prefix.external_dram_layout, *prefix.llima_args]
    return [
        {"name": "prefix", "onnx": device / "prefix.onnx", "activation": "bf16", "weights": "bf16",
         "calibration": None, "flags": flags, "persistent": False},
        # One strided convolution over the token grid: NCHW, with external HWC boundaries.
        {"name": "connector", "onnx": device / "connector.onnx", "activation": "bf16", "weights": "bf16",
         "calibration": None, "flags": ["--external-dram-layout", "HWC", "--infer-shapes"], "persistent": False,
         "layout": "NCHW"},
        {"name": "denoise_step", "onnx": device / "denoise_step.onnx", "activation": "bf16", "weights": "int8",
         "calibration": device / "denoise_step.calib.npz", "flags": ["--no-simplify", "--infer-shapes"],
         "persistent": True},
    ]


def stage_compile(build: Build) -> None:
    out = build.path("compiled")
    out.mkdir(exist_ok=True)
    results_path = out / "compile_results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    sdk, shim, env = build.settings["sdk"], str(SRC / "polima/compile/compile_onnx.py"), compiler_env(build)

    def elf_of(job) -> Path:
        work = out / job["name"]
        return (work / "persistent" if job["persistent"] else work) / "retained" / f"{job['name']}_stage1_mla.elf"

    def key(job) -> dict:
        return {"onnx": _sha(job["onnx"]), "calibration": _sha(job["calibration"]) if job["calibration"] else None,
                "activation": job["activation"], "weights": job["weights"], "flags": job["flags"],
                "persistent": job["persistent"], "layout": job.get("layout", "NHWC")}

    def one(job) -> dict:
        wanted, name, work = key(job), job["name"], out / job["name"]
        old = results.get(name, {})
        if old.get("returncode") == 0 and old.get("key") == wanted and elf_of(job).is_file():
            return old
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        common = [sdk, "-u", shim, "--model-path", str(job["onnx"]), "--build-dir", str(work / "compiled"),
                  "--model-layout", job.get("layout", "NHWC"), "--precision", "bf16", "--activation-precision", job["activation"],
                  "--weight-precision", job["weights"], *job["flags"]]
        if job["calibration"]:
            common += ["--calibration-npz", str(job["calibration"])]
        started = time.time()
        if job["persistent"]:
            with (work / "quantize.log").open("w") as log:
                code = start_compiler(common + ["--no-compile"], env, log)
            if code == 0:
                with (work / "persistent.log").open("w") as log:
                    code = start_compiler([sdk, "-u", shim, "persistent", "--source", str(work / "compiled" / name),
                                           "--name", name, "--output", str(work / "persistent"),
                                           "--onnx-outputs", "x_next"], env, log)
        else:
            with (work / "compile.log").open("w") as log:
                code = start_compiler(common + ["--retain-compile-dir", str(work / "retained"),
                                                "--exit-on-stable-elf"], env, log)
        elf = elf_of(job)
        ok = code == 0 and elf.is_file() and elf.read_bytes()[:4] == b"\x7fELF"
        return {"key": wanted, "returncode": 0 if ok else (code or 1), "elf": str(elf),
                "minutes": round((time.time() - started) / 60, 1), "elf_sha256": _sha(elf) if ok else None}

    jobs = graph_jobs(build)
    failed = []
    with ThreadPoolExecutor(max_workers=build.settings["jobs"]) as pool:
        futures = {pool.submit(one, job): job["name"] for job in jobs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:  # noqa: BLE001 - one failed graph must not stop the others
                results[name] = {"returncode": 1, "error": str(error)}
            record = results[name]
            build.log(f"compile {name}: {'ok' if record['returncode'] == 0 else 'FAILED'} "
                      f"{record.get('minutes', '')} min")
            if record["returncode"]:
                failed.append(name)
            results_path.write_text(json.dumps(results, indent=2) + "\n")
    if failed:
        raise StageError(f"{len(failed)} graph(s) failed: {', '.join(failed)}; see {out}/<graph>/*.log")


def stage_bundle(build: Build) -> None:
    from polima.bundle.importer import (
        collect_fixtures,
        collect_robot_files,
        collect_sidecars,
        detect,
    )
    from polima.bundle.pack import BundleInputs, build_bundle
    from polima.bundle.retained import ElfCandidate
    from polima.policies.registry import get_policy
    from polima.policies.smolvla import device
    from polima.policies.smolvla.runtime import IMAGE_ELEMENTS

    s = build.settings
    export, compiled = build.path("export"), build.path("compiled")
    spec = get_policy("smolvla").specialize_profile(checkpoint=s["checkpoint"], dataset_root=s["dataset"])
    results = json.loads((compiled / "compile_results.json").read_text())
    contract = json.loads((compiled / "denoise_step/persistent/contract.json").read_text())
    by_size = {device.X_ELEMENTS: "x", 720: "tbias", 2467840: "cache"}
    order = tuple(by_size[i["elements"]] for i in contract["inputs"])
    vision_elf = build.path("llima", "vision_stage1_mla.elf")
    paths = {"vision": vision_elf, **{n: Path(results[n]["elf"]) for n in ("prefix", "connector", "denoise_step")}}
    elfs = {n: ElfCandidate(n, p, n, _sha(p), p.stat().st_size) for n, p in paths.items()}
    extra = {
        "vision": {"precision": "A_BF16_W_INT8", "input_elements": IMAGE_ELEMENTS,
                   "output_elements": device.VISION_TOKENS, "dram_layout": "plain", "external_dram_layout": "HWC"},
        "connector": {"precision": "bf16", "input_elements": device.VISION_TOKENS, "output_elements": 64 * 960,
                      "dram_layout": "plain", "external_dram_layout": "HWC"},
        "denoise_step": {"precision": "A_BF16_W_INT8", "input_elements": sum(i["elements"] for i in contract["inputs"]),
                         "output_elements": device.X_ELEMENTS, "dram_layout": "plain",
                         "external_dram_layout": "HWC",
                         "persistent_input_elements": [i["elements"] for i in contract["inputs"]]},
    }
    prebuilt = detect(export)
    constants = collect_sidecars(prebuilt, spec)
    constants.update({name: export / "device/constants" / name for name in
                      json.loads((export / "device/manifest.json").read_text())["tbias"]})
    fixtures = collect_fixtures(prebuilt, spec)
    samples = export / "device/samples"
    fixtures.update({f"samples/{p.relative_to(samples)}": p for p in samples.rglob("*.f32")})
    try:
        robot_files = collect_robot_files(spec, export)
    except FileNotFoundError as error:
        build.log(f"no robot client in the bundle ({error}); inference server only")
        robot_files = {}
    from dataclasses import replace

    spec = replace(spec, smoke=replace(spec.smoke, cosine_min=ENVELOPE["cosine"], mean_abs_max=ENVELOPE["mean_abs"]))
    checkpoint = Path(s["checkpoint"])
    steps = int(checkpoint.parent.name) if checkpoint.parent.name.isdigit() else 0
    work = build.path("bundles")
    shutil.rmtree(work, ignore_errors=True)
    bundle = build_bundle(BundleInputs(
        spec=spec, elfs=elfs, dataset=s["label"], steps=steps, checkpoint=str(checkpoint), fixtures=fixtures,
        constants=constants, robot_files=robot_files, source="polima-compile-smolvla", task_label=s["label"],
        plan_rewrite=lambda plan: device.optimized_plan(plan, order), extra_graphs=extra,
        precision_label="mixed"), work)
    s["bundle"] = str(bundle.root)
    build.path("build.json").write_text(json.dumps(s, indent=2) + "\n")
    build.log(f"bundle {bundle.root}")


def board_check(build: Build, board: str) -> dict:
    """The bundle's fixture and held-out samples on a DevKit (its installed polima_cli, or
    POLIMA_CHECK_CLI) vs LeRobot, in normalized actions; mean over samples against ENVELOPE."""
    import numpy as np

    from polima.config import devkits
    from polima.config.base import BoardConfig

    bundle = Path(build.settings["bundle"])
    target = BoardConfig(host=board)
    devkits.apply(target)
    host, ssh = target.host, ["ssh", *target.ssh_options]
    cli = os.environ.get("POLIMA_CHECK_CLI", f"{target.root}/bin/polima_cli")
    workdir = f"{target.root}/compile-check"
    remote, results = f"{workdir}/bundles/{bundle.name}", f"{workdir}/results/{bundle.name}"
    samples = sorted(p.name for p in (bundle / "fixtures/samples").iterdir()) if (bundle / "fixtures/samples").is_dir() else []
    runs = [f"{cli} --bundle {remote} --input-dir {remote}/fixtures/inputs --output {results}/fixture.f32"]
    runs += [f"{cli} --bundle {remote} --input-dir {remote}/fixtures/samples/{n}/inputs --output {results}/{n}.f32"
             for n in samples]
    local = build.path("check", "results")
    local.mkdir(parents=True, exist_ok=True)
    for argv, problem in (
        ([*ssh, host, f"mkdir -p {workdir}/bundles {results}"], "could not reach {host}"),
        (["rsync", "-a", "-e", " ".join(ssh), f"{bundle}/", f"{host}:{remote}/"], "could not copy the bundle to {host}"),
        ([*ssh, host, f"flock /tmp/polima-bench.lock sh -c '{' && '.join(runs)}'"],
         "the bundle did not run on {host}; PoLiMa must be installed there and the policy server stopped"),
        (["rsync", "-a", "-e", " ".join(ssh), f"{host}:{results}/", f"{local}/"], "could not read the results from {host}"),
    ):
        if subprocess.run(argv, check=False).returncode:
            raise StageError(problem.format(host=host))
    with np.load(bundle / "fixtures/normalization_stats.npz") as stats:
        mean, std = np.asarray(stats["action_mean"], "f8"), np.asarray(stats["action_std"], "f8")
    expected = {"fixture": bundle / "fixtures/expected/normalized_actions.f32",
                **{n: bundle / "fixtures/samples" / n / "expected.f32" for n in samples}}
    per = {}
    for name, path in expected.items():
        want = (np.fromfile(path, "<f4").astype("f8").reshape(-1, mean.size) - mean) / std
        got = (np.fromfile(local / f"{name}.f32", "<f4").astype("f8").reshape(-1, mean.size) - mean) / std
        per[name] = {"cosine": float((got * want).sum() / np.linalg.norm(got) / np.linalg.norm(want)),
                     "mean_abs": float(np.abs(got - want).mean()), "finite": bool(np.isfinite(got).all())}
    report = {"samples": len(per), "cosine": float(np.mean([r["cosine"] for r in per.values()])),
              "mean_abs": float(np.mean([r["mean_abs"] for r in per.values()])),
              "worst_mean_abs": max(r["mean_abs"] for r in per.values()),
              "finite": all(r["finite"] for r in per.values()), "envelope": ENVELOPE, "per_sample": per}
    report["ok"] = bool(report["finite"] and report["cosine"] >= ENVELOPE["cosine"]
                        and report["mean_abs"] <= ENVELOPE["mean_abs"])
    build.path("check", "report.json").write_text(json.dumps(report, indent=2) + "\n")
    build.log(f"{len(per)} sample(s) on {host} vs LeRobot: mean cos {report['cosine']:.5f} "
              f"mean nMAE {report['mean_abs']:.4f} (worst {report['worst_mean_abs']:.4f}) "
              f"-> {'PASS' if report['ok'] else 'FAIL'}")
    return report


def stage_check(build: Build) -> None:
    board = build.settings.get("check_board")
    if not board:
        build.log("check skipped: pass --check-devkit to check the bundle on a DevKit")
        return
    if not board_check(build, board)["ok"]:
        raise StageError("samples outside the accuracy envelope; see check/report.json")


RUNNERS = {"export": stage_export, "llima": stage_llima, "compile": stage_compile, "bundle": stage_bundle,
           "check": stage_check}


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polima compile --policy smolvla", description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", default="smolvla", choices=("smolvla",))
    p.add_argument("--checkpoint", type=Path, required=True, help="SmolVLA pretrained_model folder")
    p.add_argument("--dataset", "--dataset-root", dest="dataset", type=Path, required=True,
                   help="the LeRobot dataset the checkpoint was trained on (calibration, held-out samples)")
    p.add_argument("--output", type=Path, required=True, help="build directory (bundle under bundles/)")
    p.add_argument("--label", help="task label in the bundle id (default: the --output name)")
    p.add_argument("--calibration-samples", dest="calibration", type=int, default=8)
    p.add_argument("--evaluation-samples", dest="evaluation", type=int, default=16,
                   help="held-out observations packed as fixtures and checked on the DevKit")
    p.add_argument("-j", "--jobs", type=int, default=2, choices=(1, 2, 3), help="graphs compiled at once")
    p.add_argument("--threads", type=int, default=8, help="compiler threads per graph")
    p.add_argument("--resume", action="store_true", help="continue an existing build directory")
    p.add_argument("--stop-after", choices=STAGES)
    p.add_argument("--check-devkit", "--check-board", dest="check_board", metavar="ADDRESS",
                   help="check the bundle on this DevKit (user@host or saved name) against LeRobot")
    return p


def main(argv: list[str] | None = None, config=None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if config is None:
        from polima.config.loader import load

        config = load()
    try:
        build = _prepare(args, config)
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
    from polima.compile.toolchain import require_compiler_python

    root = args.output.resolve()
    checkpoint, dataset = args.checkpoint.resolve(), args.dataset.resolve()
    if not (checkpoint / "config.json").is_file():
        raise StageError(f"{checkpoint} is not a SmolVLA checkpoint folder (it has no config.json)")
    if json.loads((checkpoint / "config.json").read_text()).get("type") != "smolvla":
        raise StageError(f"{checkpoint}/config.json is not a SmolVLA policy")
    if not (dataset / "meta/info.json").is_file():
        raise StageError(f"{dataset} is not a LeRobot dataset (it has no meta/info.json)")
    settings = {"checkpoint": str(checkpoint), "dataset": str(dataset), "calibration": args.calibration,
                "evaluation": args.evaluation, "sdk": str(require_compiler_python(config)), "jobs": args.jobs,
                "threads": args.threads, "check_board": args.check_board, "label": args.label or root.name}
    if root.exists() and not args.resume:
        raise StageError(f"{root} exists; pass --resume to continue it or choose a new --output")
    if args.resume and (root / "build.json").is_file():
        old = json.loads((root / "build.json").read_text())
        for key in ("checkpoint", "dataset", "calibration", "evaluation"):
            if old.get(key) != settings[key]:
                raise StageError(f"--resume with a different {key}: {old.get(key)} != {settings[key]}")
        settings = {**old, **{k: v for k, v in settings.items() if k in ("jobs", "threads", "check_board",
                                                                         "sdk", "label")}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "build.json").write_text(json.dumps(settings, indent=2) + "\n")
    build = Build(root)
    build.log(f"SmolVLA optimized build {_now()}: {checkpoint} + {dataset} -> {root}")
    return build


if __name__ == "__main__":
    raise SystemExit(main())
