"""Drive the per-graph quantize/compile loop.

The graph list comes from `PolicySpec.compile.graphs`, so a new policy needs no
controller of its own.

Three behaviours matter:

* **Precision fallback.** `GraphSpec.precisions` encodes each graph's declared
  attempt order; policies without a safe fallback stop on the primary failure.
* **ELF presence is the success condition**, not the exit code. A compile can
  return 0 having quietly placed the graph on the APU, which only shows up as a
  load failure on the DevKit.
* **Resume.** Compiles take tens of minutes; a controller that cannot resume
  gets killed and restarted from scratch.

Resume uses a content key over the ONNX, the calibration data and the compile
flags rather than "does the output file exist", so editing the export and
re-running never keeps a stale ELF, and a graph that did not change (such as a
vision backbone) need not be recompiled.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from polima.compile import mpk
from polima.util import proc
from polima.util.hashing import sha256_file, sha256_text

STATE_FILE = "compile_state.json"
SDK_VERSION_TIMEOUT_S = 300


@dataclass
class GraphResult:
    name: str
    status: str                      # compiled | reused | failed | skipped
    precision: str | None = None
    activation_precision: str | None = None
    weight_precision: str | None = None
    elf: str | None = None
    key: str | None = None
    duration_s: float = 0.0
    attempts: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("compiled", "reused")

    def summary(self) -> str:
        mark = {"compiled": "built", "reused": "reused", "failed": "FAILED",
                "skipped": "skipped"}[self.status]
        detail = f"{self.precision}" if self.precision else ""
        if self.activation_precision and self.weight_precision:
            detail = f"{detail} (A={self.activation_precision}, W={self.weight_precision})"
        if self.status == "compiled":
            detail += f"  {self.duration_s:.0f}s"
        if self.note:
            detail += f"  {self.note}"
        return f"  {mark:<8} {self.name:<24} {detail}"


class CompileError(RuntimeError):
    pass


@dataclass
class Driver:
    """Compiles every graph in a policy's `CompilePlan` into `build_dir`.

    The build tree keeps one fixed layout on purpose:

        onnx/<name>.onnx                       input, written by the export stage
        calibration/<name>.npz                 input, written by the export stage
        compiled/<precision>/<name>/           afe's own output
        retained/<name>/<name>_stage1_mla.elf  the ELF the DevKit actually runs
        models_uncompressed/<name>/share/      what bundle packing reads

    Keeping it fixed means the bundle packer consumes a freshly compiled tree
    and an imported one the same way.
    """

    spec: object                     # PolicySpec
    build_dir: Path
    compiler_python: Path
    env: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    force: bool = False
    sdk_version: str = ""
    #: Graphs compiled at once. afe is single-threaded per graph -- measured at
    #: ~100% of one core and ~1.6 GB RSS for ACT -- so a 6-graph policy uses one
    #: core of a 20-core host and takes the sum of its parts. Independent graphs
    #: have no ordering constraint: each reads its own ONNX and writes its own
    #: retained directory.
    #:
    #: Default 1 because memory, not CPU, is the real limit and it is
    #: policy-dependent: large vision and prefix stages each need substantial
    #: host RAM. Opt in with --jobs once you know the graph sizes.
    jobs: int = 1
    precision_profile: str = "policy"
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------- paths

    def onnx_path(self, name: str) -> Path:
        return self.build_dir / "onnx" / f"{name}.onnx"

    def calibration_path(self, graph) -> Path | None:
        kind = graph.calibration.kind
        if kind == "random":
            return None
        suffix = "npz" if kind == "npz" else "f32"
        return self.build_dir / "calibration" / f"{graph.name}.{suffix}"

    def retained_dir(self, name: str) -> Path:
        return self.build_dir / "retained" / name

    def elf_path(self, graph) -> Path:
        return self.retained_dir(graph.name) / graph.elf_name

    # -------------------------------------------------------------- keys

    def stage_key(self, graph) -> str:
        """Content key over everything that can change the ELF.

        Deliberately excludes absolute paths, so a build tree relocated or
        rebuilt under a different name still counts as unchanged. Includes the
        SDK version, because an afe upgrade changes code generation while every
        input stays byte-identical.
        """
        parts = [
            f"policy={getattr(self.spec, 'name', '?')}",
            f"graph={graph.name}",
            f"layout={graph.layout}",
            f"precisions={','.join(graph.precisions)}",
            f"activation_precision={graph.activation_precision or ''}",
            f"weight_precision={graph.weight_precision or ''}",
            f"tessellation={int(graph.mla_tessellation)}",
            f"exit_on_stable_elf={int(graph.exit_on_stable_elf)}",
            f"external_dram_layout={graph.external_dram_layout}",
            f"promote_rank3_hwc={int(graph.promote_rank3_hwc)}",
            f"compiler={graph.compiler}",
            f"llima_args={','.join(graph.llima_args)}",
            f"sdk={self.sdk_version}",
        ]
        onnx = self.onnx_path(graph.name)
        parts.append(f"onnx={sha256_file(onnx) if onnx.exists() else 'missing'}")
        calibration = self.calibration_path(graph)
        if calibration is not None:
            parts.append(
                f"calib={sha256_file(calibration) if calibration.exists() else 'missing'}"
            )
        return sha256_text("\n".join(parts))

    def _state(self) -> dict:
        # Callers may hold _state_lock already; this only reads.
        path = self.build_dir / STATE_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

    def _relative_elf(self, elf: str | None) -> str | None:
        """Record the ELF relative to the build tree.

        An absolute path here would undo the care taken in `stage_key`, which
        deliberately excludes paths so a relocated tree still counts as
        unchanged. The same build tree is routinely seen at two paths -- the
        Palette container mounts it as /workspace/... while the host sees
        ~/SDK/NEAT/workspace/... -- and an absolute path recorded on one side
        simply does not exist on the other, so every stage recompiles despite
        matching keys.
        """
        if not elf:
            return elf
        try:
            return str(Path(elf).resolve().relative_to(self.build_dir.resolve()))
        except ValueError:
            return elf      # outside the tree: keep it absolute, it is all we have

    def _resolve_elf(self, elf: str | None) -> Path | None:
        if not elf:
            return None
        path = Path(elf)
        return path if path.is_absolute() else self.build_dir / path

    def _record(self, result: GraphResult) -> None:
        # Serialized: with --jobs > 1 several graphs finish into the same file,
        # and a lost update means a stage recompiles next run for no reason.
        with self._state_lock:
            self._record_locked(result)

    def _record_locked(self, result: GraphResult) -> None:
        state = self._state()
        state[result.name] = {
            "key": result.key,
            "elf": self._relative_elf(result.elf),
            "precision": result.precision,
            "activation_precision": result.activation_precision,
            "weight_precision": result.weight_precision,
            "status": result.status,
            "recorded_at": time.time(),
        }
        path = self.build_dir / STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    # ----------------------------------------------------------- command

    def argv(self, graph, precision: str) -> list[str]:
        build = self.build_dir / "compiled" / precision
        command = [
            str(self.compiler_python), "-m", "polima.compile.tensor",
            "--model-path", str(self.onnx_path(graph.name)),
            "--build-dir", str(build),
            "--precision", precision,
            "--model-layout", graph.layout,
        ]
        if graph.activation_precision:
            command += ["--activation-precision", graph.activation_precision]
        if graph.weight_precision:
            command += ["--weight-precision", graph.weight_precision]
        calibration = self.calibration_path(graph)
        if calibration is not None:
            flag = "--calibration-npz" if graph.calibration.kind == "npz" else "--calibration-raw-f32"
            command += [flag, str(calibration)]
        if graph.mla_tessellation:
            command += ["--mla-tessellation"]
        # `elf_from=retained` is an output-selection contract, independent of
        # tessellation: plain-HWC graphs need a retained directory too, or a
        # successful compiler run could not be published.
        if graph.elf_from == "retained" or graph.exit_on_stable_elf:
            command += ["--retain-compile-dir", str(self.retained_dir(graph.name))]
        if graph.exit_on_stable_elf:
            command += ["--exit-on-stable-elf"]
        if graph.external_dram_layout != "compiler":
            command += ["--external-dram-layout", graph.external_dram_layout]
        if graph.promote_rank3_hwc:
            command += ["--promote-rank3-hwc"]
        command += list(graph.llima_args)
        return command

    # ------------------------------------------------------------- elf

    def locate_elf(self, graph, precision: str) -> Path | None:
        """Where the ELF ended up, honouring `GraphSpec.elf_from`.

        `retained` is the direct path: afe's retained temp dir holds the raw
        `<name>_stage1_mla.elf`. `mpk` goes through the packed archive instead.
        """
        if graph.elf_from == "retained":
            candidate = self.elf_path(graph)
            return candidate if candidate.exists() else None
        archive = mpk.find(self.build_dir / "compiled" / precision, graph.name)
        if archive is None or not mpk.has_elf(archive):
            return None
        unpacked = mpk.unpack(archive, self.build_dir / "models_uncompressed" / graph.name)
        elves = sorted((unpacked / "share").glob("*.elf"))
        return elves[0] if elves else None

    def publish(self, graph, elf: Path) -> Path:
        """Copy the ELF to `models_uncompressed/<name>/share/`, which is where
        the bundle packer reads from."""
        destination = self.build_dir / "models_uncompressed" / graph.name / "share"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / graph.elf_name
        if elf.resolve() != target.resolve():
            shutil.copy2(elf, target)
        return target

    # ------------------------------------------------------------ compile

    def compile_graph(self, graph) -> GraphResult:
        name = graph.name
        key = self.stage_key(graph)
        started = time.monotonic()

        def effective(precision: str | None) -> tuple[str | None, str | None]:
            if precision is None:
                return None, None
            return (
                graph.activation_precision or precision,
                graph.weight_precision or precision,
            )

        onnx = self.onnx_path(name)
        if not onnx.exists() and not self.dry_run:
            return GraphResult(name, "failed", key=key,
                               note=f"missing {onnx.relative_to(self.build_dir)}; run export first")

        previous = self._state().get(name, {})
        if not self.force and previous.get("key") == key and previous.get("elf"):
            elf = self._resolve_elf(previous["elf"])
            # A state file may hold an absolute path from whichever side
            # compiled it. Rather than recompile identical work, fall back to
            # where the convention says the ELF is -- and re-record it relative,
            # so the tree heals itself the first time it is used from another
            # mount.
            if (elf is None or not elf.exists()) and self.elf_path(graph).exists():
                elf = self.elf_path(graph)
            if elf is not None and elf.exists():
                self.publish(graph, elf)
                precision = previous.get("precision")
                activation, weights = effective(precision)
                outcome = GraphResult(
                    name,
                    "reused",
                    precision=precision,
                    activation_precision=previous.get("activation_precision") or activation,
                    weight_precision=previous.get("weight_precision") or weights,
                    elf=str(elf), key=key, note="unchanged",
                )
                if previous.get("elf") != self._relative_elf(str(elf)):
                    self._record(outcome)
                return outcome

        attempts: list[dict] = []
        for precision in graph.precisions:
            activation, weights = effective(precision)
            # afe writes into the retained directory; a leftover ELF from a
            # failed earlier attempt would otherwise be mistaken for success.
            retained = self.retained_dir(name)
            if retained.exists() and not self.dry_run:
                shutil.rmtree(retained)

            log = self.build_dir / "logs" / f"compile_{name}_{precision}.log"
            result = proc.run(self.argv(graph, precision), env=self.env, log_path=log,
                              echo=False, dry_run=self.dry_run)
            attempts.append({"precision": precision,
                             "activation_precision": activation,
                             "weight_precision": weights,
                             "returncode": result.returncode,
                             "log": str(log), "duration_s": round(result.duration_s, 1)})

            if self.dry_run:
                return GraphResult(name, "skipped", precision=precision,
                                   activation_precision=activation,
                                   weight_precision=weights, key=key,
                                   attempts=attempts, note="dry run")

            elf = self.locate_elf(graph, precision) if result.ok else None
            if elf is None:
                # Exit code alone is not the signal: afe can return 0 having put
                # the graph on the APU, which produces no MLA ELF at all.
                attempts[-1]["reason"] = (
                    f"exit {result.returncode}" if not result.ok else "no MLA ELF produced"
                )
                continue

            self.publish(graph, elf)
            outcome = GraphResult(name, "compiled", precision=precision,
                                  activation_precision=activation,
                                  weight_precision=weights, elf=str(elf),
                                  key=key, duration_s=time.monotonic() - started,
                                  attempts=attempts)
            self._record(outcome)
            return outcome

        return GraphResult(name, "failed", key=key, attempts=attempts,
                           duration_s=time.monotonic() - started,
                           note=f"all precisions failed ({', '.join(graph.precisions)})")

    def run(self, only: Sequence[str] | None = None) -> list[GraphResult]:
        started = time.monotonic()
        graphs = self.select(only)
        cores = os.cpu_count()
        # One independent compiler subprocess produces one ELF. More workers
        # than ELFs have nothing to do, and more workers than logical CPUs only
        # oversubscribe the host. Memory may require the caller to choose less.
        workers = min(self.jobs, len(graphs), cores or self.jobs)
        elf_label = f"{len(graphs)} ELF{'s' if len(graphs) != 1 else ''} to compile"
        core_label = (
            f"{cores} logical CPU core{'s' if cores != 1 else ''}"
            if cores else "CPU count unknown"
        )
        print(
            f"  workload  {elf_label}; {core_label}; {workers} parallel job"
            f"{'s' if workers != 1 else ''}",
            flush=True,
        )
        results = (
            self._run_sequential(graphs) if workers <= 1
            else self._run_parallel(graphs, workers)
        )
        self.write_manifest(results)
        print(f"\n  total compile time  {time.monotonic() - started:.1f}s", flush=True)
        return results

    def _run_sequential(self, graphs) -> list[GraphResult]:
        results: list[GraphResult] = []
        for graph in graphs:
            result = self.compile_graph(graph)
            print(result.summary(), flush=True)
            results.append(result)
            # Stop on the first failure: the later graphs are usually variations
            # of the same one, so continuing mostly buys more copies of the same
            # error and a longer wait before you see it.
            if result.status == "failed":
                break
        return results

    def _run_parallel(self, graphs, workers: int) -> list[GraphResult]:
        """Compile independent graphs at once.

        Threads rather than processes: each compile is a subprocess, so the work
        happens outside the interpreter and the GIL is irrelevant. Results are
        printed as they land, so a long graph does not hide the short ones that
        already finished.

        Unlike the sequential path this does not stop early -- the others are
        already running, and killing them would waste work that is nearly done.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"  running {workers} graph(s) at a time", flush=True)
        by_name: dict[str, GraphResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.compile_graph, g): g for g in graphs}
            for future in as_completed(futures):
                graph = futures[future]
                try:
                    result = future.result()
                except Exception as error:            # noqa: BLE001 - reported below
                    result = GraphResult(graph.name, "failed", note=str(error))
                by_name[graph.name] = result
                print(result.summary(), flush=True)
        # Report in plan order, not completion order, so the output is stable.
        return [by_name[g.name] for g in graphs if g.name in by_name]

    def select(self, only: Sequence[str] | None) -> list:
        graphs = list(self.spec.compile.graphs)
        if not only:
            return graphs
        wanted = set(only)
        unknown = wanted - {g.name for g in graphs}
        if unknown:
            raise CompileError(
                f"unknown graph(s) {sorted(unknown)}; "
                f"have {[g.name for g in graphs]}"
            )
        return [g for g in graphs if g.name in wanted]

    def write_manifest(self, results: Iterable[GraphResult]) -> Path:
        results = list(results)
        path = self.build_dir / "artifact_manifest.json"
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = {}

        # A targeted retry (``--graph <name>``) must not make a four-graph
        # build look like a one-graph build. Prefer the fresh result, retain
        # full records already present in the manifest, and finally recover a
        # minimal record from compile_state.json. The state file is updated per
        # graph and is the durable inventory when an old manifest was replaced.
        current = {result.name: result for result in results}
        prior_manifest = {
            item.get("name"): item
            for item in existing.get("graphs", [])
            if isinstance(item, dict) and item.get("name")
        }
        state = self._state()
        merged: list[dict] = []
        for graph in self.spec.compile.graphs:
            if graph.name in current:
                merged.append(asdict(current[graph.name]))
                continue
            if graph.name in prior_manifest:
                saved = state.get(graph.name, {})
                item = dict(prior_manifest[graph.name])
                precision = item.get("precision") or saved.get("precision") or graph.precision
                item["precision"] = precision
                item["activation_precision"] = (
                    item.get("activation_precision")
                    or saved.get("activation_precision")
                    or graph.activation_precision
                    or precision
                )
                item["weight_precision"] = (
                    item.get("weight_precision")
                    or saved.get("weight_precision")
                    or graph.weight_precision
                    or precision
                )
                recorded_elf = self._resolve_elf(item.get("elf"))
                if recorded_elf is None or not recorded_elf.exists():
                    recorded_elf = self._resolve_elf(saved.get("elf"))
                if recorded_elf is not None:
                    item["elf"] = str(recorded_elf)
                merged.append(item)
                continue
            saved = state.get(graph.name)
            if not saved:
                continue
            precision = saved.get("precision") or graph.precision
            activation = (
                saved.get("activation_precision")
                or graph.activation_precision
                or precision
            )
            weights = (
                saved.get("weight_precision")
                or graph.weight_precision
                or precision
            )
            elf = self._resolve_elf(saved.get("elf"))
            merged.append(asdict(GraphResult(
                name=graph.name,
                status=saved.get("status", "reused"),
                precision=precision,
                activation_precision=activation,
                weight_precision=weights,
                elf=str(elf) if elf is not None else None,
                key=saved.get("key"),
                note="recovered from compile_state.json",
            )))

        # Do not retain graph names the current policy no longer declares.
        known = {graph.name for graph in self.spec.compile.graphs}
        merged.extend(asdict(result) for result in results if result.name not in known)
        existing.update({
            "format": "polima-build-v1",
            "policy": getattr(self.spec, "name", "?"),
            "build_dir": str(self.build_dir),
            "sdk_version": self.sdk_version,
            "precision_profile": self.precision_profile,
            "compiled_at": time.time(),
            "graphs": merged,
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        return path


def sdk_version(compiler_python: str | Path, env: dict[str, str] | None = None) -> str:
    """afe's version string, which participates in the resume key."""
    result = proc.capture(
        [str(compiler_python), "-c",
         "from afe.apis.release_v1 import get_model_sdk_version as v; print(v())"],
        env=env,
        timeout=SDK_VERSION_TIMEOUT_S,
    )
    return result.stdout.strip().splitlines()[-1] if result.ok and result.stdout.strip() else ""
