"""`polima-compile` -- produce a deployable bundle.

Two ways in:

  --import-legacy <build_dir>   adopt an already-built compiler tree unchanged.
                                Needs no compiler, so it runs anywhere.
  --build-dir <dir>             quantize and compile the ONNX graphs in that
                                tree into ELFs, then pack a bundle.

The second path replaces the per-policy shell loops in
`ACT/scripts/compile_deploy_act_som.sh` and
`SmolVLA/scripts/compile_deploy_smolvla_som.sh`. It is verified by reproduction:
run against the tree that produced the deployed ACT bundle, it regenerates the
ELFs byte for byte.

`--stop-after compile` leaves the build tree without packing, which is what you
want while iterating on a single graph.

A compile recompiles. The per-graph content key still exists and is still
recorded, but skipping on it is opt-in via `--reuse` -- "it said reused and I
wanted a build" is a worse failure than spending the nine minutes, especially
when the export step re-runs either way and makes it look like work happened.

Run with NO arguments at a terminal, it opens an interactive session instead of
printing the flags to read: the checkpoints and build trees are on disk, so it
finds them and offers them (see polima.cli.wizard). The session only composes a
command line -- it prints the exact `polima compile ...` it is about to run and
asks first -- so there is still one code path, and using it teaches the flags.
Piped or scripted invocations are untouched: no TTY means the old message and
exit 2.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from polima.cli import wizard
from polima.compile.precision import PRECISION_PROFILES, apply_precision_profile
from polima.config.loader import load
from polima.policies.registry import get_policy
from polima.util import table


def needs_capability(argv: list[str]) -> str | None:
    """Return the external capability needed by the requested terminal stage.

    Importing legacy artifacts and stopping after checkpoint-to-ONNX export do
    not need ModelSDK. Quantization, compilation, and packing fresh artifacts
    still require the compiler venv to be reachable as a subprocess.
    """
    if any(a == "--import-legacy" or a.startswith("--import-legacy=") for a in argv):
        return None
    export_only = "--stop-after=export" in argv or any(
        current == "--stop-after" and following == "export"
        for current, following in itertools.pairwise(argv)
    )
    if export_only:
        return None
    # A bare call opens the interactive session, which offers packing (no
    # compiler needed) alongside the compile paths and reports the gap itself.
    # Gating it here would exit 3 before it could say any of that.
    if not argv:
        return None
    return "compile"



def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    if wizard.bare_invocation_is_interactive(argv):
        composed = _session(parent)
        if composed is None:
            return 130
        argv = composed

    parser = argparse.ArgumentParser(prog="polima-compile")
    parser.add_argument("--policy", default="act")
    parser.add_argument(
        "--import-legacy", metavar="BUILD_DIR", help=argparse.SUPPRESS,
    )
    parser.add_argument("--build-dir", metavar="DIR",
                        help="build tree holding onnx/ and calibration/ to compile")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="export this checkpoint into --build-dir first")
    parser.add_argument("--dataset-root", metavar="PATH",
                        help="override the dataset named in the checkpoint")
    parser.add_argument(
        "--embodiment-tag",
        help="explicit base-checkpoint embodiment (required for multi-embodiment GR00T)",
    )
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument(
        "--precision-profile",
        choices=PRECISION_PROFILES,
        default="policy",
        help="named activation/weight precision recipe (default: policy declaration)",
    )
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the onnxruntime-vs-PyTorch check after export")
    parser.add_argument("--graph", action="append", metavar="NAME",
                        help="compile only this graph (repeatable)")
    parser.add_argument("--reuse", action="store_true",
                        help="skip graphs whose content key is unchanged "
                             "(off by default: a compile recompiles)")
    parser.add_argument("--force", action="store_true",
                        help=argparse.SUPPRESS)   # now the default; kept so old
                                                  # commands and scripts still run
    parser.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                        help="compile N graphs at once (afe is ~1 core and "
                             "~1.6 GB per graph; memory is the limit, not CPU)")
    parser.add_argument("--stop-after", choices=("export", "compile", "pack"), default="pack")
    parser.add_argument("--output-root", default=None, help="where bundles are written")
    parser.add_argument("--dataset", default=None, help="override the dataset name in the id")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = load(config_file=getattr(parent, "config", None))
    output_root = Path(
        args.output_root or (config.paths.outputs or _default_outputs()) / "bundles"
    )
    dry_run = bool(getattr(parent, "dry_run", False))

    if args.import_legacy and args.build_dir:
        print("polima-compile: pass either --import-legacy or --build-dir, not both",
              file=sys.stderr)
        return 2

    if args.build_dir:
        return _compile(args, config, output_root, dry_run)

    if args.checkpoint:
        print("polima-compile: --checkpoint also needs --build-dir, naming where "
              "the exported graphs and ELFs go", file=sys.stderr)
        return 2

    if not args.import_legacy:
        print(
            "polima-compile: nothing to do. Pass --checkpoint <path> and --build-dir <dir>.",
            file=sys.stderr,
        )
        return 2

    return _import_legacy(args, output_root)


def _session(parent: argparse.Namespace | None) -> list[str] | None:
    """Compose an argv interactively, then fall through to the normal parse.

    Returns None when the user cancels, which `run` reports as 130 -- the shell
    convention for an interrupted command, so a cancelled session is
    distinguishable from a failed compile.
    """
    from polima import role
    from polima.policies.registry import load_all

    # load_all, not available(): the latter lists built-ins whose module may not
    # import here, and offering one only to raise on get_policy is worse than
    # not offering it.
    specs = load_all(strict=False)
    if not specs:
        print("polima-compile: no policy is loadable here", file=sys.stderr)
        return None
    try:
        return wizard.compose(specs, can_compile=role.detect().can_compile)
    except wizard.Cancelled:
        print("\n  cancelled")
        return None
    except KeyboardInterrupt:
        print("\n  cancelled")
        return None


# ------------------------------------------------------------------- compile


def _compile(args, config, output_root: Path, dry_run: bool) -> int:
    from polima.compile.driver import CompileError, Driver, sdk_version
    from polima.compile.toolchain import compiler_env, require_compiler_python

    spec = get_policy(args.policy)
    build_dir = Path(args.build_dir).resolve()
    try:
        spec = apply_precision_profile(
            _specialize_spec(
                spec,
                checkpoint=args.checkpoint,
                dataset_root=args.dataset_root,
                embodiment_tag=args.embodiment_tag,
                build_dir=build_dir,
            ),
            args.precision_profile,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"polima-compile: {error}", file=sys.stderr)
        return 2

    if args.checkpoint:
        code = _export(args, spec, build_dir, dry_run)
        if code or args.stop_after == "export":
            return code
        # Export can discover checkpoint-specific prompt geometry (notably
        # GR00T's camera token windows). Restore the contract it just wrote so
        # Model Compiler receives those exact shapes in this same invocation.
        if not dry_run:
            spec = apply_precision_profile(
                _specialize_spec(get_policy(args.policy), build_dir=build_dir),
                args.precision_profile,
            )


    try:
        compiler_python = require_compiler_python(config)
    except FileNotFoundError as error:
        print(f"polima-compile: {error}", file=sys.stderr)
        return 2

    # The compiler venv has no polima installed, so it needs our src/ on
    # PYTHONPATH to run `-m polima.compile.tensor`. That module imports numpy
    # and (lazily) afe only, both of which the venv has.
    env = compiler_env(compiler_python.parent, source_root=_src_root())
    version = sdk_version(compiler_python, env) if not dry_run else ""

    # Preflight the compiler rather than discovering it is broken one graph in.
    # The interpreter existing is not the same as afe working: the model-compiler
    # venv is built for the Palette `modelsdk` container, and mounting it into a
    # different image gets you a missing-shared-library error at import time.
    if not dry_run and not version:
        from polima.compile.toolchain import ACTIVATION_SCRIPT, find_activation
        from polima.util.proc import capture

        probe = capture([str(compiler_python), "-c", "import afe"], env=env)
        detail = probe.tail(3).strip() or "no output"
        activation = find_activation(compiler_python.parent)
        hint = (
            f"  Sourced {activation} and it still fails."
            if activation else
            f"  No {ACTIVATION_SCRIPT} found. In the Palette `modelsdk` container\n"
            "  that script puts the compiler's own libraries on the loader path;\n"
            "  without it afe fails on a missing libLLVM."
        )
        print(
            f"polima-compile: {compiler_python} cannot import afe.\n"
            f"  {detail}\n"
            f"{hint}\n"
            "  Set MODEL_COMPILER_BIN, or run from the modelsdk environment.",
            file=sys.stderr,
        )
        return 2

    driver = Driver(spec=spec, build_dir=build_dir, compiler_python=compiler_python,
                    env=env, dry_run=dry_run, force=not args.reuse, sdk_version=version,
                    jobs=max(1, args.jobs), precision_profile=args.precision_profile)

    print(table.section(f"compile {spec.name} in {build_dir.name}"))
    print(f"  compiler  {compiler_python}" + (f"  (ModelSDK {version})" if version else ""))
    print(f"  precision {args.precision_profile}")
    try:
        results = driver.run(only=args.graph)
    except CompileError as error:
        print(f"polima-compile: {error}", file=sys.stderr)
        return 2

    built = [r for r in results if r.status == "compiled"]
    reused = [r for r in results if r.status == "reused"]
    failed = [r for r in results if r.status == "failed"]
    print(f"\n  {len(built)} built, {len(reused)} reused, {len(failed)} failed")
    if reused and not built and not failed:
        print("  nothing changed since the last compile in this tree "
              "(--reuse); drop it to rebuild")

    if failed:
        for result in failed:
            print(f"\n  {result.name}: {result.note}", file=sys.stderr)
            for attempt in result.attempts:
                print(f"    {attempt['precision']}: {attempt.get('reason', '')}"
                      f"  see {attempt['log']}", file=sys.stderr)
        return 1

    if args.stop_after == "compile" or dry_run:
        print(f"\n  build tree {build_dir}")
        print(f"next:  polima-compile --build-dir {build_dir}   # to pack a bundle")
        return 0

    # A freshly compiled tree has the same shape as a legacy one -- that is
    # deliberate (see Driver's docstring), so the Phase-1a packer takes it as is.
    return _pack(args, build_dir, output_root, spec)


def _export(args, spec, build_dir: Path, dry_run: bool) -> int:
    """Checkpoint -> onnx/ + calibration/ + fixtures, under PoLiMa's venv.

    Torch, LeRobot and ONNX belong to PoLiMa's dedicated host venv. AFE remains
    the sole external extension and is subprocessed by the compile stage.
    """
    if dry_run:
        print(f"  [dry-run] export {args.checkpoint} -> {build_dir}")
        return 0

    try:
        from polima.export.driver import export
    except ImportError as error:      # pragma: no cover - environment dependent
        print(f"polima-compile: export needs torch + lerobot ({error}).\n"
              "  Rebuild the self-contained environment with `make venv`,\n"
              "  or compile an already-exported tree with --build-dir alone.",
              file=sys.stderr)
        return 2

    print(table.section(f"export {spec.name} from {Path(args.checkpoint).name}"))
    try:
        result = export(
            spec, args.checkpoint, build_dir,
            dataset_root=args.dataset_root,
            calibration_samples=args.calibration_samples,
            verify=not args.skip_verify,
        )
    except ModuleNotFoundError as error:   # pragma: no cover - environment dependent
        # The guard above only covers importing the driver. torch and lerobot
        # are reached later, inside the policy's own graph module, so without
        # this the wrong environment produced a bare traceback -- easy to hit now
        # that a bare `polima compile` offers checkpoints.
        print(f"\npolima-compile: export needs {error.name}, which this "
              f"interpreter does not have.\n"
              "  Rebuild the self-contained environment with `make venv`,\n"
              "  or compile an already-exported tree with --build-dir alone.",
              file=sys.stderr)
        return 2
    print(f"  {len(result.graphs)} graph(s), {result.calibration_samples} "
          f"calibration sample(s), {result.duration_s:.0f}s")
    print(f"  dataset   {result.dataset_root}")

    report = result.verification
    if report is None:
        print("  verify    skipped")
        return 0
    verdict = "PASS" if report["ok"] else "FAIL"
    print(f"  verify    {verdict}  max_abs={report['max_abs']:.3e} "
          f"mean_abs={report['mean_abs']:.3e} (atol={report.get('atol', 'n/a')})")
    if not report["ok"]:
        print("\npolima-compile: the ONNX chain does not match PyTorch, so "
              "compiling it would bake in the error.\n"
              "  This is an export bug, not a quantization one -- nothing has "
              "been quantized yet.", file=sys.stderr)
        return 1
    return 0


def _pack(args, build_dir: Path, output_root: Path, spec=None) -> int:
    from polima.bundle.import_legacy import import_legacy

    spec = spec or apply_precision_profile(
        _specialize_spec(get_policy(args.policy), build_dir=build_dir),
        args.precision_profile,
    )
    bundle = import_legacy(build_dir, spec, output_root=output_root,
                           dataset=args.dataset, steps=args.steps,
                           source="polima-compile")
    return _report(bundle, args.json)


def _import_legacy(args, output_root: Path) -> int:
    from polima.bundle.import_legacy import import_legacy

    try:
        spec = apply_precision_profile(
            _specialize_spec(
                get_policy(args.policy), build_dir=Path(args.import_legacy).resolve()
            ),
            args.precision_profile,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"polima-compile: {error}", file=sys.stderr)
        return 2
    bundle = import_legacy(args.import_legacy, spec, output_root=output_root,
                           dataset=args.dataset, steps=args.steps)
    return _report(bundle, args.json)


def _report(bundle, as_json: bool) -> int:
    if as_json:
        from polima.util.jsonio import dumps

        print(dumps(bundle.to_dict()), end="")
        return 0

    print(table.section(f"bundle {bundle.bundle_id}"))
    print(table.render(
        [[a.name, a.precision, f"{a.elf_bytes / 1048576:.1f} MiB", a.sha256[:12],
          a.input_elements, a.output_elements] for a in bundle.graphs],
        headers=["graph", "precision", "elf", "sha256", "in", "out"],
    ))
    print(f"\n  root      {bundle.root}")
    print(f"  source    {bundle.source} <- {bundle.legacy_source_dir}")
    print(f"  total     {bundle.total_elf_bytes / 1048576:.1f} MiB of ELF")
    print(f"\nnext:  polima-deploy --bundle {bundle.bundle_id}")
    return 0


def _specialize_spec(
    spec,
    *,
    checkpoint: str | Path | None = None,
    dataset_root: str | Path | None = None,
    embodiment_tag: str | None = None,
    build_dir: Path | None = None,
):
    """Restore a robot profile from a checkpoint or portable build contract."""

    profile = None
    # An explicitly supplied checkpoint always wins. Build metadata is only for
    # a later compile/pack invocation that names the build tree by itself.
    if build_dir is not None and checkpoint is None:
        contract_path = build_dir / "input_contract.json"
        try:
            contract = json.loads(contract_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            contract = {}
        profile = contract.get("profile")
        checkpoint = contract.get("checkpoint")
        dataset_root = dataset_root or contract.get("dataset_root")

    # Old build contracts did not record a profile. Preserve their historical
    # policy spec when the original checkpoint is no longer mounted.
    usable_checkpoint = checkpoint and Path(checkpoint).is_dir()
    if not profile and not usable_checkpoint:
        return spec
    context = {
        "checkpoint": checkpoint,
        "dataset_root": dataset_root,
        "profile": profile,
    }
    if embodiment_tag:
        context["embodiment_tag"] = embodiment_tag
    return spec.specialize_profile(**context)


def _src_root() -> Path:
    """The `src/` directory this package was imported from."""
    return Path(__file__).resolve().parents[2]


def _default_outputs() -> Path:
    from polima.util.paths import outputs_root

    return outputs_root()
