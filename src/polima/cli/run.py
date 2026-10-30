"""`polima run` -- run inference against a deployed bundle.

Three modes:

    --fixture            send the bundle's golden inputs and check the result
                         against expected_normalized_actions.f32
    --compare HOST:PORT  send the same fixture to a second server and compare
                         the two results
    --input-dir DIR      send arbitrary .f32 tensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from polima.bundle.contract import spec_from_bundle
from polima.bundle.layout import Bundle
from polima.config.loader import load
from polima.deploy.smoke import (
    SmokeReport,
    compare_against_reference,
    compare_normalized_actions,
    numerical_smoke,
)
from polima.policies.registry import get_policy
from polima.util import table
from polima.util.jsonio import dumps, read_json
from polima.util.paths import outputs_root
from polima.wire.client import PolimaSOMClient


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima run")
    parser.add_argument("--bundle", required=True, help="bundle id or path (for fixtures)")
    parser.add_argument("--bundles-root", default=None)
    parser.add_argument(
        "--devkit", "--board", dest="board", default=None, metavar="ADDRESS",
        help="the DevKit: user@host, a saved name or host:port (default: POLIMA_DEVKIT, else 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--fixture", action="store_true", help="use the bundle's golden inputs")
    parser.add_argument("--input-dir", default=None, help="directory of <tensor>.f32 files")
    parser.add_argument(
        "--compare",
        default=None,
        metavar="HOST:PORT",
        help="also query this server and compare the two results",
    )
    parser.add_argument("--repeat", type=int, default=1, help="send N times, report latency")
    parser.add_argument(
        "--task",
        default=None,
        help="catalog task id or exact trained instruction (required for multi-task live input)",
    )
    parser.add_argument("--output", default=None, help="write the result as .f32")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = load(
        config_file=getattr(parent, "config", None),
        overrides={"board.host": args.board, "board.port": args.port},
    )
    bundle = _resolve_bundle(args.bundle, args.bundles_root)
    spec = spec_from_bundle(get_policy(bundle.policy), bundle)
    task_id = _resolve_task(args.task, bundle, fixture=args.fixture)
    port = args.port or config.board.port or spec.wire.default_port
    # On the DevKit itself the server is local; elsewhere name the DevKit.
    host = _host_of(config.board.host) or "127.0.0.1"  # --board is in config, saved DevKit names resolved

    inputs_dir = Path(args.input_dir) if args.input_dir else bundle.fixtures_dir / "inputs"
    tensors = _load_tensors(spec, inputs_dir)

    client = PolimaSOMClient(spec, f"{host}:{port}", timeout=args.timeout)
    if not client.ping():
        print(
            f"polima run: nothing listening on {host}:{port}\n"
            f"  start it with: polima deploy --bundle {bundle.bundle_id} --port {port} --start",
            file=sys.stderr,
        )
        return 2

    latencies: list[float] = []
    actions = np.zeros(0, dtype=np.float32)
    for _ in range(max(1, args.repeat)):
        actions, latency_ms = client.predict_raw(tensors, task_id=task_id)
        latencies.append(latency_ms)

    report = SmokeReport()
    expected_path = bundle.fixtures_dir / "expected" / "normalized_actions.f32"
    if expected_path.is_file() and (args.fixture or not args.input_dir):
        if spec.name == "groot":
            stats_path = bundle.fixtures_dir / "normalization_stats.npz"
            if not stats_path.is_file():
                raise SystemExit(f"polima run: missing GR00T fixture {stats_path}")
            with np.load(stats_path, allow_pickle=False) as stats:
                result = compare_normalized_actions(
                    actions,
                    np.fromfile(expected_path, dtype="<f4"),
                    stats["action_mean"],
                    stats["action_std"],
                    state=tensors.get("state"),
                    relative_dim=int(spec.runtime_context.get("relative_action_dim", 0)),
                    name="vs-pytorch-reference",
                    cosine_min=spec.smoke.cosine_min,
                    mean_abs_max=spec.smoke.mean_abs_max,
                )
        else:
            result = numerical_smoke(
                actions,
                expected_path,
                name="vs-pytorch-reference",
                cosine_min=spec.smoke.cosine_min,
                mean_abs_max=spec.smoke.mean_abs_max,
            )
        report.add(result)

    if args.compare:
        reference_client = PolimaSOMClient(
            spec,
            args.compare if ":" in args.compare else f"{args.compare}:{port}",
            timeout=args.timeout,
        )
        if not reference_client.ping():
            print(f"polima run: cannot reach {args.compare}", file=sys.stderr)
            return 2
        reference, reference_latency = reference_client.predict_raw(tensors, task_id=task_id)
        result = report.add(
            compare_against_reference(actions, reference, name=f"vs {reference_client.address}")
        )
        result.note = f"reference latency {reference_latency:.1f} ms"

    if args.output:
        np.ascontiguousarray(actions, dtype="<f4").tofile(args.output)

    if args.json:
        print(
            dumps(
                {
                    "bundle": bundle.bundle_id,
                    "address": client.address,
                    "task_id": task_id,
                    "task": bundle.tasks[task_id] if bundle.tasks else None,
                    "elements": int(actions.size),
                    "latency_ms": {
                        "min": min(latencies),
                        "max": max(latencies),
                        "mean": sum(latencies) / len(latencies),
                    },
                    "smoke": report.to_dict(),
                }
            ),
            end="",
        )
        return 0 if report.ok else 1

    print(table.section(f"{bundle.bundle_id} @ {client.address}"))
    if bundle.tasks:
        print(f"  task       {task_id}: {bundle.tasks[task_id]}")
    print(f"  returned   {actions.size} floats")
    if len(latencies) > 1:
        print(
            f"  latency    min {min(latencies):.1f} / mean "
            f"{sum(latencies) / len(latencies):.1f} / max {max(latencies):.1f} ms "
            f"over {len(latencies)} calls"
        )
    else:
        print(f"  latency    {latencies[0]:.1f} ms")
    if report.results:
        print()
        for result in report.results:
            print(f"  {result.summary()}")
            if result.note:
                print(f"        {result.note}")
    return 0 if report.ok else 1


def _resolve_task(value: str | None, bundle: Bundle, *, fixture: bool) -> int:
    if not bundle.tasks:
        if value not in (None, "0"):
            raise SystemExit("polima run: this bundle has no task catalog")
        return 0
    if value is None:
        if fixture:
            return int(bundle.default_task)
        raise SystemExit(
            "polima run: --task is required for this multi-task bundle; "
            "use a catalog id or exact instruction"
        )
    try:
        task_id = int(value)
    except ValueError:
        matches = [index for index, instruction in enumerate(bundle.tasks) if instruction == value]
        if not matches:
            choices = "\n".join(
                f"  {index}: {instruction}" for index, instruction in enumerate(bundle.tasks)
            )
            raise SystemExit(
                f"polima run: task {value!r} is not in this bundle:\n{choices}"
            ) from None
        task_id = matches[0]
    if not 0 <= task_id < len(bundle.tasks):
        raise SystemExit(f"polima run: task id {task_id} outside [0, {len(bundle.tasks)})")
    if fixture and task_id != bundle.default_task:
        raise SystemExit(
            f"polima run: the bundled smoke fixture is for task "
            f"{bundle.default_task}, not task {task_id}"
        )
    return task_id


def _host_of(address: str) -> str:
    host = address.split("@")[-1]
    return host.split(":")[0]


def _load_tensors(spec, inputs_dir: Path) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for tensor in spec.wire.request_tensors:
        # A uint8 wire tensor is a raw camera frame: <name>.u8 beside the .f32s.
        suffix, dtype = (".u8", np.uint8) if tensor.dtype == "uint8" else (".f32", np.dtype("<f4"))
        path = inputs_dir / f"{tensor.name}{suffix}"
        if not path.is_file():
            raise SystemExit(f"polima run: missing fixture {path}")
        values = np.fromfile(path, dtype=dtype)
        if values.size != tensor.elements:
            raise SystemExit(
                f"polima run: {path.name} holds {values.size} values, "
                f"tensor {tensor.name} needs {tensor.elements}"
            )
        tensors[tensor.name] = values
    return tensors


def _resolve_bundle(reference: str, bundles_root: str | None) -> Bundle:
    candidate = Path(reference).expanduser()
    if not candidate.is_dir():
        candidate = Path(bundles_root or (outputs_root() / "bundles")) / reference
    manifest = candidate / "bundle.json"
    if not manifest.is_file():
        raise SystemExit(f"polima run: no bundle at {candidate}")
    return Bundle.from_dict(read_json(manifest), candidate)
