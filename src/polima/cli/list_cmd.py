"""`polima list` -- what policies exist and what each one compiles to."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polima.bundle.layout import parse_bundle_id
from polima.policies.registry import BUILTIN, load_all
from polima.util import table


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima list")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--graphs", action="store_true", help="show each policy's subgraphs")
    parser.add_argument("--bundles-root", type=Path, help="list packaged bundles in this directory")
    args = parser.parse_args(argv)

    if args.bundles_root is not None:
        return list_bundles(args.bundles_root, as_json=args.json)

    specs = load_all(strict=False)

    if args.json:
        print(json.dumps({
            name: (specs[name].summary() if name in specs else {"status": "not ported yet"})
            for name in BUILTIN
        }, indent=2))
        return 0

    rows = []
    for name in BUILTIN:
        spec = specs.get(name)
        if spec is None:
            rows.append([name, "-", "not ported yet", "", ""])
            continue
        rows.append([
            name,
            f"{len(spec.compile.graphs)}",
            spec.train.backend,
            f"{spec.wire.magic_ascii}:{spec.wire.default_port}",
            f"{spec.robot.actions_per_chunk}x{spec.dataset.action_dim}",
        ])
    print(table.render(rows, headers=["policy", "graphs", "checkpoints from", "wire", "chunk"]))

    if args.graphs:
        for name, spec in specs.items():
            print(table.section(f"{name}: {spec.display_name}"))
            print(table.render(
                [[g.name,
                  "x".join(str(d) for d in g.inputs[0].shape),
                  "x".join(str(d) for d in g.outputs[0].shape),
                  g.layout, "/".join(g.precisions), g.compiler]
                 for g in spec.compile.graphs],
                headers=["graph", "input", "output", "layout", "precision", "compiler"],
            ))
    return 0


def list_bundles(root: Path, *, as_json: bool = False) -> int:
    if not root.is_dir():
        raise SystemExit(f"Bundle directory does not exist: {root}")
    current = root.parent / "current"
    active = current.resolve() if current.is_symlink() else None
    records = []
    for path in sorted(root.glob("*/bundle.json")):
        manifest = json.loads(path.read_text())
        bundle_id = manifest["bundle_id"]
        try:
            identity = parse_bundle_id(bundle_id)
        except ValueError:
            identity = {}
        precisions = {g.get("precision", "unknown") for g in manifest.get("graphs", [])}
        records.append({
            "bundle": bundle_id,
            "policy": manifest.get("policy", "unknown"),
            "robot": identity.get("robot", "unknown"),
            "task": "; ".join(manifest.get("tasks", [])) or identity.get("dataset", "unknown"),
            "version": identity.get("version") or "unknown",
            "steps": identity.get("steps", "unknown"),
            "precision": identity.get("precision") or (next(iter(precisions)) if len(precisions) == 1 else "mixed"),
            "active": path.parent.resolve() == active,
        })
    if as_json:
        print(json.dumps(records, indent=2))
    else:
        columns = ["bundle", "robot", "task", "version", "steps", "precision", "active"]
        print(table.render([[str(record[k]) for k in columns] for record in records], headers=columns))
    return 0
