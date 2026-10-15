"""Install a self-contained LeRobot wheel with the pinned export stack.

The local checkout contains the policy code used by the checkpoints, but its
metadata requires newer Torch/Transformers versions than the pinned export
stack. Build from an isolated copy and relax only those metadata ranges; never
modify the training checkout itself.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPLACEMENTS = {
    '"torch>=2.7,<2.12.0"': '"torch>=2.3.1,<2.12.0"',
    '"torchvision>=0.22.0,<0.27.0"': '"torchvision>=0.18.1,<0.27.0"',
    '"numpy>=2.0.0,<2.3.0"': '"numpy>=1.26.4,<2.3.0"',
    '"huggingface-hub>=1.0.0,<2.0.0"': '"huggingface-hub>=0.36.2,<2.0.0"',
    '"packaging>=24.2,<26.0"': '"packaging>=24.2,<27.0"',
    'transformers-dep = ["transformers>=5.4.0,<5.6.0"]': (
        'transformers-dep = ["transformers>=4.57.1,<5.6.0"]'
    ),
    'av-dep = ["av>=15.0.0,<16.0.0"]': 'av-dep = ["av>=15.0.0,<19.0.0"]',
}


def _minimal_namespace(path: Path, name: str) -> None:
    """Keep an explicit-import namespace from loading every optional feature."""
    if not path.is_file():
        raise RuntimeError(f"LeRobot source changed; missing {path}")
    path.write_text(
        f'"""PoLiMa compatibility namespace; import {name} modules explicitly."""\n',
        encoding="utf-8",
    )


def _postpone_annotations(path: Path) -> None:
    """Make annotations compatible with APIs omitted by the pinned runtime."""
    text = path.read_text(encoding="utf-8")
    if "from __future__ import annotations" in text:
        return
    tree = ast.parse(text, filename=str(path))
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            line = tree.body[0].end_lineno
            assert line is not None
            lines = text.splitlines(keepends=True)
            lines.insert(line, "\nfrom __future__ import annotations\n")
            path.write_text("".join(lines), encoding="utf-8")
            return
    path.write_text("from __future__ import annotations\n\n" + text, encoding="utf-8")


def patch_staged_copy(staged: Path) -> None:
    """Apply PoLiMa's import isolation only to the temporary LeRobot copy."""
    package = staged / "src" / "lerobot"
    _minimal_namespace(package / "policies" / "__init__.py", "policy")
    _minimal_namespace(package / "datasets" / "__init__.py", "dataset")

    # LeRobot's RGB dataset reader needs pyav_utils through video_utils. Its
    # annotations name ``av.option.Option``, which is absent from PyAV 18's
    # public package namespace. Deferring annotation evaluation keeps runtime
    # video decoding intact without changing the validated PyAV pin.
    pyav_utils = package / "datasets" / "pyav_utils.py"
    if not pyav_utils.is_file():
        raise RuntimeError(f"LeRobot source changed; missing {pyav_utils}")
    _postpone_annotations(pyav_utils)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: install_lerobot_compat.py SOURCE CONSTRAINTS", file=sys.stderr)
        return 2
    source = Path(sys.argv[1]).resolve()
    constraints = Path(sys.argv[2]).resolve()
    manifest = source / "pyproject.toml"
    if not manifest.is_file():
        print(f"LeRobot checkout not found: {source}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="polima-lerobot-") as temporary:
        staged = Path(temporary) / "lerobot"
        shutil.copytree(
            source,
            staged,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
        )
        path = staged / "pyproject.toml"
        text = path.read_text(encoding="utf-8")
        for old, new in REPLACEMENTS.items():
            if old not in text:
                raise RuntimeError(f"LeRobot metadata changed; missing {old!r}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")

        # Parent package initializers run before explicit submodule imports.
        # Upstream's policy initializer imports EO1/GR00T/Evo1, while its
        # dataset initializer imports aggregation and depth tooling. Neither is
        # part of PoLiMa's explicit export path.
        patch_staged_copy(staged)

        return subprocess.call([
            sys.executable, "-m", "pip", "install",
            "-c", str(constraints), f"{staged}[smolvla]",
        ])


if __name__ == "__main__":
    raise SystemExit(main())
