"""Studio's browser code: its unit tests and the built bundle Studio serves."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
def test_review_form_logic():
    result = subprocess.run([NODE, "--test", str(ROOT / "studio-ui/src")],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
def test_the_built_bundle_parses(tmp_path):
    static = ROOT / "src/polima/studio/static"
    scripts = re.findall(r'src="/assets/([^"]+\.js)"', (static / "index.html").read_text())
    assert scripts, "index.html loads no script"
    for name in scripts:
        module = tmp_path / (name + ".mjs")  # the bundle is an ES module
        shutil.copy(static / "assets" / name, module)
        result = subprocess.run([NODE, "--check", str(module)], capture_output=True, text=True,
                                check=False)
        assert result.returncode == 0, result.stderr
