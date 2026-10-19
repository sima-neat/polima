"""The sima-add-polima-policy skill: its files are consistent and its scaffold produces a valid policy."""
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "sima-add-polima-policy"
SCAFFOLD = SKILL / "scripts" / "scaffold_policy.py"


def test_skill_metadata_and_references_are_consistent():
    text = (SKILL / "SKILL.md").read_text()
    front = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert front and "name: sima-add-polima-policy" in front.group(1)
    assert "description:" in front.group(1)
    assert "id: sima-add-polima-policy" in (SKILL / "playbook.yml").read_text()
    named = set(re.findall(r"`((?:references|scripts)/[\w./-]+)`", text))
    assert named, "SKILL.md names its references"
    for relative in named:
        assert (SKILL / relative).is_file(), f"SKILL.md names a missing file: {relative}"
    for reference in (SKILL / "references").glob("*.md"):
        for link in re.findall(r"`([\w-]+\.md)`", reference.read_text()):
            if link in ("README.md",):
                continue
            assert (SKILL / "references" / link).is_file() or (ROOT / "docs" / link).is_file(), \
                f"{reference.name} names a missing file: {link}"


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "polima" / "policies").mkdir(parents=True)
    (repo / "tests" / "unit").mkdir(parents=True)
    shutil.copy(ROOT / "src" / "polima" / "policies" / "base.py", repo / "src" / "polima" / "policies")
    return repo


def test_scaffold_generates_a_policy_whose_spec_and_plan_validate(tmp_path, monkeypatch):
    import polima.policies

    repo = _repo(tmp_path)
    result = subprocess.run([sys.executable, str(SCAFFOLD), "skilldemo", "--magic", "SKDM", "--repo", str(repo)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "registry.BUILTIN" in result.stdout  # the integration checklist is printed
    package = repo / "src" / "polima" / "policies" / "skilldemo"
    assert {p.name for p in package.iterdir()} == {"__init__.py", "runtime.py", "graphs.py"}
    assert (repo / "tests" / "unit" / "test_skilldemo.py").is_file()

    from polima.policies import registry

    monkeypatch.setattr(polima.policies, "__path__", [*polima.policies.__path__, str(package.parent)])
    try:
        module = importlib.import_module("polima.policies.skilldemo")
        spec = module.SKILLDEMO_SPEC
        spec.validate()
        plan = spec.build_runtime_plan()
        plan.validate("skilldemo", spec.compile)
        assert spec.wire.magic.to_bytes(4, "little") == b"SKDM"
        assert plan.buffers[plan.result] == spec.wire.response_elements
        with pytest.raises(NotImplementedError, match="references/graph-decomposition.md"):
            importlib.import_module("polima.policies.skilldemo.graphs").build_modules(None)
    finally:  # leave no throwaway policy behind for other tests
        registry._REGISTRY.pop("skilldemo", None)
        for name in [m for m in sys.modules if m.startswith("polima.policies.skilldemo")]:
            del sys.modules[name]


def test_scaffold_refuses_to_overwrite_and_bad_names(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src" / "polima" / "policies" / "act").mkdir()
    for name in ("act", "Bad-Name", "class"):
        result = subprocess.run([sys.executable, str(SCAFFOLD), name, "--repo", str(repo)],
                                capture_output=True, text=True, check=False)
        assert result.returncode != 0


def test_scaffold_writes_nothing_when_a_target_folder_is_missing(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests" / "unit").rmdir()
    result = subprocess.run([sys.executable, str(SCAFFOLD), "partial", "--repo", str(repo)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "does not exist" in result.stderr
    assert not (repo / "src" / "polima" / "policies" / "partial").exists()
