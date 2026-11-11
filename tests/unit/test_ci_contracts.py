"""Contracts between PoLiMa's CI and the shared sima-neat workflows."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
SHARED = "sima-neat/.github/.github/workflows/"
RUN_DEVKIT = "needs.resolve-test-scope.outputs.run_devkit"


def workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def triggers(data: dict) -> dict:
    return data.get("on", data.get(True))  # PyYAML reads the bare key `on` as True


def test_manifest_uses_the_formats_core_and_the_sysroot_resolver_accept():
    manifest = json.loads((ROOT / "deps/manifest.json").read_text())
    # sima-neat/.github scripts/resolve_sysroot.py
    assert re.fullmatch(r"3\.0\.0_master_B[1-9][0-9]*", manifest["elxr_version"])
    assert manifest["platform-version"] == "3.0.0"
    assert manifest["sdk"] == {"channel": "develop", "sysroot-version": "3.0.0"}
    # Core's manifest_dependency_spec: a ref pin, never an unsupported policy.
    assert set(manifest["llima"]) == {"ref", "spec"}
    assert re.fullmatch(r"v\d+\.\d+\.\d+", manifest["llima"]["ref"])
    assert re.fullmatch(r"[0-9a-f]{12}", manifest["llima"]["spec"])
    project = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    assert manifest["package-version"] == project.group(1)


def test_every_action_is_pinned_and_every_shared_workflow_is_sima_neat():
    for path in WORKFLOWS.glob("*.yml"):
        for uses in re.findall(r"uses:\s*(\S+)", path.read_text()):
            if uses.startswith("actions/"):
                assert re.fullmatch(r"actions/[\w-]+@[0-9a-f]{40}", uses), f"{path.name}: {uses}"
            else:
                assert uses.startswith(SHARED) and uses.endswith("@main"), f"{path.name}: {uses}"


def test_pull_requests_run_only_on_github_hosted_machines():
    source = workflow("source-ci.yml")
    assert "pull_request" in triggers(source)
    for job in source["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False
    assert "pull_request" not in triggers(workflow("vulcan-ci.yml"))


def test_source_ci_rejects_unlisted_skips():
    steps = workflow("source-ci.yml")["jobs"]["python"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "--junitxml" in commands
    assert "scripts/ci/check_skips.py" in commands and "tests/allowed_skips.txt" in commands


def test_vulcan_ci_builds_both_lanes_with_the_shared_build_workflow():
    jobs = workflow("vulcan-ci.yml")["jobs"]
    for lane, job in (("release", jobs["build-release"]), ("develop", jobs["build-develop"])):
        assert job["uses"] == SHARED + "vulcan-build.yml@main"
        assert f"scripts/ci/vulcan_host_build.sh {lane}" in job["with"]["build_command"]
        assert job["with"]["artifact_path"] == f"dist/{lane}/*"
        assert job["with"]["vulcan_env"] == "${{ needs.resolve-vulcan-config.outputs.vulcan_env }}"
    assert "verify_package.py dist/release" in jobs["build-release"]["with"]["post_build_command"]


def test_every_push_publishes_tests_and_promotes_the_package_like_llima():
    data = workflow("vulcan-ci.yml")
    jobs = data["jobs"]
    assert not triggers(data)["workflow_dispatch"], "a manual run tests everything; no inputs"
    publish = jobs["publish-vulcan-artifacts"]
    assert publish["if"] == "${{ github.event_name != 'schedule' }}"  # nightly only rebuilds
    assert publish["needs"] == ["resolve-vulcan-config", "build-release"]
    assert publish["with"]["artifact_pattern"] == "polima-release-dist"
    assert publish["with"]["environment_name"] == "${{ needs.resolve-vulcan-config.outputs.vulcan_env }}"
    scope = jobs["resolve-test-scope"]
    assert scope["runs-on"] == "ubuntu-24.04"
    assert "scripts/ci/devkit_scope.py" in "\n".join(step.get("run", "") for step in scope["steps"])
    assert f"{RUN_DEVKIT} == 'true'" in jobs["resolve-test-runner"]["if"]
    assert jobs["test-devkit"]["if"] == "${{ " + RUN_DEVKIT + " == 'true' }}"
    assert {"resolve-test-scope", "resolve-test-runner", "publish-vulcan-artifacts"} <= set(
        jobs["test-devkit"]["needs"])
    promote = jobs["promote-latest-tag"]
    assert promote["needs"][-1] == "test-devkit"
    gate = " ".join(promote["if"].split())
    for condition in ("always()", "needs.resolve-test-scope.result == 'success'",
                      "needs.publish-vulcan-artifacts.result == 'success'",
                      "needs.test-devkit.result == 'success'",
                      f"{RUN_DEVKIT} == 'false' && needs.test-devkit.result == 'skipped'"):
        assert condition in gate, condition


def test_workflows_create_no_environments_and_use_only_org_secrets():
    text = "\n".join(p.read_text() for p in WORKFLOWS.glob("*.yml"))
    assert not re.search(r"^\s+environment:", text, re.MULTILINE), "environments come from the shared workflows"
    assert set(re.findall(r"secrets\.(\w+)", text)) <= {"NEAT_RELEASES_APP_ID",
                                                        "NEAT_RELEASES_APP_PRIVATE_KEY"}


def test_devkit_job_tests_the_published_artifact_and_never_builds():
    steps = workflow("vulcan-ci.yml")["jobs"]["test-devkit"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "--download-only" in commands and "verify_package.py" in commands
    assert "scripts/ci/devkit_test.sh" in commands
    for forbidden in ("cmake", "build_runtime.sh", "vulcan_build.sh"):
        assert forbidden not in commands


def test_reference_bundles_are_the_revisions_the_benchmarks_document():
    bundles = json.loads((ROOT / "scripts/ci/reference_bundles.json").read_text())["bundles"]
    benchmarks = (ROOT / "docs/benchmarks.md").read_text()
    for bundle in bundles:
        assert f"https://huggingface.co/{bundle['repo']}/tree/{bundle['revision']}" in benchmarks


def test_customer_documentation_has_no_broken_local_links():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/ci/check_docs.py")], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
