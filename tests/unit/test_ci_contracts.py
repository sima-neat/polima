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
SHARED_REF = "a1bdd92b98bc59ff68f5ad824f6881def3ab132b"
RUN_DEVKIT = "needs.resolve-test-scope.outputs.run_devkit"


def workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def triggers(data: dict) -> dict:
    return data.get("on", data.get(True))  # PyYAML reads the bare key `on` as True


def test_manifest_uses_the_formats_core_and_the_sysroot_resolver_accept():
    manifest = json.loads((ROOT / "deps/manifest.json").read_text())
    # sima-neat/.github scripts/resolve_sysroot.py
    assert re.fullmatch(r"3\.0\.1_master_B[1-9][0-9]*", manifest["elxr_version"])
    assert manifest["platform-version"] == "3.0.1"
    assert manifest["sdk"] == {"channel": "develop", "sysroot-version": "3.0.1"}
    # Core's manifest_dependency_spec: a ref pin, never an unsupported policy.
    assert set(manifest["llima"]) == {"ref", "spec"}
    assert re.fullmatch(r"v\d+\.\d+\.\d+", manifest["llima"]["ref"])
    assert re.fullmatch(r"[0-9a-f]{12}", manifest["llima"]["spec"])
    compiler = manifest["model-compiler"]
    assert compiler["repository"] == "model-compiler/arm64"
    assert compiler["release"] == {"ref": "v3.0.0.2", "spec": "6b2d138d1927"}
    assert compiler["develop"] == {"ref": "develop", "spec": "latest"}
    project = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    assert manifest["package-version"] == project.group(1)


def test_release_version_is_consistent_and_does_not_reuse_v030():
    project = re.search(
        r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE
    ).group(1)
    ui = json.loads((ROOT / "src/polima/studio/ui/package.json").read_text())
    ui_lock = json.loads((ROOT / "src/polima/studio/ui/package-lock.json").read_text())
    main = (ROOT / "src/polima/cli/main.py").read_text()

    assert project == "0.4.0"
    assert ui["version"] == project
    assert ui_lock["version"] == project
    assert ui_lock["packages"][""]["version"] == project
    assert f'return "{project}+src"' in main


def test_release_versions_are_consistent_across_runtime_and_customer_guides():
    compatibility = (ROOT / "src/polima/compatibility.py").read_text()
    cmake = (ROOT / "runtime/CMakeLists.txt").read_text()
    guide = (ROOT / "src/polima/workbench/static/guide.html").read_text()
    install = (ROOT / "docs/install.md").read_text()

    assert 'MODELSDK_VERSION = "3.0.0.2"' in compatibility
    assert 'COMPILER_LLIMA_VERSION = "0.6.1"' in compatibility
    assert 'LLIMA_VERSION = "0.6.2"' in compatibility
    assert 'set(POLIMA_PLATFORM_VERSION "3.0.1"' in cmake
    assert "Model Compiler 3.0.0.2" in guide
    assert "DevKit runtime uses LLiMa 0.6.2" in guide
    assert "| 3.0.0.2 | 3.0.1 | 0.6.2 |" in install


def test_every_action_is_pinned_and_every_shared_workflow_is_sima_neat():
    for path in WORKFLOWS.glob("*.yml"):
        for uses in re.findall(r"uses:\s*(\S+)", path.read_text()):
            if uses.startswith("actions/"):
                assert re.fullmatch(r"actions/[\w-]+@[0-9a-f]{40}", uses), f"{path.name}: {uses}"
            else:
                assert uses.startswith(SHARED) and uses.endswith("@" + SHARED_REF), \
                    f"{path.name}: {uses}"


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
        assert job["uses"] == SHARED + "vulcan-build.yml@" + SHARED_REF
        assert f"scripts/ci/vulcan_host_build.sh {lane}" in job["with"]["build_command"]
        assert job["with"]["artifact_path"] == f"dist/{lane}/*"
        assert job["with"]["vulcan_env"] == "${{ needs.resolve-vulcan-config.outputs.vulcan_env }}"
    assert "verify_package.py dist/release" in jobs["build-release"]["with"]["post_build_command"]


def test_only_trusted_pushes_publish_and_manual_runs_are_validation_only():
    data = workflow("vulcan-ci.yml")
    jobs = data["jobs"]
    assert triggers(data)["push"]["branches"] == ["develop", "main", "release-**"]
    assert triggers(data)["workflow_dispatch"] == {}
    publish = jobs["publish-vulcan-artifacts"]
    assert publish["if"] == "${{ github.event_name == 'push' }}"
    assert publish["needs"] == ["resolve-vulcan-config", "build-release", "compiler"]
    assert publish["with"]["artifact_pattern"] == "polima-release-dist"
    assert publish["with"]["environment_name"] == "${{ needs.resolve-vulcan-config.outputs.vulcan_env }}"
    scope = jobs["resolve-test-scope"]
    assert scope["runs-on"] == "ubuntu-24.04"
    assert "scripts/ci/devkit_scope.py" in "\n".join(step.get("run", "") for step in scope["steps"])
    assert f"{RUN_DEVKIT} == 'true'" in jobs["resolve-test-runner"]["if"]
    assert f"{RUN_DEVKIT} == 'true'" in jobs["test-devkit"]["if"]
    assert "needs.publish-vulcan-artifacts.result == 'success'" in jobs["test-devkit"]["if"]
    assert {"resolve-test-scope", "resolve-test-runner", "build-release"} <= set(
        jobs["test-devkit"]["needs"])
    promote = jobs["promote-latest-tag"]
    assert promote["needs"][-1] == "test-devkit"
    gate = " ".join(promote["if"].split())
    for condition in ("always()", "github.event_name == 'push'",
                      "needs.publish-vulcan-artifacts.result == 'success'",
                      "needs.test-devkit.result == 'success'"):
        assert condition in gate, condition


def test_workflows_create_no_environments_and_use_only_org_secrets():
    text = "\n".join(p.read_text() for p in WORKFLOWS.glob("*.yml"))
    assert not re.search(r"^\s+environment:", text, re.MULTILINE), "environments come from the shared workflows"
    assert set(re.findall(r"secrets\.(\w+)", text)) <= {"NEAT_RELEASES_APP_ID",
                                                        "NEAT_RELEASES_APP_PRIVATE_KEY"}


def test_devkit_job_tests_the_published_artifact_and_never_builds():
    jobs = workflow("vulcan-ci.yml")["jobs"]
    assert jobs["test-devkit"]["env"]["SIMA_CLI_CHECK_FOR_UPDATE"] == "0"
    steps = jobs["test-devkit"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "verify_package.py" in commands
    assert "publish-vulcan-artifacts" in jobs["test-devkit"]["needs"]
    assert "sima-cli neat install" in commands
    assert "sima-cli neat install --env \"$NEAT_VULCAN_ENV\" --json" in commands
    assert "parse_resolution" in commands and "metadata_url" in commands
    assert 'polima@${GITHUB_REF_NAME}:${GITHUB_SHA:0:12}' in commands
    assert "POLIMA_ROOT" in commands and "LEROBOT_VENV" in commands
    assert "POLIMA_INSTALL_LINKS=0" in commands and "POLIMA_INSTALL_SERVICE=0" in commands
    assert "scripts/ci/devkit_test.sh" in commands
    for forbidden in ("cmake", "build_runtime.sh", "vulcan_build.sh"):
        assert forbidden not in commands


def test_devkit_qualification_serializes_accelerator_measurements():
    script = (ROOT / "scripts/ci/devkit_test.sh").read_text()
    lock = 'flock -x 9\n'
    assert 'bench_lock="${POLIMA_CI_BENCH_LOCK:-/tmp/polima-bench.lock}"' in script
    assert "os.O_RDONLY | os.O_CREAT" in script
    assert 'exec 9<"$bench_lock"' in script
    assert lock in script
    assert script.index(lock) < script.index('"$cli" --runtime-info')


def test_devkit_qualification_does_not_replace_shared_installations():
    steps = workflow("vulcan-ci.yml")["jobs"]["test-devkit"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "dpkg-query" in commands and "resolved-dependencies.lock.json" in commands
    assert "cp -a --reflink=auto" in commands
    assert "/media/nvme/lerobot" in commands
    assert 'rm -rf -- "$CANDIDATE_DIR" "$RESULTS_DIR" "$PACKAGE_DIR"' in commands


def test_model_compiler_release_and_develop_lanes_are_vulcan_jobs():
    job = workflow("vulcan-ci.yml")["jobs"]["compiler"]
    assert job["strategy"]["matrix"]["lane"] == ["release", "develop"]
    assert job["with"]["runner_profile"] == "ubuntu"
    assert job["with"]["architecture"] == "arm64"
    assert "scripts/ci/compiler_lane.sh" in job["with"]["build_command"]
    lane = (ROOT / "scripts/ci/compiler_lane.sh").read_text()
    assert "python -m polima doctor" in lane
    assert 'POLIMA_COMPILER_LANE="$lane"' in lane
    assert "--imports --no-devkit --json" in lane


def test_reference_bundles_are_the_revisions_the_benchmarks_document():
    bundles = json.loads((ROOT / "scripts/ci/reference_bundles.json").read_text())["bundles"]
    benchmarks = (ROOT / "docs/benchmarks.md").read_text()
    assert {bundle["name"]: bundle["revision"] for bundle in bundles} == {
        "act": "36f6e012b20d7c486600207ad10df62fa9ffeb8d",
        "groot-n1.6": "7279862b80b1ed906546d33eed74c5e6ac781a7d",
        "groot-n1.7": "af7076f74e189bd05b1a6200a94fe27d31ff8b11",
        "pi05": "fcb19a410a00c0c00566a83cb5e278b80cf43b79",
    }
    for bundle in bundles:
        assert f"https://huggingface.co/{bundle['repo']}/tree/{bundle['revision']}" in benchmarks
    assert re.search(r"Palette 3\.0\.0\.2,\s+LLiMa 0\.6\.2", benchmarks)
    assert "PoLiMa runtime `faa710b`" in benchmarks
    assert "Pi0.5 | PaliGemma" in benchmarks and "| 64 |" in benchmarks


def test_customer_documentation_has_no_broken_local_links():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/ci/check_docs.py")], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
