"""Contracts between PoLiMa's CI and the shared sima-neat workflows."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
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
    # NEAT Core release reinstalled on DevKits whose NEAT branch build pins another LLiMa.
    assert set(manifest["neat-core"]) == {"ref", "spec"}
    assert manifest["neat-core"]["ref"] == manifest["llima"]["ref"]  # matching release trains
    assert re.fullmatch(r"[0-9a-f]{12}", manifest["neat-core"]["spec"])
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
    steps = jobs["test-devkit"]["steps"]
    install = next(step for step in steps
                   if step.get("name") == "Install exact published package in an isolated layout")
    assert install["env"]["SIMA_CLI_CHECK_FOR_UPDATE"] == "0"
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


def test_devkit_qualification_uses_a_private_lerobot_environment():
    steps = workflow("vulcan-ci.yml")["jobs"]["test-devkit"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "dpkg-query" in commands and "resolved-dependencies.lock.json" in commands
    assert "/media/nvme/lerobot" not in commands
    assert commands.count("scripts/ci/prepare_lerobot.sh") == 2
    assert 'prepare_lerobot.sh "$CANDIDATE_DIR/lerobot"' in commands
    # devkit_test.sh imports polima with -I, so the candidate must be in the venv itself.
    assert "PYTHONPATH=$CANDIDATE_DIR" not in commands and "--target" not in commands
    assert 'POLIMA_CI_PYTHON=$CANDIDATE_DIR/lerobot/bin/python' in commands
    prepare = (ROOT / "scripts/ci/prepare_lerobot.sh").read_text()
    assert "/media/nvme/lerobot" in prepare and "cp -a --reflink=auto" in prepare
    assert "install_lerobot_modalix.sh" in prepare and "SKIP_DIALOUT=1" in prepare
    assert 'rm -rf -- "$CANDIDATE_DIR" "$RESULTS_DIR" "$PACKAGE_DIR"' in commands



def test_devkit_qualification_installs_the_llima_the_package_was_built_against():
    steps = workflow("vulcan-ci.yml")["jobs"]["test-devkit"]["steps"]
    names = [step.get("name", "") for step in steps]
    install = names.index("Install the exact LLiMa the package was built against")
    download = names.index("Download the exact published package")
    # The package's own installer refuses a DevKit whose LLiMa differs from the build.
    assert download < install < names.index("Install exact published package in an isolated layout")
    assert install < names.index("Verify firmware, package and installed LLiMa identity")
    step = steps[install]
    assert "if" not in step  # both the push and the dispatch path
    assert step["run"] == 'scripts/ci/install_exact_llima.sh "$LLIMA_LOCK" "$CANDIDATE_DIR/devkit-deps"'
    paths = steps[names.index("Prepare run-specific paths")]["run"]
    assert 'LLIMA_LOCK=${RUNNER_TEMP}/polima-package-${run}/resolved-dependencies.lock.json' in paths
    fetch = steps[download]
    assert fetch["if"] == "${{ github.event_name == 'push' }}"
    assert "--download-only" in fetch["run"]
    assert '-d "$CANDIDATE_DIR/published" "$target"' in fetch["run"]
    assert 'LLIMA_LOCK=$CANDIDATE_DIR/published/resolved-dependencies.lock.json' in fetch["run"]


LOCK = {"llima": {"ref": "v0.6.2", "resolved_spec": "a5aa79a33322", "environment": "production",
                  "version": "0.6.2"}}
BRANCH = "0.6.2+refactor-301-runtime-execution.d38bf74eb8b3"
LMM = ("sima-lmm-core", "sima-lmm-cli", "sima-lmm-dev")
NEAT_BRANCH = ("sima-neat", "0.6.2+refactor-301-runtime-execution.ccb5bedb1404",
               f"sima-lmm-core (= {BRANCH})")
needs_dpkg = pytest.mark.skipif(not shutil.which("dpkg"), reason="dpkg compares Debian versions")


def _run_install_exact_llima(tmp_path: Path, installed: dict[str, str],
                             others: tuple[tuple[str, str, str], ...] = (),
                             core_brings: str = "0.6.2"):
    """Run the helper against a fake dpkg database that the fake sima-cli updates."""
    bin_dir, package = tmp_path / "bin", tmp_path / "package"
    bin_dir.mkdir()
    package.mkdir()
    (package / "resolved-dependencies.lock.json").write_text(json.dumps(LOCK))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"neat-core": {"ref": "v0.6.2", "spec": "b1f9556caec1"}}))
    state, listing, calls = tmp_path / "state", tmp_path / "listing", tmp_path / "sima-cli.calls"
    state.write_text("".join(f"{name}={version}\n" for name, version in installed.items()))
    listing.write_text("".join(f"ii \t{n}\t{v}\t{d}\n" for n, v, d in others))
    (bin_dir / "dpkg-query").write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$3" == *Status-Abbrev* ]]; then cat {listing}; exit 0; fi\n'
        f'v="$(sed -n "s/^${{@: -1}}=//p" {state})"; [[ -n "$v" ]] && printf %s "$v"\n')
    # Installing NEAT Core replaces its branch build and brings its LLiMa; LLiMa's own
    # root artifact installs the locked release.
    (bin_dir / "sima-cli").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$* skills=$NEAT_INSTALLER_INSTALL_CODEX_SKILL,$NEAT_INSTALLER_INSTALL_CLAUDE_SKILL'
        f' sync=$NEAT_INSTALLER_SKIP_DEVKIT_SYNC" >> {calls}\n'
        'case "$*" in\n'
        f'  *core@*) printf "%s\\n" {" ".join(f"{n}={core_brings}" for n in LMM)} > {state}; : > {listing} ;;\n'
        f'  *"-d "*llima@*) printf "%s\\n" {" ".join(f"{n}=0.6.2" for n in LMM)} > {state} ;;\n'
        "esac\n")
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "scripts/ci/install_exact_llima.sh"),
         str(package / "resolved-dependencies.lock.json"), str(tmp_path / "work"), str(manifest)],
        capture_output=True, text=True, check=False, env={"PATH": f"{bin_dir}:/usr/bin:/bin"})
    lines = calls.read_text().splitlines() if calls.exists() else []
    return result, [line.split(" skills=")[0] for line in lines], lines


LLIMA_CALLS = ["neat install --env production --json llima@v0.6.2:a5aa79a33322",
               "neat install --env production -d {work}/llima llima@v0.6.2:a5aa79a33322"]


@needs_dpkg
def test_install_exact_llima_replaces_a_branch_build_with_the_locked_release(tmp_path):
    result, calls, _ = _run_install_exact_llima(tmp_path, dict.fromkeys(LMM, BRANCH))
    assert result.returncode == 0, result.stderr
    assert "Installing LLiMa 0.6.2 from llima@v0.6.2:a5aa79a33322" in result.stdout
    assert calls == [c.format(work=tmp_path / "work") for c in LLIMA_CALLS]


def test_install_exact_llima_leaves_a_matching_installation_alone(tmp_path):
    result, calls, _ = _run_install_exact_llima(tmp_path, dict.fromkeys(LMM, "0.6.2"))
    assert result.returncode == 0 and "already installed" in result.stdout and calls == []


@needs_dpkg
def test_install_exact_llima_installs_when_one_package_differs(tmp_path):
    result, calls, _ = _run_install_exact_llima(tmp_path, {"sima-lmm-core": "0.6.2",
                                                           "sima-lmm-dev": "0.6.2"})
    assert result.returncode == 0 and len(calls) == 2


@needs_dpkg
def test_install_exact_llima_reinstalls_neat_core_that_pins_a_branch_llima(tmp_path):
    result, calls, raw = _run_install_exact_llima(
        tmp_path, {"sima-lmm-core": BRANCH, "sima-lmm-cli": BRANCH}, others=(NEAT_BRANCH,))
    assert result.returncode == 0, result.stderr
    assert f"sima-neat {NEAT_BRANCH[1]} requires sima-lmm-core (= {BRANCH})" in result.stdout
    assert calls == [(f"install --neat --env production -d {tmp_path / 'work/neat-core'} "
                      "core@v0.6.2:b1f9556caec1 -t all")]
    assert raw[0].endswith("skills=OFF,OFF sync=ON")
    assert "LLiMa 0.6.2 installed with NEAT Core v0.6.2" in result.stdout


@needs_dpkg
def test_install_exact_llima_falls_back_when_neat_core_brings_another_llima(tmp_path):
    result, calls, _ = _run_install_exact_llima(
        tmp_path, dict.fromkeys(LMM, BRANCH), others=(NEAT_BRANCH,), core_brings="0.6.1")
    assert result.returncode == 0, result.stderr
    assert calls[0].startswith("install --neat") and calls[1:] == [
        c.format(work=tmp_path / "work") for c in LLIMA_CALLS]


@needs_dpkg
def test_install_exact_llima_never_replaces_other_packages(tmp_path):
    result, calls, _ = _run_install_exact_llima(
        tmp_path, dict.fromkeys(LMM, BRANCH),
        others=(("someone-elses-app", "1.0", f"sima-lmm-core (= {BRANCH})"),))
    assert result.returncode == 1 and calls == []
    assert "refusing to replace someone-elses-app" in result.stderr


def test_install_exact_llima_requires_the_lock(tmp_path):
    result = subprocess.run([str(ROOT / "scripts/ci/install_exact_llima.sh"),
                             str(tmp_path / "missing.json"), str(tmp_path / "work")],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 1 and "no LLiMa lock" in result.stderr


def _llima_conflicts():
    spec = importlib.util.spec_from_file_location(
        "llima_conflicts", ROOT / "scripts/ci/llima_conflicts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_dpkg
@pytest.mark.parametrize(("depends", "blocks"), [
    (f"libc6, sima-lmm-core (= {BRANCH})", True),            # NEAT Core branch build
    ("sima-lmm-core (= 0.6.2), sima-lmm-dev (= 0.6.2)", False),  # the matching release
    ("sima-lmm-core (>= 0.6.0)", False),
    (f"sima-lmm-core (>= {BRANCH})", True),                  # 0.6.2 sorts before 0.6.2+…
    ("sima-lmm-core (<< 0.6.2)", True),
    ("sima-lmm-core", False),                                # unversioned
    (f"other-llima | sima-lmm-core (= {BRANCH})", False),    # satisfiable by another package
    ("libc6 (>= 2.36)", False),
])
def test_llima_conflicts_compares_sima_lmm_relations_with_dpkg(depends, blocks):
    listing = f"ii \tsima-neat\t1\t{depends}\n"
    assert bool(_llima_conflicts().conflicts(listing, "0.6.2")) is blocks


@needs_dpkg
def test_llima_conflicts_ignores_removed_packages_and_llima_itself():
    listing = (f"rc \tsima-neat\t1\tsima-lmm-core (= {BRANCH})\n"
               f"ii \tsima-lmm-cli\t{BRANCH}\tsima-lmm-core (= {BRANCH})\n")
    assert _llima_conflicts().conflicts(listing, "0.6.2") == []


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
