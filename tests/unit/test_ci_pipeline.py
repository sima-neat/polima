"""Behaviour of the CI scripts in scripts/ci (no SDK, Vulcan or DevKit needed)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resolve_llima = load("resolve_llima")
package_version = load("package_version")
check_skips = load("check_skips")
fetch_bundle = load("fetch_bundle")
verify_package = load("verify_package")
lane_summary = load("lane_summary")
devkit_scope = load("devkit_scope")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------- LLiMa lock

def test_manifest_selector_reads_the_ref_pin_like_core():
    assert resolve_llima.manifest_selector({"llima": {"ref": "v0.6.0", "spec": "88bfaffc1bf1"}}) \
        == "v0.6.0:88bfaffc1bf1"
    assert resolve_llima.manifest_selector({"llima": {"ref": "v0.6.0"}}) == "v0.6.0:latest"
    with pytest.raises(ValueError):
        resolve_llima.manifest_selector({"llima": {"policy": "release", "version": "0.6.0"}})


def test_resolution_is_read_after_the_sima_cli_banner():
    text = 'warning banner\n{"ref": "develop", "resolved_spec": "9ab65e87d04c", ' \
           '"metadata_url": "https://x/m.json", "environment": "production"}'
    assert resolve_llima.parse_resolution(text)["resolved_spec"] == "9ab65e87d04c"
    with pytest.raises(ValueError):
        resolve_llima.parse_resolution('{"ref": "develop"}')


def _fake_llima(tmp_path: Path, version="0.6.0+develop.9ab65e87d04c", commit="9ab65e87d04c" + "0" * 28):
    debs = {f"sima-lmm-{version}-Linux-{kind}.deb": f"{kind} payload".encode()
            for kind in ("core", "dev", "cli")}
    store = tmp_path / "store"
    store.mkdir()
    for name, data in debs.items():
        (store / name).write_bytes(data)
    metadata = {"version": version, "branch": "develop", "commit": commit,
                "resources": sorted(debs), "resources-checksum": {n: sha(d) for n, d in debs.items()}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    resolution = {"environment": "production", "ref": "develop", "resolved_spec": commit[:12],
                  "metadata_url": (tmp_path / "metadata.json").as_uri()}
    cli = tmp_path / "sima-cli"
    cli.write_text(f"""#!{sys.executable}
import json, shutil, sys
from pathlib import Path
args = sys.argv[1:]
if "--json" in args:
    print("Environment: host")
    print(json.dumps({resolution!r}))
elif "--download-only" in args:
    target = Path(args[args.index("-d") + 1])
    assert args[-1] == "llima/debs@develop:{commit[:12]}", args[-1]
    for name in {sorted(debs)!r}:
        shutil.copy(Path({str(store)!r}) / name, target / name)
""")
    cli.chmod(0o755)
    return cli, store, metadata


def test_resolve_downloads_verifies_and_locks_the_exact_artifact(tmp_path):
    cli, _, metadata = _fake_llima(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"llima": {"ref": "v0.6.0", "spec": "88bfaffc1bf1"}}))
    lock_path = tmp_path / "lock.json"
    assert resolve_llima.main(["--manifest", str(manifest), "--lane", "develop", "--ref", "develop:latest",
                               "--sima-cli", str(cli), "--output", str(tmp_path / "debs"),
                               "--lock", str(lock_path)]) == 0
    lock = json.loads(lock_path.read_text())
    assert lock["lane"] == "develop"
    assert lock["llima"]["requested"] == "develop:latest"
    assert lock["llima"]["commit"] == metadata["commit"]
    assert set(lock["llima"]["packages"]) == {n for n in metadata["resources"] if "-cli." not in n}


def test_resolve_rejects_a_download_that_differs_from_the_published_checksum(tmp_path):
    cli, store, _ = _fake_llima(tmp_path)
    next(store.glob("*-core.deb")).write_bytes(b"tampered")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"llima": {"ref": "v0.6.0"}}))
    with pytest.raises(ValueError, match="checksum"):
        resolve_llima.main(["--manifest", str(manifest), "--lane", "develop", "--ref", "develop:latest",
                            "--sima-cli", str(cli), "--output", str(tmp_path / "debs"),
                            "--lock", str(tmp_path / "lock.json")])


def test_the_release_lane_cannot_override_the_manifest_pin(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"llima": {"ref": "v0.6.0"}}))
    with pytest.raises(SystemExit):
        resolve_llima.main(["--manifest", str(manifest), "--lane", "release", "--ref", "develop:latest",
                            "--output", str(tmp_path / "d"), "--lock", str(tmp_path / "l.json")])


# ------------------------------------------------------- versions and skips

def test_package_version_follows_the_published_llima_scheme():
    manifest = {"package-version": "0.3.0"}
    assert package_version.package_version(manifest, {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.3.0"}) \
        == "0.3.0"
    assert package_version.package_version(
        manifest, {"GITHUB_REF_TYPE": "branch", "GITHUB_REF_NAME": "feature/SDK-3.0",
                   "GITHUB_SHA": "609100f661585a02359322816f594477fd141e8f"}) \
        == "0.3.0+feature.sdk.3.0.609100f66158"
    with pytest.raises(ValueError):
        package_version.package_version(manifest, {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.4.0"})


JUNIT = """<testsuites><testsuite>
<testcase classname="tests.unit.test_plan" name="a"><skipped message="no imported bundle; run: x"/></testcase>
<testcase classname="tests.unit.test_pi05" name="b"><skipped message="could not import 'torch'"/></testcase>
<testcase classname="tests.unit.test_pi05" name="c"/>
</testsuite></testsuites>"""


def test_only_listed_skips_are_allowed():
    allow = "# comment\ntests/unit/test_plan.py | no imported bundle\n"
    assert check_skips.unexpected(JUNIT, allow) == ["tests/unit/test_pi05.py: could not import 'torch'"]


def test_the_committed_allow_list_is_well_formed():
    rules = check_skips.allowed_rules((ROOT / "tests/allowed_skips.txt").read_text())
    assert rules and all((ROOT / path).is_file() for path, _ in rules)


def test_bundle_checksums_reject_paths_outside_the_bundle():
    good = fetch_bundle.parse_sums(f"{'a' * 64}  bundle/bundle.json\n")
    assert good == {"bundle/bundle.json": "a" * 64}
    assert fetch_bundle.parse_sums(f"{'b' * 64}  ./bundle/bundle.json\n") == {"bundle/bundle.json": "b" * 64}


def test_bundle_fetch_verifies_only_the_bundle_files(tmp_path, monkeypatch):
    """The Hub rewrites .gitattributes and model cards after SHA256SUMS is written."""
    files = {"bundle/bundle.json": b"{}", "README.md": b"edited card", ".gitattributes": b"*.elf lfs"}
    digest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    digest.update(dict.fromkeys(("README.md", ".gitattributes"), hashlib.sha256(b"stale").hexdigest()))
    sums = "".join(f"{digest[name]}  ./{name}\n" for name in files)
    requested = []

    def download(url, destination):
        name = url.split("/resolve/rev/", 1)[1]
        requested.append(name)
        destination.write_bytes(sums.encode() if name == "SHA256SUMS" else files[name])

    monkeypatch.setattr(fetch_bundle, "download", download)
    bundle = fetch_bundle.fetch("org/repo", "rev", tmp_path)
    assert bundle == tmp_path / "org--repo@rev" / "bundle" and (bundle / "bundle.json").read_bytes() == b"{}"
    assert requested == ["SHA256SUMS", "bundle/bundle.json"]
    for line in (f"{'a' * 64}  ../etc/passwd", f"{'a' * 64}  /etc/passwd", "abc  bundle/x"):
        with pytest.raises(ValueError):
            fetch_bundle.parse_sums(line)


# ------------------------------------------------------- package verification

def elf(rpath: bool = False, machine: int = 183) -> bytes:
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", header, 18, machine)
    if not rpath:
        return bytes(header)
    struct.pack_into("<Q", header, 32, 64)            # e_phoff
    struct.pack_into("<HH", header, 54, 56, 1)        # e_phentsize, e_phnum
    phdr = bytearray(56)
    struct.pack_into("<I", phdr, 0, 2)                # PT_DYNAMIC
    struct.pack_into("<Q", phdr, 8, 120)              # p_offset
    struct.pack_into("<Q", phdr, 32, 32)              # p_filesz
    dynamic = struct.pack("<qQqQ", 29, 1, 0, 0)       # DT_RUNPATH, DT_NULL
    return bytes(header + phdr + dynamic)


def tar_gz(members: dict[str, tuple[bytes, int]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, (data, mode) in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


LOCK = {"schema_version": 1, "lane": "release", "llima": {
    "repository": "llima/debs", "requested": "v0.6.0:88bfaffc1bf1", "environment": "production",
    "ref": "v0.6.0", "resolved_spec": "88bfaffc1bf1", "commit": "88bfaffc1bf1" + "5" * 28,
    "version": "0.6.0", "metadata_url": "https://artifacts/llima/m.json",
    "packages": {"sima-lmm-0.6.0-Linux-core.deb": "c" * 64, "sima-lmm-0.6.0-Linux-dev.deb": "d" * 64}}}
MANIFEST = {"package-version": "0.3.0", "platform-version": "3.0.0"}
COMMIT = "609100f661585a02359322816f594477fd141e8f"


def make_package(tmp_path: Path, *, runtime_overrides=None, cli=None, lane="release") -> Path:
    version = "0.3.0+feature.sdk.3.0.609100f66158"
    package = tmp_path / "package"
    package.mkdir()
    lock = dict(LOCK, lane=lane)
    runtime = {"schema_version": 2, "polima_version": version, "source_commit": COMMIT,
               "platform_version": "3.0.0", "llima_package_version": "0.6.0", "lane": lane,
               "llima": {k: LOCK["llima"][k] for k in verify_package.LOCK_FIELDS}}
    runtime.update(runtime_overrides or {})
    runtime_name = f"polima-runtime-{version}-modalix.tar.gz"
    files = {
        runtime_name: tar_gz({
            "runtime/bin/polima_cli": (cli or elf(), 0o755),
            "runtime/bin/polima_server": (elf(), 0o755),
            "runtime/tests/polima_kernel_tests": (elf(), 0o755),
            "runtime/runtime.json": (json.dumps(runtime).encode(), 0o644)}),
        f"polima-source-{version}.tar.gz": tar_gz(
            {name: (b"x", 0o644) for name in verify_package.SOURCE_MEMBERS if name != "deps/manifest.json"}
            | {"deps/manifest.json": (json.dumps(MANIFEST).encode(), 0o644)}),
        "resolved-dependencies.lock.json": json.dumps(lock).encode(),
        "polima-package-manifest.json": json.dumps(MANIFEST).encode(),
        "install_polima.sh": b"#!/bin/sh\n",
    }
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("polima-0.3.0.dist-info/METADATA", "Name: polima\nVersion: 0.3.0\n")
        archive.writestr("polima/__main__.py", "")
        archive.writestr("polima/studio/static/index.html", "")
    files["polima-0.3.0-py3-none-any.whl"] = wheel.getvalue()
    files[f"{runtime_name}.sha256"] = f"{sha(files[runtime_name])}  {runtime_name}\n".encode()
    for name, data in files.items():
        (package / name).write_bytes(data)
    metadata = {"name": "gh:sima-neat/polima", "version": version, "release": "",
                "platforms": [{"type": "board", "compatible_with": ["modalix"], "version": "==3.0.0"}],
                "resources": sorted(files), "resources-checksum": {n: sha(d) for n, d in files.items()},
                "installation": {"script": "./install_polima.sh"},
                "branch": "feature/sdk-3.0", "commit": COMMIT, "commit_folder": COMMIT[:12]}
    (package / "metadata.json").write_text(json.dumps(metadata))
    return package


def test_a_consistent_package_verifies(tmp_path):
    summary = verify_package.verify(make_package(tmp_path), COMMIT, "feature/sdk-3.0")
    assert summary["llima_commit"] == LOCK["llima"]["commit"]


@pytest.mark.parametrize("kwargs, message", [
    ({"cli": elf(rpath=True)}, "RPATH"),
    ({"cli": elf(machine=62)}, "AArch64"),
    ({"runtime_overrides": {"llima_package_version": "0.6.1"}}, "LLiMa version"),
    ({"runtime_overrides": {"llima": {}}}, "!= lock"),
    ({"lane": "develop"}, "release lane"),
])
def test_an_inconsistent_package_is_rejected(tmp_path, kwargs, message):
    with pytest.raises(verify_package.PackageError, match=message):
        verify_package.verify(make_package(tmp_path, **kwargs), COMMIT, "feature/sdk-3.0")


def test_a_changed_file_or_wrong_commit_is_rejected(tmp_path):
    package = make_package(tmp_path)
    with pytest.raises(verify_package.PackageError, match="commit"):
        verify_package.verify(package, "0" * 40)
    (package / "install_polima.sh").write_text("changed")
    with pytest.raises(verify_package.PackageError, match="checksum"):
        verify_package.verify(package)


def test_lane_summary_records_the_resolved_llima(tmp_path):
    package = make_package(tmp_path)
    text = lane_summary.summary(package)
    assert LOCK["llima"]["commit"] in text and "llima/debs@v0.6.0:88bfaffc1bf1" in text


# ------------------------------------------------------------- DevKit scope

@pytest.mark.parametrize("event, ref_type, ref, changed, run", [
    ("push", "branch", "feature/docs", ["docs/robot.md", "./README.md", ".github/ISSUE_TEMPLATE/bug.md",
                                        ".github/workflows/source-ci.yml"], False),
    ("push", "branch", "feature/x", ["docs/robot.md", "runtime/src/cli_main.cpp"], True),
    ("push", "branch", "feature/x", ["src/polima/policies/groot/n17.py"], True),
    ("push", "branch", "feature/x", ["examples/so101/setup/install_lerobot_modalix.sh"], True),
    ("push", "branch", "feature/x", ["a/new/unlisted.file"], True),
    ("push", "branch", "feature/x", None, True),
    ("push", "branch", "develop", ["docs/robot.md"], True),
    ("push", "branch", "main", ["docs/robot.md"], True),
    ("push", "branch", "release-0.3", ["docs/robot.md"], True),
    ("push", "tag", "v0.3.0", ["docs/robot.md"], True),
    ("workflow_dispatch", "branch", "feature/docs", ["docs/robot.md"], True),
])
def test_only_documentation_only_branches_skip_the_devkit_test(event, ref_type, ref, changed, run):
    decision, reason = devkit_scope.decide(event, ref_type, ref, changed)
    assert decision is run and reason


def test_scope_compares_the_branch_with_its_base_and_writes_the_job_output(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=ci", "-c", "user.email=ci@example.com", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "develop")
    (tmp_path / "README.md").write_text("base")
    git("add", "."), git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "feature/docs")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/robot.md").write_text("docs")
    git("add", "."), git("commit", "-q", "-m", "docs")
    monkeypatch.chdir(tmp_path)
    output, summary = tmp_path / "output", tmp_path / "summary"
    args = ["--event", "push", "--ref-type", "branch", "--ref", "feature/docs", "--base", "develop",
            "--github-output", str(output), "--github-summary", str(summary)]
    assert devkit_scope.main(args) == 0
    assert output.read_text().splitlines()[0] == "run_devkit=false"
    (tmp_path / "runtime.cpp").write_text("code")
    git("add", "."), git("commit", "-q", "-m", "code")
    output.unlink()
    assert devkit_scope.main(args) == 0
    assert output.read_text().splitlines()[0] == "run_devkit=true"
    assert "DevKit test" in summary.read_text()
