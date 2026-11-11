"""Customer installation assets remain complete and relocatable."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_devkit_installer_includes_pi05_tokenizer_runtime():
    installer = (ROOT / "scripts/install_polima_modalix.sh").read_text()
    project = (ROOT / "pyproject.toml").read_text()
    assert 'TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-0.22.2}"' in installer
    assert '"tokenizers==$TOKENIZERS_VERSION"' in installer
    assert 'import lerobot, cv2, flask, tokenizers, waitress, torch, polima.studio' in installer
    assert '"tokenizers==0.22.2"' in project


@pytest.mark.parametrize("metadata, accepted", [
    ({"schema_version": 2, "polima_version": "0.3.0", "source_commit": "609100f",
      "platform_version": "3.0.0", "llima_package_version": "0.6.0", "llima": {"ref": "v0.6.0"}}, True),
    ({"schema_version": 1, "polima_version": "0.3.0", "source_commit": "609100f",
      "platform_version": "3.0.0", "llima_package_version": "0.6.0"}, False),
    ({"schema_version": 2, "polima_version": "", "source_commit": "609100f",
      "platform_version": "3.0.0", "llima_package_version": "0.6.0", "llima": {}}, False),
])
def test_devkit_installer_accepts_only_a_complete_packaged_runtime(tmp_path, metadata, accepted):
    installer = (ROOT / "scripts/install_polima_modalix.sh").read_text()
    assert 'sha256sum -c "$(basename -- "$RUNTIME_ARCHIVE").sha256"' in installer
    assert 'extractall(sys.argv[2], filter="data")' in installer
    check = installer.split("<<'PY_METADATA'\n", 1)[1].split("\nPY_METADATA", 1)[0]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(metadata))
    result = subprocess.run([sys.executable, "-", str(path)], input=check, text=True,
                            capture_output=True, check=False)
    assert (result.returncode == 0) is accepted, result.stderr


def test_source_install_embeds_the_exact_installed_runtime_identity():
    installer = (ROOT / "scripts/install_polima_modalix.sh").read_text()
    assert "dpkg-query -W -f='${Version}' sima-lmm-core" in installer
    assert "dpkg-query -W -f='${Version}' sima-lmm-dev" in installer
    assert '[[ -n "$llima_core" && "$llima_core" == "$llima_dev" ]]' in installer
    assert "DISTRO_VERSION[[:space:]]*=" in installer
    assert '-DPOLIMA_PLATFORM_VERSION="$platform_version"' in installer
    assert '-DPOLIMA_LLIMA_PACKAGE_VERSION="$llima_core"' in installer


def test_studio_service_uses_selected_installation(tmp_path):
    installer = (ROOT / "scripts/install_polima_modalix.sh").read_text()
    renderer = installer.split("<<'PY_SERVICE'\n", 1)[1].split("\nPY_SERVICE", 1)[0]
    destination = tmp_path / "studio.service"
    subprocess.run([
        sys.executable, "-", str(ROOT / "scripts/systemd/polima-studio.service"),
        str(destination), "/opt/customer/polima", "/opt/customer/venv", "operator", "robot",
    ], input=renderer, text=True, check=True)
    service = destination.read_text()
    assert "POLIMA_ROOT=/opt/customer/polima" in service
    assert "LEROBOT_VENV=/opt/customer/venv /opt/customer/venv/bin/python" in service
    assert "ReadWritePaths=/opt/customer/polima" in service
    assert "User=operator\nGroup=robot" in service
    assert "/media/nvme/lerobot" not in service


def test_customer_document_links_resolve():
    documents = [ROOT / "README.md", *ROOT.glob("docs/*.md"),
                 ROOT / "examples/so101/README.md", ROOT / "runtime/README.md",
                 ROOT / "src/polima/studio/README.md", ROOT / "src/polima/workbench/README.md"]
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if ":" in link or link.startswith("#"):
                continue
            assert (document.parent / link.split("#", 1)[0]).exists(), (document, link)


def test_upgrade_replaces_shadow_package_and_keeps_backup(tmp_path):
    installer = (ROOT / "scripts/install_polima_modalix.sh").read_text()
    migration = installer.split("<<'PY_PACKAGE'\n", 1)[1].split("\nPY_PACKAGE", 1)[0]
    root = tmp_path / "install"
    shadow = root / "python/polima"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("old version")
    installed = tmp_path / "site-packages/polima"
    installed.mkdir(parents=True)
    (installed / "__main__.py").write_text("new version")
    subprocess.run([sys.executable, "-", str(root), str(installed)], input=migration, text=True, check=True)
    assert shadow.resolve() == installed
    backups = list((root / "var/backups").glob("python-polima-*"))
    assert len(backups) == 1
    assert (backups[0] / "__init__.py").read_text() == "old version"
    subprocess.run([sys.executable, "-", str(root), str(installed)], input=migration, text=True, check=True)
    assert len(list((root / "var/backups").glob("python-polima-*"))) == 1


def test_published_package_installs_its_wheel_and_records_llima():
    installer = (ROOT / "scripts/install_polima_modalix.sh").read_text()
    assert '"${POLIMA_WHEEL:-$POLIMA_SOURCE}"' in installer
