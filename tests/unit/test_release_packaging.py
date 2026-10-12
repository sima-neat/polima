"""Customer installation assets remain complete and relocatable."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
