"""Keep tests independent of installed packages and saved customer connections."""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_customer_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("POLIMA_DEVKITS", str(tmp_path / "devkits.json"))
    monkeypatch.setenv("POLIMA_WORKBENCH_HOME", str(tmp_path / "workbench"))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    for name in ("POLIMA_DEVKIT", "POLIMA_BOARD", "POLIMA_BOARD_HOST",
                 "POLIMA_DEVKIT_URLS", "POLIMA_BOARDS", "POLIMA_MODELS", "POLIMA_BUNDLES"):
        monkeypatch.delenv(name, raising=False)
