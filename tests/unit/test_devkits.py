from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
import urllib.request

import pytest
from pathlib import Path

from polima.config import devkits
from polima.config.loader import load
from polima.workbench.app import Workbench, serve


@pytest.fixture(autouse=True)
def saved(tmp_path, monkeypatch):
    monkeypatch.setenv("POLIMA_DEVKITS", str(tmp_path / "devkits.json"))
    return tmp_path / "devkits.json"


def test_devkits_are_saved_replaced_by_name_and_removed(tmp_path, saved):
    key = tmp_path / "id_lab"
    key.write_text("")
    devkits.add("sima@10.0.0.2", str(key), "lab")
    devkits.add("sima@10.0.0.3", name="lab")
    devkits.add(" sima@192.168.1.20 ")
    assert [(d["name"], d["host"], "key" in d) for d in devkits.load()] == [
        ("lab", "sima@10.0.0.3", False), ("192.168.1.20", "sima@192.168.1.20", False)]
    devkits.remove("lab")
    assert [d["name"] for d in json.loads(saved.read_text())] == ["192.168.1.20"]
    with pytest.raises(LookupError):
        devkits.remove("lab")


@pytest.mark.parametrize("host, key, name", [
    ("10.0.0.2", "", ""), ("sima@10.0.0.2; reboot", "", ""), ("sima@-oProxyCommand=x", "", "bad name!"),
    ("-oProxyCommand@x", "", ""), ("sima@-x", "", ""), ("sima@10.0.0.2", "/no/such/key", ""),
])
def test_bad_addresses_names_and_keys_are_refused(host, key, name):
    with pytest.raises(ValueError):
        devkits.add(host, key, name)
    assert devkits.load() == []


def test_a_saved_name_resolves_to_its_address_and_key_in_every_command(tmp_path):
    key = tmp_path / "id_lab"
    key.write_text("")
    devkits.add("sima@10.0.0.2", str(key), "lab")
    board = load(overrides={"board.host": "lab"}, use_env=False).board
    assert board.host == "sima@10.0.0.2"
    assert board.ssh_options[-4:] == ("-i", str(key), "-o", "IdentitiesOnly=yes")
    assert load(overrides={"board.host": "sima@10.9.9.9"}, use_env=False).board.host == "sima@10.9.9.9"


def test_run_connects_to_a_saved_devkit_by_name(monkeypatch):
    from types import SimpleNamespace

    from polima.cli import run as run_cli

    devkits.add("sima@10.0.0.2", name="lab")
    seen = []

    class Client:
        def __init__(self, spec, address, timeout):
            seen.append(address)

        def ping(self):
            return False

    stub = SimpleNamespace(policy="groot", bundle_id="b", fixtures_dir=Path("."))
    monkeypatch.setattr(run_cli, "_resolve_bundle", lambda *a: stub)
    monkeypatch.setattr(run_cli, "get_policy", lambda name: None)
    monkeypatch.setattr(run_cli, "spec_from_bundle", lambda *a: SimpleNamespace(
        wire=SimpleNamespace(default_port=8093)))
    monkeypatch.setattr(run_cli, "_resolve_task", lambda *a, **k: 0)
    monkeypatch.setattr(run_cli, "_load_tensors", lambda *a: {})
    monkeypatch.setattr(run_cli, "PolimaSOMClient", Client)
    assert run_cli.run(["--bundle", "b", "--board", "lab"]) == 2
    assert seen == ["10.0.0.2:8093"]


def test_probe_reports_what_ssh_finds_and_why_it_failed(monkeypatch):
    calls = []

    def fake(argv, **_):
        calls.append(argv)
        if argv[-2] == "sima@down":
            return subprocess.CompletedProcess(argv, 255, "", "ssh: connect to host down port 22: No route to host\n")
        return subprocess.CompletedProcess(argv, 0, "ARCH=aarch64\nFREE=330\nPOLIMA=yes\n", "")

    monkeypatch.setattr(devkits.subprocess, "run", fake)
    assert devkits.probe({"name": "up", "host": "sima@up"}, ("-o", "BatchMode=yes")) == {
        "reachable": True, "modalix": True, "free_gb": 330, "polima": True, "server": False}
    assert devkits.probe({"name": "down", "host": "sima@down"}, ()) == {
        "reachable": False, "error": "ssh: connect to host down port 22: No route to host"}
    assert calls[0][:3] == ["ssh", "-o", "BatchMode=yes"] and calls[0][-1] == devkits.PROBE


def test_simultaneous_additions_all_land(tmp_path):
    names = [f"kit{i}" for i in range(12)]
    threads = [threading.Thread(target=devkits.add, args=(f"sima@10.0.0.{i}", "", name))
               for i, name in enumerate(names)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(d["name"] for d in devkits.load()) == sorted(names)


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    monkeypatch.setattr(devkits, "probe", lambda entry, base: {"reachable": True, "modalix": True,
                                                               "free_gb": 1, "polima": True, "server": False})
    server = serve(Workbench([tmp_path], [tmp_path], []), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def post(base, path, body, headers):
    request = urllib.request.Request(base + path, json.dumps(body).encode(), headers)
    return json.load(urllib.request.urlopen(request, timeout=10))


def session(base):
    reply = urllib.request.urlopen(base + "/api/v1/session", timeout=10)
    token = json.load(reply)["csrf_token"]
    assert f"polima_csrf={token}" in reply.headers["Set-Cookie"] and "SameSite=Strict" in reply.headers["Set-Cookie"]
    return {"Content-Type": "application/json", "X-CSRF-Token": token, "Cookie": f"polima_csrf={token}"}


def test_the_page_adds_lists_and_removes_devkits(workbench):
    headers = session(workbench)
    assert post(workbench, "/api/v1/devkits", {"host": "sima@127.0.0.1", "name": "here"}, headers)["ssh"]["reachable"]
    listed = json.load(urllib.request.urlopen(workbench + "/api/v1/snapshot", timeout=10))["boards"]
    with pytest.raises(urllib.error.HTTPError) as refused:
        post(workbench, "/api/v1/devkits", {"host": "not an address"}, headers)
    post(workbench, "/api/v1/devkits/remove", {"name": "here"}, headers)
    assert [(b["name"], b["host"], b["ssh"]["polima"]) for b in listed] == [("here", "sima@127.0.0.1", True)]
    assert refused.value.code == 400 and devkits.load() == []


@pytest.mark.parametrize("headers", [
    {"Content-Type": "text/plain", "Origin": "http://elsewhere.example"},  # a cross-site form or fetch
    {"Content-Type": "application/json", "X-CSRF-Token": "guess", "Cookie": "polima_csrf=guess"},
])
def test_writes_without_this_sessions_token_are_refused(workbench, headers):
    with pytest.raises(urllib.error.HTTPError) as refused:
        post(workbench, "/api/v1/devkits", {"host": "sima@10.0.0.2", "name": "evil"}, headers)
    assert refused.value.code == 403 and devkits.load() == []


def test_polima_loads_where_fcntl_does_not_exist(monkeypatch):
    """Native Windows has no fcntl; the config (and so every command) must still import."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "fcntl", None)
    importlib.reload(devkits)
    assert devkits.load() == []
