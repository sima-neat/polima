"""Hub imports with a fake Hub: no network, no token."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from polima.workbench import app, hub, inputs

CHECKPOINT = {"pretrained_model/config.json": b'{"type":"act"}',
              "pretrained_model/model.safetensors": b"weights",
              "pretrained_model/train_config.json": b'{"dataset":{"repo_id":"lab/so101-pick"}}'}
DATASET = {"meta/info.json": b'{"codebase_version":"v3.0"}', "data/chunk-000/file-000.parquet": b"rows"}


class GatedRepoError(Exception):
    pass


class FakeHub:
    def __init__(self, files, error=None):
        self.files, self.error, self.calls = files, error, []
        self.sha = "a" * 40

    def auth_check(self, repo_id, repo_type):
        if self.error:
            raise self.error

    def repo_info(self, repo_id, repo_type, revision, files_metadata):
        return SimpleNamespace(sha=self.sha, siblings=[SimpleNamespace(size=len(b)) for b in self.files.values()])

    def download(self, repo_id, repo_type, revision, local_dir):
        self.calls.append((repo_id, repo_type, revision))
        for name, content in self.files.items():
            path = Path(local_dir, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return local_dir


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("POLIMA_WORKBENCH_HOME", str(tmp_path / "workbench"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return inputs.Inputs()


def use(storage, fake):
    storage.hub._api, storage.hub._download = fake, fake.download
    return fake


def wait(storage):
    deadline = time.time() + 5
    while storage.hub.status()["status"] == "running" and time.time() < deadline:
        time.sleep(0.01)
    return storage.hub.status()


def test_checkpoint_from_the_hub_is_listed_like_an_upload(storage):
    fake = use(storage, FakeHub(CHECKPOINT))
    storage.hub.start("checkpoint", "lab/act-so101", "v1")
    state = wait(storage)
    assert state["status"] == "completed" and state["received"] == state["total"]
    assert fake.calls == [("lab/act-so101", "model", fake.sha)]
    found = app.Workbench([], [], []).scan()[0]
    assert [c["path"] for c in found] == [state["item"]["path"]]
    assert found[0]["hub_dataset"] == "lab/so101-pick"
    assert state["item"]["name"] == "lab/act-so101@v1" and found[0]["run"] == "act-so101"


def test_dataset_reports_its_format(storage):
    use(storage, FakeHub(DATASET))
    storage.hub.start("dataset", "lab/so101-pick")
    state = wait(storage)
    assert state["item"]["format"] == {"version": "v3.0", "groot": False,
                                       "note": "GR00T needs LeRobot v2.0/v2.1 with meta/modality.json."}
    listed = [i for i in storage.items() if i["kind"] == "dataset"]
    assert listed[0]["repo_id"] == "lab/so101-pick" and listed[0]["format"]["version"] == "v3.0"


def test_same_repository_resumes_in_the_same_folder(storage):
    use(storage, FakeHub(DATASET))
    first = storage.hub.folder("dataset", "lab/so101-pick", "")
    assert first == storage.hub.folder("dataset", "lab/so101-pick", "")
    assert first != storage.hub.folder("dataset", "lab/so101-pick", "v2")
    assert first != storage.hub.folder("checkpoint", "lab/so101-pick", "")


def test_wrong_repository_content_is_reported(storage):
    use(storage, FakeHub(DATASET))
    storage.hub.start("checkpoint", "lab/not-a-model")
    state = wait(storage)
    assert state["status"] == "failed" and "one PoLiMa checkpoint" in state["error"]
    assert app.Workbench([], [], []).scan()[0] == []


def test_gated_repository_explains_access_without_starting(storage):
    use(storage, FakeHub(CHECKPOINT, error=GatedRepoError("403")))
    with pytest.raises(ValueError, match="gated.*hf auth login"):
        storage.hub.start("checkpoint", "lab/private")
    assert storage.hub.status() is None


def test_not_enough_space_refuses_before_downloading(storage, monkeypatch):
    fake = use(storage, FakeHub(DATASET))
    monkeypatch.setattr(hub.shutil, "disk_usage", lambda _p: SimpleNamespace(free=hub.MARGIN))
    with pytest.raises(ValueError, match="Not enough space"):
        storage.hub.start("dataset", "lab/so101-pick")
    assert fake.calls == []


@pytest.mark.parametrize("kind, repo, revision", [
    ("bundle", "lab/x", ""), ("dataset", "no-owner", ""), ("dataset", "../etc/passwd", ""),
    ("dataset", "lab/x", "../../main"), ("dataset", "lab/x y", ""),
])
def test_invalid_requests_are_refused(storage, kind, repo, revision):
    use(storage, FakeHub(DATASET))
    with pytest.raises(ValueError):
        storage.hub.start(kind, repo, revision)


def test_one_download_at_a_time(storage):
    fake = use(storage, FakeHub(DATASET))
    storage.hub.current = {"status": "running", "path": "/nonexistent", "total": 1}
    with pytest.raises(ValueError, match="already running"):
        storage.hub.start("dataset", "lab/so101-pick")
    assert fake.calls == []


@pytest.mark.parametrize("config, expected", [
    ({"dataset": {"repo_id": "lab/so101-pick"}}, "lab/so101-pick"),
    ({"dataset": {"repo_id": "local/so101-pick"}}, ""),
    ({"dataset": {"repo_id": "/data/so101"}}, ""),
    ({}, ""),
])
def test_training_dataset_offered_only_when_it_is_on_the_hub(tmp_path, config, expected):
    (tmp_path / "train_config.json").write_text(json.dumps(config))
    assert hub.hub_dataset(tmp_path) == expected


def test_hub_routes_require_the_session(storage):
    workbench = app.Workbench([], [], [])
    use(workbench.inputs, FakeHub(DATASET))
    server = app.serve(workbench, port=0)
    import threading
    import urllib.error
    import urllib.request
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        request = urllib.request.Request(base + "/api/v1/hub/download", method="POST",
                                         data=b'{"kind":"dataset","repo_id":"lab/so101-pick"}')
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request)
        assert refused.value.code == 403
        with urllib.request.urlopen(base + "/api/v1/session") as response:
            token = json.load(response)["csrf_token"]
        request.add_header("X-CSRF-Token", token)
        request.add_header("Cookie", f"polima_workbench_csrf={token}")
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["repo_id"] == "lab/so101-pick"
        wait(workbench.inputs)
        assert workbench.snapshot()["hub"]["status"] == "completed"
    finally:
        server.shutdown()


def test_parallel_requests_reserve_slot_before_network(storage):
    fake = use(storage, FakeHub(DATASET))
    entered, release = threading.Event(), threading.Event()
    def auth(*args, **kwargs):
        entered.set()
        assert release.wait(5)
    fake.auth_check = auth
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(storage.hub.start, "dataset", "lab/first")
        assert entered.wait(5)
        try:
            second = pool.submit(storage.hub.start, "dataset", "lab/second")
            with pytest.raises(ValueError, match="already running"):
                second.result(timeout=1)
        finally:
            release.set()
            first.result(timeout=5)
    assert wait(storage)["status"] == "completed"


def test_failed_refresh_preserves_published_checkpoint_and_resumes(storage):
    fake = use(storage, FakeHub(CHECKPOINT))
    storage.hub.start("checkpoint", "lab/model")
    original = wait(storage)["item"]
    old = Path(original["path"])
    fake.sha = "b" * 40
    attempted = []
    def interrupted(repo_id, **kwargs):
        attempted.append(kwargs)
        folder = Path(kwargs["local_dir"]) / "pretrained_model"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text('{"type":"Gr00tN1d7"}')
        raise OSError("connection lost")
    storage.hub._download = interrupted
    storage.hub.start("checkpoint", "lab/model")
    assert wait(storage)["status"] == "failed"
    assert storage.items() == [original]
    assert json.loads((old / "config.json").read_text()) == {"type": "act"}
    assert attempted[0]["revision"] == fake.sha
    fake.files = {**CHECKPOINT, "pretrained_model/config.json": b'{"type":"Gr00tN1d7"}'}
    storage.hub._download = fake.download
    storage.hub.start("checkpoint", "lab/model")
    completed = wait(storage)
    assert completed["status"] == "completed"
    assert completed["tree"] == attempted[0]["local_dir"]
    assert completed["item"]["path"] != original["path"]
    assert storage.items() == [completed["item"]]
    assert json.loads((old / "config.json").read_text()) == {"type": "act"}


def test_preflight_failure_releases_slot_and_completed_revision_is_reused(storage):
    fake = use(storage, FakeHub(DATASET, error=GatedRepoError("403")))
    with pytest.raises(ValueError):
        storage.hub.start("dataset", "lab/data")
    fake.error = None
    storage.hub.start("dataset", "lab/data")
    first = wait(storage)["item"]
    storage.hub.start("dataset", "lab/data")
    assert wait(storage)["item"] == first
    assert len(fake.calls) == 1


def test_failed_publication_preserves_previous_import(storage, monkeypatch):
    fake = use(storage, FakeHub(DATASET))
    storage.hub.start("dataset", "lab/data")
    old = wait(storage)["item"]
    fake.sha = "c" * 40
    original = Path.replace
    def fail(source, target):
        if source.name == "input.json.pending":
            raise OSError("disk full")
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail)
        storage.hub.start("dataset", "lab/data")
        assert wait(storage)["status"] == "failed"
        assert storage.items() == [old]
    storage.hub.start("dataset", "lab/data")
    assert wait(storage)["status"] == "completed"
    assert storage.items()[0]["commit"] == fake.sha


@pytest.mark.parametrize('kind, url, expected', [
    ('checkpoint', 'https://huggingface.co/nvidia/GR00T-H-N1.7', 'nvidia/GR00T-H-N1.7'),
    ('checkpoint', 'https://huggingface.co/lab/model/?a=b#files', 'lab/model'),
    ('dataset', 'https://huggingface.co/datasets/lab/data', 'lab/data'),
])
def test_repository_urls_are_normalized(storage, kind, url, expected):
    fake = use(storage, FakeHub(CHECKPOINT if kind == 'checkpoint' else DATASET))
    storage.hub.start(kind, url)
    assert wait(storage)['status'] == 'completed'
    assert fake.calls[0][0] == expected


@pytest.mark.parametrize('kind, url', [
    ('checkpoint', 'https://huggingface.co.evil.test/lab/model'),
    ('checkpoint', 'https://huggingface.co/datasets/lab/data'),
    ('dataset', 'https://huggingface.co/lab/model'),
    ('checkpoint', 'https://huggingface.co/lab/model/tree/main'),
    ('checkpoint', 'https://user:password@huggingface.co/lab/model'),
])
def test_non_repository_or_wrong_type_urls_are_refused(storage, kind, url):
    fake = use(storage, FakeHub(CHECKPOINT))
    with pytest.raises(ValueError):
        storage.hub.start(kind, url)
    assert fake.calls == []
