"""Checkpoints and datasets fetched from the Hugging Face Hub by the Workbench host.

The download runs on the host, with the token saved there by `hf auth login`;
the browser only names the repository. Results land beside browser uploads, so
listing and prerequisite checks treat them the same way. Compile stays offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,96}")
REVISION = re.compile(r"[\w.\-/]{1,128}")
KINDS = {"checkpoint": "model", "dataset": "dataset"}
MARGIN = 1 << 30  # keep 1 GiB free beyond the repository's size


def repository_id(kind: str, value: str) -> str:
    value = value.strip()
    if "://" not in value:
        return value
    url = urlsplit(value)
    if url.scheme != "https" or url.netloc != "huggingface.co":
        raise ValueError("Use a https://huggingface.co repository URL or owner/name.")
    parts = url.path.strip("/").split("/")
    dataset = parts[0] == "datasets"
    if dataset != (kind == "dataset"):
        raise ValueError("Choose Dataset for dataset URLs or Checkpoint for model URLs.")
    if dataset:
        parts = parts[1:]
    if len(parts) != 2:
        raise ValueError("Paste the repository's main URL; enter its revision separately.")
    return "/".join(parts)


def validate(kind: str, repo_id: str, revision: str) -> None:
    if kind not in KINDS:
        raise ValueError("Choose a checkpoint or dataset repository.")
    if not REPO_ID.fullmatch(repo_id) or ".." in repo_id:
        raise ValueError("Enter a Hugging Face repository URL or owner/name.")
    if revision and (not REVISION.fullmatch(revision) or ".." in revision):
        raise ValueError("The revision may contain letters, digits, '.', '-', '_' and '/'.")


def locate(kind: str, tree: Path) -> list[Path]:
    """Checkpoint or dataset folders inside a fetched or uploaded tree."""
    if kind == "checkpoint":
        from polima.workbench.app import checkpoint
        return [p.parent for p in tree.rglob("config.json")
                if ".cache" not in p.parts and checkpoint(p.parent)]
    return [p.parent.parent for p in tree.rglob("meta/info.json")
            if ".cache" not in p.parts and (p.parent.parent / "data").is_dir()]


def dataset_format(root: Path) -> dict[str, Any]:
    """The LeRobot format a dataset declares, and which policies can calibrate on it."""
    try:
        info = json.loads((root / "meta/info.json").read_text())
    except (OSError, ValueError):
        info = {}
    version = str(info.get("codebase_version", "")) if isinstance(info, dict) else ""
    groot = version in ("v2.0", "v2.1") and (root / "meta/modality.json").is_file()
    return {"version": version or "unknown", "groot": groot,
            "note": "" if groot else "GR00T needs LeRobot v2.0/v2.1 with meta/modality.json."}


def hub_dataset(checkpoint: str | Path) -> str:
    """The Hub dataset a LeRobot checkpoint was trained on, if its train_config.json names one."""
    try:
        config = json.loads((Path(checkpoint) / "train_config.json").read_text())
        repo_id = str((config.get("dataset") or {}).get("repo_id") or "")
    except (OSError, ValueError, AttributeError):
        return ""
    return repo_id if REPO_ID.fullmatch(repo_id) and not repo_id.startswith("local/") else ""


def explain(error: Exception) -> str:
    """A message for the page; never the token or a traceback."""
    name = type(error).__name__
    if name == "GatedRepoError":
        return ("This repository is gated. Accept its terms on huggingface.co with the account "
                "logged in on the Workbench host (hf auth login).")
    if name == "RepositoryNotFoundError":
        return ("Repository not found, or it is private and the Workbench host has no access. "
                "Check the id, then run `hf auth login` on the host.")
    if name == "RevisionNotFoundError":
        return "That revision does not exist in the repository."
    if name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled") or os.environ.get("HF_HUB_OFFLINE") == "1":
        return "The Workbench host cannot reach the Hub (offline, or HF_HUB_OFFLINE=1)."
    return f"Download failed: {error}"


def tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            total += path.stat().st_size if path.is_file() else 0
        except OSError:
            continue
    return total


class HubDownloads:
    """One Hub download at a time, resumable: the same repository reuses its folder."""

    def __init__(self, root: Path, api=None, download=None):
        self.root = root  # the workspace's imports/ folder
        self._api, self._download = api, download
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.current: dict[str, Any] | None = None

    def _hub(self):
        if self._api is None or self._download is None:
            try:
                from huggingface_hub import HfApi, snapshot_download
            except ImportError as error:
                raise ValueError("huggingface_hub is not installed in the Workbench environment. "
                                 "Run `make venv` on the host.") from error
            self._api = self._api or HfApi()
            self._download = self._download or snapshot_download
        return self._api, self._download

    def folder(self, kind: str, repo_id: str, revision: str) -> Path:
        key = hashlib.sha256(f"{kind}\0{repo_id}\0{revision}".encode()).hexdigest()[:16]
        return self.root / f"hub-{key}"

    def start(self, kind: str, repo_id: str, revision: str = "") -> dict[str, Any]:
        if not self.start_lock.acquire(blocking=False):
            raise ValueError("A Hub download is already running. Wait for it to finish.")
        try:
            return self._start(kind, repo_id, revision)
        finally:
            self.start_lock.release()

    def _start(self, kind: str, repo_id: str, revision: str) -> dict[str, Any]:
        repo_id, revision = repository_id(kind, repo_id), revision.strip()
        validate(kind, repo_id, revision)
        with self.lock:
            if self.current and self.current["status"] == "running":
                raise ValueError("A Hub download is already running. Wait for it to finish.")
        api, download = self._hub()
        try:  # auth_check refuses gated and private repositories the host's token cannot read
            api.auth_check(repo_id, repo_type=KINDS[kind])
            info = api.repo_info(repo_id, repo_type=KINDS[kind], revision=revision or None,
                                 files_metadata=True)
        except Exception as error:  # the Hub's errors are mapped to a readable message
            raise ValueError(explain(error)) from error
        total = sum(getattr(f, "size", 0) or 0 for f in getattr(info, "siblings", None) or [])
        base = self.folder(kind, repo_id, revision)
        commit = str(info.sha)
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("The Hub did not return a valid commit for this revision.")
        tree = base / "snapshots" / commit / repo_id.split("/")[1]
        tree.mkdir(parents=True, exist_ok=True)
        have = tree_bytes(tree)
        free = shutil.disk_usage(base).free
        if total - have + MARGIN > free:
            raise ValueError(f"Not enough space on the Workbench host: the repository needs "
                             f"{(total - have) / 1e9:.1f} GB more and {free / 1e9:.1f} GB is free.")
        (base / "kind").write_text(kind)
        state = {"kind": kind, "repo_id": repo_id, "revision": revision, "status": "running",
                 "total": total, "started_at": time.time(), "path": str(tree), "tree": str(tree),
                 "commit": commit}
        with self.lock:
            self.current = state
        try:
            threading.Thread(target=self._run, args=(download, base, state), daemon=True).start()
        except Exception:
            with self.lock:
                state.update(status="failed", error="Could not start the download worker.")
            raise
        return self.status()

    def _run(self, download, base: Path, state: dict[str, Any]) -> None:
        kind, tree = state["kind"], Path(state["tree"])
        try:
            published = base / "input.json"
            try:
                previous = json.loads(published.read_text())
            except (OSError, ValueError):
                previous = {}
            if isinstance(previous, dict) and previous.get("commit") == state["commit"] and Path(previous.get("path", "")).is_dir():
                with self.lock:
                    state.update(status="completed", item=previous, ended_at=time.time())
                return
            download(state["repo_id"], repo_type=KINDS[kind], revision=state["commit"],
                     local_dir=str(tree))
            found = locate(kind, tree)
            if len(found) != 1:
                raise ValueError("The repository does not hold one PoLiMa checkpoint "
                                 "(config.json with safetensors)." if kind == "checkpoint" else
                                 "The repository is not one LeRobot dataset with meta/info.json and data/.")
            label = state["repo_id"] + (f"@{state['revision']}" if state["revision"] else "")
            item = {"kind": kind, "path": str(found[0]), "name": label, "source": "hub",
                    "repo_id": state["repo_id"], "revision": state["revision"], "commit": state["commit"]}
            if kind == "dataset":
                item["format"] = dataset_format(found[0])
            pending = base / "input.json.pending"
            pending.write_text(json.dumps(item))
            pending.replace(published)
            update = {"status": "completed", "item": item}
        except Exception as error:  # noqa: BLE001 - reported on the page; the folder is kept for resuming
            update = {"status": "failed", "error": error.args[0] if isinstance(error, ValueError)
                      else explain(error)}
        with self.lock:
            state.update(update, ended_at=time.time())

    def status(self) -> dict[str, Any] | None:
        with self.lock:
            state = dict(self.current) if self.current else None
        if state:
            state["received"] = min(tree_bytes(Path(state["path"])), state["total"]) \
                if state["status"] != "completed" else state["total"]
        return state
