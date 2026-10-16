"""Browser imports and compiler prerequisites for a user's Workbench workspace."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from polima.studio.transfer import Uploads
from polima.workbench.hub import HubDownloads, dataset_format, locate


def workspace() -> Path:
    return Path(os.environ.get('POLIMA_WORKBENCH_HOME', Path.home() / '.local/share/polima/workbench')).expanduser()


class Inputs:
    def __init__(self, root: Path | None = None):
        self.root = root or workspace()
        self.uploads = Uploads(self.root / 'imports')
        self.hub = HubDownloads(self.root / 'imports')

    def create(self, kind: str, files: list) -> dict:
        if not isinstance(files, list) or not files or not all(isinstance(f, dict) for f in files):
            raise ValueError('Select a folder containing checkpoint or dataset files.')
        if any(not isinstance(f.get('path'), str) or type(f.get('size')) is not int or f['size'] < 0 for f in files):
            raise ValueError('Each file must have a relative path and a nonnegative size.')
        if kind not in ('checkpoint', 'dataset'):
            raise ValueError('Select a checkpoint or dataset folder.')
        if any(str(f.get('path', '')).lower().endswith(('.tar', '.tgz', '.tar.gz', '.zip')) for f in files):
            raise ValueError('Upload an extracted folder, not an archive.')
        self.root.mkdir(parents=True, exist_ok=True)
        result = self.uploads.create(files, shutil.disk_usage(self.root).free)
        (self.uploads.path(result['id']) / 'kind').write_text(kind)
        return result

    def finish(self, upload_id: str) -> dict:
        state = self.uploads.status(upload_id)
        if not state['complete']:
            raise ValueError('Upload is incomplete. Retry the upload to resume.')
        base = self.uploads.path(upload_id)
        tree = base / 'files'
        kind = (base / 'kind').read_text()
        found = locate(kind, tree)
        if len(found) != 1:
            raise ValueError('Select one complete checkpoint folder.' if kind == 'checkpoint'
                             else 'Select one LeRobot dataset with meta/info.json and data/.')
        item = {'kind': kind, 'path': str(found[0]), 'name': found[0].name}
        (base / 'input.json').write_text(json.dumps(item))
        return item

    def items(self) -> list:
        result = []
        for path in (self.root / 'imports').glob('*/input.json'):
            try:
                item = json.loads(path.read_text())
                if isinstance(item, dict) and item.get('kind') in ('checkpoint', 'dataset') and isinstance(item.get('path'), str):
                    if item['kind'] == 'dataset' and 'format' not in item:
                        item['format'] = dataset_format(Path(item['path']))
                    result.append(item)
            except (OSError, ValueError):
                continue
        return result


def prerequisites(policy: str, dataset: str = '') -> list[dict]:
    from polima.compile.toolchain import find_compiler_bin
    from polima.config.loader import load
    config = load()
    rows = [{'label': 'Linux x86_64 build host', 'ok': platform.system() == 'Linux' and platform.machine() in ('x86_64', 'AMD64'),
             'help': 'Open Workbench on a Linux x86_64 host to compile. macOS can import and manage files.'},
            {'label': 'SiMa ModelSDK', 'ok': bool(find_compiler_bin(config)),
             'help': 'Install the licensed ModelSDK on the build host and set MODEL_COMPILER_BIN.'}]
    if policy == 'ACT':
        rows.append({'label': 'PyTorch export dependency',
                     'ok': importlib.util.find_spec('torch') is not None,
                     'help': 'Install the ACT export dependencies in the Python environment running Workbench. See Compiler setup.'})
        root = Path(dataset).expanduser() if dataset else None
        try:
            info = json.loads((root / 'meta/info.json').read_text()) if root else None
        except (OSError, ValueError):
            info = None
        rows.append({'label': 'Representative dataset',
                     'ok': bool(root and isinstance(info, dict) and (root / 'data').is_dir()),
                     'help': 'Select an uploaded LeRobot dataset with meta/info.json, data/ and referenced videos. ACT export uses sample observations for calibration and validation; no retraining.'})
    if policy == 'Pi0.5':
        python = getattr(config.paths, 'pi05_python', None)
        path = Path(python).expanduser() if python else None
        ready = bool(path and path.is_file() and os.access(path, os.X_OK))
        rows.append({'label': 'Pi0.5 export environment', 'ok': ready,
                     'help': 'Save paths.pi05_python (an interpreter with torch, transformers 5 and LeRobot[pi]) '
                             'in your user configuration. See Compiler setup.'})
        if ready:
            try:
                probe = subprocess.run([str(path), '-c', 'from lerobot.policies.pi05.modeling_pi05 import PI05Policy'],
                                       capture_output=True, text=True, timeout=60, check=False)
                ready, detail = probe.returncode == 0, (probe.stderr.strip().splitlines() or [''])[-1][:240]
            except (OSError, subprocess.TimeoutExpired):
                ready, detail = False, 'The export environment could not complete its dependency check.'
            rows.append({'label': 'Pi0.5 export dependencies', 'ok': ready,
                         'help': 'Install LeRobot with the [pi] and [dataset] extras and transformers 5. ' + detail})
        root = Path(dataset).expanduser() if dataset else None
        rows.append({'label': 'Calibration dataset', 'ok': bool(root and (root / 'meta/info.json').is_file()
                                                              and (root / 'data').is_dir()),
                     'help': 'Select a LeRobot dataset recorded with the checkpoint\'s cameras: export calibrates '
                             'INT8 weights on its observations and builds the fixture from one of them.'})
    if policy.startswith('GR00T'):
        tag = '16' if policy == 'GR00T N1.6' else '17'
        paths = {}
        for suffix, title in [('python', 'Python environment'), ('source', 'Isaac-GR00T source')]:
            value = getattr(config.paths, f'groot{tag}_{suffix}', None)
            path = Path(value).expanduser() if value else None
            ready = bool(path and (path.is_file() and os.access(path, os.X_OK) if suffix == 'python'
                                   else (path / 'gr00t/__init__.py').is_file()))
            paths[suffix] = path if ready else None
            rows.append({'label': f'{policy} {title}', 'ok': ready,
                         'help': f'Save paths.groot{tag}_{suffix} in your user configuration. See Compiler setup.'})
        if all(paths.values()):
            env = {**os.environ, 'PYTHONPATH': str(paths['source']),
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            try:
                probe = subprocess.run([str(paths['python']), '-c',
                    'import torch, transformers, onnx, onnxruntime, gr00t'],
                    env=env, capture_output=True, text=True, timeout=30, check=False)
                ready = probe.returncode == 0
                detail = (probe.stderr.strip().splitlines() or ['Export dependency imports failed.'])[-1][:240]
            except (OSError, subprocess.TimeoutExpired):
                ready, detail = False, 'The export environment could not complete its dependency check.'
            rows.append({'label': f'{policy} export dependencies', 'ok': ready,
                         'help': 'Install the matching GR00T export dependencies. ' + (detail if not ready else '')})
        if tag == '17':
            value = config.paths.groot17_backbone
            path = Path(value).expanduser() if value else None
            ready = bool(path and (path / 'config.json').is_file() and
                         (any(path.glob('*.safetensors')) or any(path.glob('*.bin'))))
            rows.append({'label': 'Local N1.7 backbone assets', 'ok': ready,
                         'help': 'Download the checkpoint’s backbone and save its local directory as paths.groot17_backbone. See Compiler setup.'})
        root = Path(dataset).expanduser() if dataset else None
        try:
            info = json.loads((root / 'meta/info.json').read_text()) if root else {}
        except (OSError, ValueError):
            info = {}
        ready = bool(root and isinstance(info, dict) and info.get('codebase_version') in ('v2.0', 'v2.1') and
                     (root / 'meta/modality.json').is_file() and (root / 'data').is_dir())
        rows.append({'label': 'Compatible LeRobot dataset', 'ok': ready,
                     'help': 'Upload the training dataset in v2.0/v2.1 format, including meta/, data/ and referenced videos.'})
    return rows
