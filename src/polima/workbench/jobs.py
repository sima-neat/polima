"""One browser-requested CLI operation at a time, with retained output."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class WorkflowJobs:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current: dict[str, Any] | None = None

    def start(self, kind: str, argv: list[str], output: Path) -> dict[str, Any]:
        with self.lock:
            if self.current and self.current['status'] == 'running':
                raise ValueError('A workflow is already running. Wait for it to finish.')
            output.mkdir(parents=True, exist_ok=False)
            log = output / 'workflow.log'
            env = os.environ.copy()
            source = str(Path(__file__).resolve().parents[2])
            env['PYTHONPATH'] = os.pathsep.join([source, env.get('PYTHONPATH', '')])
            env['PYTHONUNBUFFERED'] = '1'
            self.current = {'kind': kind, 'status': 'running', 'started_at': time.time(),
                            'command': argv, 'output': str(output), 'log_path': str(log)}
            try:
                with log.open('w') as stream:
                    process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                               env=env, stdin=subprocess.DEVNULL)
            except OSError as error:
                self.current.update(status='failed', error=str(error), ended_at=time.time())
                raise ValueError(f'Could not start {kind}: {error}') from error
            threading.Thread(target=self._finish, args=(process,), daemon=True).start()
            return dict(self.current)

    def _finish(self, process: subprocess.Popen) -> None:
        code = process.wait()
        with self.lock:
            self.current.update(status='completed' if code == 0 else 'failed',
                                returncode=code, ended_at=time.time())

    def snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            result = dict(self.current) if self.current else None
        if result:
            try:
                with Path(result['log_path']).open('rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 65536))
                    result['log'] = stream.read().decode('utf-8', errors='replace')
            except OSError:
                result['log'] = ''
        return result


def compile_command(checkpoint: dict, dataset: str, output: Path) -> list[str]:
    policy = checkpoint['policy']
    argv = [sys.executable, '-m', 'polima', 'compile', '--policy']
    if policy == 'ACT':
        argv += ['act', '--checkpoint', checkpoint['path'], '--build-dir', str(output / 'build'),
                 '--output-root', str(output / 'bundles'), '--jobs', '3']
        if dataset:
            argv += ['--dataset-root', dataset]
    elif policy in ('GR00T N1.6', 'GR00T N1.7'):
        if not dataset or not Path(dataset).is_dir():
            raise ValueError('Select a calibration dataset for GR00T.')
        argv += ['groot', '--checkpoint', checkpoint['path'], '--dataset', dataset,
                 '--output', str(output / 'build'), '--jobs', '3']
    else:
        raise ValueError(f'Compilation is unavailable for {policy}.')
    return argv
