"""Compiler evidence and browser workflow contracts, without a DevKit."""
import json
import sys

import pytest

from polima.workbench.app import Workbench, bundle


def saved_bundle(tmp_path):
    root = tmp_path / 'build'
    root.mkdir()
    (root / 'bundle.json').write_text(json.dumps({'bundle_id': 'act-demo', 'policy': 'act',
        'graphs': [{'name': 'vision', 'precision': 'bf16', 'elf_bytes': 123,
                    'elf': 'models/vision/model.elf', 'input_elements': 64}]}))
    (root / 'plan.json').write_text(json.dumps({'steps': [
        {'op': 'run_elf', 'out': 'camera0', 'args': {'graph': 'vision'}},
        {'op': 'run_elf', 'out': 'camera1', 'args': {'graph': 'vision'}},
        {'op': 'gather_strided', 'out': 'actions', 'args': {'src': 'camera1'}}]}))
    return root


def test_graph_inspection_preserves_manifest_and_runtime_steps(tmp_path):
    item = bundle(saved_bundle(tmp_path))
    assert 'inspection' in item
    graph = item['inspection']['graphs'][0]
    assert graph['name'] == 'vision' and graph['precision'] == 'bf16'
    assert graph['elf_bytes'] == 123
    assert [step['out'] for step in graph['runtime_steps']] == ['camera0', 'camera1']
    assert graph['placement'] == 'MLA (runtime plan)'
    assert graph['memory_bytes'] is None  # ELF size is not resident memory
    assert graph['operators'] is None  # runtime operations are not ONNX operators


def test_missing_graph_details_are_not_invented(tmp_path):
    root = saved_bundle(tmp_path)
    (root / 'bundle.json').write_text(json.dumps({'graphs': [{'name': 'unknown'}]}))
    item = bundle(root)
    assert 'inspection' in item
    graph = item['inspection']['graphs'][0]
    assert graph['precision'] is None and graph['placement'] is None


def test_compile_request_uses_only_a_discovered_checkpoint(tmp_path, monkeypatch):
    workbench = Workbench([], [], [])
    assert hasattr(workbench, 'start_job')
    with pytest.raises(ValueError, match='checkpoint'):
        workbench.start_job({'kind': 'compile', 'checkpoint': '/not/discovered'})
    with pytest.raises(ValueError, match='operation'):
        workbench.start_job({'kind': 'shell', 'command': 'echo unwanted'})


def test_compile_arguments_use_current_interpreter_and_groot_contract(tmp_path, monkeypatch):
    monkeypatch.setattr('polima.workbench.app.prerequisites', lambda *args: [])
    workbench = Workbench([], [], [])
    assert hasattr(workbench, 'start_job')
    checkpoint = tmp_path / 'model with spaces; echo NO'
    checkpoint.mkdir()
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    monkeypatch.setattr(workbench, 'scan', lambda *args, **kwargs: ([{'path': str(checkpoint), 'policy': 'GR00T N1.7'}], []))
    monkeypatch.setenv('POLIMA_OUTPUTS', str(tmp_path / 'outputs'))
    captured = {}
    def start(kind, argv, output):
        captured.update(kind=kind, argv=argv, output=output)
        return {'status': 'running'}
    monkeypatch.setattr(workbench.jobs, 'start', start)
    workbench.start_job({'kind': 'compile', 'checkpoint': str(checkpoint), 'dataset': str(dataset)})
    argv = captured['argv']
    assert argv[:4] == [sys.executable, '-m', 'polima', 'compile']
    assert argv[argv.index('--checkpoint') + 1] == str(checkpoint)
    assert argv[argv.index('--dataset') + 1] == str(dataset)
    assert '--output' in argv and '--build-dir' not in argv
    assert argv[argv.index('--jobs') + 1] == '3'


def test_deploy_reuses_cli_with_saved_devkit_and_explicit_server_choice(tmp_path, monkeypatch):
    from polima.workbench import app
    workbench = Workbench([], [], [])
    assert hasattr(workbench, 'start_job')
    root = saved_bundle(tmp_path)
    monkeypatch.setattr(workbench, 'scan', lambda *args, **kwargs: ([], [bundle(root)]))
    monkeypatch.setattr(app.devkits, 'load', lambda: [{'name': 'test-kit', 'host': 'sima@test-kit'}])
    monkeypatch.setenv('POLIMA_OUTPUTS', str(tmp_path / 'outputs'))
    captured = []
    monkeypatch.setattr(workbench.jobs, 'start', lambda kind, argv, output: captured.append(argv))
    workbench.start_job({'kind': 'deploy', 'bundle': str(root), 'devkit': 'test-kit', 'start_server': False})
    assert '--start' not in captured[-1]
    workbench.start_job({'kind': 'deploy', 'bundle': str(root), 'devkit': 'test-kit', 'start_server': True})
    assert '--start' in captured[-1] and '--devkit' in captured[-1]
    with pytest.raises(ValueError, match='DevKit'):
        workbench.start_job({'kind': 'deploy', 'bundle': str(root), 'devkit': 'not-saved'})


def test_workflow_records_exit_and_rejects_overlap(tmp_path, monkeypatch):
    import threading

    from polima.workbench import jobs
    from polima.workbench.jobs import WorkflowJobs
    done = threading.Event()
    entered = threading.Event()
    class Process:
        def wait(self):
            entered.set()
            done.wait(2)
            return 7
    def popen(argv, **kwargs):
        assert kwargs.get('shell', False) is False
        kwargs['stdout'].write('compiler diagnostic\n')
        return Process()
    monkeypatch.setattr(jobs.subprocess, 'Popen', popen)
    work = WorkflowJobs()
    assert work.start('compile', ['compiler', 'path with spaces'], tmp_path / 'first')['status'] == 'running'
    assert entered.wait(1)
    with pytest.raises(ValueError, match='already running'):
        work.start('deploy', ['deploy'], tmp_path / 'second')
    done.set()
    import time
    deadline = time.monotonic() + 2
    while work.snapshot()['status'] == 'running' and time.monotonic() < deadline:
        time.sleep(.01)
    result = work.snapshot()
    assert result['status'] == 'failed' and result['returncode'] == 7
    assert 'compiler diagnostic' in result['log']
    assert not (tmp_path / 'second').exists()


def test_completed_workflow_bundles_are_discovered_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv('POLIMA_OUTPUTS', str(tmp_path))
    root = tmp_path / 'workbench/compile-example/build/bundles/act'
    root.mkdir(parents=True)
    (root / 'bundle.json').write_text(json.dumps({'bundle_id': 'act-example', 'graphs': []}))
    staging = tmp_path / 'workbench/compile-example/build/bundles-work/duplicate'
    staging.mkdir(parents=True)
    (staging / 'bundle.json').write_text(json.dumps({'bundle_id': 'act-example', 'graphs': []}))
    assert [b['id'] for b in Workbench([], [], []).scan()[1]] == ['act-example']


def test_workflow_endpoint_requires_session_and_returns_job(tmp_path, monkeypatch):
    import http.cookiejar
    import threading
    import urllib.error
    import urllib.request

    from polima.workbench.app import serve
    workbench = Workbench([], [], [])
    calls = []
    monkeypatch.setattr(workbench, 'start_job', lambda body: calls.append(body) or {'status': 'running'})
    server = serve(workbench, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    body = json.dumps({'kind': 'compile', 'checkpoint': 'selected'}).encode()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(urllib.request.Request(base + '/api/v1/workflow', data=body), timeout=2)
        assert error.value.code == 403 and calls == []
        session = json.load(opener.open(base + '/api/v1/session', timeout=2))
        request = urllib.request.Request(base + '/api/v1/workflow', data=body,
                                         headers={'X-CSRF-Token': session['csrf_token']})
        assert json.load(opener.open(request, timeout=2))['status'] == 'running'
        assert calls == [{'kind': 'compile', 'checkpoint': 'selected'}]
        assert json.load(opener.open(base + '/api/v1/workflow', timeout=2)) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_chain_mapping_uses_runtime_opcode_and_preserves_all_graph_calls():
    from polima.workbench.app import graph_inspection
    data = graph_inspection({'graphs': [{'name': 'vision'}]}, {'steps': [
        {'op': 'run_elf_chain', 'out': 'camera0', 'args': {'graphs': ['vision', 'norm']}},
        {'op': 'run_elf_chain', 'out': 'camera1', 'args': {'graphs': ['vision', 'norm']}}]})
    assert data['graphs'][0]['placement'] == 'MLA (runtime plan)'
    assert len(data['graphs'][0]['runtime_steps']) == 2


def test_build_reports_require_matching_graph_hashes(tmp_path):
    from polima.workbench.app import build_reports
    root = tmp_path / 'build'
    folder = root / 'bundles/model'
    folder.mkdir(parents=True)
    (root / 'compiled').mkdir()
    (root / 'export').mkdir()
    (root / 'check').mkdir()
    (root / 'build.json').write_text(json.dumps({'bundle': str(folder), 'checkpoint': '/model', 'dataset': '/dataset'}))
    (root / 'compiled/compile_results.json').write_text(json.dumps({'vision': {'elf_sha256': 'abc', 'minutes': 2}}))
    (root / 'export/onnx_verification_report.json').write_text(json.dumps({'ok': True, 'max_abs': .01}))
    (root / 'check/report.json').write_text(json.dumps({'ok': True, 'all': {'cosine': .999, 'mean_abs': .001}}))
    manifest = {'graphs': [{'name': 'vision', 'sha256': 'abc'}]}
    reports = build_reports(folder, manifest)
    assert reports['export']['max_abs'] == .01
    assert reports['compiled']['cosine'] == .999
    assert reports['graphs']['vision']['compile_seconds'] == 120
    manifest['graphs'][0]['sha256'] = 'different'
    reports = build_reports(folder, manifest)
    assert reports['export'] is None and reports['compiled'] is None and reports['graphs'] == {}


def test_graph_match_does_not_establish_full_policy_validation(tmp_path):
    from polima.workbench.app import build_reports
    root = tmp_path / 'build'
    folder = root / 'bundles/model'
    folder.mkdir(parents=True)
    (root / 'compiled').mkdir()
    (root / 'check').mkdir()
    (root / 'compiled/compile_results.json').write_text(json.dumps({'vision': {'elf_sha256': 'abc'}}))
    (root / 'build.json').write_text(json.dumps({'bundle': str(root / 'other-model')}))
    (root / 'check/report.json').write_text(json.dumps({'ok': True, 'all': {'cosine': .999}}))
    assert build_reports(folder, {'graphs': [{'name': 'vision', 'sha256': 'abc'}]})['compiled'] is None


@pytest.mark.parametrize('matching', [True, False])
def test_act_sibling_build_reports(tmp_path, matching):
    from hashlib import sha256

    from polima.workbench.app import build_reports
    folder = tmp_path / 'bundles/model'
    folder.mkdir(parents=True)
    build = tmp_path / 'build'
    build.mkdir()
    elf = build / 'vision.elf'
    elf.write_bytes(b'compiled model')
    (build / 'artifact_manifest.json').write_text(json.dumps({
        'graphs': [{'name': 'vision', 'elf': str(elf), 'duration_s': 12.5}],
        'export': {'dataset_root': '/uploaded/data', 'checkpoint': '/uploaded/model'}}))
    (build / 'onnx_verification_report.json').write_text(json.dumps({
        'ok': True, 'mean_abs': .000001, 'max_abs': .000002}))
    manifest = {'graphs': [{'name': 'vision', 'sha256': sha256(elf.read_bytes()).hexdigest() if matching else 'wrong'}]}
    result = build_reports(folder, manifest)
    if matching:
        assert result['export']['ok'] is True
        assert result['export']['mean_abs'] == .000001
        assert result['graphs']['vision']['compile_seconds'] == 12.5
        assert result['dataset'] == '/uploaded/data'
        assert result['compiled'] is None
    else:
        assert result['export'] is None and result['graphs'] == {}
