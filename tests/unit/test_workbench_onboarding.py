"""Customer onboarding with temporary storage and no device access."""
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from polima.workbench import app, inputs


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv('POLIMA_WORKBENCH_HOME', str(tmp_path / 'workbench'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    return inputs.Inputs()


def test_folder_upload_can_resume_and_be_discovered(storage):
    files = {'checkpoint/config.json': b'{"type":"act"}', 'checkpoint/model.safetensors': b'fixture'}
    state = storage.create('checkpoint', [{'path': p, 'size': len(b)} for p, b in files.items()])
    name, content = next(iter(files.items()))
    storage.uploads.receive(state['id'], name, 0, io.BytesIO(content[:3]), 3)
    with pytest.raises(ValueError, match='incomplete'):
        storage.finish(state['id'])
    assert storage.uploads.status(state['id'])['files'][0]['received'] == 3
    storage.uploads.receive(state['id'], name, 3, io.BytesIO(content[3:]), len(content)-3)
    for name, content in list(files.items())[1:]:
        storage.uploads.receive(state['id'], name, 0, io.BytesIO(content), len(content))
    item = storage.finish(state['id'])
    assert app.Workbench([], [], []).scan()[0][0]['path'] == item['path']
    assert Path(item['path'], 'model.safetensors').read_bytes() == b'fixture'


@pytest.mark.parametrize('files', [[], None, ['invalid'], [{'path':'../escape', 'size':1}], [{'path':'data.zip', 'size':1}]])
def test_upload_rejects_invalid_input(storage, files):
    with pytest.raises(ValueError):
        storage.create('dataset', files)


def test_missing_prerequisites_do_not_start_a_compile(storage, monkeypatch):
    work = app.Workbench([], [], [])
    monkeypatch.setattr(work, 'scan', lambda: ([{'path':'chosen', 'policy':'GR00T N1.7'}], []))
    monkeypatch.setattr(app, 'prerequisites', lambda *a: [{'ok':False, 'help':'Install the matching export environment.'}])
    with pytest.raises(ValueError, match='Setup required'):
        work.start_job({'kind':'compile', 'checkpoint':'chosen'})
    assert work.jobs.snapshot() is None


def test_macos_has_an_explicit_compiler_requirement(storage, monkeypatch):
    monkeypatch.setattr(inputs.platform, 'system', lambda: 'Darwin')
    checks = inputs.prerequisites('ACT')
    assert checks[0]['ok'] is False
    assert 'Linux' in checks[0]['help']


@pytest.mark.parametrize("bind_host", ["127.0.0.1", "127.0.0.2"])
def test_managed_workbench_start_status_stop_without_install(storage, tmp_path, bind_host):
    source = str(Path(app.__file__).resolve().parents[2])
    env = {k:v for k,v in os.environ.items() if not k.startswith('POLIMA_')}
    env.update(PYTHONPATH=source, POLIMA_WORKBENCH_HOME=str(storage.root), XDG_CONFIG_HOME=str(tmp_path/'config'), NO_PROXY='*', no_proxy='*')
    def run(*args):
        return subprocess.run([sys.executable, '-m', 'polima', 'workbench', *args], env=env, cwd=tmp_path,
                              text=True, capture_output=True, timeout=20, check=False)
    try:
        started = run('start', '--host', bind_host, '--port', '0', '--models', str(tmp_path/'empty'))
        assert started.returncode == 0, started.stderr
        assert 'Workbench started' in started.stdout
        assert f'http://{bind_host}:' in run('status').stdout
        again = run('start')
        assert 'Workbench started' not in again.stdout
    finally:
        stopped = run('stop')
    assert stopped.returncode == 0, stopped.stderr
    assert 'stopped' in run('status').stdout


def test_invalid_import_metadata_does_not_hide_valid_models(storage):
    base = storage.root / 'imports'
    for name, text in [('partial', '{'), ('list', '[]'), ('invalid', '{"kind":"checkpoint","path":null}')]:
        (base / name).mkdir(parents=True)
        (base / name / 'input.json').write_text(text)
    assert storage.items() == []


@pytest.mark.parametrize('metadata', ['[]', 'null', '42', '{'])
def test_invalid_dataset_metadata_is_a_failed_prerequisite(storage, tmp_path, metadata):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(metadata)
    checks = inputs.prerequisites('GR00T N1.7', str(tmp_path))
    assert checks[-1]['ok'] is False


def test_stop_network_failure_has_actionable_message(storage, monkeypatch, capsys):
    from polima.cli import workbench
    monkeypatch.setattr(workbench, 'state', lambda: {'url':'http://127.0.0.1:1'})
    assert workbench.manage('stop', []) == 1
    assert 'Could not contact Workbench' in capsys.readouterr().err


def test_saved_process_identity_is_verified(storage, monkeypatch):
    from polima.cli import workbench
    storage.root.mkdir(parents=True)
    (storage.root / 'service.json').write_text('{"url":"http://127.0.0.1:1","instance":"expected"}')
    class Response(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    from types import SimpleNamespace
    monkeypatch.setattr(workbench.urllib.request, "build_opener",
                        lambda *a: SimpleNamespace(open=lambda *a, **k: Response(b'{"instance":"other"}')))
    assert workbench.state() == {}


@pytest.fixture
def browser_api(storage):
    import http.cookiejar
    import json
    import threading
    import urllib.error
    import urllib.request
    server = app.serve(app.Workbench([], [], []), host='127.0.0.1', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with opener.open(base + '/api/v1/session') as response:
        token = json.load(response)['csrf_token']
    def request(path, body, method='POST', authenticated=True):
        headers = {'X-CSRF-Token': token} if authenticated else {}
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        try:
            response = opener.open(urllib.request.Request(base + path, data=data, method=method, headers=headers), timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)
    try:
        yield request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_browser_import_requires_session_and_rejects_bad_manifest(browser_api):
    body = {'kind':'checkpoint', 'files':[{'path':'config.json', 'size':2}]}
    assert browser_api('/api/v1/imports', body, authenticated=False)[0] == 403
    body['files'][0]['size'] = None
    status, result = browser_api('/api/v1/imports', body)
    assert status == 400 and 'size' in result['error']


def test_browser_chunk_retries_and_path_confinement(browser_api, storage):
    status, upload = browser_api('/api/v1/imports', {'kind':'dataset', 'files':[{'path':'data.bin', 'size':4}]})
    assert status == 200
    path = f'/api/v1/imports/chunk?id={upload["id"]}&path=data.bin&offset=0'
    assert browser_api(path, b'ab', method='PUT', authenticated=False)[0] == 403
    assert browser_api(path, b'ab', method='PUT')[1]['received'] == 2
    assert browser_api(path, b'ab', method='PUT')[1] == {'ok':False, 'path':'data.bin', 'received':2}
    escape = f'/api/v1/imports/chunk?id={upload["id"]}&path=../escape&offset=0'
    assert browser_api(escape, b'ab', method='PUT')[0] == 400
    assert browser_api(path.replace('offset=0','offset=2'), b'cd', method='PUT')[1]['received'] == 4
    assert browser_api('/api/v1/imports/status', {'id':upload['id']})[1]['complete'] is True
    assert browser_api('/api/v1/imports/cancel', {'id':upload['id']})[0] == 200
    assert not list((storage.root / 'imports').iterdir())


@pytest.mark.parametrize('error', [subprocess.TimeoutExpired('ssh', 1), subprocess.CalledProcessError(1, 'ssh-keygen')])
def test_devkit_setup_failure_is_visible_and_not_saved(browser_api, monkeypatch, error):
    def fail(*args):
        raise error
    monkeypatch.setattr(app.devkits, 'install_key', fail)
    status, result = browser_api('/api/v1/devkits', {'host':'sima@new-devkit', 'password':'test-only'})
    assert status == 400 and 'SSH' in result['error']
    assert app.devkits.load() == []


def test_failed_authentication_preserves_empty_customer_config(browser_api, monkeypatch):
    monkeypatch.setattr(app.devkits, 'probe', lambda *a: {'reachable':False, 'error':'SSH authentication failed.'})
    status, result = browser_api('/api/v1/devkits', {'host':'sima@new-devkit'})
    assert status == 400 and 'authentication' in result['error']
    assert app.devkits.load() == []


def test_devkit_file_setting_is_not_a_studio_url(storage, monkeypatch):
    from types import SimpleNamespace

    from polima.cli import workbench
    monkeypatch.setenv('POLIMA_DEVKITS', str(storage.root / 'saved-devkits.json'))
    monkeypatch.setenv('POLIMA_DEVKIT_URLS', 'kit-one:8080,kit-two:8080')
    monkeypatch.setattr(workbench.role, 'is_board', lambda: False)
    seen = {}
    def serve(model, *args):
        seen['boards'] = model.boards
        return SimpleNamespace(server_address=('127.0.0.1', 1234), serve_forever=lambda:None, server_close=lambda:None)
    monkeypatch.setattr(workbench, 'serve', serve)
    assert workbench.run(['serve', '--host', '127.0.0.1']) == 0
    assert seen['boards'] == ['kit-one:8080', 'kit-two:8080']
    monkeypatch.delenv('POLIMA_DEVKIT_URLS')
    monkeypatch.delenv('POLIMA_BOARDS', raising=False)
    assert workbench.run(['serve', '--host', '127.0.0.1']) == 0
    assert seen['boards'] == []


def test_workbench_defaults_to_local_access(storage, monkeypatch, capsys):
    from types import SimpleNamespace

    from polima.cli import workbench

    monkeypatch.setattr(workbench.role, 'is_board', lambda: False)
    seen = []
    def serve(model, host, port):
        seen.append(host)
        return SimpleNamespace(server_address=(host, 1234), serve_forever=lambda: None,
                               server_close=lambda: None)
    monkeypatch.setattr(workbench, 'serve', serve)
    assert workbench.run(['serve']) == 0
    assert seen == ['127.0.0.1']
    assert 'this computer' in capsys.readouterr().out
    assert workbench.run(['serve', '--host', '0.0.0.0']) == 0
    assert seen[-1] == '0.0.0.0'


@pytest.mark.parametrize('available', [True, False])
def test_act_prerequisites_detect_missing_torch(storage, monkeypatch, available):
    monkeypatch.setattr(inputs.importlib.util, 'find_spec', lambda name: object() if available else None)
    checks = inputs.prerequisites('ACT')
    torch = next(row for row in checks if row['label'] == 'PyTorch export dependency')
    assert torch['ok'] is available
    assert 'Workbench' in torch['help']


@pytest.mark.parametrize('kind', ['missing', 'invalid', 'valid'])
def test_act_requires_selected_dataset(storage, tmp_path, kind):
    dataset = tmp_path / 'dataset'
    if kind != 'missing':
        (dataset / 'meta').mkdir(parents=True)
        (dataset / 'data').mkdir()
        (dataset / 'meta/info.json').write_text('{}' if kind == 'valid' else 'invalid')
    checks = inputs.prerequisites('ACT', str(dataset) if kind != 'missing' else '')
    selected = next(row for row in checks if row['label'] == 'Representative dataset')
    assert selected['ok'] is (kind == 'valid')


def test_act_without_selected_dataset_does_not_start_job(storage, monkeypatch):
    work = app.Workbench([], [], [])
    monkeypatch.setattr(work, 'scan', lambda: ([{'path':'chosen', 'policy':'ACT'}], []))
    with pytest.raises(ValueError, match='Select an uploaded LeRobot dataset'):
        work.start_job({'kind':'compile', 'checkpoint':'chosen'})
    assert work.jobs.snapshot() is None


@pytest.mark.parametrize('method', ['POST', 'PUT'])
def test_browser_renews_expired_session_once(tmp_path, method):
    import shutil

    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    page = (app.STATIC / 'index.html').read_text()
    helpers = page[page.index('  var session = null;'):page.index('  var pendingUpload = null;')]
    script = tmp_path / 'session.cjs'
    script.write_text(helpers + """
const assert = require('node:assert/strict');
(async () => {
  let sessions=0, writes=0, rejected=true;
  global.fetch=async (path, options) => {
    if (path==='/api/v1/session') return {ok:true,json:async()=>({csrf_token:'token'+(++sessions)})};
    writes++;
    assert.equal(options.method, process.argv[2]);
    assert.equal(options.headers['X-CSRF-Token'], 'token'+sessions);
    if (rejected) { rejected=false; return {ok:false,status:403,json:async()=>({error:'invalid CSRF token'})}; }
    return {ok:true,status:200,json:async()=>({ok:true,received:16})};
  };
  assert.equal((await write('/upload', {method:process.argv[2],body:'chunk'})).received,16);
  assert.equal(sessions,2); assert.equal(writes,2);
  writes=0;
  global.fetch=async path => path==='/api/v1/session'
    ? {ok:true,json:async()=>({csrf_token:'new'})}
    : (writes++, {ok:false,status:403,json:async()=>({error:'Invalid session'})});
  await assert.rejects(write('/upload',{method:process.argv[2]}), /Session expired/);
  assert.equal(writes,2);
  writes=0;
  global.fetch=async path => path==='/api/v1/session'
    ? {ok:true,json:async()=>({csrf_token:'new'})}
    : (writes++, {ok:false,status:400,json:async()=>({error:'Invalid file'})});
  await assert.rejects(write('/upload',{method:process.argv[2]}), /Invalid file/);
  assert.equal(writes,1);
})().catch(error=>{console.error(error);process.exit(1)});
""")
    result = subprocess.run([node, str(script), method], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('outcome', ['ready', 'missing', 'timeout'])
def test_groot_checks_actual_export_dependencies(storage, tmp_path, monkeypatch, outcome):
    from types import SimpleNamespace

    python = tmp_path / 'python'
    python.write_text('')
    python.chmod(0o700)
    (tmp_path / 'gr00t').mkdir()
    (tmp_path / 'gr00t/__init__.py').write_text('')
    monkeypatch.setenv('POLIMA_GROOT17_PYTHON', str(python))
    monkeypatch.setenv('POLIMA_GROOT17_SOURCE', str(tmp_path))
    def probe(command, **kwargs):
        assert command[0] == str(python)
        assert kwargs['env']['PYTHONPATH'] == str(tmp_path)
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(command, 30)
        return SimpleNamespace(returncode=0 if outcome == 'ready' else 1,
                               stderr="ModuleNotFoundError: No module named 'torch'")
    monkeypatch.setattr(inputs.subprocess, 'run', probe)
    checks = inputs.prerequisites('GR00T N1.7')
    dependencies = next(row for row in checks if row['label'].endswith('export dependencies'))
    assert dependencies['ok'] is (outcome == 'ready')
    assert next(row for row in checks if row['label'] == 'Local N1.7 backbone assets')['ok'] is False


def test_groot_rejects_unrelated_source_directory(storage, tmp_path, monkeypatch):
    monkeypatch.setenv('POLIMA_GROOT17_SOURCE', str(tmp_path))
    checks = inputs.prerequisites('GR00T N1.7')
    assert next(row for row in checks if row['label'].endswith('Isaac-GR00T source'))['ok'] is False


def test_shutdown_acknowledges_before_stopping_server(storage, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv('POLIMA_WORKBENCH_INSTANCE', 'test-instance')
    monkeypatch.setattr(app.secrets, 'token_urlsafe', lambda _: 'test-token')
    server = app.serve(app.Workbench([], [], []), port=0)
    events = []

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(app, 'threading', SimpleNamespace(Thread=ImmediateThread))
    monkeypatch.setattr(server, 'shutdown', lambda: events.append('shutdown'))
    handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
    handler.server = server
    handler.path = '/api/v1/shutdown'
    handler.headers = {'Cookie': 'polima_workbench_csrf=test-token',
                       'X-CSRF-Token': 'test-token', 'Content-Length': '2'}
    handler.rfile = io.BytesIO(b'{}')
    handler.wfile = io.BytesIO()
    handler.send = lambda *args: events.append('response')
    try:
        handler.do_POST()
    finally:
        server.server_close()
    assert events == ['response', 'shutdown']


@pytest.mark.parametrize('layout,on_board,expected', [
    ('checkout', False, True), ('checkout', True, False),
    ('wheel', False, False), ('unrelated', False, False),
])
def test_default_checkpoint_discovery_layout(tmp_path, monkeypatch, layout, on_board, expected):
    from polima.cli import workbench
    from polima.workbench.app import walk

    monkeypatch.setattr(workbench, 'package_root', lambda: tmp_path)
    if layout != 'wheel':
        (tmp_path / 'pyproject.toml').write_text('[project]\nname="polima"\n')
    if layout == 'checkout':
        (tmp_path / 'src/polima').mkdir(parents=True)
        (tmp_path / 'src/polima/__init__.py').touch()
    roots = workbench.default_models(on_board)
    assert roots == ([tmp_path / 'models'] if expected else [])
    assert walk(roots, 'config.json') == []
    checkpoint = tmp_path / 'models/run/checkpoints/100000/pretrained_model'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}')
    assert walk(roots, 'config.json') == ([checkpoint] if expected else [])


@pytest.mark.parametrize('explicit,environment,expected', [
    (['chosen'], 'environment', ['chosen']),
    (None, 'first:second', ['first', 'second']),
    (None, '', ['models']),
])
def test_checkpoint_folder_override_precedence(monkeypatch, explicit, environment, expected):
    from pathlib import Path

    from polima.workbench.app import folders

    monkeypatch.setenv('POLIMA_MODELS', environment)
    assert folders(explicit, 'POLIMA_MODELS', [Path('models')]) == [Path(p) for p in expected]
