import { render } from "preact";
import { useEffect, useMemo, useState } from "preact/hooks";
import "./styles.css";
import "./tools.css";
import { benchmarkSelection, meanChange } from "./metrics.js";
import { createApi } from "./api.js";
import { BundleInstaller, BundleLibrary, DeviceFiles, duration, when } from "./files.jsx";
import { RobotRunReviews } from "./reviews.jsx";

const blank = { bundle: "", task: "", robot_port: "", overhead_camera: "", wrist_camera: "", preview: true, repeat: true, autocomplete: null, speed: "fast", fps: null };
const idleStates = ["idle", "fault"];
const shortDevice = value => value ? value.split("/").pop().replace("-video-index0", "") : "Not connected";
const leaseKey = "polima-controller-lease";
const browserKey = "polima-browser-id";
const stored = key => { try { return localStorage.getItem(key) || ""; } catch (_) { return ""; } };
const browserId = (() => {
  const existing = stored(browserKey);
  const value = existing || (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`);
  try { localStorage.setItem(browserKey, value); } catch (_) {}
  return value;
})();

function Icon({ name }) {
  const paths = {
    operate: <><path d="M4 19v-7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v7"/><path d="M8 10V7a4 4 0 0 1 8 0v3M9 15h6M12 12v6"/></>,
    performance: <><path d="M4 18a8 8 0 1 1 16 0"/><path d="m12 18 4-6M7 18h10"/></>,
    manage: <><rect x="3" y="4" width="18" height="6" rx="2"/><rect x="3" y="14" width="18" height="6" rx="2"/><path d="M7 7h.01M7 17h.01"/></>,
    activity: <><path d="M4 19V5M4 19h16"/><path d="m7 15 3-4 3 2 5-7"/></>,
    files: <><path d="M12 15V3M7 10l5 5 5-5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></>,
    camera: <><rect x="3" y="6" width="18" height="13" rx="2"/><circle cx="12" cy="12.5" r="3.5"/><path d="m8 6 1-2h6l1 2"/></>,
    sun: <><circle cx="12" cy="12" r="3.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></>,
    moon: <path d="M20 15.2A8 8 0 0 1 8.8 4 8.1 8.1 0 1 0 20 15.2Z"/>,
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function Studio() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || "dark");
  const [snapshot, setSnapshot] = useState(null);
  const [csrf, setCsrf] = useState("");
  const [lease, setLease] = useState(() => stored(leaseKey));
  const [config, setConfig] = useState(blank);
  const [log, setLog] = useState([]);
  const [error, setError] = useState("");
  const [arming, setArming] = useState(null);
  const [benchmarks, setBenchmarks] = useState([]);
  const [history, setHistory] = useState([]);
  const [benchmarkConfig, setBenchmarkConfig] = useState({ iterations: 50, warmup: 3 });
  const controller = Boolean(lease);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("polima-theme", next);
    setTheme(next);
  };

  const api = createApi(fetch, () => ({ "X-CSRF-Token": csrf, "X-Controller-Lease": lease, "X-Browser-ID": browserId }), setCsrf);

  const refresh = () => fetch("/api/v1/snapshot").then(r => r.json()).then(data => {
    setSnapshot(data);
    setConfig(old => {
      if (old.bundle) return old;
      const saved = data.last_run_config || {};
      const savedExists = data.bundles.some(bundle => bundle.id === saved.bundle);
      const bundle = (savedExists && saved.bundle) || data.bundles.find(item => item.active)?.id || data.bundles[0]?.id || "";
      const selected = data.bundles.find(item => item.id === bundle);
      return {
        ...blank,
        ...(savedExists ? saved : {}),
        bundle,
        autocomplete: selected?.policy === "groot" ? null : (savedExists ? saved.autocomplete ?? null : null),
        task: (savedExists && saved.task) || selected?.default_task || "",
        robot_port: saved.robot_port || data.hardware.arms[0] || "",
        overhead_camera: saved.overhead_camera || data.hardware.cameras.find(c => /046d|c920/i.test(c)) || data.hardware.cameras[0] || "",
        wrist_camera: saved.wrist_camera || data.hardware.cameras.find(c => /sonix|cam1/i.test(c)) || data.hardware.cameras[1] || "",
      };
    });
  });
  const refreshLists = () => Promise.all([
    fetch("/api/v1/benchmarks").then(r => r.json()).then(data => setBenchmarks(data.items || [])),
    fetch("/api/v1/history").then(r => r.json()).then(data => setHistory(data.items || [])),
  ]);

  useEffect(() => {
    fetch("/api/v1/session").then(r => r.json()).then(body => setCsrf(body.csrf_token));
    refresh();
    refreshLists();
    const events = new EventSource("/api/v1/events");
    events.onmessage = ({ data }) => {
      const event = JSON.parse(data);
      if (event.type === "log") setLog(old => [...old.slice(-299), event.line]);
      if (event.type === "preview") {
        setSnapshot(old => old ? { ...old, preview_url: event.url } : old);
      }
      if (event.type === "job") {
        setSnapshot(old => old ? { ...old, job: event.job } : old);
        if (event.job.status !== "running") refresh();
      }
      if (["state", "halt", "bundle", "server", "calibration"].includes(event.type)) { refresh(); refreshLists(); }
      if (event.type === "benchmark") refreshLists();
    };
    return () => events.close();
  }, []);

  useEffect(() => {
    if (!csrf || !lease) return;
    const renew = setInterval(() => api("/lease", { method: "POST", body: JSON.stringify({ token: lease }) }).then(body => rememberLease(body.token)).catch(error => { forgetLease(); setError(`Control session could not be renewed: ${error.message}. Select Take control to reconnect.`); }), 15000);
    return () => clearInterval(renew);
  }, [csrf, lease]);

  const rememberLease = token => { try { localStorage.setItem(leaseKey, token); } catch (_) {} setLease(token); };
  const forgetLease = () => { try { localStorage.removeItem(leaseKey); } catch (_) {} setLease(""); };
  const mutate = async fn => { setError(""); try { await fn(); await refresh(); } catch (e) { setError(e.message); } };
  const claim = () => mutate(async () => rememberLease((await api("/lease", { method: "POST", body: JSON.stringify({ token: lease || null }) })).token));
  const arm = () => mutate(async () => setArming(await api("/robot/arm", { method: "POST", body: JSON.stringify(config) })));
  const start = () => mutate(async () => { await api("/robot/start", { method: "POST", body: JSON.stringify({ config, arming_token: arming.arming_token }) }); setArming(null); });
  const stop = emergency => mutate(() => api(`/robot/${emergency ? "halt" : "stop"}`, { method: "POST", body: "{}" }));
  const preview = () => mutate(() => api("/preview/start", { method: "POST", body: JSON.stringify(config) }));
  // Needs no controller lease: it is the way out of a wedged studio, and one
  // of the things it clears IS a stale lease held by some other tab.
  const resetSystem = () => mutate(() => api("/system/reset", { method: "POST" }));
  const activate = () => activateBundle(config.bundle);
  const activateBundle = id => mutate(() => api(`/bundles/${encodeURIComponent(id)}/activate`, { method: "POST", body: "{}" }));
  const uploadHeaders = () => ({ "X-CSRF-Token": csrf, "X-Controller-Lease": lease, "X-Browser-ID": browserId });
  const activeBundle = snapshot?.bundles.find(item => item.active)?.id;
  const copyCalibration = id => api(`/bundles/${encodeURIComponent(id)}/calibration`, { method: "POST", body: JSON.stringify({ from_bundle: activeBundle }) });
  const selectForRun = id => { changeBundle(id); document.getElementById("operate")?.scrollIntoView(); };
  const saveFrame = role => {
    const image = document.querySelector(`img[data-role="${role}"]`);
    if (!image?.naturalWidth) return setError("No camera frame to save yet");
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    canvas.getContext("2d").drawImage(image, 0, 0);
    canvas.toBlob(blob => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${role}-${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, "image/jpeg", 0.95);
  };
  const policyServer = action => mutate(() => api(`/server/${action}`, { method: "POST", body: JSON.stringify({ bundle: config.bundle }) }));
  const calibrate = () => mutate(() => api("/calibration/start", { method: "POST", body: JSON.stringify(config) }));
  const calibrationInput = value => mutate(() => api("/calibration/input", { method: "POST", body: JSON.stringify({ value }) }));
  const runBenchmark = () => mutate(() => api("/benchmarks/start", { method: "POST", body: JSON.stringify({ bundle: config.bundle, ...benchmarkConfig }) }));
  const stopBenchmark = () => mutate(() => api("/benchmarks/stop", { method: "POST", body: "{}" }));
  const changeConfig = values => { setArming(null); setConfig(old => ({ ...old, ...values })); };
  const changeBundle = id => {
    const bundle = snapshot?.bundles.find(item => item.id === id);
    changeConfig({ bundle: id, autocomplete: null, task: bundle?.default_task || "" });
  };

  const active = snapshot?.state || "loading";
  const canPrepare = controller && idleStates.includes(active);
  const selected = useMemo(() => snapshot?.bundles.find(b => b.id === config.bundle), [snapshot, config.bundle]);
  const { latest: latestBenchmark, previous: previousBenchmark } = benchmarkSelection(benchmarks, config.bundle);
  const benchmarkDelta = meanChange(latestBenchmark, previousBenchmark);
  const stageTotal = (latestBenchmark?.result?.stages || []).reduce((sum, stage) => sum + Math.max(0, stage.ms || 0), 0);
  const readyCount = snapshot ? [snapshot.bundles.length > 0, snapshot.hardware.arms.length > 0, snapshot.hardware.cameras.length >= 2].filter(Boolean).length : 0;

  if (!snapshot) return <main class="loading"><span class="spinner"></span>Connecting to PoLiMa Studio…</main>;
  return <div class="shell">
    <header class="app-header">
      <div class="brand"><img class="app-brand-mark" src="/assets/neat-mark.svg" alt="PoLiMa"/><div><h1>PoLiMa Studio</h1><small>Robot operations · SiMa.ai</small></div></div>
      <div class="header-status"><button class="theme-toggle" onClick={toggleTheme} title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`} aria-label={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}><Icon name={theme === "dark" ? "sun" : "moon"}/></button><span class={`state ${active}`}><i></i>{active}</span><button class="halt compact" onClick={() => stop(true)}>Emergency Halt</button></div>
    </header>
    <aside>
      <div>
        <p class="nav-label">Workspace</p>
        <nav>
          <a href="#operate"><Icon name="operate"/>Operate</a>
          <a href="#files"><Icon name="files"/>Bundles & files</a>
          <a href="#performance"><Icon name="performance"/>Performance</a>
          <a href="#manage"><Icon name="manage"/>Device & policy</a>
          <a href="#activity"><Icon name="activity"/>Activity</a>
        </nav>
      </div>
      <div class="device"><span class="pulse"></span><div><b>Modalix DevKit</b><small>{snapshot.server.running ? `Policy server · PID ${snapshot.server.pid}` : "Policy server stopped"}</small></div></div>
    </aside>
    <main>
      {error && <div class="alert" role="alert"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss">×</button></div>}
      {snapshot.fault && <div class="alert" role="alert"><span><strong>Studio Fault:</strong> {snapshot.fault}</span></div>}
      <section class="hero" id="operate">
        <div><p class="eyebrow">ROBOT OPERATIONS</p><h2>Run a policy</h2><p>Configure the model and hardware before launch.</p></div>
        {!controller ? <button class="primary" onClick={claim}>Take control</button> : <span class="lease"><i></i> This browser has control</span>}
      </section>

      <section class="readiness" aria-label="Run readiness">
        <div class={snapshot.bundles.length ? "ready" : "missing"}><b>1</b><span><strong>Select policy</strong><small>{selected ? `${selected.policy.toUpperCase()} · ${selected.active ? "active" : "installed"}` : <a href="#files">No policy installed — install one</a>}</small></span></div>
        <div class={config.task ? "ready" : "missing"}><b>2</b><span><strong>Confirm task</strong><small>{config.task || "Instruction required"}</small></span></div>
        <div class={readyCount === 3 ? "ready" : "missing"}><b>3</b><span><strong>Check hardware</strong><small>{snapshot.hardware.arms.length ? `${snapshot.hardware.cameras.length} camera devices · arm port detected` : "Follower arm missing"}</small></span></div>
        <div class={arming ? "ready" : "pending"}><b>4</b><span><strong>Arm & run</strong><small>{arming ? "Preflight passed" : "Motion requires confirmation"}</small></span></div>
      </section>

      <div class="operate-grid">
        <section class="card setup">
          <div class="cardhead"><div><span class="step-label">STEP 1</span><h3>Policy and task</h3><p>Task from the selected bundle.</p></div></div>
          <label>Policy bundle<select autoComplete="off" value={config.bundle} onChange={e => changeBundle(e.target.value)}>{snapshot.bundles.map(bundle => <option value={bundle.id} key={bundle.id}>{bundle.id}</option>)}</select></label>
          <div class="contract"><span>{selected?.policy || "unknown"}</span><span>{selected?.active ? "ACTIVE" : "INSTALLED"}</span><span>{selected?.tasks?.length || 1} TASK</span></div>
          <label>Task instruction
            {selected?.tasks?.length > 1
              ? <select autoComplete="off" value={config.task} onChange={e => changeConfig({ task: e.target.value })}>{selected.tasks.map(task => <option value={task} key={task}>{task}</option>)}</select>
              : <textarea autoComplete="off" value={config.task} onInput={e => changeConfig({ task: e.target.value })} />}
            <small class="field-help">Loaded from this bundle. Edit only when intentionally overriding its trained instruction.</small>
          </label>
        </section>

        <section class="card hardware-card">
          <div class="cardhead"><div><span class="step-label">STEP 2</span><h3>Hardware</h3><p>Detected devices are selected automatically.</p></div></div>
          <label>Follower arm<select autoComplete="off" value={config.robot_port} onChange={e => changeConfig({ robot_port: e.target.value })}><option value="">Not connected</option>{snapshot.hardware.arms.map(x => <option value={x} key={x}>{shortDevice(x)}</option>)}</select></label>
          <label>Overhead camera<select autoComplete="off" value={config.overhead_camera} onChange={e => changeConfig({ overhead_camera: e.target.value })}><option value="">Not connected</option>{snapshot.hardware.cameras.map(x => <option value={x} key={x}>{shortDevice(x)}</option>)}</select></label>
          <label>Wrist camera<select autoComplete="off" value={config.wrist_camera} onChange={e => changeConfig({ wrist_camera: e.target.value })}><option value="">Not connected</option>{snapshot.hardware.cameras.map(x => <option value={x} key={x}>{shortDevice(x)}</option>)}</select></label>
          <div class="toggles"><label><input autoComplete="off" type="checkbox" checked={config.preview} onChange={e => changeConfig({ preview: e.target.checked })}/><span>Live preview during run<small>Browser camera view</small></span></label><label><input autoComplete="off" type="checkbox" checked={config.repeat} onChange={e => changeConfig({ repeat: e.target.checked })}/><span>Repeat task<small>Continue after return to rest</small></span></label><label><input autoComplete="off" type="checkbox" checked={config.autocomplete ?? (selected?.policy !== "groot")} onChange={e => changeConfig({ autocomplete: e.target.checked })}/><span>Automatic return<small>Gripper-based estimate; disabled by default for GR00T</small></span></label></div>
          <label>Speed<select autoComplete="off" value={config.speed || (config.fps === 15 ? "slow" : "fast")} onChange={e => changeConfig({ speed: e.target.value, fps: null })}><option value="fast">Fast · 30 commands/s</option><option value="slow">Slow · 15 commands/s</option></select></label>
        </section>

        <section class="card cameras">
          <div class="cardhead"><div><span class="step-label">OPTIONAL</span><h3>Inspect camera views</h3><p>Camera-only preview never connects to the arm or policy server.</p></div><button disabled={!canPrepare || !config.overhead_camera || !config.wrist_camera} onClick={preview}><Icon name="camera"/> Start Preview</button></div>
          {snapshot.preview_url ? <div class="feeds">{["overhead", "wrist"].map(role => <figure key={role}><img data-role={role} src={`/api/v1/cameras/${role}.mjpg`}/><figcaption><i></i>{role}<button class="snap" onClick={() => saveFrame(role)}>Save Photo</button></figcaption></figure>)}</div> : <div class="cameraempty"><Icon name="camera"/><b>Camera preview is off</b><p>Use preview to frame the scene before enabling robot motion.</p></div>}
        </section>

        <section class="card launch-card">
          <div class="cardhead"><div><span class="step-label">STEP 3</span><h3>Preflight</h3><p>Review preflight results before enabling motion.</p></div></div>
          <div class="run-summary"><div><span>Policy</span><b>{selected?.policy?.toUpperCase() || "—"}</b></div><div><span>Task</span><b>{config.task || "—"}</b></div><div><span>State</span><b class={`text-${active}`}>{snapshot.fault || active}</b></div></div>
          {!arming ? <button class="primary wide" disabled={!canPrepare || !config.bundle || !config.task} onClick={arm}>Run Preflight Check</button> : <div class="armed"><div><b>Preflight complete</b><small>Authorization expires in 15 seconds</small></div>{arming.preflight.map(check => <span class={check.ok ? "ok" : "bad"} key={check.name}>{check.ok ? "✓" : "×"} {check.name}</span>)}<button class="danger wide" onClick={start}>Confirm And Start Robot</button></div>}
          <div class="stop-row"><button disabled={!controller || !["running", "previewing"].includes(active)} onClick={() => stop(false)}>Graceful Stop</button><button class="halt" onClick={() => stop(true)}>Emergency Halt</button><button onClick={resetSystem} title="Stop everything, reset the accelerator, clear the controller lease">Reset runtime</button></div>
        </section>
      </div>

      <section class="section-heading" id="files"><div><p class="eyebrow">BUNDLES & FILES</p><h2><span class="section-icon" aria-hidden="true">📦</span>Install bundles, download files</h2><p>Send compiled bundles to the devkit and bring logs and calibrations back, with no manual file copying.</p></div></section>
      <div class="transfer-grid">
        <BundleInstaller api={api} headers={uploadHeaders} controller={controller} onClaim={claim} job={snapshot.job} idle={idleStates.includes(active)} onSelect={selectForRun} onCopyCalibration={copyCalibration}/>
        <BundleLibrary snapshot={snapshot} api={api} controller={controller} idle={idleStates.includes(active)} selected={config.bundle} onSelect={selectForRun} onActivate={activateBundle} onCopyCalibration={copyCalibration}/>
        <DeviceFiles/>
      </div>

      <section class="section-heading" id="performance"><div><p class="eyebrow">PERFORMANCE</p><h2><span class="section-icon" aria-hidden="true">⚡</span>Benchmark policy inference</h2><p>Run stored fixture inputs on the MLA without cameras or a follower arm.</p></div></section>
      <section class="card benchmark">
        <div class="cardhead"><div><h3>Inference benchmark</h3><p class="data-value">{selected?.policy?.toUpperCase()} · {config.bundle}</p></div>{active === "benchmarking" ? <button onClick={stopBenchmark}>Stop Benchmark</button> : <button class="primary" disabled={!canPrepare || snapshot.server.running || !config.bundle} onClick={runBenchmark}>Run Benchmark</button>}</div>
        <div class="benchsetup"><label>Measured runs<input autoComplete="off" type="number" min="1" max="1000" value={benchmarkConfig.iterations} onInput={e => setBenchmarkConfig({...benchmarkConfig, iterations:Number(e.target.value)})}/></label><label>Warm-up<input autoComplete="off" type="number" min="0" max="100" value={benchmarkConfig.warmup} onInput={e => setBenchmarkConfig({...benchmarkConfig, warmup:Number(e.target.value)})}/></label><span>{snapshot.server.running ? "Stop the policy server to benchmark in isolation." : "No arm movement. No camera capture. The bundle’s fixtures are replayed on-device."}</span></div>
        {active === "benchmarking" && <div class="benchbusy"><span></span><b>Benchmarking {config.bundle}</b><small>Loading once, warming up, then measuring inference…</small></div>}
        {latestBenchmark?.result && <div class="benchresult"><div class="benchmetrics"><div><small><span aria-hidden="true">⏱</span> MEAN</small><strong>{latestBenchmark.result.mean.toFixed(1)}<i>ms</i></strong></div><div><small><span aria-hidden="true">📈</span> P95</small><strong>{latestBenchmark.result.p95.toFixed(1)}<i>ms</i></strong></div><div><small><span aria-hidden="true">📉</span> P99</small><strong>{latestBenchmark.result.p99.toFixed(1)}<i>ms</i></strong></div><div><small><span aria-hidden="true">⚡</span> THROUGHPUT</small><strong>{latestBenchmark.result.throughput_hz.toFixed(1)}<i>Hz</i></strong></div></div><div class="benchmeta"><span>{latestBenchmark.bundle}</span><span>{latestBenchmark.iterations} runs</span>{benchmarkDelta !== null && <span class={benchmarkDelta <= 0 ? "ok" : "bad"}>{benchmarkDelta <= 0 ? "↓" : "↑"} {Math.abs(benchmarkDelta).toFixed(1)}% mean latency vs previous</span>}<span class={latestBenchmark.result.validation.status === "pass" ? "ok" : "bad"}>{latestBenchmark.result.validation.status === "pass" ? "✓ output verified" : latestBenchmark.result.validation.status}</span></div>{latestBenchmark.result.stages.length > 0 && <div class="stages" aria-label="Recorded stage profile, separate from multi-run mean">{latestBenchmark.result.stages.map((stage, index) => <div key={`${stage.name}-${index}`}><code>{stage.name}</code><span style={{width:`${Math.max(2, Math.min(100, stage.ms / (stageTotal || 1) * 100))}%`}}></span><b>{stage.ms.toFixed(2)} ms</b></div>)}</div>}</div>}
        {latestBenchmark?.status === "failed" && <div class="inline-error">{latestBenchmark.error || "Benchmark failed. Check runtime events."}</div>}
        {!latestBenchmark?.result && active !== "benchmarking" && <div class="empty-small"><b>No benchmark results yet</b><p>Select a bundle and run an isolated fixture benchmark.</p></div>}
        {benchmarks.length > 1 && <div class="table-list"><div class="table-head"><span>When</span><span>Bundle</span><span>Mean</span><span>P95</span><span>Result</span><span></span></div>{benchmarks.slice(0, 8).map(item => <div key={item.id}><span>{when(item.started_at)}</span><span class="mono">{item.bundle}</span><span>{item.result ? `${item.result.mean.toFixed(1)} ms` : "—"}</span><span>{item.result ? `${item.result.p95.toFixed(1)} ms` : "—"}</span><span class={item.status === "completed" ? "ok" : item.status === "failed" ? "bad" : ""}>{item.status}</span><a href={`/api/v1/logs/benchmarks/${item.id}`} target="_blank" rel="noopener">Log</a></div>)}</div>}
      </section>

      <section class="section-heading" id="manage"><div><p class="eyebrow">DEVICE & POLICY</p><h2><span class="section-icon" aria-hidden="true">⚙️</span>Runtime</h2><p>Activate bundles, control the policy server, and calibrate the follower arm.</p></div></section>
      <div class="management-grid">
        <section class="card"><div class="cardhead"><div><h3>Selected bundle</h3><p class="data-value">{config.bundle}</p></div><span class={`badge ${selected?.active ? "success" : ""}`}>{selected?.active ? "Active" : "Installed"}</span></div><div class="toolrow"><span>Make this the default policy</span><button disabled={!canPrepare || selected?.active} onClick={activate}>Activate Bundle</button></div></section>
        <section class="card"><div class="cardhead"><div><h3>Policy server</h3><p>Inference endpoint for the selected bundle</p></div><span class={`badge ${snapshot.server.running ? "success" : ""}`}>{snapshot.server.running ? "Running" : "Stopped"}</span></div><div class="toolrow"><span>{snapshot.server.running ? `Process ${snapshot.server.pid}` : "Safe to benchmark"}</span><div>{!controller && <button class="primary" onClick={claim}>Take control</button>}<button disabled={!controller || snapshot.server.running} title={!controller ? "Take control to start the policy server" : "Start model inference"} onClick={() => policyServer("start")}>Start</button><button disabled={!controller || !snapshot.server.running} onClick={() => policyServer("stop")}>Stop</button></div></div></section>
        <section class="card"><div class="cardhead"><div><h3>Follower calibration</h3><p>Guided SO-ARM101 calibration with automatic backup</p></div></div><div class="toolrow"><span>{active === "calibrating" ? "Choose existing calibration or begin manual calibration" : config.robot_port ? shortDevice(config.robot_port) : "Connect an arm to continue"}</span>{active === "calibrating" ? <div><button onClick={() => calibrationInput("")}>Use Existing (Enter)</button><button class="primary" onClick={() => calibrationInput("c")}>Manual Calibration</button><button onClick={() => stop(false)}>Cancel</button></div> : <button disabled={!canPrepare || !config.robot_port} onClick={calibrate}>Start Calibration</button>}</div></section>
      </div>

      <section class="section-heading" id="activity"><div><p class="eyebrow">ACTIVITY</p><h2><span class="section-icon" aria-hidden="true">📊</span>Run history</h2><p>Review recent sessions and live process output.</p></div></section>
      <RobotRunReviews api={api} refreshKey={`${history[0]?.id}:${history[0]?.status}`}/>
      <div class="activity-grid">
        <section class="card history"><div class="cardhead"><div><h3>Recent robot runs</h3><p>Latest persisted sessions</p></div></div>{history.length ? <div class="history-list">{history.slice(0, 12).map(run => <div key={run.id}><span class={`run-status ${run.status}`}></span><div><b>{run.task}</b><small>{run.bundle}</small><small>{when(run.started_at)} · {run.ended_at ? duration(run.ended_at - run.started_at) : "in progress"}{run.error ? ` · ${run.error}` : ""}</small></div><span class="run-meta"><time>{run.status}</time><a href={`/api/v1/logs/runs/${run.id}`} target="_blank" rel="noopener">Log</a></span></div>)}</div> : <div class="empty-small"><b>No robot runs recorded</b><p>Completed runs will appear here.</p></div>}</section>
        <section class="card console"><div class="cardhead"><div><h3>Runtime events</h3><p>Live output from the active process</p></div><div class="toolrow-buttons"><a class="button" href="/api/v1/logs/server" target="_blank" rel="noopener">Server Log</a><button onClick={() => setLog([])}>Clear</button></div></div><pre>{log.length ? log.join("\n") : "Waiting for runtime events…"}</pre></section>
      </div>
      <footer><strong>SiMa.ai</strong><span>PoLiMa Studio · Physical AI control for Modalix</span></footer>
    </main>
  </div>;
}

render(<Studio />, document.getElementById("app"));
