import { useEffect, useRef, useState } from "preact/hooks";

const CHUNK = 16 << 20;
// The files `polima deploy`'s rsync leaves behind: the board never needs them.
const skipped = path => /(^|\/)(__pycache__|\.DS_Store|Thumbs\.db)(\/|$)|\.(pyc|onnx|tar\.gz)$/i.test(path);
const isArchive = name => /\.(tar|tar\.gz|tgz)$/i.test(name);
const post = body => ({ method: "POST", body: JSON.stringify(body || {}) });
const fail = (message, extra) => Object.assign(new Error(message), extra);

export const bytes = value => {
  if (value == null) return "—";
  const index = Math.min(4, Math.max(0, Math.floor(Math.log10(value || 1) / 3)));
  const number = value / 1000 ** index;
  return `${number.toFixed(number >= 100 || !index ? 0 : 1)} ${["B", "KB", "MB", "GB", "TB"][index]}`;
};
export const duration = s => s == null || !isFinite(s) ? "—" : s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.floor(s / 3600)} h ${Math.round(s % 3600 / 60)} min`;
export const when = epoch => epoch ? new Date(epoch * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";

async function readEntry(entry, prefix = "") {
  if (entry.isFile) return [{ path: prefix + entry.name, file: await new Promise((ok, no) => entry.file(ok, no)) }];
  const reader = entry.createReader(), children = [];
  for (let batch; (batch = await new Promise((ok, no) => reader.readEntries(ok, no))).length;) children.push(...batch);
  return (await Promise.all(children.map(child => readEntry(child, `${prefix}${entry.name}/`)))).flat();
}
const fromDrop = async data => (await Promise.all([...data.items].map(item => item.webkitGetAsEntry?.()).filter(Boolean).map(entry => readEntry(entry)))).flat();
const fromInput = files => [...files].map(file => ({ path: file.webkitRelativePath || file.name, file }));

function sendChunk(url, blob, headers, onProgress, abort) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    abort.current = () => xhr.abort();
    xhr.open("PUT", url);
    Object.entries({ ...headers, "Content-Type": "application/octet-stream" }).forEach(([key, value]) => xhr.setRequestHeader(key, value));
    xhr.upload.onprogress = event => onProgress(event.loaded);
    xhr.onload = () => {
      const body = (() => { try { return JSON.parse(xhr.responseText); } catch (_) { return {}; } })();
      xhr.status < 300 ? resolve(body) : reject(fail(body.error || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(fail("Network interruption", { retry: true }));
    xhr.onabort = () => reject(fail("Upload paused", { cancelled: true }));
    xhr.send(blob);
  });
}

const stages = { starting: "Starting", extracting: "Unpacking archive", verifying: "Checking ELF checksums", installing: "Installing", activating: "Activating", done: "Done" };

function Progress({ label, done, total, children }) {
  return <div class="transfer-progress">
    <div class="transfer-progress-head"><b>{label}</b><span>{total > 1 ? `${bytes(done)} of ${bytes(total)}` : ""}</span></div>
    <div class="bar"><span style={{ width: `${total ? Math.min(100, done / total * 100) : 0}%` }}></span></div>
    {children}
  </div>;
}

export function BundleInstaller({ api, headers, controller, onClaim, job, idle, onSelect, onCopyCalibration }) {
  const [picked, setPicked] = useState(null);
  const [activate, setActivate] = useState(false);
  const [upload, setUpload] = useState(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState(null);
  const abort = useRef(null), paused = useRef(false), waiting = useRef(false);
  const folderInput = useRef(null), archiveInput = useRef(null);

  useEffect(() => {
    if (!waiting.current || job?.kind !== "install" || job.status === "running") return;
    job.status === "completed" ? (setResult(job.result), setPicked(null)) : setError(job.error || "Install failed");
    setUpload(null);
    waiting.current = false;
  }, [job]);

  const choose = entries => {
    setError(""); setResult(null);
    if (entries.length === 1 && isArchive(entries[0].path)) return setPicked({ name: entries[0].path, files: entries, skipped: 0 });
    const files = entries.filter(entry => !skipped(entry.path));
    if (!files.some(entry => entry.path.endsWith("bundle.json"))) return setError("That folder has no bundle.json. Choose a PoLiMa bundle folder or a .tar of one.");
    setPicked({ name: files[0].path.split("/")[0], files, skipped: entries.length - files.length });
  };

  // Resumable: a session whose files match is continued from what the board already holds.
  const start = async () => {
    setError(""); setResult(null); paused.current = false;
    const files = picked.files, total = files.reduce((sum, entry) => sum + entry.file.size, 0);
    const listing = files.map(entry => ({ path: entry.path, size: entry.file.size }));
    const signature = list => list.map(item => `${item.path}:${item.size}`).sort().join("|");
    try {
      const pending = (await fetch("/api/v1/uploads").then(r => r.json())).items;
      const session = pending.find(item => signature(item.files) === signature(listing)) || await api("/uploads", post({ files: listing }));
      const received = Object.fromEntries(session.files.map(item => [item.path, item.received]));
      let done = session.received;
      const began = Date.now(), resumed = done;
      const report = extra => {
        const speed = (done + extra - resumed) / Math.max(0.5, (Date.now() - began) / 1000);
        setUpload({ id: session.id, done: done + extra, total, speed, eta: (total - done - extra) / speed, resumed: resumed > 0 });
      };
      for (const { path, file } of files) {
        for (let offset = received[path], failures = 0; offset < file.size;) {
          if (paused.current) throw fail("Upload paused", { cancelled: true });
          const url = `/api/v1/uploads/${session.id}/chunk?path=${encodeURIComponent(path)}&offset=${offset}`;
          try {
            const body = await sendChunk(url, file.slice(offset, offset + CHUNK), headers(), report, abort);
            done += body.received - offset;
            offset = body.received;
            failures = 0;
          } catch (e) {
            if (e.cancelled || !e.retry || ++failures > 6) throw e;
            await new Promise(ok => setTimeout(ok, 1000 * failures));
            const now = (await fetch(`/api/v1/uploads/${session.id}`).then(r => r.json())).files.find(item => item.path === path).received;
            done += now - offset;
            offset = now;
          }
        }
      }
      setUpload(old => ({ ...old, installing: true }));
      waiting.current = true;
      await api(`/uploads/${session.id}/install`, post({ activate }));
    } catch (e) {
      waiting.current = false;
      setUpload(null);
      setError(`${e.message}. Choose the same files again to resume.`);
    }
  };

  const pause = () => { paused.current = true; abort.current?.(); };
  const discard = () => { pause(); upload && api(`/uploads/${upload.id}/cancel`, post()).catch(() => {}); };
  const installing = job?.kind === "install" && job.status === "running" ? job : upload?.installing && { stage: "starting" };
  const busy = Boolean(upload || installing);
  const note = (tone, text) => <p class={`note ${tone}-note`}>{text}</p>;

  return <section class="card installer">
    <div class="cardhead"><div><h3>Install a bundle</h3><p>Uploaded from this browser and checked with the same ELF checksums as polima deploy.</p></div></div>
    <div class={`dropzone ${dragging ? "dragging" : ""} ${busy ? "busy" : ""}`} onDragOver={e => { e.preventDefault(); setDragging(!busy); }} onDragLeave={() => setDragging(false)} onDrop={async e => { e.preventDefault(); setDragging(false); if (!busy) choose(await fromDrop(e.dataTransfer)); }}>
      {!picked && !busy && <>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>
        <b>Drop a bundle folder or .tar here</b>
        <p>Copy a folder from <code>outputs/bundles</code> on the build host to this computer, then drop it here.</p>
        <div class="dropactions"><button onClick={() => folderInput.current.click()}>Choose Folder</button><button onClick={() => archiveInput.current.click()}>Choose .tar</button></div>
      </>}
      {picked && !busy && <div class="picked">
        <b class="mono">{picked.name}</b>
        <small>{picked.files.length} files · {bytes(picked.files.reduce((sum, entry) => sum + entry.file.size, 0))}{picked.skipped ? ` · skipping ${picked.skipped} ONNX, cache and archive files` : ""}</small>
        <label class="inline-check"><input type="checkbox" checked={activate} onChange={e => setActivate(e.target.checked)}/> Activate after install (only when nothing is running)</label>
        <div class="dropactions">{controller ? <button class="primary" onClick={start}>Upload And Install</button> : <button class="primary" onClick={onClaim}>Take Control To Install</button>}<button onClick={() => setPicked(null)}>Clear</button></div>
      </div>}
      {upload && !upload.installing && <Progress label={`${upload.resumed ? "Resuming" : "Uploading"} ${picked?.name}`} done={upload.done} total={upload.total}>
        <small>{upload.speed ? `${bytes(upload.speed)}/s · about ${duration(upload.eta)} left` : "Measuring speed…"}</small>
        <div class="dropactions"><button onClick={pause}>Pause</button><button onClick={discard}>Cancel And Discard</button></div>
      </Progress>}
      {installing && <Progress label={stages[installing.stage] || installing.stage} done={installing.done} total={installing.total}/>}
    </div>
    <input ref={folderInput} type="file" webkitdirectory multiple hidden onChange={e => { choose(fromInput(e.target.files)); e.target.value = ""; }}/>
    <input ref={archiveInput} type="file" accept=".tar,.tgz,.gz" hidden onChange={e => { choose(fromInput(e.target.files)); e.target.value = ""; }}/>
    {error && <div class="inline-error">{error}</div>}
    {result && <div class="install-result">
      <p><b>Installed <span class="mono">{result.bundle}</span></b> · {result.graphs} graphs verified{result.replaced ? " · replaced the previous copy" : ""}{result.activated ? " · activated" : ""}</p>
      {result.kept_calibration.length > 0 && note("ok", `Kept this board's calibration (${result.kept_calibration.join(", ")}).`)}
      {result.activation_skipped && note("warn", `${result.activation_skipped}.`)}
      {result.activation_error && note("warn", `Activation failed: ${result.activation_error}`)}
      {result.warnings.map(text => note("warn", text))}
      {result.calibration === "same" && note("ok", "Follower calibration matches the active bundle.")}
      {["missing", "different"].includes(result.calibration) && <p class="note warn-note">{result.calibration === "missing" ? "This bundle has no follower calibration." : "Its follower calibration differs from the active bundle's."} If it drives the same arm, copy the calibration in use (the current file is backed up first).<button disabled={!controller || !idle} onClick={() => onCopyCalibration(result.bundle).then(() => setResult({ ...result, calibration: "same" }), e => setError(e.message))}>Copy Active Calibration</button></p>}
      <div class="dropactions"><button class="primary" onClick={() => onSelect(result.bundle)}>Use For Next Run</button><button onClick={() => setResult(null)}>Done</button></div>
    </div>}
  </section>;
}

const compared = {
  same: ["ok", "Matches the active bundle"], different: ["bad", "Differs from the active bundle"], missing: ["bad", "Missing"],
  "is-active": ["ok", "In use by the active bundle"], "no-active": ["", "No active bundle to compare"], "no-active-calibration": ["", "The active bundle has none"],
};

function BundleRow({ bundle, api, controller, idle, job, selected, onSelect, onActivate, onCopyCalibration, onError }) {
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState(null);
  const calibrationInput = useRef(null);
  const url = `/bundles/${encodeURIComponent(bundle.id)}`;
  const load = () => fetch(`/api/v1${url}/details`).then(r => r.ok && r.json()).then(body => body && setDetails(body));
  useEffect(() => { if (open) load(); }, [open, bundle.active, job?.status]);

  const act = fn => async () => { onError(""); try { await fn(); await load(); } catch (e) { onError(e.message); } };
  const remove = () => confirm(`Delete ${bundle.id} from the board? Its calibration is backed up first.`) && api(`${url}/delete`, post()).catch(e => onError(e.message));
  const uploadCalibration = file => act(async () => api(`${url}/calibration`, post({ calibration: JSON.parse(await file.text()) })))();
  const verified = job?.kind === "verify" && job.result?.bundle === bundle.id ? job.result : null;
  const calibration = details?.calibration;
  const [tone, label] = compared[calibration?.compared_to_active] || ["", "—"];
  const can = controller && idle;

  return <div class={`bundle-row ${selected ? "selected" : ""}`}>
    <button class="bundle-summary" onClick={() => setOpen(!open)} aria-expanded={open}>
      <span class="chev">{open ? "▾" : "▸"}</span><span class="mono bundle-name">{bundle.id}</span><span class="badge">{bundle.policy}</span>
      {bundle.active && <span class="badge success">Active</span>}{selected && <span class="badge">Selected</span>}
    </button>
    {open && <div class="bundle-details">{!details ? <small>Loading…</small> : <>
      <dl>
        <div><dt>Size</dt><dd>{bytes(details.size_bytes)} · {details.files} files · {when(details.modified)}</dd></div>
        <div><dt>Graphs</dt><dd>{details.graphs.map(g => `${g.name} (${g.precision})`).join(", ") || "Hand-built deploy root"}</dd></div>
        <div><dt>Robot client</dt><dd class={details.robot_client ? "" : "bad"}>{details.robot_client ? "Included" : "Missing: benchmark only"}</dd></div>
        <div><dt>Calibration</dt><dd class={tone}>{!calibration.applicable ? `Handled by the ${calibration.robot} client` : calibration.present ? label : "Missing"}</dd></div>
        {details.checkpoint && <div><dt>Checkpoint</dt><dd class="mono">{details.checkpoint}</dd></div>}
      </dl>
      {job?.kind === "verify" && job.status === "running" && <Progress label="Checking ELF checksums" done={job.done} total={job.total}/>}
      {verified && <p class={`note ${verified.ok ? "ok" : "warn"}-note`}>{verified.ok ? `All ${verified.graphs} ELF checksums match bundle.json.` : verified.problems.join("; ")}</p>}
      <div class="bundle-actions">
        <button onClick={() => onSelect(bundle.id)} disabled={selected}>Use For Next Run</button>
        <button onClick={onActivate} disabled={!can || bundle.active}>Activate</button>
        <button onClick={act(() => api(`${url}/verify`, post()))} disabled={!controller || !details.managed || job?.status === "running"}>Check Integrity</button>
        {calibration.present && <a class="button" href={`/api/v1${url}/calibration`}>Download Calibration</a>}
        {calibration.applicable && <button onClick={() => calibrationInput.current.click()} disabled={!can}>Upload Calibration</button>}
        {calibration.applicable && calibration.active_bundle && !bundle.active && <button onClick={act(() => onCopyCalibration(bundle.id))} disabled={!can}>Copy Active Calibration</button>}
        {details.managed && !bundle.active && <button class="danger-outline" onClick={remove} disabled={!can}>Delete</button>}
      </div>
      <input ref={calibrationInput} type="file" accept=".json" hidden onChange={e => { e.target.files[0] && uploadCalibration(e.target.files[0]); e.target.value = ""; }}/>
    </>}</div>}
  </div>;
}

export function BundleLibrary({ snapshot, ...props }) {
  const [error, setError] = useState("");
  const { free_bytes: free, total_bytes: total } = snapshot.storage;
  return <section class="card library">
    <div class="cardhead"><div><h3>Installed bundles</h3><p>{snapshot.bundles.length} in the model store</p></div>
      <div class="disk"><small>{bytes(free)} free</small><div class="bar"><span class={free < total / 10 ? "full" : ""} style={{ width: `${(1 - free / total) * 100}%` }}></span></div></div>
    </div>
    {error && <div class="inline-error">{error}</div>}
    <div class="bundle-list">{snapshot.bundles.length ? snapshot.bundles.map(bundle => <BundleRow key={bundle.id} {...props} bundle={bundle} job={snapshot.job} selected={bundle.id === props.selected} onActivate={() => props.onActivate(bundle.id)} onError={setError}/>) : <div class="empty-small"><b>No bundles installed</b><p>Install one with the panel on the left.</p></div>}</div>
  </section>;
}

const places = { logs: "Logs", "calibration-backups": "Calibration backups", models: "Installed bundles" };

export function DeviceFiles() {
  const [roots, setRoots] = useState([]);
  const [root, setRoot] = useState("logs");
  const [path, setPath] = useState("");
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState("");
  const load = () => fetch(`/api/v1/files/${root}/list?path=${encodeURIComponent(path)}`).then(async r => {
    const body = await r.json();
    r.ok ? (setEntries(body.entries), setError("")) : setError(body.error);
  });
  useEffect(() => { fetch("/api/v1/files").then(r => r.json()).then(body => setRoots(body.items)); }, []);
  useEffect(() => { load(); }, [root, path]);

  const current = roots.find(item => item.key === root);
  const parts = path ? path.split("/") : [];
  const join = name => path ? `${path}/${name}` : name;
  const href = sub => `/api/v1/files/${root}/download?path=${encodeURIComponent(sub)}`;

  return <section class="card device-files">
    <div class="cardhead"><div><h3>Download from the devkit</h3><p>Logs, calibration backups and bundles. Folders download as .tar.</p></div><button onClick={load}>Refresh</button></div>
    <div class="root-tabs">{roots.map(item => <button key={item.key} class={item.key === root ? "active" : ""} title={item.path} onClick={() => { setRoot(item.key); setPath(""); }}>{places[item.key] || item.key}</button>)}</div>
    {current && <div class="crumbs">
      <button onClick={() => setPath("")}>{current.path}</button>
      {parts.map((part, index) => <span key={index}>/<button onClick={() => setPath(parts.slice(0, index + 1).join("/"))}>{part}</button></span>)}
      {entries.length > 0 && <a class="button" href={href(path)} download>Download {path ? "Folder" : "All"}</a>}
    </div>}
    {error && <div class="inline-error">{error}</div>}
    <div class="file-list">
      {!error && !entries.length && <div class="empty-small"><b>Nothing here yet</b></div>}
      {entries.map(entry => <div key={entry.name} class="file-row">
        {entry.dir ? <button class="file-name" onClick={() => setPath(join(entry.name))}>📁 {entry.name}</button> : <span class="file-name">{entry.name}</span>}
        <small>{entry.dir ? "folder" : bytes(entry.size)}</small><small>{when(entry.modified)}</small>
        <a class="button" href={href(join(entry.name))} download>{entry.dir ? ".tar" : "Download"}</a>
      </div>)}
    </div>
  </section>;
}
