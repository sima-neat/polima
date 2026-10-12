import { useEffect, useState } from "preact/hooks";
import { when } from "./files.jsx";
import { fetchManifest, formFrom, manifestFilename, manifestSummary, pendingNote, reviewState, saveReview } from "./review.js";

const label = tag => tag.replace(/_/g, " ");

export function RobotRunReviews({ api, refreshKey }) {
  const [runs, setRuns] = useState([]);
  const [choices, setChoices] = useState({ outcomes: [], failure_tags: [] });
  const [open, setOpen] = useState(null);
  const [form, setForm] = useState(null);
  const [status, setStatus] = useState(null);
  const [withSuccesses, setWithSuccesses] = useState(false);
  const [exported, setExported] = useState(null);

  const load = () => fetch("/api/v1/robot-runs?limit=12").then(r => r.json()).then(data => {
    setRuns(data.items || []);
    setChoices({ outcomes: data.outcomes || [], failure_tags: data.failure_tags || [] });
  }).catch(() => {});
  useEffect(() => { load(); }, [refreshKey]);

  const edit = run => { setOpen(run.run_id); setForm(formFrom(run.annotation)); setStatus(null); };
  // Updates build on the latest form, so changes made in quick succession all survive.
  const change = (field, value) => setForm(current => ({ ...current, [field]: value }));
  const toggle = tag => setForm(current => ({ ...current, failure_tags: current.failure_tags.includes(tag)
    ? current.failure_tags.filter(t => t !== tag) : [...current.failure_tags, tag] }));
  const save = async () => {
    setStatus({ saving: true });
    try {
      const summary = await saveReview(api, open, form);
      setRuns(list => list.map(run => run.run_id === summary.run_id ? summary : run));
      setStatus({ saved: `Saved. ${reviewState(summary).label}.` });
    } catch (error) {
      setStatus({ error: error.message });
    }
  };

  const exportManifest = async () => {
    try {
      const manifest = await fetchManifest(fetch, withSuccesses);
      setExported({ text: manifestSummary(manifest) });
      if (!manifest.counts?.exported_runs) return;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2) + "\n"], { type: "application/json" }));
      link.download = manifestFilename(manifest);
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    } catch (error) {
      setExported({ error: error.message });
    }
  };
  const pending = pendingNote(runs);

  return <section class="card reviews">
    <div class="cardhead"><div><h3>Robot run reviews</h3><p>Mark how each run went and why, for later improvement</p></div><button onClick={load}>Refresh</button></div>
    <div class="manifest-export">
      <button class="primary" onClick={exportManifest}>Export improvement manifest</button>
      <label class="inline-check"><input type="checkbox" checked={withSuccesses} onChange={e => setWithSuccesses(e.currentTarget.checked)}/>Include successes</label>
      {pending && <span class="field-help" data-pending>{pending}</span>}
      {exported?.text && <span class="note ok-note" role="status" data-manifest>{exported.text}</span>}
      {exported?.error && <span class="note warn-note" role="alert">{exported.error}</span>}
    </div>
    {runs.length ? <div class="history-list">{runs.map(run => {
      const review = reviewState(run);
      const latency = run.latency_ms?.model?.p50;
      return <div key={run.run_id} class="review-row">
        <div class="review-summary">
          <span class={`run-status ${run.status}`}></span>
          <div><b>{run.task?.instruction || run.bundle_id}</b><small>{run.bundle_id}</small>
            <small>{when(run.started_at)} · {run.status} · {run.cycles} cycles · {run.actions} actions{latency != null ? ` · ${latency} ms model` : ""}</small>
            {run.failure_tags?.length ? <small>Why: {run.failure_tags.map(label).join(", ")}</small> : null}</div>
          <span class="run-meta">{review.label && <span class={`badge ${review.kind}`} data-review={run.run_id}>{review.label}</span>}
            <button onClick={() => open === run.run_id ? setOpen(null) : edit(run)}>{open === run.run_id ? "Close" : "Review"}</button></span>
        </div>
        {open === run.run_id && form && <div class="review-form">
          <label>Outcome<select value={form.outcome} onChange={e => change("outcome", e.currentTarget.value)}>
            {choices.outcomes.map(outcome => <option value={outcome}>{outcome}</option>)}</select></label>
          <fieldset><legend>Why</legend>{choices.failure_tags.map(tag => <label class="inline-check" key={tag}>
            <input type="checkbox" checked={form.failure_tags.includes(tag)} onChange={() => toggle(tag)}/>{label(tag)}</label>)}</fieldset>
          <label>Notes<textarea value={form.notes} maxLength={4000} onInput={e => change("notes", e.currentTarget.value)}></textarea></label>
          <label>Selected ranges <span class="field-help">optional JSON, e.g. [{"{"}"unit": "time", "start": 4, "end": 7{"}"}]</span>
            <textarea value={form.ranges} onInput={e => change("ranges", e.currentTarget.value)}></textarea></label>
          <div class="toolrow-buttons"><button class="primary" disabled={status?.saving} onClick={save}>Save review</button><button onClick={() => setOpen(null)}>Cancel</button></div>
          {status?.saved && <p class="note ok-note" role="status">{status.saved}</p>}
          {status?.error && <p class="note warn-note" role="alert">{status.error}</p>}
        </div>}
      </div>;
    })}</div> : <div class="empty-small"><b>No robot runs recorded yet</b><p>Each run on the arm appears here for review.</p></div>}
  </section>;
}
