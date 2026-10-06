// Reviewing a robot run: what the operator marks, and what is sent to Studio.

const ENDED = ["completed", "failed", "interrupted"];

export function reviewState(run) {
  if (run.outcome && run.outcome !== "unknown") return { label: `Reviewed: ${run.outcome}`, kind: run.outcome };
  if (ENDED.includes(run.status)) return { label: "Needs review", kind: "pending" };
  return { label: "", kind: "" };
}

export function formFrom(annotation = {}) {
  return {
    outcome: annotation.outcome || "unknown",
    failure_tags: [...(annotation.failure_tags || [])],
    notes: annotation.notes || "",
    ranges: annotation.selected_ranges?.length ? JSON.stringify(annotation.selected_ranges, null, 2) : "",
  };
}

export function reviewPayload(form) {
  let selected_ranges = [];
  if (form.ranges.trim()) {
    try {
      selected_ranges = JSON.parse(form.ranges);
    } catch {
      throw new Error('Selected ranges must be JSON, for example [{"unit": "time", "start": 4, "end": 7}]');
    }
  }
  return { outcome: form.outcome, failure_tags: form.failure_tags, notes: form.notes, selected_ranges };
}

// Resolves with the run's updated summary; rejects with Studio's own error message.
export async function saveReview(api, runId, form) {
  return api(`/robot-runs/${encodeURIComponent(runId)}/annotation`, {
    method: "POST",
    body: JSON.stringify(reviewPayload(form)),
  });
}

// -- Improvement manifest: reviewed failures, ready for a later data-repair step --

export function manifestPath(includeSuccesses) {
  return `/api/v1/robot-runs/improvement-manifest${includeSuccesses ? "?all=1" : ""}`;
}

export async function fetchManifest(fetchImpl, includeSuccesses) {
  const response = await fetchImpl(manifestPath(includeSuccesses));
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Export failed (${response.status})`);
  return body;
}

export function manifestSummary(manifest) {
  const counts = manifest.counts || {};
  if (!counts.exported_runs) {
    return manifest.filters?.outcomes?.includes("success") ? "No reviewed runs to export" : "No reviewed failures to export";
  }
  const parts = ["failed", "partial", "success"].filter(o => o in counts).map(o => `${counts[o]} ${o}`);
  return `Exported ${counts.exported_runs} of ${counts.total_runs_considered} runs: ${parts.join(", ")}`;
}

export function manifestFilename(manifest) {
  const stamp = new Date((manifest.created_at || 0) * 1000).toISOString().replace(/[-:]/g, "").slice(0, 15);
  return `improvement-manifest-${manifest.source_board || "board"}-${stamp}.json`;
}

export function pendingNote(runs) {
  return runs.some(run => reviewState(run).label === "Needs review") ? "Some runs still need review" : "";
}
