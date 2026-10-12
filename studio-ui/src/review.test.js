// node --test studio-ui/src  (run by tests/unit/test_studio_ui.py)
import assert from "node:assert/strict";
import { test } from "node:test";
import { createApi } from "./api.js";
import { fetchManifest, formFrom, manifestFilename, manifestPath, manifestSummary, pendingNote, reviewPayload, reviewState, saveReview } from "./review.js";

function fakeFetch(status, body) {
  const calls = [];
  const fetchImpl = async (url, options) => {
    calls.push({ url, options });
    return { ok: status < 400, status, json: async () => body };
  };
  return { calls, fetchImpl };
}

const form = { outcome: "failed", failure_tags: ["missed_grasp"], notes: "closed beside it",
  ranges: '[{"unit": "time", "start": 4, "end": 7}]' };

test("a saved review is posted with the session token and shows as reviewed", async () => {
  const summary = { run_id: "20261002-120000-abcdef", status: "completed", outcome: "failed" };
  const { calls, fetchImpl } = fakeFetch(200, summary);
  const api = createApi(fetchImpl, () => ({ "X-CSRF-Token": "token-1" }));

  const saved = await saveReview(api, summary.run_id, form);

  assert.equal(calls[0].url, "/api/v1/robot-runs/20261002-120000-abcdef/annotation");
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.headers["X-CSRF-Token"], "token-1");
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    outcome: "failed", failure_tags: ["missed_grasp"], notes: "closed beside it",
    selected_ranges: [{ unit: "time", start: 4, end: 7 }],
  });
  assert.equal(reviewState(saved).label, "Reviewed: failed");
});

test("a refused review rejects with Studio's own message", async () => {
  const { fetchImpl } = fakeFetch(400, { error: "unknown failure tags ['bad']" });
  const api = createApi(fetchImpl, () => ({ "X-CSRF-Token": "token-1" }));
  await assert.rejects(saveReview(api, "20261002-120000-abcdef", form), /unknown failure tags/);
});

test("ranges that are not JSON are caught before anything is sent", async () => {
  let sent = false;
  const api = async () => { sent = true; };
  await assert.rejects(saveReview(api, "x", { ...form, ranges: "4 to 7" }), /must be JSON/);
  assert.equal(sent, false);
  assert.deepEqual(reviewPayload({ ...form, ranges: "  " }).selected_ranges, []);
});

test("finished runs nobody reviewed need review; live ones do not", () => {
  assert.equal(reviewState({ status: "interrupted", outcome: "unknown" }).label, "Needs review");
  assert.equal(reviewState({ status: "running", outcome: "unknown" }).label, "");
  assert.equal(reviewState({ status: "failed", outcome: "partial" }).label, "Reviewed: partial");
});

test("the form starts from the run's existing review", () => {
  const start = formFrom({ outcome: "partial", failure_tags: ["jitter"], notes: "n",
    selected_ranges: [{ unit: "event", start: 1, end: 2 }] });
  assert.equal(start.outcome, "partial");
  assert.deepEqual(JSON.parse(start.ranges), [{ unit: "event", start: 1, end: 2 }]);
  assert.equal(formFrom().outcome, "unknown");
});

const manifest = (counts, outcomes = ["failed", "partial"]) => ({
  schema_version: "polima-improvement-manifest-v1", created_at: 1790956800, source_board: "modalix",
  filters: { outcomes }, counts, runs: [] });

test("the export reads the existing manifest endpoint; Include successes adds ?all=1", async () => {
  assert.equal(manifestPath(false), "/api/v1/robot-runs/improvement-manifest");
  assert.equal(manifestPath(true), "/api/v1/robot-runs/improvement-manifest?all=1");
  const body = manifest({ total_runs_considered: 5, exported_runs: 2, failed: 1, partial: 1 });
  const { calls, fetchImpl } = fakeFetch(200, body);
  assert.deepEqual(await fetchManifest(fetchImpl, true), body);
  assert.equal(calls[0].url, "/api/v1/robot-runs/improvement-manifest?all=1");
});

test("the export says what it holds, or that there is nothing to export", () => {
  assert.equal(manifestSummary(manifest({ total_runs_considered: 5, exported_runs: 2, failed: 1, partial: 1 })),
    "Exported 2 of 5 runs: 1 failed, 1 partial");
  assert.equal(manifestSummary(manifest({ total_runs_considered: 4, exported_runs: 3, failed: 1, partial: 0, success: 2 },
    ["failed", "partial", "success"])), "Exported 3 of 4 runs: 1 failed, 0 partial, 2 success");
  assert.equal(manifestSummary(manifest({ total_runs_considered: 3, exported_runs: 0, failed: 0, partial: 0 })),
    "No reviewed failures to export");
  assert.equal(manifestFilename(manifest({})), "improvement-manifest-modalix-20261002T160000.json");
});

test("a refused export shows the server's message", async () => {
  const { fetchImpl } = fakeFetch(500, { error: "disk unreadable" });
  await assert.rejects(fetchManifest(fetchImpl, false), /disk unreadable/);
});

test("unreviewed finished runs prompt a note", () => {
  assert.equal(pendingNote([{ status: "completed", outcome: "unknown" }, { status: "failed", outcome: "failed" }]),
    "Some runs still need review");
  assert.equal(pendingNote([{ status: "running", outcome: "unknown" }]), "");
});
