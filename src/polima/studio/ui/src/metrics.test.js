import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { benchmarkSelection, meanChange } from './metrics.js';

const run = (bundle, mean, status = 'completed') => ({ bundle, status, result: mean == null ? null : { mean } });
test('benchmark cards follow the selected bundle, including its latest failure', () => {
  const records = [run('groot', 80), run('act', null, 'failed'), run('act', 20), run('act', 30)];
  assert.equal(benchmarkSelection(records, 'act').latest, records[1]);
  assert.equal(benchmarkSelection(records, 'missing').latest, undefined);
  assert.equal(benchmarkSelection(records, 'groot').previous, undefined);
});
test('comparison uses the preceding completed measurement of the same bundle', () => {
  const records = [run('act', 20), run('groot', 70), run('act', 10, 'failed'), run('act', 25)];
  assert.equal(benchmarkSelection(records, 'act').previous, records[3]);
  assert.equal(meanChange(records[0], records[3]), -20);
});
test('unavailable or zero baselines never produce misleading percentages', () => {
  for (const value of [0, null, undefined, NaN, Infinity, -1]) {
    assert.equal(meanChange(run('act', 20), run('act', value)), null);
  }
  assert.equal(meanChange(run('act', 20), run('groot', 30)), null);
  assert.equal(meanChange(run('act', null), run('act', 30)), null);
  assert.equal(meanChange(run('act', 30), run('act', 20)), 50);
  assert.equal(meanChange(run('act', 30, 'failed'), run('act', 20)), null);
});

function cards(builds) {
  const html = readFileSync(new URL('../../../workbench/static/index.html', import.meta.url), 'utf8');
  const script = html.match(/<script id="comparison-renderer">([\s\S]*?)<\/script>/);
  assert.ok(script, 'Workbench provides a measured comparison renderer');
  const context = {};
  vm.runInNewContext(script[1], context);
  return context.comparisonCards(builds);
}
test('comparison renders measured latency and throughput with board provenance', () => {
  const html = cards([{id:'act-a', policy:'ACT', evidence:{median_ms:20, p99_ms:25, rate_hz:50, board:'board-a', iterations:50, check:'pass'}}]);
  assert.match(html, /20\.0/);
  assert.match(html, /50\.0/);
  assert.match(html, /board-a/);
  assert.match(html, /50 runs/);
  assert.match(html, /Output verified/);
});
test('missing measurements remain unavailable; failed validation stays visible', () => {
  const html = cards([{id:'groot-a', policy:'GR00T', evidence:null}, {id:'groot-a', policy:'GR00T N1.6', evidence:{median_ms:40, rate_hz:null, check:'fail'}}]);
  assert.match(html, /Not measured/);
  assert.match(html, /Output check failed/);
  assert.doesNotMatch(html, /NaN|Infinity|0\.0 Hz/);
  assert.match(cards([]), /No measured bundles/);
});
test('comparison escapes bundle metadata and handles nonfinite measurements', () => {
  const html = cards([{id:'<img src=x onerror=alert(1)>', policy:'<script>', evidence:{median_ms:Infinity, rate_hz:-2, board:'<svg>'}}]);
  assert.doesNotMatch(html, /<img|<svg|<script|Infinity|NaN|width:-/);
  assert.match(html, /&lt;img/);
});

test('recorded event rates and gaps use timestamps, never configured command rate', async () => {
  const { eventTiming } = await import('./metrics.js');
  assert.deepEqual(eventTiming([], 'action_sent'), {rate:null, gap:null});
  const events = [{event:'action_sent',t:10},{event:'action_sent',t:10.1},{event:'inference_completed',t:10.2},{event:'action_sent',t:10.4}];
  assert.ok(Math.abs(eventTiming(events, 'action_sent').rate - 5) < 1e-8);
  assert.ok(Math.abs(eventTiming(events, 'action_sent').gap - 300) < 1e-8);
  assert.equal(eventTiming(events, 'inference_completed').rate, null);
  assert.equal(eventTiming([{event:'action_sent',t:1},{event:'action_sent',t:1}], 'action_sent').rate, null);
});

test('validation separates export fidelity, compiled fidelity and unavailable reports', () => {
  const html = readFileSync(new URL('../../../workbench/static/index.html', import.meta.url), 'utf8');
  const context = {};
  vm.runInNewContext(html.match(/<script id="comparison-renderer">([\s\S]*?)<\/script>/)[1], context);
  const rendered = context.validationTable([{id:'test-bundle',policy:'GR00T',precision:['bf16'],validation:{export:{ok:true,action_cosine_similarity:.99999946,max_abs:.001,source:'<export>'},compiled:{ok:false,cosine:.8,mean_abs:.04,source:'Normalized actions'}}}]);
  assert.match(rendered, /0\.99999946/);
  assert.match(rendered, /ONNX \/ PyTorch/);
  assert.match(rendered, /Compiled \/ PyTorch/);
  assert.match(rendered, /Fail/);
  assert.match(rendered, /&lt;export&gt;/);
  assert.doesNotMatch(rendered, /Runtime \/ reference/);
});
