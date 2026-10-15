import test from 'node:test';
import assert from 'node:assert/strict';
import { initialConfig } from './selection.js';
const data = { bundles: [{id:'act', policy:'act'}, {id:'groot', policy:'groot', active:true, default_task:'pick'}], hardware:{arms:[], cameras:[]}, last_run_config:{bundle:'act', task:'old task', fps:15, autocomplete:true} };
test('active GR00T takes precedence over saved ACT and its motion settings', () => {
  const result = initialConfig(data, {fps:null, speed:'fast'});
  assert.equal(result.bundle, 'groot');
  assert.equal(result.task, 'pick');
  assert.equal(result.fps, null);
  assert.equal(result.autocomplete, null);
});
test('saved settings apply when they belong to the selected active bundle', () => {
  const result = initialConfig({...data, last_run_config:{bundle:'groot',task:'custom',fps:15}}, {});
  assert.equal(result.task, 'custom');
  assert.equal(result.fps,15);
});
test('empty bundle list has no stale selection', () => {
  assert.equal(initialConfig({...data,bundles:[]}, {}).bundle, '');
});
