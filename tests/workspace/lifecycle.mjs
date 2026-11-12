// Real CLI and native-request lifecycle over an isolated Ster workspace.
// Inputs name the candidate, executable digest and a real canonical pair set.
// This source does not build Ster or substitute another implementation.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { isAbsolute, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inspect } from 'node:util';

// Fixed result/process identities: https://ster.wisent.com/docs/request-api#events-and-exit-status
const SUCCESS = 0;
// Fixed result/process identities: https://ster.wisent.com/docs/request-api#events-and-exit-status
const FAILURE = 1;
// Fixed result/process identities: https://ster.wisent.com/docs/request-api#events-and-exit-status
const USAGE = 2;
const repository = fileURLToPath(new URL('../../', import.meta.url));
function required(name) {
  const value = process.env[name];
  assert.ok(value, `${name} is required; state the real candidate input before running this journey`);
  return value;
}
const binary = realpathSync(required('STER'));
const revision = required('STER_TEST_SOURCE_REVISION');
const expectedDigest = required('STER_TEST_BINARY_SHA256');
const source = realpathSync(required('STER_TEST_PAIRS'));
const sourceBytes = readFileSync(source);
const input = JSON.parse(sourceBytes);
assert.equal(typeof input.trait_name, 'string', 'STER_TEST_PAIRS must explicitly name its trait');
assert.ok(Array.isArray(input.pairs), 'STER_TEST_PAIRS must be a canonical pair set');
const root = join(repository, '.build', 'workspace', randomUUID());
const fixture = join(root, 'fixture');
const home = join(fixture, 'home');
const data = join(fixture, 'data');
const stateHome = join(fixture, 'state');
const manifest = join(data, 'ster', 'workspace.json');
mkdirSync(home, { recursive: true });
mkdirSync(data, { recursive: true });
const isolatedEnvironment = { HOME: home, XDG_DATA_HOME: data, XDG_STATE_HOME: stateHome };
const environment = { ...process.env, ...isolatedEnvironment };
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { revision, binary, expectedDigest, observedDigest: digest(readFileSync(binary)), source, environment: isolatedEnvironment, commands: [], states: [], verdict: 'running' };
const detailed = error => inspect(error, { depth: null, maxArrayLength: null, maxStringLength: null, customInspect: false });

async function capture(program, args, stdin = '') {
  const command = { program, args, cwd: repository, stdin };
  const stdout = [], stderr = [];
  await new Promise(resolve => {
    const child = spawn(program, args, { cwd: repository, env: environment, stdio: ['pipe', 'pipe', 'pipe'] });
    child.stdout.on('data', chunk => stdout.push(chunk));
    child.stderr.on('data', chunk => stderr.push(chunk));
    child.on('error', error => { command.error = detailed(error); });
    child.stdin.on('error', error => { command.stdinError = detailed(error); });
    child.on('close', (status, signal) => {
      Object.assign(command, { status, signal, stdout: Buffer.concat(stdout).toString('utf8'), stderr: Buffer.concat(stderr).toString('utf8') });
      resolve();
    });
    child.stdin.end(stdin);
  });
  report.commands.push(command);
  assert.equal(command.error, undefined, `cannot start ${program}: ${command.error}`);
  assert.equal(command.stdinError, undefined, `cannot send input to ${program}: ${command.stdinError}`);
  assert.equal(command.signal, null, `${program} ended by ${command.signal}`);
  return command;
}
async function cli(args, status, stdin = '') {
  const result = await capture(binary, args, stdin);
  assert.equal(result.status, status, `${args.join(' ')}: ${result.stderr}\n${result.stdout}`);
  return result;
}
async function request(operation, body, status) {
  const result = await cli(['request', operation], status, typeof body === 'string' ? body : JSON.stringify(body));
  const events = result.stdout.split('\n').filter(line => line.trim()).map(line => JSON.parse(line));
  const final = events.at(-1);
  assert.deepEqual(events.filter(event => event.type === 'result'), [final], 'exactly one final result is required');
  assert.equal(final.type, 'result');
  assert.equal(final.status, status, 'wire status must describe the same outcome as the process');
  return { document: final.json, events };
}
function state(label) {
  const bytes = readFileSync(manifest, 'utf8');
  const document = JSON.parse(bytes);
  const copies = document.pair_sets.map(entry => {
    const path = realpathSync(entry.path);
    const child = relative(data, path);
    assert.ok(child && !child.startsWith('..') && !isAbsolute(child), `workspace copy escaped isolation: ${path}`);
    return { path, contents: readFileSync(path, 'utf8') };
  });
  const snapshot = { manifest: bytes, copies };
  report.states.push({ label, ...snapshot });
  return snapshot;
}
function unchanged(before, label) {
  assert.deepEqual(state(label), before, `${label} changed durable workspace state`);
  assert.deepEqual(readFileSync(source), sourceBytes, `${label} changed the caller's source`);
}
const summary = async () => JSON.parse((await cli(['workspace', 'show'], SUCCESS)).stdout);
let failure;
try {
  assert.equal(report.observedDigest, expectedDigest, 'the executable differs from the declared candidate artifact');
  const head = await capture('git', ['rev-parse', 'HEAD']);
  assert.equal(head.status, SUCCESS);
  assert.equal(head.stdout.trim(), revision, 'the checkout differs from the declared source revision');
  const clean = await capture('git', ['status', '--porcelain']);
  assert.equal(clean.status, SUCCESS);
  assert.equal(clean.stdout, '', 'qualification requires the actual clean committed candidate');
  const ignored = await capture('git', ['check-ignore', fixture]);
  assert.equal(ignored.status, SUCCESS, 'the fixture must be under the checkout ignored build directory');
  const first = `first-${randomUUID()}`, second = `second-${randomUUID()}`;
  const local = join(fixture, 'source.json');
  writeFileSync(local, sourceBytes);
  const accepted = JSON.parse((await cli(['workspace', 'import', local, '--name', first], SUCCESS)).stdout);
  assert.equal(accepted.status, 'imported');
  assert.equal(accepted.id, first);
  assert.equal(accepted.pair_count, input.pairs.length);
  const persisted = JSON.parse(readFileSync(accepted.path, 'utf8'));
  assert.deepEqual(persisted.pairs, input.pairs);
  assert.equal(persisted.trait_name, input.trait_name);
  assert.equal((await summary()).active_pair_set, first);
  const imported = state('first imported');
  const onboarding = JSON.parse(readFileSync(join(stateHome, 'ster', 'first-use.json'), 'utf8'));
  assert.equal(onboarding.status, 'completed');
  assert.equal(onboarding.evidence.contrastive_pair_set_imported, true);
  assert.equal(onboarding.evidence.pair_set_path, accepted.path);
  report.onboarding = onboarding;

  const repeated = JSON.parse((await cli(['workspace', 'import', local, '--name', second], SUCCESS)).stdout);
  assert.equal(repeated.status, 'unchanged');
  assert.equal(repeated.id, first);
  assert.equal(repeated.path, accepted.path);
  unchanged(imported, 'repeated content');
  await cli(['workspace', 'import-pairs', local], USAGE);
  unchanged(imported, 'retired CLI command');
  const retired = await request('workspace/import-pairs', { source: local }, USAGE);
  assert.equal(retired.document.error, 'unknown operation: workspace/import-pairs');
  unchanged(imported, 'retired request operation');
  const missing = await request('workspace/import', {}, USAGE);
  assert.equal(missing.document.error, 'workspace import requires a source path');
  const malformed = await request('workspace/import', '{', USAGE);
  assert.ok(malformed.document.error.startsWith('request body is not valid JSON:'), malformed.document.error);
  unchanged(imported, 'invalid request bodies');
  await cli(['workspace', 'import'], USAGE);
  unchanged(imported, 'missing CLI source');

  const changed = structuredClone(input);
  for (const pair of changed.pairs) pair.positive += ` ${randomUUID()}`;
  const other = join(fixture, 'other.json');
  writeFileSync(other, JSON.stringify(changed));
  const conflicting = await request('workspace/import', { source: other, name: first }, FAILURE);
  assert.equal(conflicting.document.status, 'conflicting');
  assert.equal(conflicting.document.path, accepted.path);
  assert.equal(conflicting.document.reason, `pair-set name conflicts with existing content; choose another --name to preserve ${accepted.path}`);
  assert.ok(conflicting.events.some(event => event.type === 'log' && event.chunk === `error: ${conflicting.document.reason}\n`));
  unchanged(imported, 'conflicting API import');
  const invalid = join(fixture, 'invalid.json');
  writeFileSync(invalid, '{');
  const rejected = await request('workspace/import', { source: invalid }, FAILURE);
  assert.equal(rejected.document.status, 'rejected');
  assert.ok(rejected.document.reason.includes(invalid), rejected.document.reason);
  unchanged(imported, 'rejected API import');
  const rejectedCli = JSON.parse((await cli(['workspace', 'import', invalid], FAILURE)).stdout);
  assert.equal(rejectedCli.status, 'rejected');
  unchanged(imported, 'rejected CLI import');

  const next = await request('workspace/import', { source: other, name: second }, SUCCESS);
  assert.equal(next.document.status, 'imported');
  assert.equal(next.document.id, second);
  assert.deepEqual(JSON.parse(readFileSync(next.document.path, 'utf8')).pairs, changed.pairs);
  assert.equal((await summary()).active_pair_set, second);
  const selected = await request('workspace/select', { id: first }, SUCCESS);
  assert.equal(selected.document.active_pair_set, first);
  assert.equal(JSON.parse(readFileSync(manifest, 'utf8')).active_pair_set, first);
  const selectedState = state('first selected');
  const duplicateSelection = await cli(['workspace', 'select', first], FAILURE);
  assert.ok(duplicateSelection.stderr.includes(`pair set ${first} is already the active one`));
  unchanged(selectedState, 'already active selection');
  await cli(['workspace', 'remove', first], SUCCESS);
  assert.equal(existsSync(accepted.path), false, 'removal left its managed copy');
  const remaining = await summary();
  assert.equal(remaining.active_pair_set, null, 'removal selected a replacement nobody requested');
  assert.deepEqual(remaining.pair_sets.map(entry => entry.id), [second]);
  await request('workspace/remove', { id: second }, SUCCESS);
  assert.equal(existsSync(next.document.path), false);
  const empty = await summary();
  assert.deepEqual(empty.pair_sets, []);
  assert.equal(empty.active_pair_set, null);
  const final = state('all imported sets removed');
  const unknown = await cli(['workspace', 'select', first], FAILURE);
  assert.ok(unknown.stderr.includes(`the Ster workspace holds no pair set named ${first}; it holds none — import one with \`ster workspace import\``));
  unchanged(final, 'unknown id after removal');
  assert.deepEqual(readFileSync(local), sourceBytes);
  report.verdict = 'passed';
} catch (error) {
  failure = error;
  report.verdict = 'failed';
  report.error = detailed(error);
} finally {
  try {
    rmSync(fixture, { recursive: true, force: true });
    assert.equal(existsSync(fixture), false, 'the isolated fixture was not removed');
    report.fixtureRemoved = true;
  } catch (error) {
    report.verdict = 'failed';
    report.cleanupError = detailed(error);
    failure = failure ? new AggregateError([failure, error], 'workspace journey and cleanup failed') : error;
  }
  writeFileSync(join(root, 'report.json'), JSON.stringify(report, null, '\t'));
  console.log(`${report.verdict}: ${join(root, 'report.json')}`);
}
if (failure) throw failure;
