// Real evaluation of a caller-supplied checkpoint, adapter and held-out examples.
// Mutates only owned copies of the adapter pair; never changes the input files.
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { copyFile, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { dirname, join, parse, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const required = name => {
  const value = process.env[name];
  assert.ok(value?.trim(), `${name} is required`);
  return value;
};
const repository = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const binary = resolve(required('STER'));
const revision = required('STER_TEST_SOURCE_REVISION');
// The status a program that succeeded exits with, measured on this machine
// from `true` rather than asked of whoever runs the test.
const success = spawnSync('true').status;
assert.ok(Number.isInteger(success), 'true did not exit with a status');
const adapter = resolve(required('STER_TEST_ADAPTER'));
const sidecar = join(dirname(adapter), `${parse(adapter).name}.json`);
assert.notEqual(adapter, sidecar, 'name the safetensors weights, not the sidecar');
const model = required('STER_TEST_MODEL');
const examples = resolve(required('STER_TEST_EXAMPLES'));
const limits = ['--max-sequence', required('STER_TEST_MAX_SEQUENCE'), '--batch-size', required('STER_TEST_BATCH_SIZE')];
const root = join(repository, 'target', 'real-tests', 'provenance', randomUUID());
await mkdir(root, { recursive: true });
const owned = join(root, 'inputs');
await mkdir(owned);
const weights = join(owned, 'adapter.safetensors');
const metadata = join(owned, 'adapter.json');
const report = { revision, binary, model, examples, commands: [], checks: [], verdict: 'failed' };
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

async function command(program, args) {
  const child = spawn(program, args, { cwd: repository, stdio: ['ignore', 'pipe', 'pipe'] });
  const stdout = [];
  const stderr = [];
  const errors = [];
  child.stdout.on('data', chunk => stdout.push(chunk));
  child.stderr.on('data', chunk => stderr.push(chunk));
  child.stdout.on('error', error => errors.push(String(error)));
  child.stderr.on('error', error => errors.push(String(error)));
  child.on('error', error => errors.push(String(error)));
  const result = await new Promise(resolve => child.once('close', (status, signal) => {
    resolve({ program, args, status, signal, stdout: Buffer.concat(stdout).toString(), stderr: Buffer.concat(stderr).toString(), errors });
  }));
  report.commands.push(result);
  assert.deepEqual(errors, [], 'the process and both output streams must complete');
  assert.equal(result.signal, null, 'a signal is not a product refusal');
  return result;
}
const evaluate = () => command(binary, ['tune', 'evaluate', '--model', model, '--examples', examples,
  '--adapter', weights, ...limits]);
const check = (name, condition) => {
  assert.ok(condition, name);
  report.checks.push(name);
};

try {
  const head = await command('git', ['rev-parse', 'HEAD']);
  assert.equal(head.status, success);
  assert.equal(head.stdout.trim(), revision);
  const clean = await command('git', ['status', '--porcelain']);
  assert.equal(clean.status, success);
  assert.equal(clean.stdout, '', 'commit the qualification source before execution');
  report.binarySha256 = hash(await readFile(binary));
  const originalWeights = hash(await readFile(adapter));
  const originalMetadata = await readFile(sidecar);
  report.inputSha256 = { adapter: originalWeights, sidecar: hash(originalMetadata), examples: hash(await readFile(examples)) };
  await copyFile(adapter, weights);
  await writeFile(metadata, originalMetadata);
  const baseline = await evaluate();
  assert.equal(baseline.status, success, baseline.stderr);
  const baselineReport = JSON.parse(baseline.stdout);
  check('baseline evaluates every supplied held-out example', baselineReport.evaluated === JSON.parse(await readFile(examples)).examples.length);
  check('baseline has a finite measured loss', Number.isFinite(baselineReport.loss));

  await rm(metadata);
  await symlink(metadata, metadata);
  const loop = await evaluate();
  assert.notEqual(loop.status, success);
  check('sidecar inspection refuses with its path and operating-system cause',
    loop.stderr.includes(metadata) && loop.stderr.includes('failed to inspect adapter sidecar') && loop.stderr.includes('os error'));
  assert.equal(loop.stdout, '');
  await rm(metadata);
  await mkdir(metadata);
  const directory = await evaluate();
  assert.notEqual(directory.status, success);
  check('sidecar read refuses with its path and operating-system cause',
    directory.stderr.includes(metadata) && directory.stderr.includes('failed to read adapter sidecar') && directory.stderr.includes('os error'));
  assert.equal(directory.stdout, '');
  await rm(metadata, { recursive: true });
  await writeFile(metadata, '{');
  const malformed = await evaluate();
  assert.notEqual(malformed.status, success);
  check('malformed metadata retains the parse location and path',
    malformed.stderr.includes(metadata) && malformed.stderr.includes('invalid adapter sidecar JSON') && malformed.stderr.includes('line') && malformed.stderr.includes('column'));
  assert.equal(malformed.stdout, '');

  await writeFile(metadata, originalMetadata);
  const recovered = await evaluate();
  assert.equal(recovered.status, success, recovered.stderr);
  assert.deepEqual(JSON.parse(recovered.stdout), baselineReport, 'restored adapter metadata recovers the exact evaluation');
  assert.equal(hash(await readFile(adapter)), originalWeights, 'input weights changed');
  assert.deepEqual(await readFile(sidecar), originalMetadata, 'input sidecar changed');
  assert.equal(hash(await readFile(weights)), originalWeights, 'read failures changed owned weights');
  report.verdict = 'passed';
} catch (error) {
  report.error = String(error.stack ?? error);
} finally {
  try { await rm(owned, { recursive: true }); }
  catch (error) { report.cleanupError = String(error); report.verdict = 'failed'; }
  await writeFile(join(root, 'report.json'), JSON.stringify(report, null, '\t'));
}
assert.equal(report.verdict, 'passed', JSON.stringify(report));
