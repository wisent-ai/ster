import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inspect } from 'node:util';

// Native process and Ster request status identities, not experiment settings:
// https://ster.wisent.com/docs/request-api#events-and-exit-status
export const SUCCESS = 0;
// https://ster.wisent.com/docs/request-api#events-and-exit-status
export const FAILURE = 1;
// https://ster.wisent.com/docs/request-api#events-and-exit-status
export const USAGE = 2;
export const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const detail = error => inspect(error, { depth: null, maxArrayLength: null, maxStringLength: null, customInspect: false });

export async function qualify(area, flow) {
    const repo = fileURLToPath(new URL('../..', import.meta.url));
    const build = join(repo, '.build');
    const root = resolve(build, area, randomUUID());
    assert.ok(root.startsWith(`${build}${sep}`), 'qualification output must remain under the checkout build directory');
    const fixture = join(root, 'fixture');
    const isolatedEnvironment = {
        HOME: join(fixture, 'home'),
        XDG_DATA_HOME: join(fixture, 'data'),
        XDG_STATE_HOME: join(fixture, 'state'),
        XDG_CACHE_HOME: join(fixture, 'cache'),
        TMPDIR: join(fixture, 'temporary'),
        HF_HOME: join(fixture, 'huggingface'),
        TORCH_HOME: join(fixture, 'torch'),
        TORCHINDUCTOR_CACHE_DIR: join(fixture, 'torchinductor'),
        TRITON_CACHE_DIR: join(fixture, 'triton'),
        CUDA_CACHE_PATH: join(fixture, 'cuda'),
        VLLM_CACHE_ROOT: join(fixture, 'vllm'),
    };
    mkdirSync(fixture, { recursive: true });
    for (const path of Object.values(isolatedEnvironment)) mkdirSync(path);
    const environment = { ...process.env, ...isolatedEnvironment };
    const report = { area, startedAt: new Date().toISOString(), environment: isolatedEnvironment, inputs: [], commands: [], states: [], verdict: 'running' };
    const input = name => {
        const value = process.env[name];
        assert.ok(typeof value === 'string' && value.trim() !== '', `qualification requires environment input ${name}`);
        return value;
    };
    async function capture(command, args, stdin, control) {
        assert.equal(typeof stdin, 'string', 'each command must declare its actual stdin');
        assert.ok(control === undefined || typeof control === 'function', 'process control must be an explicit child lifecycle function');
        const record = { command, args, stdin, stdout: '', stderr: '', status: null, signal: null, startError: null, stdinError: null, readErrors: [], controlError: null, controlStopError: null };
        report.commands.push(record);
        let controlTask;
        await new Promise(resolveCommand => {
            // These are native Node spawn stream identities:
            // https://nodejs.org/api/child_process.html#optionsstdio
            const child = spawn(command, args, { cwd: repo, env: environment, stdio: ['pipe', 'pipe', 'pipe'] });
            child.stdout.setEncoding('utf8');
            child.stderr.setEncoding('utf8');
            child.stdout.on('data', value => { record.stdout += value; });
            child.stderr.on('data', value => { record.stderr += value; });
            child.stdout.on('error', error => { record.readErrors.push({ stream: 'stdout', error: detail(error) }); });
            child.stderr.on('error', error => { record.readErrors.push({ stream: 'stderr', error: detail(error) }); });
            child.stdin.on('error', error => { record.stdinError = detail(error); });
            child.on('error', error => { record.startError = detail(error); });
            child.on('close', (status, signal) => {
                record.status = status;
                record.signal = signal;
                resolveCommand();
            });
            if (control) {
                controlTask = (async () => {
                    try {
                        await control(child);
                    } catch (error) {
                        record.controlError = detail(error);
                        if (child.pid !== undefined && child.exitCode === null && child.signalCode === null) {
                            try {
                                if (!child.kill('SIGTERM')) {
                                    throw new Error(`Cannot terminate controlled child ${child.pid}; exit=${child.exitCode}, signal=${child.signalCode}`);
                                }
                            } catch (stopError) {
                                record.controlStopError = detail(stopError);
                            }
                        }
                    }
                })();
            }
            child.stdin.end(stdin);
        });
        if (controlTask) await controlTask;
        return record;
    }
    async function expect(command, args, stdin, expectedStatus) {
        const record = await capture(command, args, stdin);
        record.expectedStatus = expectedStatus;
        assert.equal(record.startError, null, `cannot start ${command}: ${record.startError}`);
        assert.equal(record.stdinError, null, `cannot write ${command} stdin: ${record.stdinError}`);
        assert.deepEqual(record.readErrors, [], `cannot retain complete ${command} output`);
        assert.equal(record.controlError, null, `process lifecycle observation failed: ${record.controlError}`);
        assert.equal(record.controlStopError, null, `controlled child cleanup failed: ${record.controlStopError}`);
        assert.equal(record.signal, null, `${command} ended from signal ${record.signal}`);
        assert.equal(record.status, expectedStatus, `${command} ${args.join(' ')}: ${record.stderr}\n${record.stdout}`);
        return record;
    }
    let failure;
    try {
        const revision = input('STER_TEST_SOURCE_REVISION');
        const binary = realpathSync(input('STER'));
        const expectedDigest = input('STER_TEST_BINARY_SHA256');
        const observedDigest = digest(readFileSync(binary));
        report.artifact = { revision, binary, expectedDigest, observedDigest };
        assert.equal(observedDigest, expectedDigest, 'the executable differs from the declared qualification artifact');
        const head = await expect('git', ['rev-parse', 'HEAD'], '', SUCCESS);
        assert.equal(head.stdout.trim(), revision, 'the checkout differs from the artifact source revision');
        const clean = await expect('git', ['status', '--porcelain'], '', SUCCESS);
        assert.equal(clean.stdout, '', 'qualification requires the exact clean committed candidate');
        await expect('git', ['check-ignore', fixture], '', SUCCESS);
        const cli = (args, status) => expect(binary, args, '', status);
        const api = async (operation, body, status) => {
            const command = await expect(binary, ['request', operation], JSON.stringify(body), status);
            const events = command.stdout.trim().split('\n').map(line => JSON.parse(line));
            const results = events.filter(event => event.type === 'result');
            assert.deepEqual(results, [events.at(-1)], 'a request must end with exactly one result');
            const result = events.at(-1);
            assert.equal(result.status, status, 'the native result and process statuses must agree');
            return { document: result.json, events };
        };
        await flow({ repo, root, fixture, input, report, cli, api, capture, expect, binary });
    } catch (error) {
        failure = error;
        report.failure = detail(error);
    } finally {
        try {
            rmSync(fixture, { recursive: true });
            report.cleanup = { removed: true };
        } catch (error) {
            report.cleanup = { removed: false, error: detail(error) };
            if (failure) failure = new AggregateError([failure, error], 'qualification and owned fixture cleanup failed');
            else failure = error;
        }
        report.verdict = failure ? 'failed' : 'passed';
        report.finishedAt = new Date().toISOString();
        writeFileSync(join(root, 'report.json'), JSON.stringify(report, null, '\t'));
        console.log(`${report.verdict}: ${join(root, 'report.json')}`);
    }
    if (failure) throw failure;
}
