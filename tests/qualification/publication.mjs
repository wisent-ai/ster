import assert from 'node:assert/strict';
import { lstatSync, readFileSync, watch } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { digest } from './runtime.mjs';

// These are the product's persisted checkpoint identities, not progress words:
// https://github.com/wisent-ai/ster/blob/main/src/tune/checkpoint/mod.rs
export function stopAfterPublication(directory) {
    const pointer = join(directory, 'latest');
    const watcher = watch(directory);
    let observation;
    let started = false;
    const control = child => {
        assert.equal(started, false, 'a publication observer belongs to one actual child');
        started = true;
        return new Promise((resolveControl, rejectControl) => {
            let failure;
            const fail = error => {
                failure = error;
                watcher.close();
                rejectControl(error);
            };
            watcher.once('error', cause => fail(new Error(`Cannot observe checkpoint directory ${directory}`, { cause })));
            watcher.on('change', () => {
                if (observation) return;
                try {
                    const state = lstatSync(pointer, { throwIfNoEntry: false });
                    if (state === undefined) return;
                    assert.ok(state.isFile(), `Published checkpoint pointer ${pointer} is not a regular file`);
                    observation = { pointer, child: child.pid, observedAt: new Date().toISOString() };
                    assert.ok(child.kill('SIGTERM'),
                        `Cannot interrupt child ${child.pid} after publication; exit=${child.exitCode}, signal=${child.signalCode}`);
                    watcher.close();
                } catch (cause) {
                    fail(new Error(`Cannot interrupt the actual writer after checkpoint publication at ${pointer}`, { cause }));
                }
            });
            child.once('close', () => {
                watcher.close();
                if (failure) rejectControl(failure);
                else if (observation) resolveControl(observation);
                else rejectControl(new Error(
                    `Child ${child.pid} ended before a published checkpoint was observed at ${pointer}; exit=${child.exitCode}, signal=${child.signalCode}`));
            });
        });
    };
    return { control, close: () => watcher.close(), observation: () => observation };
}

export function publishedCheckpoint(directory) {
    const pointer = join(directory, 'latest');
    const name = readFileSync(pointer, 'utf8').trim();
    assert.equal(basename(name), name, `${pointer} does not name one checkpoint directory`);
    const path = resolve(directory, name);
    assert.equal(dirname(path), resolve(directory), `${pointer} escapes its actual checkpoint root`);
    const statePath = join(path, 'state.json');
    const stateBytes = readFileSync(statePath);
    const state = JSON.parse(stateBytes);
    assert.equal(state.kind, 'ster tune checkpoint', `${statePath} is not an actual native checkpoint`);
    assert.ok(Number.isSafeInteger(state.position.step), `${statePath} has an unrepresentable checkpoint step`);
    const tensorsPath = join(path, 'tensors.safetensors');
    const tensors = readFileSync(tensorsPath);
    return { path, pointer, name, state, statePath, tensorsPath,
        stateDigest: digest(stateBytes), tensorsDigest: digest(tensors) };
}
