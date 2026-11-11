import assert from 'node:assert/strict';
import { readFileSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { SUCCESS, digest } from './runtime.mjs';

export async function createCheckpoint(run, name) {
    const declaration = realpathSync(run.input('STER_TEST_MODEL_SPEC'));
    const declarationBytes = readFileSync(declaration);
    run.report.inputs.push({ path: declaration, sha256: digest(declarationBytes) });
    const model = join(run.fixture, name);
    const created = JSON.parse((await run.cli(['model', 'fixture', 'create', '--from', declaration, '--output', model], SUCCESS)).stdout);
    const path = join(model, 'config.json');
    const bytes = readFileSync(path);
    const config = JSON.parse(bytes);
    run.report.states.push({ kind: 'checkpoint', model, creation: created, configuration: { path, sha256: digest(bytes), config } });
    return { model, config, declaration, declarationBytes };
}

export function nativeInteger(value) {
    assert.equal(typeof JSON.rawJSON, 'function', 'qualification requires native JSON.rawJSON support to preserve integer request inputs');
    return JSON.rawJSON(BigInt(value).toString());
}

export function greedyGeneration(run) {
    const maxNewTokens = run.input('STER_TEST_MAX_NEW_TOKENS');
    const seed = run.input('STER_TEST_SEED');
    // Native argmax boundary, not a token-budget or seed preset:
    // https://ster.wisent.com/docs/cli/generate
    const temperature = 0;
    const experiment = { maxNewTokens, seed, temperature, device: 'cpu', precision: 'f32', chatTemplate: 'off' };
    run.report.generation = experiment;
    return {
        flags: ['--max-new-tokens', maxNewTokens, '--temperature', String(temperature), '--seed', seed,
            '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'],
        request: { maxNewTokens: nativeInteger(maxNewTokens), temperature,
            seed: nativeInteger(seed), device: 'cpu', precision: 'f32', chatTemplate: 'off' },
    };
}

// Preserve the native numeric lexeme before a JavaScript Number can round it:
// https://tc39.es/ecma262/multipage/structured-data.html#sec-json.parse
export function nativeDocument(bytes, label) {
    assert.equal(typeof JSON.rawJSON, 'function', `${label} requires native JSON.rawJSON support`);
    try {
        return JSON.parse(bytes.toString('utf8'), (key, value, context) => {
            if (typeof value !== 'number') return value;
            assert.ok(context && typeof context.source === 'string',
                `${label} field ${key} requires JSON.parse source-context support`);
            return JSON.rawJSON(context.source);
        });
    } catch (cause) {
        throw new Error(`Cannot preserve native numeric inputs from ${label}`, { cause });
    }
}

/// Training flags and native fields use the same declared kebab/camel convention.
export function trainingFlags(request) {
    assert.equal(typeof JSON.isRawJSON, 'function', 'training qualification requires JSON.isRawJSON support');
    return Object.entries(request).flatMap(([field, value]) => {
        const flag = `--${field.replace(/[A-Z]/g, letter => `-${letter.toLowerCase()}`)}`;
        if (typeof value === 'string') return [flag, value];
        assert.ok(JSON.isRawJSON(value), `${field} must be a declared string or native JSON number`);
        return [flag, value.rawJSON];
    });
}

export function tuningInputs(run, objective, bindings) {
    const path = realpathSync(run.input('STER_TEST_TUNING_SPEC'));
    const bytes = readFileSync(path);
    const declaration = nativeDocument(bytes, path);
    assert.ok(declaration && typeof declaration === 'object' && !Array.isArray(declaration),
        `${path} must map native training operation names to their explicit parameters`);
    assert.ok(Object.hasOwn(declaration, objective), `${path} must declare ${objective} parameters`);
    const settings = declaration[objective];
    assert.ok(settings && typeof settings === 'object' && !Array.isArray(settings),
        `${path} ${objective} parameters must be a native request-field object`);
    assert.ok(!Object.hasOwn(settings, 'checkpoint') && !Object.hasOwn(settings, 'resume'),
        `${path} ${objective} checkpoint/resume paths must come from owned qualification bindings, not an operator-data profile`);
    for (const field of Object.keys(bindings)) {
        assert.ok(!Object.hasOwn(settings, field),
            `${path} ${objective}.${field} conflicts with an isolated qualification binding`);
    }
    const request = { ...settings, ...bindings };
    run.report.inputs.push({ path, sha256: digest(bytes), objective });
    run.report.states.push({ kind: 'training-input', objective, request });
    return { request, flags: trainingFlags(request), path, bytes };
}
