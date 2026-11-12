import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint } from '../../qualification/inputs.mjs';

await qualify('pairs/merge', async run => {
    const source = join(run.repo, 'docs', 'examples', 'pairs.json');
    const originalBytes = readFileSync(source);
    const input = JSON.parse(originalBytes);
    const [first, ...remaining] = input.pairs;
    assert.ok(first && remaining.length, 'the real merge/training case needs a pair to mirror and unmatched directions');
    run.report.inputs.push({ path: source, sha256: digest(originalBytes) });
    const swapped = { trait_name: 'restless', pairs: [{ positive: first.negative, negative: first.positive }] };
    const opposite = join(run.fixture, 'opposite.json');
    writeFileSync(opposite, JSON.stringify(swapped));
    const oppositeBytes = readFileSync(opposite);
    const trait = 'weather';
    const expected = { trait_name: trait, pairs: [...input.pairs, ...swapped.pairs] };
    const output = join(run.fixture, 'merged.json');
    const result = JSON.parse((await run.cli(['pairs', 'merge', '--pairs', source, '--pairs', opposite, '--trait', trait, '--output', output], SUCCESS)).stdout);
    const mergedBytes = readFileSync(output);
    assert.deepEqual(JSON.parse(mergedBytes), expected, 'the persisted merge must preserve every pair in source order under the declared trait');
    assert.deepEqual(result.report.sources, [
        { path: source, trait_name: input.trait_name, pairs: input.pairs.length },
        { path: opposite, trait_name: swapped.trait_name, pairs: swapped.pairs.length },
    ]);
    assert.equal(result.report.pair_count, expected.pairs.length);
    const { model, config, declaration, declarationBytes } = await createCheckpoint(run, 'checkpoint');
    const direction = join(run.fixture, 'weather.ster.json');
    await run.cli(['vector', 'train', '--model', model, '--pairs', output, '--output', direction, '--layers', 'all',
        '--method', 'caa', '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    const directionBytes = readFileSync(direction);
    const fitted = JSON.parse(directionBytes);
    assert.equal(fitted.trait_name, trait);
    const inspection = JSON.parse((await run.cli(['vector', 'inspect', direction], SUCCESS)).stdout);
    const layers = Array.from({ length: config.num_hidden_layers }, (_, layer) => ({ layer, width: config.hidden_size }));
    assert.deepEqual(inspection.layers.map(({ layer, width }) => ({ layer, width })), layers);
    assert.deepEqual(fitted.vectors.map(({ layer, values }) => ({ layer, width: values.length })), layers);
    for (const { values } of fitted.vectors) assert.ok(values.every(Number.isFinite));

    const single = join(run.fixture, 'single.json');
    await run.cli(['pairs', 'merge', '--pairs', source, '--trait', trait, '--output', single], FAILURE);
    assert.ok(!existsSync(single));
    const unnamed = join(run.fixture, 'unnamed.json');
    await run.cli(['pairs', 'merge', '--pairs', source, '--pairs', opposite, '--trait', ' ', '--output', unnamed], FAILURE);
    assert.ok(!existsSync(unnamed));
    await run.api('pairs/merge', { sources: [source], traitName: trait, output: single }, FAILURE);
    await run.api('pairs/merge', { sources: [source, opposite], traitName: ' ', output: unnamed }, FAILURE);
    await run.api('pairs/merge', { sources: [source, opposite], traitName: trait, output: '' }, USAGE);
    assert.ok(!existsSync(single) && !existsSync(unnamed), 'native request refusals wrote a merge artifact');
    const requested = join(run.fixture, 'requested.json');
    const request = await run.api('pairs/merge', { sources: [source, opposite], traitName: trait, output: requested }, SUCCESS);
    const requestedBytes = readFileSync(requested);
    assert.deepEqual(JSON.parse(requestedBytes), expected, 'the native request must persist the same real merged dataset');
    assert.deepEqual(request.document.report, result.report);
    assert.deepEqual(readFileSync(output), mergedBytes);
    assert.deepEqual(readFileSync(direction), directionBytes);
    assert.deepEqual(readFileSync(opposite), oppositeBytes);
    assert.deepEqual(readFileSync(source), originalBytes);
    assert.deepEqual(readFileSync(declaration), declarationBytes);
    run.report.states.push({ path: output, sha256: digest(mergedBytes), merged: expected, report: result.report },
        { path: requested, sha256: digest(requestedBytes), requested: request.document },
        { path: direction, sha256: digest(directionBytes), fitted, inspection },
        { path: single, exists: existsSync(single) }, { path: unnamed, exists: existsSync(unnamed) });
});
