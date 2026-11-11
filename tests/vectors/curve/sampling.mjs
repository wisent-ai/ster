import assert from 'node:assert/strict';
import { readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, nativeDocument, nativeInteger } from '../../qualification/inputs.mjs';

await qualify('vectors/curve', async run => {
    const { model, config } = await createCheckpoint(run, 'model');
    const pairs = realpathSync(run.input('STER_TEST_PAIRS'));
    const pairBytes = readFileSync(pairs);
    const pairSet = JSON.parse(pairBytes);
    const holdout = nativeDocument(run.input('STER_TEST_HOLDOUT'), 'STER_TEST_HOLDOUT');
    const sizes = nativeDocument(run.input('STER_TEST_CURVE_SIZES'), 'STER_TEST_CURVE_SIZES');
    const method = run.input('STER_TEST_CURVE_METHOD');
    assert.ok(JSON.isRawJSON(holdout), 'the holdout must be a caller-declared native number');
    assert.ok(Array.isArray(sizes) && sizes.every(JSON.isRawJSON), 'sizes must be a caller-declared JSON array of native integers');
    const numericSizes = sizes.map(size => Number(size.rawJSON));
    assert.ok(numericSizes.every(Number.isSafeInteger), 'sizes must be exact integral pair counts');
    const fraction = Number(holdout.rawJSON);
    const heldCount = Math.round(pairSet.pairs.length * fraction);
    const fitCount = pairSet.pairs.length - heldCount;
    const layers = Array.from({ length: config.num_hidden_layers }, (_, layer) => layer);
    const frozen = new Map(['config.json', 'tokenizer.json', 'model.safetensors'].map(file => [file, digest(readFileSync(join(model, file)))]));
    const request = { model, pairs, layers: 'all', method, holdout, sizes, device: 'cpu', precision: 'f32', chatTemplate: 'off' };
    const common = ['vector', 'curve', '--model', model, '--pairs', pairs, '--layers', 'all', '--method', method,
        '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'];
    const sizeFlags = ['--sizes', sizes.map(size => size.rawJSON).join(',')];
    const report = JSON.parse((await run.cli([...common, '--holdout', holdout.rawJSON, ...sizeFlags], SUCCESS)).stdout);
    assert.equal(report.trait_name, pairSet.trait_name);
    assert.equal(report.method, method);
    assert.deepEqual(report.holdout, { fraction, fit_pairs: fitCount, holdout_pairs: heldCount });
    assert.deepEqual(report.points.map(({ layer, size }) => ({ layer, size })),
        layers.flatMap(layer => numericSizes.map(size => ({ layer, size }))), 'the curve lost declared sizes, order or model layers');
    for (const point of report.points) {
        assert.ok(Number.isFinite(point.holdout_accuracy) && Number.isFinite(point.holdout_margin));
        assert.ok(point.holdout_effect_size === null || Number.isFinite(point.holdout_effect_size));
    }
    const expectedEnough = layers.map(layer => {
        const points = report.points.filter(point => point.layer === layer);
        const bestAccuracy = Math.max(...points.map(point => point.holdout_accuracy));
        return { layer, best_accuracy: bestAccuracy,
            smallest_size: Math.min(...points.filter(point => point.holdout_accuracy === bestAccuracy).map(point => point.size)) };
    });
    assert.deepEqual(report.enough, expectedEnough, 'the published minimum size did not reach the actual best accuracy');
    const heldPath = join(run.fixture, 'held-pairs.json');
    writeFileSync(heldPath, JSON.stringify({ ...pairSet, pairs: pairSet.pairs.slice(fitCount) }));
    for (const size of new Set(numericSizes)) {
        const fitting = join(run.fixture, `fit-${size}.pairs.json`);
        const artifact = join(run.fixture, `fit-${size}.ster.json`);
        // The first prefix starts at the array origin, not an experimental preset:
        // https://tc39.es/ecma262/multipage/indexed-collections.html#sec-array.prototype.slice
        writeFileSync(fitting, JSON.stringify({ ...pairSet, pairs: pairSet.pairs.slice(0, size) }));
        await run.cli(['vector', 'train', '--model', model, '--pairs', fitting, '--output', artifact, '--layers', 'all',
            '--method', method, '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
        const scored = JSON.parse((await run.cli(['vector', 'evaluate', '--model', model, '--pairs', heldPath, '--vector', artifact,
            '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS)).stdout);
        assert.equal(scored.pair_count, heldCount);
        assert.deepEqual(scored.layers.map(row => row.layer), layers);
        const measured = new Map(scored.layers.map(row => [row.layer, row]));
        for (const point of report.points.filter(point => point.size === size)) {
            assert.deepEqual(measured.get(point.layer),
                { layer: point.layer, accuracy: point.holdout_accuracy, margin: point.holdout_margin },
                'independent persisted prefix fits did not reproduce held-out curve scores');
        }
        run.report.states.push({ kind: 'independent-prefix-fit', fitting, artifact,
            sha256: digest(readFileSync(artifact)), heldPath, scored });
    }
    const native = await run.api('vector/curve', request, SUCCESS);
    assert.deepEqual(native.document, report, 'native learning-curve selection disagreed with CLI measurements');
    run.report.inputs.push({ path: pairs, sha256: digest(pairBytes), method, holdout, sizes });
    run.report.states.push({ kind: 'learning-curve', report, native: native.document });
    const allPairs = nativeInteger(String(pairSet.pairs.length));
    await run.cli([...common, '--holdout', holdout.rawJSON, '--sizes', allPairs.rawJSON], FAILURE);
    await run.api('vector/curve', { ...request, sizes: [allPairs] }, FAILURE);
    // Exact excluded fraction endpoints in the native split contract:
    // https://github.com/wisent-ai/ster/blob/main/src/workflow/optimize.rs
    for (const boundary of [0, 1]) {
        await run.cli([...common, '--holdout', String(boundary), ...sizeFlags], FAILURE);
        await run.api('vector/curve', { ...request, holdout: nativeInteger(String(boundary)) }, FAILURE);
    }
    await run.cli([...common, '--holdout', holdout.rawJSON], USAGE);
    const { sizes: omitted, ...unsized } = request;
    const missing = await run.api('vector/curve', unsized, USAGE);
    assert.deepEqual(missing.events.filter(event => event.type === 'log'), []);
    assert.deepEqual(readFileSync(pairs), pairBytes);
    for (const [file, sha256] of frozen) assert.equal(digest(readFileSync(join(model, file))), sha256, `curve changed ${file}`);
});
