import assert from 'node:assert/strict';
import { existsSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, nativeDocument, nativeInteger, trainingFlags } from '../../qualification/inputs.mjs';

await qualify('vectors/optimize', async run => {
    const { model, config } = await createCheckpoint(run, 'model');
    const pairs = realpathSync(run.input('STER_TEST_PAIRS'));
    const pairBytes = readFileSync(pairs);
    const pairSet = JSON.parse(pairBytes);
    const holdout = nativeDocument(run.input('STER_TEST_HOLDOUT'), 'STER_TEST_HOLDOUT');
    assert.ok(JSON.isRawJSON(holdout), 'holdout must be a declared native numeric input');
    const fraction = Number(holdout.rawJSON);
    const heldCount = Math.round(pairSet.pairs.length * fraction);
    const fitCount = pairSet.pairs.length - heldCount;
    const layers = Array.from({ length: config.num_hidden_layers }, (_, layer) => layer);
    const output = join(run.fixture, 'selected.json');
    const frozen = new Map(['config.json', 'tokenizer.json', 'model.safetensors'].map(file => [file, digest(readFileSync(join(model, file)))]));
    const request = { model, pairs, output, layers: 'all', holdout, device: 'cpu', precision: 'f32', chatTemplate: 'off' };
    const report = JSON.parse((await run.cli(['vector', 'optimize', ...trainingFlags(request)], SUCCESS)).stdout);
    assert.deepEqual(report.selection.holdout, { fraction, fit_pairs: fitCount, holdout_pairs: heldCount });
    const artifactBytes = readFileSync(output);
    const artifact = JSON.parse(artifactBytes);
    const fitting = join(run.fixture, 'fitting.pairs.json');
    const held = join(run.fixture, 'held.pairs.json');
    // Ordered prefix and suffix are the actual split contract; zero is the
    // array origin: https://github.com/wisent-ai/ster/blob/main/src/workflow/optimize.rs
    writeFileSync(fitting, JSON.stringify({ ...pairSet, pairs: pairSet.pairs.slice(0, fitCount) }));
    writeFileSync(held, JSON.stringify({ ...pairSet, pairs: pairSet.pairs.slice(fitCount) }));
    async function fit(method) {
        const path = join(run.fixture, `${method}.json`);
        await run.cli(['vector', 'train', '--model', model, '--pairs', fitting, '--output', path, '--layers', 'all',
            '--method', method, '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
        const bytes = readFileSync(path);
        const trained = JSON.parse(bytes);
        assert.equal(trained.method, method);
        assert.deepEqual(trained.vectors.map(vector => vector.layer), layers);
        const scored = JSON.parse((await run.cli(['vector', 'evaluate', '--model', model, '--pairs', held, '--vector', path,
            '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS)).stdout);
        assert.equal(scored.pair_count, heldCount);
        assert.deepEqual(scored.layers.map(row => row.layer), layers);
        run.report.states.push({ kind: 'independent-candidate', path, sha256: digest(bytes), scored });
        return { method: trained.method, layers: scored.layers };
    }
    // Independently exercise every native optimization method through its real
    // artifact producer and consumer, not a word classifier or an invented count:
    // https://github.com/wisent-ai/ster/blob/main/src/workflow/optimize.rs
    const caa = await fit('caa');
    const pca = await fit('pca');
    const logistic = await fit('logistic');
    const fitted = [caa, pca, logistic];
    const expected = layers.flatMap(layer => fitted.map(fit => {
        const score = fit.layers.find(score => score.layer === layer);
        return { layer, method: fit.method, accuracy: score.accuracy, margin: score.margin };
    }));
    assert.deepEqual(report.selection.candidates.map(row => ({ layer: row.layer, method: row.method,
        accuracy: row.holdout_accuracy, margin: row.holdout_margin })), expected,
    'optimization did not measure every actual layer/method on independently held-out pairs');
    for (const candidate of report.selection.candidates) {
        assert.ok(Number.isFinite(candidate.holdout_accuracy) && Number.isFinite(candidate.holdout_margin));
        assert.ok(candidate.holdout_effect_size === null || Number.isFinite(candidate.holdout_effect_size));
        // Signed Cohen effect and signed mean margin share the mathematical origin:
        // https://github.com/wisent-ai/ster/blob/main/src/workflow/optimize.rs
        if (candidate.holdout_effect_size !== null && candidate.holdout_margin !== 0) {
            // https://github.com/wisent-ai/ster/blob/main/src/workflow/optimize.rs
            assert.ok(candidate.holdout_effect_size * candidate.holdout_margin > 0);
        }
    }
    const [first, ...remaining] = report.selection.candidates;
    const winner = remaining.reduce((best, next) => next.holdout_accuracy > best.holdout_accuracy
        || (next.holdout_accuracy === best.holdout_accuracy && next.holdout_margin > best.holdout_margin) ? next : best, first);
    assert.deepEqual(report.selection.candidates.filter(candidate => candidate.selected), [winner],
        'selection did not publish exactly the highest-accuracy, then highest-margin candidate');
    assert.equal(artifact.method, winner.method);
    assert.deepEqual(artifact.vectors.map(vector => vector.layer), [winner.layer]);
    const refit = join(run.fixture, 'independent-refit.json');
    await run.cli(['vector', 'train', '--model', model, '--pairs', pairs, '--output', refit, '--layers', String(winner.layer),
        '--method', winner.method, '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    assert.deepEqual(artifact.vectors, JSON.parse(readFileSync(refit)).vectors,
        'the published winner was not refitted on the entire pair set');
    const nativeOutput = join(run.fixture, 'native-selected.json');
    const native = await run.api('vector/optimize', { ...request, output: nativeOutput }, SUCCESS);
    assert.deepEqual(native.document.selection, report.selection);
    assert.deepEqual(JSON.parse(readFileSync(nativeOutput)).vectors, artifact.vectors);
    const consumed = await run.api('vector/evaluate', { model, pairs, vector: output,
        device: 'cpu', precision: 'f32', chatTemplate: 'off' }, SUCCESS);
    assert.deepEqual(consumed.document.layers.map(row => row.layer), [winner.layer]);
    assert.ok(consumed.document.layers.every(row => Number.isFinite(row.accuracy) && Number.isFinite(row.margin)));
    run.report.inputs.push({ path: pairs, sha256: digest(pairBytes), holdout });
    run.report.states.push({ kind: 'optimization', output, sha256: digest(artifactBytes), report, winner,
        native: native.document, refit, refitSha256: digest(readFileSync(refit)), consumed: consumed.document });

    // Excluded fraction endpoints in the native split contract:
    // https://github.com/wisent-ai/ster/blob/main/src/workflow/optimize.rs
    for (const boundary of [0, 1]) {
        const refusedOutput = join(run.fixture, `refused-${boundary}.json`);
        const refused = { ...request, output: refusedOutput, holdout: nativeInteger(String(boundary)) };
        await run.cli(['vector', 'optimize', ...trainingFlags(refused)], FAILURE);
        await run.api('vector/optimize', refused, FAILURE);
        assert.ok(!existsSync(refusedOutput), 'a refused split published a steering artifact');
    }
    const missingOutput = join(run.fixture, 'missing-holdout.json');
    const { holdout: omitted, ...withoutHoldout } = { ...request, output: missingOutput };
    await run.cli(['vector', 'optimize', ...trainingFlags(withoutHoldout)], USAGE);
    const missing = await run.api('vector/optimize', withoutHoldout, USAGE);
    assert.deepEqual(missing.events.filter(event => event.type === 'log'), []);
    assert.ok(!existsSync(missingOutput));
    assert.deepEqual(readFileSync(pairs), pairBytes);
    assert.deepEqual(readFileSync(output), artifactBytes);
    for (const [file, sha256] of frozen) assert.equal(digest(readFileSync(join(model, file))), sha256, `optimization changed ${file}`);
});
