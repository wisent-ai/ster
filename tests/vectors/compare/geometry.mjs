import assert from 'node:assert/strict';
import { readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, nativeInteger } from '../../qualification/inputs.mjs';

await qualify('vectors/compare', async run => {
    const { model, config } = await createCheckpoint(run, 'model');
    const source = realpathSync(run.input('STER_TEST_PAIRS'));
    const sourceBytes = readFileSync(source);
    const pairSet = JSON.parse(sourceBytes);
    run.report.inputs.push({ path: source, sha256: digest(sourceBytes) });
    const reversePath = join(run.fixture, 'reverse-pairs.json');
    const pcaPath = join(run.fixture, 'pca-pairs.json');
    const labels = [pairSet.trait_name, `${pairSet.trait_name}-reversed`, `${pairSet.trait_name}-pca`];
    const [originalLabel, reverseLabel, pcaLabel] = labels;
    writeFileSync(reversePath, JSON.stringify({ ...pairSet, trait_name: reverseLabel,
        pairs: pairSet.pairs.map(pair => ({ positive: pair.negative, negative: pair.positive })) }));
    writeFileSync(pcaPath, JSON.stringify({ ...pairSet, trait_name: pcaLabel }));
    async function fit(name, pairs, method) {
        const path = join(run.fixture, `${name}.json`);
        await run.cli(['vector', 'train', '--model', model, '--pairs', pairs, '--method', method,
            '--layers', 'all', '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off', '--output', path], SUCCESS);
        return path;
    }
    const original = await fit('original', source, 'caa');
    const reversed = await fit('reversed', reversePath, 'caa');
    const pca = await fit('pca', pcaPath, 'pca');
    const artifacts = [original, reversed, pca];
    const protectedArtifacts = new Map(artifacts.map(path => [path, readFileSync(path)]));
    const selectedLayers = JSON.parse(protectedArtifacts.get(original)).vectors.map(vector => vector.layer);
    const clusters = artifacts.length; // The actual maximum grouping boundary: one group per artifact.
    const report = JSON.parse((await run.cli(['vector', 'compare', ...artifacts, '--clusters', String(clusters)], SUCCESS)).stdout);
    assert.deepEqual(report.labels, labels, 'the compared artifacts lost their actual trait identity or order');
    assert.deepEqual(report.layers, selectedLayers);
    assert.deepEqual([report.most_different.a, report.most_different.b], [originalLabel, reverseLabel]);
    // Opposite unit directions have a negative normalized dot product:
    // https://github.com/wisent-ai/ster/blob/main/src/workflow/compare/mod.rs
    assert.ok(report.most_different.cosine < 0, 'the comparison did not identify the sign reversal');
    const indices = new Map(report.labels.map((label, index) => [label, index]));
    const originalDistances = report.distance[indices.get(originalLabel)];
    assert.ok(originalDistances[indices.get(reverseLabel)] <= originalDistances[indices.get(pcaLabel)],
        'sign-blind geometry did not place the opposite CAA directions on the same axis');
    assert.equal(new Set(report.clusters).size, artifacts.length, 'maximal grouping must put every artifact in its own group');
    assert.equal(report.silhouette, null, 'singleton groups have no defined silhouette');
    assert.deepEqual(report.uniqueness.map(item => item.label), labels);
    assert.ok(report.uniqueness.every(item => Number.isFinite(item.score)));
    assert.deepEqual(report.per_layer.map(item => item.layer), selectedLayers);
    for (const matrix of [report.similarity, report.distance, ...report.per_layer.map(item => item.similarity)]) {
        assert.equal(matrix.length, artifacts.length);
        for (const row of matrix) {
            assert.equal(row.length, artifacts.length);
            assert.ok(row.every(Number.isFinite), 'the actual comparison emitted a non-finite matrix');
        }
    }
    const native = await run.api('vector/compare', { artifacts, clusters: nativeInteger(String(clusters)) }, SUCCESS);
    assert.deepEqual(native.document, report, 'native comparison changed the measured geometry or grouping');
    run.report.states.push({ kind: 'comparison', artifacts, report, native: native.document });

    const single = [original];
    await run.cli(['vector', 'compare', ...single, '--clusters', String(single.length)], FAILURE);
    const tooFew = await run.api('vector/compare', { artifacts: single, clusters: nativeInteger(String(single.length)) }, USAGE);
    assert.deepEqual(tooFew.events.filter(event => event.type === 'log'), [], 'too few artifacts must refuse before native execution');
    // The next count beyond the native maximum is a refusal boundary, not a preset:
    // https://github.com/wisent-ai/ster/blob/main/src/workflow/compare/mod.rs
    const excessive = artifacts.length + 1;
    await run.cli(['vector', 'compare', ...artifacts, '--clusters', String(excessive)], FAILURE);
    await run.api('vector/compare', { artifacts, clusters: nativeInteger(String(excessive)) }, FAILURE);
    await run.cli(['vector', 'compare', ...artifacts, '--clusters', String(clusters), '--layers', String(config.num_hidden_layers)], FAILURE);
    await run.api('vector/compare', { artifacts, clusters: nativeInteger(String(clusters)),
        layers: [nativeInteger(String(config.num_hidden_layers))] }, FAILURE);
    run.report.states.push({ kind: 'comparison-refusals', tooFew: tooFew.document, excessiveClusters: excessive, absentLayer: config.num_hidden_layers });
    for (const [path, bytes] of protectedArtifacts) {
        assert.deepEqual(readFileSync(path), bytes, `comparison changed the actual learned input ${path}`);
    }
    assert.deepEqual(readFileSync(source), sourceBytes, 'comparison qualification changed the operator pair set');
});
