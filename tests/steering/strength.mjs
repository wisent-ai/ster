import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../qualification/runtime.mjs';
import { createCheckpoint, nativeInteger } from '../qualification/inputs.mjs';

await qualify('steering/strength', async run => {
    const { model, config, declaration, declarationBytes } = await createCheckpoint(run, 'checkpoint');
    // Empty/additive identity and unit probability/unique selection boundaries:
    // https://ster.wisent.com/docs/steering
    const zero = 0;
    // https://ster.wisent.com/docs/steering
    const unit = 1;
    const raw = run.input('STER_TEST_STRENGTHS');
    const strengths = raw.split(',').map(value => {
        assert.notEqual(value.trim(), '', 'each declared strength must contain a number');
        const strength = Number(value);
        assert.ok(Number.isFinite(strength), 'the successful selection needs finite declared candidates');
        return strength;
    });
    assert.ok(new Set(strengths).size > unit, 'qualification requires distinct alternatives for a real selection');
    const batch = run.input('STER_TEST_BATCH_SIZE');
    const sequence = run.input('STER_TEST_MAX_SEQUENCE');
    const layer = config.num_hidden_layers - unit;
    run.report.experiment = { strengths, batch, sequence, layer };
    const pairs = join(run.repo, 'docs', 'examples', 'pairs.json');
    const originalBytes = readFileSync(pairs);
    const input = JSON.parse(originalBytes);
    run.report.inputs.push({ path: pairs, sha256: digest(originalBytes) });
    const vector = join(run.fixture, 'calm.ster.json');
    await run.cli(['vector', 'train', '--model', model, '--pairs', pairs, '--output', vector, '--layers', String(layer),
        '--method', 'caa', '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    const vectorBytes = readFileSync(vector);
    const base = ['vector', 'evaluate', '--model', model, '--pairs', pairs, '--vector', vector,
        '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'];
    function selection(report) {
        const measured = report.strength;
        assert.deepEqual(measured.candidates.map(candidate => candidate.strength), strengths);
        assert.deepEqual(measured.layers, [layer]);
        assert.equal(measured.pairs, input.pairs.length);
        assert.equal(measured.scored_pairs + measured.skipped_long, input.pairs.length);
        assert.ok(measured.scored_pairs > zero, 'the selection must measure actual pairs');
        for (const candidate of measured.candidates) {
            assert.ok(Number.isFinite(candidate.ordered) && candidate.ordered >= zero && candidate.ordered <= unit);
            assert.ok(Number.isFinite(candidate.mean_shift));
        }
        const selected = measured.candidates.filter(candidate => candidate.selected);
        assert.equal(selected.length, unit, 'exactly one measured candidate must be selected');
        const [winner] = selected;
        assert.equal(measured.selected_strength, winner.strength);
        assert.ok(measured.candidates.every(candidate => winner.ordered > candidate.ordered
            || (winner.ordered === candidate.ordered && winner.mean_shift >= candidate.mean_shift)),
        'selection must maximize ordered share, then mean shift, rather than merely name a row');
        assert.deepEqual(report.layers.map(item => item.layer), [layer], 'strength selection must retain the actual representation measurement');
        for (const item of report.layers) {
            assert.ok(Number.isFinite(item.accuracy) && item.accuracy >= zero && item.accuracy <= unit);
            assert.ok(Number.isFinite(item.margin));
        }
        return measured.selected_strength;
    }
    const measured = JSON.parse((await run.cli([...base, '--strengths', raw, '--batch-size', batch, '--max-sequence', sequence], SUCCESS)).stdout);
    const selected = selection(measured);
    const body = { model, pairs, vector, strengths, batchSize: nativeInteger(batch), maxSequence: nativeInteger(sequence),
        device: 'cpu', precision: 'f32', chatTemplate: 'off' };
    const requested = await run.api('vector/evaluate', body, SUCCESS);
    assert.equal(selection(requested.document), selected, 'the request must select the same actual candidate as the CLI');
    async function refuse(args, status) {
        const result = await run.cli([...base, ...args], status);
        assert.equal(result.stdout, '', 'a refused selection must not print a successful measurement');
    }
    await refuse(['--strengths', raw, '--max-sequence', sequence], FAILURE);
    await refuse(['--strengths', raw, '--batch-size', batch], FAILURE);
    await refuse(['--batch-size', batch], USAGE);
    const [firstStrength] = strengths;
    await refuse(['--strengths', `${firstStrength},inf`, '--batch-size', batch, '--max-sequence', sequence], FAILURE);
    await refuse(['--strengths', raw, '--batch-size', String(zero), '--max-sequence', sequence], FAILURE);
    await refuse(['--strengths', raw, '--batch-size', batch, '--max-sequence', String(unit)], FAILURE);
    const unsized = { ...body };
    delete unsized.batchSize;
    await run.api('vector/evaluate', unsized, USAGE);
    const unlimited = { ...body };
    delete unlimited.maxSequence;
    await run.api('vector/evaluate', unlimited, USAGE);
    await run.api('vector/evaluate', { ...body, strengths: [] }, USAGE);
    await run.api('vector/evaluate', { ...body, batchSize: zero }, FAILURE);
    assert.deepEqual(readFileSync(vector), vectorBytes, 'measurement or refusal changed the learned input artifact');
    run.report.states.push({ path: vector, sha256: digest(vectorBytes), artifact: JSON.parse(vectorBytes), measured, requested: requested.document });
    assert.deepEqual(readFileSync(pairs), originalBytes);
    assert.deepEqual(readFileSync(declaration), declarationBytes);
});
