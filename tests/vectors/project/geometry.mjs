import assert from 'node:assert/strict';
import { readFileSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, nativeDocument, nativeInteger, trainingFlags } from '../../qualification/inputs.mjs';

await qualify('vectors/project', async run => {
    const { model, config } = await createCheckpoint(run, 'model');
    const pairs = realpathSync(run.input('STER_TEST_PAIRS'));
    const pairBytes = readFileSync(pairs);
    const pairSet = JSON.parse(pairBytes);
    const vector = join(run.fixture, 'direction.json');
    await run.cli(['vector', 'train', '--model', model, '--pairs', pairs, '--output', vector, '--layers', 'all',
        '--method', 'caa', '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    const vectorBytes = readFileSync(vector);
    const frozen = new Map(['config.json', 'tokenizer.json', 'model.safetensors'].map(file => [file, digest(readFileSync(join(model, file)))]));
    const layer = nativeDocument(run.input('STER_TEST_PROJECTION_LAYER'), 'STER_TEST_PROJECTION_LAYER');
    const strength = nativeDocument(run.input('STER_TEST_PROJECTION_STRENGTH'), 'STER_TEST_PROJECTION_STRENGTH');
    const components = nativeDocument(run.input('STER_TEST_PROJECTION_COMPONENTS'), 'STER_TEST_PROJECTION_COMPONENTS');
    assert.ok([layer, strength, components].every(JSON.isRawJSON), 'projection parameters must be explicit native numeric inputs');
    const request = { model, pairs, vector, layer, strength, components, device: 'cpu', precision: 'f32', chatTemplate: 'off' };
    const report = JSON.parse((await run.cli(['vector', 'project', ...trainingFlags(request)], SUCCESS)).stdout);
    assert.equal(report.trait_name, pairSet.trait_name);
    assert.equal(report.layer, Number(layer.rawJSON));
    assert.equal(report.strength, Number(strength.rawJSON));
    assert.equal(report.pair_count, pairSet.pairs.length);
    assert.ok(report.explained.length <= Number(components.rawJSON), 'projection exceeded the requested component budget');
    // Exact variance-share bounds: https://github.com/wisent-ai/ster/blob/main/src/workflow/project/mod.rs
    assert.ok(report.explained.every(share => Number.isFinite(share) && share > 0 && share <= 1));
    const steeringStates = [false, true];
    const expectedKeys = new Set(pairSet.pairs.flatMap(({ positive, negative }, pair) =>
        Object.entries({ positive, negative }).flatMap(([side, prompt]) => {
            assert.equal(typeof prompt, 'string', `pair ${pair}.${side} is not a native prompt field`);
            return steeringStates.map(steered => `${pair}/${side}/${steered}`);
        })));
    const actualKeys = report.points.map(point => `${point.pair}/${point.side}/${point.steered}`);
    assert.equal(actualKeys.length, expectedKeys.size, 'projection duplicated or omitted actual pair-side states');
    assert.deepEqual(new Set(actualKeys), expectedKeys, 'projection lost pair identity, side or steering state');
    for (const point of report.points) {
        assert.equal(point.coordinates.length, report.explained.length, 'a projected state lost an actual fitted component');
        assert.ok(point.coordinates.every(Number.isFinite));
    }
    // A fitted axis of varying reads cannot project every state to the origin:
    // https://github.com/wisent-ai/ster/blob/main/src/workflow/project/mod.rs
    assert.ok(report.points.some(point => point.coordinates.some(value => value !== 0)), 'projection discarded the actual variation');
    assert.ok(Number.isFinite(report.shift_along_direction), 'the learned read-layer direction has no measured shift');
    assert.ok(Object.values(report.negative_distance_to_positive).every(Number.isFinite));
    assert.notEqual(report.negative_distance_to_positive.steered, report.negative_distance_to_positive.unsteered,
        'the declared projection experiment did not move the measured negative states');
    // Exact fraction bounds, not success thresholds: https://github.com/wisent-ai/ster/blob/main/src/workflow/project/mod.rs
    assert.ok(Number.isFinite(report.negatives_moved_toward_positive) && report.negatives_moved_toward_positive >= 0 && report.negatives_moved_toward_positive <= 1);
    const native = await run.api('vector/project', request, SUCCESS);
    assert.deepEqual(native.document, report, 'native projection changed actual state coordinates or steering measurements');
    run.report.inputs.push({ path: pairs, sha256: digest(pairBytes), layer, strength, components });
    run.report.states.push({ kind: 'projection', vector, sha256: digest(vectorBytes), report, native: native.document });

    const absentLayer = { ...request, layer: nativeInteger(String(config.num_hidden_layers)) };
    await run.cli(['vector', 'project', ...trainingFlags(absentLayer)], FAILURE);
    await run.api('vector/project', absentLayer, FAILURE);
    // NonZeroUsize rejects the empty component budget at schema parsing:
    // https://github.com/wisent-ai/ster/blob/main/src/request/requests/vectors/mod.rs
    const emptyComponents = { ...request, components: nativeInteger('0') };
    await run.cli(['vector', 'project', ...trainingFlags(emptyComponents)], USAGE);
    const empty = await run.api('vector/project', emptyComponents, USAGE);
    assert.deepEqual(empty.events.filter(event => event.type === 'log'), []);
    for (const field of ['layer', 'strength', 'components']) {
        const omitted = { ...request };
        delete omitted[field];
        await run.cli(['vector', 'project', ...trainingFlags(omitted)], USAGE);
        const missing = await run.api('vector/project', omitted, USAGE);
        assert.deepEqual(missing.events.filter(event => event.type === 'log'), [], `${field} omission entered native execution`);
    }
    assert.deepEqual(readFileSync(pairs), pairBytes);
    assert.deepEqual(readFileSync(vector), vectorBytes);
    for (const [file, sha256] of frozen) assert.equal(digest(readFileSync(join(model, file))), sha256, `projection changed ${file}`);
});
