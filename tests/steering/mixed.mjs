import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../qualification/runtime.mjs';
import { createCheckpoint, greedyGeneration } from '../qualification/inputs.mjs';

await qualify('steering/mixed', async run => {
    const { model, config, declaration, declarationBytes } = await createCheckpoint(run, 'checkpoint');
    const generation = greedyGeneration(run);
    const prompt = JSON.parse(declarationBytes).verification.prompt;
    const strength = Number(run.input('STER_TEST_STRENGTH'));
    // Additive identity and zero-based final layer are mathematical/native boundaries:
    // https://ster.wisent.com/docs/steering
    const zeroStrength = 0;
    assert.ok(Number.isFinite(strength) && strength !== zeroStrength, 'qualification needs an explicit finite nonzero steering strength');
    const doubledStrength = strength + strength;
    assert.ok(Number.isFinite(doubledStrength), 'the declared doubled steering experiment must remain finite');
    // Native model layers are indexed from zero through count minus one:
    // https://ster.wisent.com/docs/cli/vector-train
    const lastLayer = config.num_hidden_layers - 1;
    run.report.experiment = { prompt, strength, doubledStrength, lastLayer };
    const source = join(run.repo, 'docs', 'examples', 'pairs.json');
    const originalBytes = readFileSync(source);
    const input = JSON.parse(originalBytes);
    run.report.inputs.push({ path: source, sha256: digest(originalBytes) });
    const opposite = join(run.fixture, 'opposite.json');
    writeFileSync(opposite, JSON.stringify({ ...input, trait_name: 'restless',
        pairs: input.pairs.map(({ positive, negative, ...metadata }) => ({ ...metadata, positive: negative, negative: positive })) }));
    const calm = join(run.fixture, 'calm.ster.json');
    const restless = join(run.fixture, 'restless.ster.json');
    for (const [pairs, output] of [[source, calm], [opposite, restless]]) {
        await run.cli(['vector', 'train', '--model', model, '--pairs', pairs, '--output', output,
            '--layers', String(lastLayer), '--method', 'caa', '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    }
    const artifacts = [calm, restless].map(path => ({ path, bytes: readFileSync(path) }));
    const generate = parts => run.cli(['generate', '--model', model, '--prompt', prompt, ...generation.flags, ...parts], SUCCESS);
    const plain = await generate([]);
    const cancelled = await generate(['--vector', calm, '--strength', String(strength), '--vector', restless, '--strength', String(strength)]);
    assert.equal(cancelled.stdout, plain.stdout, 'opposite learned directions at equal strengths must cancel');
    const doubled = await generate(['--vector', calm, '--strength', String(doubledStrength)]);
    assert.notEqual(doubled.stdout, plain.stdout, 'qualification must observe active steering, not only equal answers from an ineffective experiment');
    const duplicate = await generate(['--vector', calm, '--strength', String(strength), '--vector', calm, '--strength', String(strength)]);
    assert.equal(duplicate.stdout, doubled.stdout, 'duplicate directions must compose to the sum of their strengths');
    const body = { model, prompt, ...generation.request, steering: [{ vector: calm, strength }, { vector: calm, strength }] };
    const requested = await run.api('generate', body, SUCCESS);
    assert.equal(duplicate.stdout, `${requested.document.text}\n`, 'native request generation must apply the same actual mixture');
    const missing = await run.cli(['generate', '--model', model, '--prompt', prompt, ...generation.flags,
        '--vector', calm, '--vector', restless, '--strength', String(strength)], FAILURE);
    assert.equal(missing.stdout, '', 'a missing-strength refusal must not print a generated answer');
    const zero = await run.cli(['generate', '--model', model, '--prompt', prompt, ...generation.flags,
        '--vector', calm, '--strength', String(zeroStrength), '--vector', restless, '--strength', String(zeroStrength)], FAILURE);
    assert.equal(zero.stdout, '', 'an all-zero refusal must not print a generated answer');
    await run.api('generate', { ...body, steering: [{ vector: calm }] }, USAGE);
    await run.api('generate', { ...body, steering: [{ vector: calm, strength: zeroStrength }] }, FAILURE);
    for (const { path, bytes } of artifacts) {
        assert.deepEqual(readFileSync(path), bytes, 'generation or refusal changed a learned input artifact');
        run.report.states.push({ path, sha256: digest(bytes), artifact: JSON.parse(bytes) });
    }
    run.report.states.push({ plain: plain.stdout, cancelled: cancelled.stdout, doubled: doubled.stdout, duplicate: duplicate.stdout,
        requested: requested.document });
    assert.deepEqual(readFileSync(source), originalBytes);
    assert.deepEqual(readFileSync(declaration), declarationBytes);
});
