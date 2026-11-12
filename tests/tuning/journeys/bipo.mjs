import assert from 'node:assert/strict';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, greedyGeneration, nativeInteger, trainingFlags, tuningInputs } from '../../qualification/inputs.mjs';

await qualify('tuning/bipo', async run => {
    const { model, config, declarationBytes } = await createCheckpoint(run, 'model');
    const pairPath = realpathSync(run.input('STER_TEST_PAIRS'));
    const pairBytes = readFileSync(pairPath);
    const pairs = JSON.parse(pairBytes);
    run.report.inputs.push({ path: pairPath, sha256: digest(pairBytes) });
    const base = Object.fromEntries(['config.json', 'tokenizer.json', 'model.safetensors']
        .map(file => [file, digest(readFileSync(join(model, file)))]));
    const output = join(run.fixture, 'learned.json');
    const profile = tuningInputs(run, 'bipo', { model, pairs: pairPath, output });
    const expectedLayer = Number(JSON.stringify(profile.request.layer));
    const trained = JSON.parse((await run.cli(['tune', 'bipo', ...profile.flags], SUCCESS)).stdout);
    const artifactBytes = readFileSync(output);
    const artifact = JSON.parse(artifactBytes);
    assert.equal(trained.path, output);
    assert.equal(artifact.method, 'bipo', 'the persisted direction must identify its actual producer');
    assert.equal(artifact.model, model);
    assert.deepEqual(artifact.vectors.map(vector => vector.layer), [expectedLayer]);
    for (const vector of artifact.vectors) {
        assert.equal(vector.values.length, config.hidden_size, 'the saved direction must fit the actual consumer width');
        assert.ok(vector.values.every(Number.isFinite), 'the saved direction contains non-finite coefficients');
    }
    assert.equal(trained.report.trained_pairs, pairs.pairs.length, 'this qualification profile must train every supplied pair');
    assert.equal(trained.report.layer, expectedLayer);
    run.report.states.push({ kind: 'learned-direction', path: output, sha256: digest(artifactBytes), report: trained.report, vectors: artifact.vectors });

    const nativePath = join(run.fixture, 'native-learned.json');
    const repeated = await run.api('tune/bipo', { ...profile.request, output: nativePath }, SUCCESS);
    const nativeArtifact = JSON.parse(readFileSync(nativePath));
    assert.equal(repeated.document.path, nativePath);
    assert.deepEqual(nativeArtifact.vectors, artifact.vectors, 'identical declared inputs must train the same direction across frontends');
    assert.equal(repeated.document.report.final_loss, trained.report.final_loss);
    run.report.states.push({ kind: 'native-learned-direction', path: nativePath, sha256: digest(readFileSync(nativePath)), report: repeated.document.report });

    const prompt = JSON.parse(declarationBytes).verification.prompt;
    const generation = greedyGeneration(run);
    const plain = JSON.parse((await run.cli(['generate', '--model', model, '--prompt', prompt, ...generation.flags], SUCCESS)).stdout);
    const steered = JSON.parse((await run.cli(['generate', '--model', model, '--prompt', prompt,
        '--vector', output, '--strength', String(trained.report.strength), ...generation.flags], SUCCESS)).stdout);
    assert.notEqual(steered.text, plain.text,
        'qualification requires an observably active learned direction, not an accepted no-op; supply a training profile and fixture prompt that exercise generation');
    const nativeGeneration = await run.api('generate', { model, prompt, ...generation.request,
        steering: [{ vector: output, strength: trained.report.strength }] }, SUCCESS);
    assert.equal(nativeGeneration.document.text, steered.text, 'the native consumer must apply the same learned direction');
    run.report.states.push({ kind: 'generation-effect', prompt, baseline: plain, steered, native: nativeGeneration.document });

    async function refusal(name, request, status) {
        await run.cli(['tune', 'bipo', ...trainingFlags(request)], status);
        assert.ok(!existsSync(request.output), `${name} CLI refusal wrote a steering artifact`);
        const native = await run.api('tune/bipo', request, status);
        assert.equal(typeof native.document.error, 'string');
        assert.ok(!existsSync(request.output), `${name} native refusal wrote a steering artifact`);
        if (status === USAGE) {
            assert.deepEqual(native.events.filter(event => event.type === 'log'), [], 'schema refusals must precede model execution');
        }
        run.report.states.push({ kind: 'refusal', name, output: request.output, exists: existsSync(request.output), document: native.document });
    }
    await refusal('outside-model', { ...profile.request, layer: nativeInteger(String(config.num_hidden_layers)),
        output: join(run.fixture, 'outside.json') }, FAILURE);
    const missingWarmup = { ...profile.request, output: join(run.fixture, 'missing-warmup.json') };
    delete missingWarmup.warmupSteps;
    await refusal('missing-warmup', missingWarmup, USAGE);
    const missingLayer = { ...profile.request, output: join(run.fixture, 'missing-layer.json') };
    delete missingLayer.layer;
    await refusal('missing-layer', missingLayer, USAGE);
    assert.deepEqual(readFileSync(output), artifactBytes, 'a later operation changed the completed learned direction');
    assert.deepEqual(readFileSync(pairPath), pairBytes, 'qualification changed the supplied pair set');
    assert.deepEqual(readFileSync(profile.path), profile.bytes, 'qualification changed the supplied training profile');
    for (const [file, sha256] of Object.entries(base)) {
        assert.equal(digest(readFileSync(join(model, file))), sha256, `training or generation changed the frozen base ${file}`);
    }
});
