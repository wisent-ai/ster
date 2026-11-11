import assert from 'node:assert/strict';
import { existsSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../qualification/runtime.mjs';
import { nativeDocument } from '../qualification/inputs.mjs';
import { float32Tensors } from '../qualification/tensors.mjs';

await qualify('fixture', async run => {
    const declaration = realpathSync(run.input('STER_TEST_MODEL_SPEC'));
    const declarationBytes = readFileSync(declaration);
    const spec = JSON.parse(declarationBytes);
    const nativeSpec = nativeDocument(declarationBytes, declaration);
    const configPath = realpathSync(resolve(dirname(declaration), spec.config));
    const declaredTokenizer = Object.hasOwn(spec.tokenizer, 'from');
    const tokenizerSource = realpathSync(resolve(dirname(declaration), declaredTokenizer ? spec.tokenizer.from : spec.tokenizer));
    const tokenizerSourceBytes = readFileSync(tokenizerSource);
    const configBytes = readFileSync(configPath);
    let tokenizerPath = tokenizerSource;
    if (declaredTokenizer) {
        tokenizerPath = join(run.fixture, 'standalone-tokenizer.json');
        const preview = await run.api('model/tokenizer/create', { from: tokenizerSource, output: tokenizerPath }, SUCCESS);
        run.report.states.push({ path: tokenizerPath, nativeTokenizer: preview.document, sha256: digest(readFileSync(tokenizerPath)) });
    }
    const tokenizerBytes = readFileSync(tokenizerPath);
    const config = JSON.parse(configBytes);
    const nativeConfig = nativeDocument(configBytes, configPath);
    const inputs = [
        { path: declaration, bytes: declarationBytes },
        { path: configPath, bytes: configBytes },
        { path: tokenizerSource, bytes: tokenizerSourceBytes },
    ];
    run.report.inputs = inputs.map(({ path, bytes }) => ({ path, sha256: digest(bytes) }));

    function checkpoint(path, report) {
        assert.equal(report.path, path, 'the result must name the actual output directory');
        assert.deepEqual(readFileSync(join(path, 'config.json')), configBytes, 'the native config must preserve the supplied document');
        assert.deepEqual(readFileSync(join(path, 'tokenizer.json')), tokenizerBytes, 'the native tokenizer must preserve the supplied document');
        assert.deepEqual(report.verification.map(item => item.layer), spec.verification.layers);
        assert.deepEqual(report.verification.map(item => item.width), spec.verification.layers.map(() => config.hidden_size));
        const bytes = readFileSync(join(path, 'model.safetensors'));
        const tensors = Array.from(
            float32Tensors(bytes, join(path, 'model.safetensors')).values(),
            ({ name, shape, sha256 }) => ({ name, shape, sha256 }));
        assert.equal(report.tensor_count, tensors.length, 'the result count must agree with the actual checkpoint');
        const state = { path, config: digest(readFileSync(join(path, 'config.json'))), tokenizer: digest(readFileSync(join(path, 'tokenizer.json'))), weights: digest(bytes), tensors };
        run.report.states.push(state);
        return state;
    }
    function unchanged(before) {
        assert.equal(digest(readFileSync(join(before.path, 'config.json'))), before.config);
        assert.equal(digest(readFileSync(join(before.path, 'tokenizer.json'))), before.tokenizer);
        assert.equal(digest(readFileSync(join(before.path, 'model.safetensors'))), before.weights);
    }
    function variant(name, changes) {
        const value = { ...nativeSpec, config: configPath, tokenizer: tokenizerPath, ...changes };
        const path = join(run.fixture, `${name}.json`);
        writeFileSync(path, JSON.stringify(value));
        return path;
    }
    async function refuse(name, from) {
        const output = join(run.fixture, name);
        const result = await run.api('model/fixture/create', { from, output }, FAILURE);
        assert.equal(typeof result.document.error, 'string');
        assert.ok(!existsSync(output), `failed ${name} left an output directory`);
        run.report.states.push({ path: output, exists: existsSync(output), failure: result.document });
    }

    const first = join(run.fixture, 'first');
    const created = JSON.parse((await run.cli(['model', 'fixture', 'create', '--from', declaration, '--output', first], SUCCESS)).stdout);
    const firstState = checkpoint(first, created);
    const second = join(run.fixture, 'second');
    const repeated = await run.api('model/fixture/create', { from: declaration, output: second }, SUCCESS);
    const secondState = checkpoint(second, repeated.document);
    assert.equal(secondState.weights, firstState.weights, 'the same declaration must reproduce the native checkpoint across real processes');

    await run.cli(['model', 'fixture', 'create', '--from', declaration, '--output', first], FAILURE);
    unchanged(firstState);
    await run.api('model/fixture/create', { from: declaration, output: first }, FAILURE);
    unchanged(firstState);
    const absent = join(run.fixture, 'absent');
    await run.cli(['model', 'fixture', 'create', '--output', absent], USAGE);
    await run.cli(['model', 'fixture', 'create', '--from', declaration], USAGE);
    await run.cli(['toy-model', absent], USAGE);
    assert.ok(!existsSync(absent), 'invalid or retired command syntax wrote output');
    await run.api('model/fixture/create', { output: absent }, USAGE);
    await run.api('model/fixture/create', { from: '', output: absent }, USAGE);
    await run.api('model/fixture/create', { from: declaration, output: absent, unexpected: true }, USAGE);
    assert.ok(!existsSync(absent), 'a refused request body wrote output');

    const missingSeed = { ...nativeSpec.initialization };
    delete missingSeed.seed;
    await refuse('missing-seed', variant('missing-seed-spec', { initialization: missingSeed }));
    await refuse('unknown-field', variant('unknown-field-spec', { unexpected: true }));
    await refuse('missing-declaration', join(run.fixture, 'missing-declaration.json'));
    const malformed = join(run.fixture, 'malformed.json');
    writeFileSync(malformed, '{');
    await refuse('malformed-declaration', malformed);
    await refuse('malformed-tokenizer', variant('malformed-tokenizer-spec', { tokenizer: malformed }));
    await refuse('out-of-range-layer', variant('out-of-range-spec', { verification: { ...nativeSpec.verification, layers: [config.num_hidden_layers] } }));

    const biasedConfig = join(run.fixture, 'unsupported-bias-config.json');
    writeFileSync(biasedConfig, JSON.stringify({ ...nativeConfig, mlp_bias: true }));
    await refuse('unsupported-bias', variant('unsupported-bias-spec', { config: biasedConfig }));

    // A zero residual with zero epsilon is the singular RMS boundary, not
    // a chosen experiment threshold. Drive it through the real decoder.
    const singularConfig = join(run.fixture, 'singular-config.json');
    // Native RMS normalization's singular boundary:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    writeFileSync(singularConfig, JSON.stringify({ ...nativeConfig, rms_norm_eps: 0 }));
    // Native normal sampling with zero scale produces its stated mean:
    // https://docs.rs/rand_distr/0.5.1/rand_distr/struct.Normal.html
    const zero = { mean: 0, scale: 0 };
    const singular = variant('singular-spec', {
        config: singularConfig,
        // The zero residual and normalization scale exercise that same boundary:
        // https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        initialization: { ...nativeSpec.initialization, embedding: zero, projection: zero, normalization: 0, bias: zero },
    });
    await refuse('singular-forward', singular);
    unchanged(firstState);
    unchanged(secondState);
    for (const original of inputs) {
        assert.deepEqual(readFileSync(original.path), original.bytes, `qualification changed operator input ${original.path}`);
    }
});
