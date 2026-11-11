import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, greedyGeneration, nativeDocument } from '../../qualification/inputs.mjs';
import { float32Tensors } from '../../qualification/tensors.mjs';

await qualify('vectors/ablate', async run => {
    const { model, declarationBytes } = await createCheckpoint(run, 'model');
    const { model: otherModel } = await createCheckpoint(run, 'other-model');
    const pairs = realpathSync(run.input('STER_TEST_PAIRS'));
    const pairBytes = readFileSync(pairs);
    run.report.inputs.push({ path: pairs, sha256: digest(pairBytes) });
    const artifactPath = join(run.fixture, 'direction.json');
    await run.cli(['vector', 'train', '--model', model, '--pairs', pairs, '--output', artifactPath,
        '--method', 'caa', '--layers', 'all', '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    const artifactBytes = readFileSync(artifactPath);
    const artifact = JSON.parse(artifactBytes);
    const sourceBytes = readFileSync(join(model, 'model.safetensors'));
    const source = float32Tensors(sourceBytes, model);
    const frozen = Object.fromEntries(['config.json', 'tokenizer.json', 'model.safetensors']
        .map(file => [file, digest(readFileSync(join(model, file)))]));
    // Actual Llama checkpoint identities produced by the declared fixture:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    const expectedNames = new Set(artifact.vectors.flatMap(vector => [
        `model.layers.${vector.layer}.self_attn.o_proj.weight`,
        `model.layers.${vector.layer}.mlp.down_proj.weight`,
    ]));
    const strengthText = run.input('STER_TEST_ABLATION_STRENGTH');
    const strength = nativeDocument(strengthText, 'STER_TEST_ABLATION_STRENGTH');
    const output = join(run.fixture, 'ablated');
    const report = JSON.parse((await run.cli(['vector', 'ablate', '--model', model, '--vector', artifactPath,
        '--strength', strengthText, '--output', output], SUCCESS)).stdout);
    function inspectRewrite(path, observed) {
        assert.equal(observed.output, path);
        assert.deepEqual(new Set(observed.rewritten), expectedNames, 'the actual residual-stream writes differ from the learned layers');
        assert.equal(new Set(observed.rewritten).size, observed.rewritten.length, 'one tensor was reported rewritten more than once');
        assert.equal(observed.total_tensors, source.size);
        assert.equal(observed.copied_tensors + observed.rewritten.length, source.size);
        assert.deepEqual(readdirSync(path).sort(), [...observed.files].sort(), 'the rewrite report does not account for the written checkpoint');
        const tensors = float32Tensors(readFileSync(join(path, 'model.safetensors')), path);
        assert.deepEqual(new Set(tensors.keys()), new Set(source.keys()), 'the rewrite lost or invented model tensors');
        for (const [name, before] of source) {
            const after = tensors.get(name);
            assert.deepEqual(after.shape, before.shape, `ablation changed tensor shape ${name}`);
            if (expectedNames.has(name)) {
                assert.notEqual(after.sha256, before.sha256, `the declared intervention did not actually rewrite ${name}`);
            } else {
                assert.equal(after.sha256, before.sha256, `ablation changed unrelated tensor ${name}`);
            }
        }
        assert.deepEqual(readFileSync(join(path, 'config.json')), readFileSync(join(model, 'config.json')));
        assert.deepEqual(readFileSync(join(path, 'tokenizer.json')), readFileSync(join(model, 'tokenizer.json')));
        run.report.states.push({ kind: 'rewritten-checkpoint', path, report: observed,
            tensors: Array.from(tensors.values(), ({ name, shape, sha256 }) => ({ name, shape, sha256 })) });
        return tensors;
    }
    const rewritten = inspectRewrite(output, report);
    const nativeOutput = join(run.fixture, 'native-ablated');
    const native = await run.api('vector/ablate', { model, vector: artifactPath, strength, output: nativeOutput }, SUCCESS);
    const nativeTensors = inspectRewrite(nativeOutput, native.document);
    for (const [name, tensor] of rewritten) {
        assert.equal(nativeTensors.get(name).sha256, tensor.sha256, `native and CLI rewrites disagree on actual tensor ${name}`);
    }

    const generation = greedyGeneration(run);
    const prompt = JSON.parse(declarationBytes).verification.prompt;
    const baseline = JSON.parse((await run.cli(['generate', '--model', model, '--prompt', prompt, ...generation.flags], SUCCESS)).stdout);
    const generated = JSON.parse((await run.cli(['generate', '--model', output, '--prompt', prompt, ...generation.flags], SUCCESS)).stdout);
    assert.notEqual(generated.text, baseline.text,
        'qualification requires an active permanently rewritten model, not only changed bookkeeping; use a declaration/prompt/strength that exercises inference');
    const consumed = await run.api('generate', { model: nativeOutput, prompt, ...generation.request }, SUCCESS);
    assert.equal(consumed.document.text, generated.text, 'the actual native-written checkpoint is not equivalent in the generation consumer');
    run.report.states.push({ kind: 'permanent-generation-effect', prompt, baseline, generated, native: consumed.document });

    // The zero projection coefficient is the mathematical no-op refusal boundary:
    // https://github.com/wisent-ai/ster/blob/main/src/tune/artifact/ablate.rs
    const zero = 0;
    const unchanged = join(run.fixture, 'zero-strength');
    await run.cli(['vector', 'ablate', '--model', model, '--vector', artifactPath,
        '--strength', String(zero), '--output', unchanged], FAILURE);
    await run.api('vector/ablate', { model, vector: artifactPath, strength: zero, output: unchanged }, FAILURE);
    assert.ok(!existsSync(unchanged), 'a zero-strength refusal wrote a checkpoint');
    const incompatible = join(run.fixture, 'incompatible');
    await run.cli(['vector', 'ablate', '--model', otherModel, '--vector', artifactPath,
        '--strength', strengthText, '--output', incompatible], FAILURE);
    await run.api('vector/ablate', { model: otherModel, vector: artifactPath, strength, output: incompatible }, FAILURE);
    assert.ok(!existsSync(incompatible), 'incompatible-model ablation wrote a checkpoint');
    const missingOutput = join(run.fixture, 'missing-strength');
    await run.cli(['vector', 'ablate', '--model', model, '--vector', artifactPath,
        '--output', missingOutput], USAGE);
    const missing = await run.api('vector/ablate', { model, vector: artifactPath, output: missingOutput }, USAGE);
    assert.deepEqual(missing.events.filter(event => event.type === 'log'), []);
    assert.ok(!existsSync(missingOutput));
    assert.deepEqual(readFileSync(artifactPath), artifactBytes);
    assert.deepEqual(readFileSync(pairs), pairBytes);
    for (const [file, sha256] of Object.entries(frozen)) {
        assert.equal(digest(readFileSync(join(model, file))), sha256, `ablation changed source ${file}`);
    }
});
