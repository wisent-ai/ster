import assert from 'node:assert/strict';
import { copyFileSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../qualification/runtime.mjs';
import { createCheckpoint, nativeInteger, trainingFlags } from '../qualification/inputs.mjs';

await qualify('tuning/system', async run => {
    const { model: bareModel } = await createCheckpoint(run, 'bare-model');
    const { model: chatModel } = await createCheckpoint(run, 'chat-model');
    const source = realpathSync(run.input('STER_TEST_EXAMPLES'));
    const sourceBytes = readFileSync(source);
    const examples = JSON.parse(sourceBytes);
    assert.ok(examples.examples.every(example => !Object.hasOwn(example, 'system')),
        'system qualification requires a baseline example set without system turns');
    const template = realpathSync(run.input('STER_TEST_CHAT_CONFIG'));
    const templateBytes = readFileSync(template);
    copyFileSync(template, join(chatModel, 'tokenizer_config.json'));
    run.report.inputs.push({ path: source, sha256: digest(sourceBytes) }, { path: template, sha256: digest(templateBytes) });
    const system = run.input('STER_TEST_SYSTEM_PROMPT');
    const served = join(run.fixture, 'served.json');
    const blank = join(run.fixture, 'blank.json');
    writeFileSync(served, JSON.stringify({ ...examples, examples: examples.examples.map(example => ({ ...example, system })) }));
    writeFileSync(blank, JSON.stringify({ ...examples, examples: examples.examples.map(example => ({ ...example, system: ' \t ' })) }));
    const servedBytes = readFileSync(served);
    const protectedModels = new Map([bareModel, chatModel].map(model => [model,
        Object.fromEntries(['config.json', 'tokenizer.json', 'model.safetensors']
            .map(file => [file, digest(readFileSync(join(model, file)))]))]));
    const scoring = {
        maxSequence: nativeInteger(run.input('STER_TEST_MAX_SEQUENCE')),
        batchSize: nativeInteger(run.input('STER_TEST_BATCH_SIZE')),
        device: 'cpu', precision: 'f32', chatTemplate: 'auto',
    };
    async function evaluate(model, path) {
        const request = { model, examples: path, ...scoring };
        const cli = JSON.parse((await run.cli(['tune', 'evaluate', ...trainingFlags(request)], SUCCESS)).stdout);
        const native = await run.api('tune/evaluate', request, SUCCESS);
        assert.equal(native.document.loss, cli.loss, 'the actual native and CLI scorers disagree');
        assert.deepEqual(native.document.entries, cli.entries);
        assert.equal(cli.chat_template, 'applied', 'the actual consumer did not apply the declared template');
        assert.equal(cli.evaluated, examples.examples.length, 'this qualification profile must score every supplied example');
        assert.deepEqual(cli.entries.map(entry => entry.prompt), examples.examples.map(entry => entry.prompt));
        assert.deepEqual(cli.entries.map(entry => entry.completion), examples.examples.map(entry => entry.completion));
        assert.ok(Number.isFinite(cli.loss) && Number.isFinite(cli.perplexity));
        run.report.states.push({ kind: 'evaluation', model, examples: path, cli, native: native.document });
        return cli;
    }
    const unconditioned = await evaluate(chatModel, source);
    const conditioned = await evaluate(chatModel, served);
    assert.deepEqual(unconditioned.entries.map(entry => Object.hasOwn(entry, 'system')),
        examples.examples.map(() => false), 'an absent system turn must remain absent, not become a substituted prompt');
    assert.deepEqual(conditioned.entries.map(entry => entry.system), examples.examples.map(() => system));
    assert.equal(conditioned.completion_tokens, unconditioned.completion_tokens,
        'the system prefix became part of the scored completion target');
    assert.deepEqual(conditioned.entries.map(entry => entry.completion_tokens), unconditioned.entries.map(entry => entry.completion_tokens));
    assert.notEqual(conditioned.loss, unconditioned.loss,
        'qualification requires observable system conditioning, not a template that accepts but ignores the system turn');

    for (const request of [
        { model: chatModel, examples: blank, ...scoring },
        { model: bareModel, examples: served, ...scoring },
        { model: chatModel, examples: served, ...scoring, chatTemplate: 'off' },
    ]) {
        await run.cli(['tune', 'evaluate', ...trainingFlags(request)], FAILURE);
        const refused = await run.api('tune/evaluate', request, FAILURE);
        assert.equal(typeof refused.document.error, 'string');
        run.report.states.push({ kind: 'system-refusal', request, document: refused.document });
    }
    const missingSequence = { model: chatModel, examples: served, ...scoring };
    delete missingSequence.maxSequence;
    await run.cli(['tune', 'evaluate', ...trainingFlags(missingSequence)], USAGE);
    const missing = await run.api('tune/evaluate', missingSequence, USAGE);
    assert.deepEqual(missing.events.filter(event => event.type === 'log'), [], 'missing scoring limits must refuse before execution');
    for (const [model, files] of protectedModels) {
        for (const [file, sha256] of Object.entries(files)) {
            assert.equal(digest(readFileSync(join(model, file))), sha256, `evaluation changed frozen ${model}/${file}`);
        }
    }
    assert.deepEqual(readFileSync(served), servedBytes);
    assert.deepEqual(readFileSync(source), sourceBytes);
    assert.deepEqual(readFileSync(template), templateBytes);
    assert.deepEqual(readFileSync(join(chatModel, 'tokenizer_config.json')), templateBytes);
});
