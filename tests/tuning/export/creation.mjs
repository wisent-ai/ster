import assert from 'node:assert/strict';
import { existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, writeFileSync } from 'node:fs';
import { constants as HTTP } from 'node:http2';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, greedyGeneration, nativeDocument, trainingFlags, tuningInputs } from '../../qualification/inputs.mjs';

await qualify('tuning/export', async run => {
    const { model, config, declarationBytes } = await createCheckpoint(run, 'model');
    const examples = realpathSync(run.input('STER_TEST_EXAMPLES'));
    const exampleBytes = readFileSync(examples);
    run.report.inputs.push({ path: examples, sha256: digest(exampleBytes) });
    const weights = join(run.fixture, 'learned.safetensors');
    const profile = tuningInputs(run, 'sft', { model, examples, output: weights });
    const trained = JSON.parse((await run.cli(['tune', 'sft', ...profile.flags], SUCCESS)).stdout);
    assert.equal(trained.path, weights);
    const sidecar = JSON.parse(readFileSync(join(run.fixture, 'learned.json')));
    const frozen = Object.fromEntries(['config.json', 'tokenizer.json', 'model.safetensors']
        .map(file => [file, digest(readFileSync(join(model, file)))]));
    const adapterDigest = digest(readFileSync(weights));
    const output = join(run.fixture, 'peft');
    const exported = JSON.parse((await run.cli(['tune', 'export', '--model', model,
        '--adapter', weights, '--format', 'peft', '--output', output], SUCCESS)).stdout);
    assert.equal(exported.report.output, output);
    assert.deepEqual(readdirSync(output).sort(), [...exported.report.files].sort(), 'the export report must account for the actual persisted files');
    const adapterConfig = JSON.parse(readFileSync(join(output, 'adapter_config.json')));
    assert.equal(adapterConfig.r, sidecar.rank, 'the real consumer must receive the trained factor dimension');
    assert.equal(adapterConfig.lora_alpha, sidecar.alpha, 'the real consumer must receive the trained update scale');
    assert.equal(adapterConfig.base_model_name_or_path, model);
    assert.deepEqual(readFileSync(join(output, 'tokenizer.json')), readFileSync(join(model, 'tokenizer.json')));
    const fileDigests = Object.fromEntries(exported.report.files.map(file => [file, digest(readFileSync(join(output, file)))]));
    run.report.states.push({ kind: 'export', path: output, report: exported.report, configuration: adapterConfig, files: fileDigests });

    const nativeOutput = join(run.fixture, 'native-peft');
    const native = await run.api('tune/export', { model, adapter: weights, format: 'peft', output: nativeOutput }, SUCCESS);
    assert.equal(native.document.report.output, nativeOutput);
    assert.deepEqual(readdirSync(nativeOutput).sort(), [...exported.report.files].sort());
    for (const [file, sha256] of Object.entries(fileDigests)) {
        assert.equal(digest(readFileSync(join(nativeOutput, file))), sha256, `native export changed the actual portable ${file}`);
    }
    run.report.states.push({ kind: 'native-export', path: nativeOutput, report: native.document.report });

    // Consume the real adapter through vLLM's supported public batch CLI,
    // without owned Python code, inline scripts, a server restart or a mock:
    // https://docs.vllm.ai/en/latest/cli/run-batch/
    // https://docs.vllm.ai/en/v0.18.0/api/vllm/entrypoints/openai/run_batch/
    const consumer = realpathSync(run.input('STER_TEST_VLLM'));
    const consumerVersion = await run.expect(consumer, ['--version'], '', SUCCESS);
    const settingsPath = realpathSync(run.input('STER_TEST_VLLM_SPEC'));
    const settingsBytes = readFileSync(settingsPath);
    const settings = nativeDocument(settingsBytes, settingsPath);
    assert.ok(settings && typeof settings === 'object' && !Array.isArray(settings), 'vLLM resource inputs must be an explicit native option object');
    assert.ok(Object.hasOwn(settings, 'maxLoraRank'), 'the vLLM qualification declaration must state its supported maxLoraRank');
    const template = realpathSync(run.input('STER_TEST_VLLM_TEMPLATE'));
    const templateBytes = readFileSync(template);
    const batchInput = join(run.root, 'consumer-input.jsonl');
    const batchOutput = join(run.root, 'consumer-output.jsonl');
    const staging = join(run.fixture, 'batch-staging');
    mkdirSync(staging);
    const baseName = 'qualification-base';
    const adapterName = 'qualification-adapter';
    const bindings = {
        model, tokenizer: model, servedModelName: baseName, dtype: 'float32',
        maxModelLen: String(config.max_position_embeddings),
        loraModules: JSON.stringify({ name: adapterName, path: output, base_model_name: baseName }),
        chatTemplate: template, inputFile: batchInput, outputFile: batchOutput, outputTmpDir: staging,
    };
    for (const field of Object.keys({ ...bindings, enableLora: true, enforceEager: true })) {
        assert.ok(!Object.hasOwn(settings, field), `${settingsPath}.${field} conflicts with an isolated consumer binding`);
    }
    const generation = greedyGeneration(run);
    const prompt = JSON.parse(declarationBytes).verification.prompt;
    const jobs = [
        { custom_id: 'baseline', model: baseName },
        { custom_id: 'adapted', model: adapterName },
    ].map(({ custom_id, model: selectedModel }) => ({
        custom_id, method: 'POST', url: '/v1/chat/completions',
        body: { model: selectedModel, messages: [{ role: 'user', content: prompt }],
            max_tokens: generation.request.maxNewTokens, temperature: generation.request.temperature,
            seed: generation.request.seed },
    }));
    writeFileSync(batchInput, `${jobs.map(job => JSON.stringify(job)).join('\n')}\n`);
    const consumerInputs = { ...settings, ...bindings };
    run.report.inputs.push({ path: settingsPath, sha256: digest(settingsBytes) }, { path: template, sha256: digest(templateBytes) });
    run.report.consumer = { executable: consumer, sha256: digest(readFileSync(consumer)), version: consumerVersion.stdout,
        stderr: consumerVersion.stderr, inputs: consumerInputs, batchInput, batchOutput };
    await run.expect(consumer, ['run-batch', ...trainingFlags(consumerInputs), '--enable-lora', '--enforce-eager'], '', SUCCESS);
    const rows = readFileSync(batchOutput, 'utf8').trim().split('\n').map(line => JSON.parse(line));
    const responses = new Map(rows.map(row => [row.custom_id, row]));
    assert.equal(responses.size, rows.length, 'the real batch consumer returned duplicate identities');
    assert.deepEqual(new Set(responses.keys()), new Set(jobs.map(job => job.custom_id)));
    const answers = new Map();
    for (const job of jobs) {
        const row = responses.get(job.custom_id);
        assert.equal(row.error, null, JSON.stringify(row));
        assert.equal(row.response.status_code, HTTP.HTTP_STATUS_OK, JSON.stringify(row));
        assert.equal(row.response.body.model, job.body.model, 'the consumer must report the requested base or loaded adapter identity');
        const [choice] = row.response.body.choices;
        assert.deepEqual(row.response.body.choices, [choice], 'this qualification requires one completion per declared request');
        assert.equal(typeof choice.message.content, 'string');
        answers.set(job.custom_id, choice.message.content);
    }
    assert.notEqual(answers.get('adapted'), answers.get('baseline'),
        'qualification requires an active externally consumed adapter, not merely written files or a successful process; choose inputs that exercise its effect');
    run.report.states.push({ kind: 'real-peft-consumer', input: batchInput, output: batchOutput, responses: rows });

    const { model: otherModel } = await createCheckpoint(run, 'other-model');
    const wrongOutput = join(run.fixture, 'wrong-model');
    await run.cli(['tune', 'export', '--model', otherModel, '--adapter', weights,
        '--format', 'peft', '--output', wrongOutput], FAILURE);
    const wrong = await run.api('tune/export', { model: otherModel, adapter: weights, format: 'peft', output: wrongOutput }, FAILURE);
    assert.equal(typeof wrong.document.error, 'string');
    assert.ok(!existsSync(wrongOutput), 'wrong-model export wrote a portable artifact');
    const absent = join(run.fixture, 'refused-format');
    await run.cli(['tune', 'export', '--model', model, '--adapter', weights, '--output', absent], USAGE);
    await run.cli(['tune', 'export', '--model', model, '--adapter', weights,
        '--format', 'unsupported', '--output', absent], USAGE);
    const missing = await run.api('tune/export', { model, adapter: weights, output: absent }, USAGE);
    const invalid = await run.api('tune/export', { model, adapter: weights, format: 'unsupported', output: absent }, USAGE);
    assert.deepEqual(missing.events.filter(event => event.type === 'log'), []);
    assert.deepEqual(invalid.events.filter(event => event.type === 'log'), []);
    assert.ok(!existsSync(absent), 'a refused native format wrote output');
    run.report.states.push({ kind: 'export-refusals', wrong: wrong.document, missing: missing.document, invalid: invalid.document });
    assert.equal(digest(readFileSync(weights)), adapterDigest, 'export changed the trained source weights');
    for (const [file, sha256] of Object.entries(fileDigests)) {
        assert.equal(digest(readFileSync(join(output, file))), sha256, `consumer or refusal changed portable ${file}`);
    }
    for (const [file, sha256] of Object.entries(frozen)) {
        assert.equal(digest(readFileSync(join(model, file))), sha256, `export or inference changed frozen base ${file}`);
    }
    assert.deepEqual(readFileSync(examples), exampleBytes);
    assert.deepEqual(readFileSync(profile.path), profile.bytes);
    assert.deepEqual(readFileSync(settingsPath), settingsBytes);
    assert.deepEqual(readFileSync(template), templateBytes);
});
