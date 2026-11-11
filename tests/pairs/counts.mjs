import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, digest, qualify } from '../qualification/runtime.mjs';
import { createCheckpoint } from '../qualification/inputs.mjs';

await qualify('pairs/counts', async run => {
    const { model, config, declaration, declarationBytes } = await createCheckpoint(run, 'checkpoint');
    const seed = run.input('STER_TEST_SEED');
    run.report.experiment = { seed };
    const pairsPath = join(run.repo, 'docs', 'examples', 'pairs.json');
    const promptsPath = join(run.repo, 'docs', 'examples', 'prompts.json');
    const pairsBytes = readFileSync(pairsPath);
    const promptsBytes = readFileSync(promptsPath);
    const pairs = JSON.parse(pairsBytes);
    const prompts = JSON.parse(promptsBytes);
    run.report.inputs.push({ path: pairsPath, sha256: digest(pairsBytes) }, { path: promptsPath, sha256: digest(promptsBytes) });
    // Native BiPO CSV fields, not a behavior classifier or a question-prefix rule:
    // https://ster.wisent.com/docs/cli/pairs-import
    const unanswered = [{ question: 'Is it calm?', matching: '', not_matching: '' },
        { question: 'Is it quiet?', matching: '', not_matching: '' }];
    function csv(name, rows) {
        const [first] = rows;
        const columns = Object.keys(first);
        const quote = value => `"${value.replaceAll('"', '""')}"`;
        const text = [columns, ...rows.map(row => columns.map(column => row[column]))]
            .map(row => row.map(quote).join(',')).join('\n');
        const path = join(run.fixture, name);
        writeFileSync(path, `${text}\n`);
        run.report.states.push({ path, rows, sha256: digest(readFileSync(path)) });
        return path;
    }
    async function refusal(args, output) {
        await run.cli(args, FAILURE);
        assert.ok(!existsSync(output), 'a refused count operation wrote an artifact');
        run.report.states.push({ path: output, exists: existsSync(output) });
    }
    const unansweredPath = csv('unanswered.csv', unanswered);
    const none = join(run.fixture, 'none.pairs.json');
    await refusal(['pairs', 'import', '--benchmark', 'bipo', '--source', unansweredPath, '--seed', seed, '--output', none], none);
    const [first, ...rest] = unanswered;
    const partialRows = [{ ...first, matching: 'Yes', not_matching: 'No' }, ...rest];
    const partial = csv('partial.csv', partialRows);
    const few = join(run.fixture, 'few.pairs.json');
    await refusal(['pairs', 'import', '--benchmark', 'bipo', '--source', partial, '--count', String(partialRows.length),
        '--seed', seed, '--output', few], few);

    const emptyPairs = join(run.fixture, 'empty.pairs.json');
    writeFileSync(emptyPairs, JSON.stringify({ ...pairs, pairs: [] }));
    const emptyDirection = join(run.fixture, 'empty.ster.json');
    await refusal(['vector', 'train', '--model', model, '--pairs', emptyPairs, '--output', emptyDirection, '--layers', 'all'], emptyDirection);
    const emptyPrompts = join(run.fixture, 'empty.prompts.json');
    writeFileSync(emptyPrompts, JSON.stringify({ ...prompts, prompts: [] }));
    const emptyActivations = join(run.fixture, 'empty.activations.json');
    await refusal(['extract', '--model', model, '--input', emptyPrompts, '--output', emptyActivations, '--layers', 'all'], emptyActivations);

    const expectedLayers = Array.from({ length: config.num_hidden_layers }, (_, layer) => String(layer));
    const activationsPath = join(run.fixture, 'activations.json');
    await run.cli(['extract', '--model', model, '--input', promptsPath, '--output', activationsPath, '--layers', 'all',
        '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    const activationBytes = readFileSync(activationsPath);
    const activations = JSON.parse(activationBytes);
    assert.deepEqual(activations.records.map(record => record.prompt), prompts.prompts, 'activation records must preserve each prompt and its order');
    for (const record of activations.records) {
        assert.deepEqual(new Set(Object.keys(record.layers)), new Set(expectedLayers), 'every record must hold exactly every requested model layer');
        for (const vector of Object.values(record.layers)) {
            assert.equal(vector.length, config.hidden_size);
            assert.ok(vector.every(Number.isFinite), 'persisted activations must be finite');
        }
    }
    run.report.states.push({ path: activationsPath, sha256: digest(activationBytes), activations });
    const directionPath = join(run.fixture, 'calm.ster.json');
    await run.cli(['vector', 'train', '--model', model, '--pairs', pairsPath, '--output', directionPath, '--layers', 'all',
        '--method', 'caa', '--device', 'cpu', '--precision', 'f32', '--chat-template', 'off'], SUCCESS);
    const directionBytes = readFileSync(directionPath);
    const direction = JSON.parse(directionBytes);
    assert.deepEqual(direction.vectors.map(vector => String(vector.layer)), expectedLayers,
        'the persisted direction must contain every requested native layer exactly once');
    for (const { values } of direction.vectors) {
        assert.equal(values.length, config.hidden_size);
        assert.ok(values.every(Number.isFinite));
    }
    const inspection = JSON.parse((await run.cli(['vector', 'inspect', directionPath], SUCCESS)).stdout);
    assert.deepEqual(inspection.layers.map(({ layer, width }) => ({ layer, width })),
        expectedLayers.map(layer => ({ layer: Number(layer), width: config.hidden_size })), 'the actual artifact consumer must see every persisted layer and width');
    run.report.states.push({ path: directionPath, sha256: digest(directionBytes), direction, inspection });
    assert.deepEqual(readFileSync(pairsPath), pairsBytes);
    assert.deepEqual(readFileSync(promptsPath), promptsBytes);
    assert.deepEqual(readFileSync(declaration), declarationBytes);
});
