import assert from 'node:assert/strict';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, digest, qualify } from '../qualification/runtime.mjs';
import { createCheckpoint, greedyGeneration } from '../qualification/inputs.mjs';

await qualify('generate', async run => {
    const { model, declaration, declarationBytes } = await createCheckpoint(run, 'checkpoint');
    const source = join(run.repo, 'docs', 'examples', 'prompts.json');
    const originalBytes = readFileSync(source);
    const input = JSON.parse(originalBytes);
    assert.notDeepEqual(input.prompts, [], 'qualification needs the real nonempty prompt set');
    const generation = greedyGeneration(run);
    run.report.inputs.push({ path: source, sha256: digest(originalBytes) });
    const output = join(run.fixture, 'answers.json');
    const batch = await run.cli(['generate', '--model', model, '--prompts', source, '--output', output, ...generation.flags], SUCCESS);
    assert.equal(batch.stdout, `${output}\n`);
    const bytes = readFileSync(output);
    const answers = JSON.parse(bytes);
    const expected = [];
    for (const prompt of input.prompts) {
        const result = await run.api('generate', { model, prompt, ...generation.request }, SUCCESS);
        assert.equal(typeof result.document.text, 'string');
        const single = await run.cli(['generate', '--model', model, '--prompt', prompt, ...generation.flags], SUCCESS);
        assert.equal(single.stdout, `${result.document.text}\n`, 'single-prompt CLI and native request results disagree');
        expected.push({ prompt, model_output: result.document.text });
    }
    assert.deepEqual(answers, expected, 'persisted batch must preserve all prompts, their order and each independently generated answer');
    run.report.states.push({ path: output, sha256: digest(bytes), answers });

    const empty = join(run.fixture, 'empty.json');
    writeFileSync(empty, JSON.stringify({ ...input, prompts: [] }));
    const refusedOutput = join(run.fixture, 'refused.json');
    await run.cli(['generate', '--model', model, '--prompts', empty, '--output', refusedOutput, ...generation.flags], FAILURE);
    assert.ok(!existsSync(refusedOutput), 'empty-set refusal wrote a batch artifact');
    assert.deepEqual(readFileSync(output), bytes, 'refusal changed the completed batch');
    run.report.states.push({ path: refusedOutput, exists: existsSync(refusedOutput) });
    assert.deepEqual(readFileSync(source), originalBytes, 'qualification changed the checked-in prompt set');
    assert.deepEqual(readFileSync(declaration), declarationBytes, 'qualification changed the supplied fixture declaration');
});
