import assert from 'node:assert/strict';
import { cpSync, existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../../qualification/runtime.mjs';
import { createCheckpoint, nativeInteger, trainingFlags, tuningInputs } from '../../qualification/inputs.mjs';
import { publishedCheckpoint, stopAfterPublication } from '../../qualification/publication.mjs';

await qualify('tuning/checkpoints', async run => {
    const { model, declaration, declarationBytes } = await createCheckpoint(run, 'model');
    const sources = new Map();
    const supplied = new Map();
    for (const [field, variable] of [
        ['examples', 'STER_TEST_EXAMPLES'], ['pairs', 'STER_TEST_PAIRS'],
        ['prompts', 'STER_TEST_PROMPTS'], ['decisions', 'STER_TEST_DECISIONS'],
    ]) {
        const path = realpathSync(run.input(variable));
        const bytes = readFileSync(path);
        sources.set(field, path);
        supplied.set(path, bytes);
        run.report.inputs.push({ path, sha256: digest(bytes), field });
    }
    const frozen = Object.fromEntries(['config.json', 'tokenizer.json', 'model.safetensors']
        .map(file => [file, digest(readFileSync(join(model, file)))]));
    const every = nativeInteger(run.input('STER_TEST_CHECKPOINT_EVERY'));
    const compareWeights = (actual, expected) => assert.deepEqual(readFileSync(actual), readFileSync(expected),
        'resumed trained weights differ from actual uninterrupted training');
    const compareDirections = (actual, expected) => assert.deepEqual(
        JSON.parse(readFileSync(actual)).vectors, JSON.parse(readFileSync(expected)).vectors,
        'resumed learned direction differs from actual uninterrupted training');
    // Actual public training operation and input-schema identities, not classifiers.
    const cases = [
        { objective: 'sft', input: { examples: sources.get('examples') }, suffix: '.safetensors', compare: compareWeights },
        { objective: 'dpo', input: { pairs: sources.get('pairs') }, suffix: '.safetensors', compare: compareWeights },
        { objective: 'reward', input: { pairs: sources.get('pairs') }, suffix: '.safetensors', compare: compareWeights },
        { objective: 'grpo', input: { prompts: sources.get('prompts') }, suffix: '.safetensors', compare: compareWeights },
        { objective: 'bipo', input: { pairs: sources.get('pairs') }, suffix: '.json', compare: compareDirections },
        { objective: 'decide', input: { examples: sources.get('decisions') }, suffix: '.safetensors', compare: compareWeights },
    ];
    const completed = new Map();
    for (const flow of cases) {
        const directory = join(run.fixture, flow.objective);
        mkdirSync(directory);
        const wholePath = join(directory, `whole${flow.suffix}`);
        const profile = tuningInputs(run, flow.objective, { model, ...flow.input, output: wholePath });
        supplied.set(profile.path, profile.bytes);
        const whole = JSON.parse((await run.cli(['tune', flow.objective, ...profile.flags], SUCCESS)).stdout);
        assert.equal(whole.path, wholePath);
        assert.ok(Number.isSafeInteger(whole.report.steps), `${flow.objective} returned an unrepresentable step count`);
        assert.ok(BigInt(every.rawJSON) < BigInt(whole.report.steps),
            `${flow.objective} qualification requires an actual intermediate checkpoint; interval=${every.rawJSON}, observed steps=${whole.report.steps}`);
        const checkpoint = join(directory, 'checkpoints');
        mkdirSync(checkpoint);
        const stoppedPath = join(directory, `stopped${flow.suffix}`);
        const interruptedRequest = { ...profile.request, output: stoppedPath, checkpoint, checkpointEvery: every };
        const observer = stopAfterPublication(checkpoint);
        let interrupted;
        try {
            interrupted = await run.capture(run.binary,
                ['tune', flow.objective, ...trainingFlags(interruptedRequest)], '', observer.control);
        } finally {
            observer.close();
        }
        assert.equal(interrupted.startError, null, interrupted.startError);
        assert.equal(interrupted.stdinError, null, interrupted.stdinError);
        assert.deepEqual(interrupted.readErrors, []);
        assert.equal(interrupted.controlError, null, interrupted.controlError);
        assert.equal(interrupted.controlStopError, null, interrupted.controlStopError);
        assert.equal(interrupted.signal, 'SIGTERM', 'a completed process is not interruption evidence');
        assert.equal(interrupted.status, null);
        assert.ok(!existsSync(stoppedPath), 'the supposedly interrupted run already wrote its completed output');
        const saved = publishedCheckpoint(checkpoint);
        assert.ok(saved.state.position.step < whole.report.steps,
            `${flow.objective} completed all optimizer steps before interruption`);
        run.report.states.push({ kind: 'interrupted-checkpoint', objective: flow.objective,
            observation: observer.observation(), directory: checkpoint, saved });

        // An actual valid unreferenced legacy-named attempt must not be overwritten.
        const orphan = join(checkpoint, `step-${whole.report.steps}`);
        cpSync(saved.path, orphan, { recursive: true, force: false, errorOnExist: true });
        const orphanState = digest(readFileSync(join(orphan, 'state.json')));
        const orphanTensors = digest(readFileSync(join(orphan, 'tensors.safetensors')));
        const resumedPath = join(directory, `resumed${flow.suffix}`);
        const resumed = await run.api(`tune/${flow.objective}`, { ...profile.request,
            output: resumedPath, resume: checkpoint, checkpoint, checkpointEvery: every }, SUCCESS);
        assert.equal(resumed.document.path, resumedPath);
        assert.equal(resumed.document.report.resumed_from, saved.state.position.step);
        assert.equal(resumed.document.report.steps, whole.report.steps);
        assert.equal(resumed.document.report.final_loss, whole.report.final_loss);
        flow.compare(resumedPath, wholePath);
        assert.equal(digest(readFileSync(join(orphan, 'state.json'))), orphanState,
            'resume overwrote an unreferenced checkpoint state');
        assert.equal(digest(readFileSync(join(orphan, 'tensors.safetensors'))), orphanTensors,
            'resume overwrote unreferenced optimizer tensors');
        const latest = publishedCheckpoint(checkpoint);
        run.report.states.push({ kind: 'resumed-training', objective: flow.objective,
            whole: whole.report, resumed: resumed.document.report,
            wholePath, resumedPath, weights: digest(readFileSync(resumedPath)), latest });
        completed.set(flow.objective, { profile, directory, checkpoint, whole, latest });
    }

    const sft = completed.get('sft');
    const request = sft.profile.request;
    const differentSeed = nativeInteger(run.input('STER_TEST_DIFFERENT_SEED'));
    assert.notEqual(BigInt(differentSeed.rawJSON), BigInt(request.seed.rawJSON),
        'the mismatch qualification seed must actually differ from the run declaration');
    async function refuse(name, body, status) {
        await run.cli(['tune', 'sft', ...trainingFlags(body)], status);
        assert.ok(!existsSync(body.output), `${name} CLI refusal wrote trained output`);
        const native = await run.api('tune/sft', body, status);
        assert.equal(typeof native.document.error, 'string');
        assert.ok(!existsSync(body.output), `${name} native refusal wrote trained output`);
        if (status === USAGE) assert.deepEqual(native.events.filter(event => event.type === 'log'), []);
        run.report.states.push({ kind: 'checkpoint-refusal', name, request: body, document: native.document });
    }
    await refuse('different-seed', { ...request, seed: differentSeed, resume: sft.checkpoint,
        output: join(sft.directory, 'different.safetensors') }, FAILURE);
    await refuse('missing-interval', { ...request, checkpoint: join(sft.directory, 'half'),
        output: join(sft.directory, 'half.safetensors') }, USAGE);
    await refuse('missing-directory', { ...request, checkpointEvery: every,
        output: join(sft.directory, 'no-directory.safetensors') }, USAGE);
    await refuse('missing-state', { ...request, resume: join(sft.directory, 'missing'),
        output: join(sft.directory, 'missing.safetensors') }, FAILURE);

    const valid = publishedCheckpoint(sft.checkpoint);
    const outside = join(dirname(sft.checkpoint), 'step-outside');
    cpSync(valid.path, outside, { recursive: true, force: false, errorOnExist: true });
    const outsideState = digest(readFileSync(join(outside, 'state.json')));
    const outsideTensors = digest(readFileSync(join(outside, 'tensors.safetensors')));
    // This relative path resolves to the actual valid copied checkpoint above.
    const escaping = '../step-outside\n';
    writeFileSync(valid.pointer, escaping);
    await refuse('escaping-resume', { ...request, resume: sft.checkpoint,
        output: join(sft.directory, 'escaping-resume.safetensors') }, FAILURE);
    await refuse('escaping-save', { ...request, checkpoint: sft.checkpoint, checkpointEvery: every,
        output: join(sft.directory, 'escaping-save.safetensors') }, FAILURE);
    assert.equal(digest(readFileSync(join(outside, 'state.json'))), outsideState,
        'refused checkpoint traversal changed state outside its checkpoint root');
    assert.equal(digest(readFileSync(join(outside, 'tensors.safetensors'))), outsideTensors,
        'refused checkpoint traversal changed tensors outside its checkpoint root');
    assert.equal(readFileSync(valid.pointer, 'utf8'), escaping, 'a refused save replaced the invalid latest pointer');
    run.report.states.push({ kind: 'protected-outside-checkpoint', path: outside,
        state: outsideState, tensors: outsideTensors, pointer: escaping });
    for (const [path, bytes] of supplied) assert.deepEqual(readFileSync(path), bytes, `qualification changed operator input ${path}`);
    assert.deepEqual(readFileSync(declaration), declarationBytes);
    for (const [file, sha256] of Object.entries(frozen)) {
        assert.equal(digest(readFileSync(join(model, file))), sha256, `a training lifecycle changed frozen base ${file}`);
    }
});
