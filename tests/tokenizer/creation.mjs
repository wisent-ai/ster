import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { existsSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAILURE, SUCCESS, USAGE, digest, qualify } from '../qualification/runtime.mjs';

await qualify('tokenizer', async run => {
    const declaration = realpathSync(run.input('STER_TEST_TOKENIZER_SPEC'));
    const originalBytes = readFileSync(declaration);
    const input = JSON.parse(originalBytes);
    run.report.inputs = [{ path: declaration, sha256: digest(originalBytes) }];
    function state(path, result) {
        const bytes = readFileSync(path);
        const native = JSON.parse(bytes);
        assert.equal(result.path, path);
        // This is the actual supported native model identity, not a classifier:
        // https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.WordLevel
        assert.equal(native.model.type, 'WordLevel');
        const declaredVocabulary = Object.fromEntries(input.modelTokens.map((token, index) => [token.content, index]));
        assert.deepEqual(native.model.vocab, declaredVocabulary, 'persisted native IDs must preserve declared token order');
        const actualMetadata = Object.fromEntries(native.added_tokens.map(({ id, ...metadata }) => [metadata.content, metadata]));
        assert.deepEqual(actualMetadata, Object.fromEntries(input.addedTokens.map(token => [token.content, token])),
            'every declared native added-token flag must survive persistence without extra or missing records');
        const vocabulary = new Map(Object.entries(native.model.vocab));
        for (const token of native.added_tokens) vocabulary.set(token.content, token.id);
        assert.equal(result.vocabulary_size, vocabulary.size, 'measured vocabulary size must count overlapping native identities once');
        assert.equal(result.verification.text, input.verification.text);
        assert.deepEqual(result.verification.token_ids, result.verification.tokens.map(token => {
            assert.ok(vocabulary.has(token), `encoded token ${token} is absent from the saved vocabulary`);
            return vocabulary.get(token);
        }));
        const snapshot = { path, sha256: digest(bytes), native, verification: result.verification };
        run.report.states.push(snapshot);
        return snapshot;
    }
    const first = join(run.fixture, 'nested', 'first.json');
    const created = JSON.parse((await run.cli(['model', 'tokenizer', 'create', '--from', declaration, '--output', first], SUCCESS)).stdout);
    const firstState = state(first, created);
    const second = join(run.fixture, 'second.json');
    const repeated = await run.api('model/tokenizer/create', { from: declaration, output: second }, SUCCESS);
    const secondState = state(second, repeated.document);
    assert.equal(secondState.sha256, firstState.sha256, 'the same declaration must produce the same native artifact across real processes');

    await run.cli(['model', 'tokenizer', 'create', '--from', declaration, '--output', first], FAILURE);
    assert.equal(digest(readFileSync(first)), firstState.sha256, 'an existing tokenizer was overwritten');
    await run.api('model/tokenizer/create', { from: declaration, output: first }, FAILURE);
    assert.equal(digest(readFileSync(first)), firstState.sha256);
    const absent = join(run.fixture, 'absent.json');
    await run.cli(['model', 'tokenizer', 'create', '--output', absent], USAGE);
    await run.cli(['model', 'tokenizer', 'create', '--from', declaration], USAGE);
    await run.api('model/tokenizer/create', { from: '', output: absent }, USAGE);
    await run.api('model/tokenizer/create', { from: declaration, output: absent, unexpected: true }, USAGE);
    assert.ok(!existsSync(absent), 'refused tokenizer syntax wrote output');

    async function refuse(name, value) {
        const from = join(run.fixture, `${name}.spec.json`);
        writeFileSync(from, JSON.stringify(value));
        const output = join(run.fixture, `${name}.json`);
        const result = await run.api('model/tokenizer/create', { from, output }, FAILURE);
        assert.equal(typeof result.document.error, 'string');
        assert.ok(!existsSync(output), `failed ${name} left a tokenizer artifact`);
        run.report.states.push({ path: output, exists: existsSync(output), failure: result.document });
    }
    const [firstToken] = input.modelTokens;
    assert.ok(firstToken, 'qualification requires the actual nonempty WordLevel declaration');
    await refuse('duplicate-token', { ...input, modelTokens: [...input.modelTokens, firstToken] });
    await refuse('unknown-token', { ...input, unknownToken: randomUUID() });
    await refuse('empty-token', { ...input, modelTokens: [...input.modelTokens, { content: '' }] });
    await refuse('unknown-field', { ...input, unexpected: true });
    const [firstAdded] = input.addedTokens;
    assert.ok(firstAdded, 'qualification needs declared added-token metadata');
    await refuse('empty-added-token', { ...input, addedTokens: [...input.addedTokens, { ...firstAdded, content: '' }] });
    await refuse('duplicate-added-token', { ...input, addedTokens: [...input.addedTokens, firstAdded] });
    await refuse('conflicting-added-token', { ...input, addedTokens: [...input.addedTokens, { ...firstAdded, special: !firstAdded.special }] });
    await refuse('unknown-added-field', { ...input, addedTokens: [{ ...firstAdded, unexpected: true }] });
    const assigned = firstState.native.added_tokens.find(token => token.content === firstAdded.content);
    await refuse('caller-assigned-id', { ...input, addedTokens: [{ ...firstAdded, id: assigned.id }] });
    const { normalized, ...missingNormalization } = firstAdded;
    await refuse('missing-added-normalization', { ...input, addedTokens: [missingNormalization] });
    const missingStage = structuredClone(input);
    delete missingStage.pipeline.preTokenizer;
    await refuse('missing-pipeline-stage', missingStage);
    const missingVerification = structuredClone(input);
    delete missingVerification.verification;
    await refuse('missing-verification', missingVerification);

    const malformed = join(run.fixture, 'malformed.spec.json');
    writeFileSync(malformed, '{');
    await run.api('model/tokenizer/create', { from: malformed, output: absent }, FAILURE);
    assert.ok(!existsSync(absent));
    const blockedParent = join(run.fixture, 'blocked-parent');
    const marker = randomUUID();
    writeFileSync(blockedParent, marker);
    await run.api('model/tokenizer/create', { from: declaration, output: join(blockedParent, 'tokenizer.json') }, FAILURE);
    assert.equal(readFileSync(blockedParent, 'utf8'), marker, 'a failed parent operation changed the blocking file');

    const special = input.addedTokens.find(token => token.special);
    assert.ok(special, 'qualification requires a declared special token to exercise native decoding semantics');
    const text = `${special.content} ${input.verification.text}`;
    async function decode(name, skipSpecialTokens) {
        const from = join(run.fixture, `${name}.spec.json`);
        writeFileSync(from, JSON.stringify({ ...input, verification: { ...input.verification, text, skipSpecialTokens } }));
        const output = join(run.fixture, `${name}.json`);
        const result = await run.api('model/tokenizer/create', { from, output }, SUCCESS);
        run.report.states.push({ path: output, sha256: digest(readFileSync(output)), verification: result.document.verification });
        return result.document.verification;
    }
    const kept = await decode('keep-special', false);
    const skipped = await decode('skip-special', true);
    assert.deepEqual(kept.token_ids, skipped.token_ids, 'decoding policy must not change the encoded identities');
    assert.ok(kept.decoded.includes(special.content), 'native decoding did not retain the requested special token');
    assert.ok(!skipped.decoded.includes(special.content), 'native decoding did not apply the requested special-token omission');
    assert.equal(digest(readFileSync(first)), firstState.sha256);
    assert.equal(digest(readFileSync(second)), secondState.sha256);
    assert.deepEqual(readFileSync(declaration), originalBytes, 'qualification changed the operator tokenizer declaration');
});
