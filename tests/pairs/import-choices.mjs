// Real test of `ster pairs import --benchmark choices` through the built
// binary, on multiple-choice rows it writes under this run's directory in the
// shapes ARC (labels) and HellaSwag (an index held as text) publish.
//
// It imports both, reads each written pair set back, and checks that every
// row whose answer resolves became a pair with the correct choice as its
// positive and another choice of the same row as its negative, and that every
// other row is skipped with its row and reason. Then it checks the refusals:
// row flags on another benchmark, choices without them, a missing
// --answer-form, label form without --labels, and a pointer without its
// leading slash, none of which may write a pair set. Every command, its exit
// status and output go to the run's report.json. The seed is this run's
// process id, so any incorrect choice of the row is accepted as a negative.
//
// Usage: STER=target/debug/ster STER_TEST_SUCCESS_EXIT=0 node tests/pairs/import-choices.mjs
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

function required(name, why) {
  const value = process.env[name];
  assert.ok(value, `${name} is required: ${why}`);
  return value;
}
const binary = required('STER', 'the built ster binary this test runs, e.g. target/debug/ster');
const successExit = Number(required('STER_TEST_SUCCESS_EXIT', 'the exit status ster answers an accepted command with'));
assert.ok(Number.isInteger(successExit));
const repository = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const seed = String(process.pid);
const root = join(repository, 'target', 'real-tests', 'pairs-import-choices', `${new Date().toISOString().replace(/[:.]/g, '-')}-${seed}`);
mkdirSync(root, { recursive: true });
const git = args => spawnSync('git', args, { cwd: repository, encoding: 'utf8' }).stdout.trim();
const report = { revision: git(['rev-parse', 'HEAD']), dirty: git(['status', '--porcelain']) !== '', binary, seed, commands: [], checks: [], verdict: 'failed' };

// The rows as the datasets publish them, then what each row must become: a
// pair whose positive is `answer`, or a skip at `row` for `reason`.
const arcRows = `{"question": "Which gas do plants take in?", "choices": {"text": ["They take in oxygen.", "They take in carbon dioxide.", "They take in helium."], "label": ["A", "B", "C"]}, "answerKey": "B"}
{"question": "What melts ice?", "choices": {"text": ["Heat melts it.", "Cold melts it."], "label": ["A", "B"]}, "answerKey": "D"}

{"question": "Which is a mammal?", "choices": {"text": ["The whale is.", "The shark is."], "label": ["A", "B"]}}
{"question": "Which planet is red?", "choices": {"text": ["Mars is red.", "Venus is red."], "label": ["A", "B"]}, "answerKey": "A"}
`;
const arcExpected = [
  { answer: 'They take in carbon dioxide.' },
  { row: '2', reason: 'answer is not one of the labels' },
  { row: '4', reason: 'no answer' },
  { answer: 'Mars is red.' },
];
const hellaswagRows = `[
  {"ctx": "She picks up the violin and", "endings": ["eats it.", "plays a tune.", "throws it away."], "label": "1"},
  {"ctx": "He opens the umbrella because", "endings": ["it is raining.", "it is sunny."], "label": "7"},
  {"ctx": "The dog fetches", "endings": ["the ball.", ""], "label": "0"}
]`;
const hellaswagExpected = [
  { answer: 'plays a tune.' },
  { row: '2', reason: 'answer index is outside the choices' },
  { row: '3', reason: 'choices are not a list of non-empty text' },
];

function importing(args) {
  const result = spawnSync(binary, ['pairs', 'import', ...args], { cwd: repository, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  report.commands.push({ args, status: result.status, signal: result.signal, error: result.error?.message, stdout: result.stdout, stderr: result.stderr });
  return result;
}
function check(name, actual, expected) {
  assert.deepEqual(actual, expected, name);
  report.checks.push({ name, actual });
}
const side = (question, answer) => `Question: ${question}\nAnswer: ${answer}`;

// Import `text` as `name`, then hold the answer and the written set to
// `expected`. `parts` reads a published row's question and choices.
function accepted(name, file, text, expected, parts, args) {
  const source = join(root, file);
  writeFileSync(source, text);
  const output = join(root, `${name}.pairs.json`);
  const result = importing(['--benchmark', 'choices', '--source', source, ...args, '--seed', seed, '--output', output]);
  assert.equal(result.status, successExit, `${name} import was refused: ${result.stderr}`);
  const answer = JSON.parse(result.stdout);
  const set = JSON.parse(readFileSync(output, 'utf8'));
  const published = file.endsWith('.jsonl') ? text.split('\n').filter(line => line.trim()).map(line => JSON.parse(line)) : JSON.parse(text);
  check(`${name}: rows read`, answer.report.rows, published.length);
  const paired = expected.map((row, index) => ({ ...row, ...parts(published[index]) })).filter(row => row.answer);
  check(`${name}: pairs written`, set.pairs.length, paired.length);
  paired.forEach((row, index) => {
    check(`${name}: pair ${index} positive is the stated answer`, set.pairs[index].positive, side(row.question, row.answer));
    const negatives = row.choices.filter(choice => choice !== row.answer).map(choice => side(row.question, choice));
    assert.ok(negatives.includes(set.pairs[index].negative), `${name}: pair ${index} negative ${set.pairs[index].negative} is not another choice`);
    report.checks.push({ name: `${name}: pair ${index} negative is another choice`, actual: set.pairs[index].negative });
  });
  check(`${name}: skipped rows`, answer.report.skipped, expected.filter(row => row.reason));
  return { set, source };
}
function refusedWith(name, args, sentence) {
  const output = join(root, 'refused.pairs.json');
  const result = importing([...args, '--seed', seed, '--output', output]);
  assert.notEqual(result.status, successExit, `${name} was accepted`);
  assert.ok(result.stderr.includes(sentence), `${name}: expected a refusal containing ${sentence}, got ${result.stderr}`);
  assert.ok(!existsSync(output), `${name} wrote ${output}`);
  report.checks.push({ name, actual: result.stderr.trim() });
}

try {
  const labelled = ['--question', '/question', '--choices', '/choices/text', '--answer', '/answerKey'];
  const arc = accepted('arc', 'arc.jsonl', arcRows, arcExpected, row => ({ question: row.question, choices: row.choices.text }),
    [...labelled, '--answer-form', 'label', '--labels', '/choices/label']);
  check('arc: the trait is the benchmark name', arc.set.trait_name, 'choices');
  const hellaswag = accepted('hellaswag', 'hellaswag.json', hellaswagRows, hellaswagExpected, row => ({ question: row.ctx, choices: row.endings }),
    ['--trait', 'plausible-endings', '--question', '/ctx', '--choices', '/endings', '--answer', '/label', '--answer-form', 'index']);
  check('hellaswag: the trait is the one given', hellaswag.set.trait_name, 'plausible-endings');
  refusedWith('row flags on another benchmark', ['--benchmark', 'truthfulqa', '--source', arc.source, '--question', '/question'],
    '--question, --choices, --answer, --answer-form and --labels apply only to --benchmark choices');
  refusedWith('choices without row flags', ['--benchmark', 'choices', '--source', arc.source],
    '--benchmark choices needs --question, --choices, --answer and --answer-form');
  refusedWith('a missing --answer-form', ['--benchmark', 'choices', '--source', arc.source, ...labelled],
    'multiple-choice rows need --answer-form');
  refusedWith('label form without --labels', ['--benchmark', 'choices', '--source', arc.source, ...labelled, '--answer-form', 'label'],
    '--answer-form label needs --labels');
  refusedWith('a pointer without its slash', ['--benchmark', 'choices', '--source', arc.source, '--question', 'question',
    '--choices', '/choices/text', '--answer', '/answerKey', '--answer-form', 'text'], '--question "question" is not a JSON pointer');
  report.verdict = 'passed';
} catch (error) {
  report.error = String(error.stack ?? error);
} finally {
  writeFileSync(join(root, 'report.json'), JSON.stringify(report, null, '\t'));
  console.log(`${report.verdict}: ${join(root, 'report.json')}`);
}
if (report.verdict !== 'passed') throw new Error(report.error);
