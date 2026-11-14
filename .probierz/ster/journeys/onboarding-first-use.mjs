#!/usr/bin/env node
// onboarding-first-use: Ster's first use (2026-09-05.1,
// contrastive_pair_set_imported). With state and a workspace of the
// journey's own (XDG_STATE_HOME, XDG_DATA_HOME), the walkthrough shows its
// three screens and names `ster onboarding --import-pairs`, saving progress
// without completing. A pair set that is not one Ster reads is rejected with
// nothing written; importing a real pair set copies it into the workspace,
// counts its pairs and makes it active, and only that completes first use;
// the same set again is unchanged, and a different set under the same name
// is a conflict that keeps the first.
//
// The run names TUI_CMD (the ster binary).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

function required(name) {
  const value = process.env[name]?.trim();
  if (!value) {
    process.stderr.write(`${name} is not set: run this journey through \`probierz run tui --app ster\`\n`);
    process.exit(2);
  }
  return value;
}

const cli = required('TUI_CMD');
const own = join(required('PROBIERZ_ARTIFACTS'), 'ster-first-use');
const env = { ...process.env, XDG_STATE_HOME: join(own, 'state'), XDG_DATA_HOME: join(own, 'data') };
mkdirSync(env.XDG_STATE_HOME, { recursive: true });
mkdirSync(env.XDG_DATA_HOME, { recursive: true });
const pairs = {
  trait_name: 'plain-spoken',
  pairs: [
    { positive: 'The release ships on Monday.', negative: 'We are thrilled to unveil an unprecedented release journey this Monday!' },
    { positive: 'It failed because the key expired.', negative: 'Unfortunately, various unforeseen circumstances may have contributed to the failure.' },
  ],
};
const set = join(own, 'pairs.json');
const changed = join(own, 'changed', 'pairs.json');
const malformed = join(own, 'malformed.json');
mkdirSync(join(own, 'changed'), { recursive: true });
writeFileSync(set, `${JSON.stringify(pairs)}\n`);
writeFileSync(changed, `${JSON.stringify({ ...pairs, pairs: pairs.pairs.slice(0, 1) })}\n`);
writeFileSync(malformed, `${JSON.stringify({ trait_name: 'x', pairs: [{ positive: 'only one side' }], extra: true })}\n`);

const transcript = [];
function ster(args) {
  const result = spawnSync(cli, args, { env, encoding: 'utf8', maxBuffer: Infinity });
  transcript.push({ args, status: result.status, stdout: result.stdout, stderr: result.stderr });
  return result;
}
// The answer is the JSON document printed first; a line for a person may follow it.
const answer = (result) => {
  try {
    return JSON.parse(result.stdout.slice(result.stdout.indexOf('{'), result.stdout.lastIndexOf('}') + 1));
  } catch {
    return {};
  }
};

const broken = [];
try {
  const walkthrough = ster(['onboarding']);
  for (const title of ['Bring your existing pair set into Ster', 'Validate first, preserve what is there', 'Import the data you want to use']) {
    if (!walkthrough.stdout.includes(title)) broken.push(`onboarding does not show "${title}"`);
  }
  if (!walkthrough.stdout.includes('ster onboarding --import-pairs')) broken.push('onboarding does not name the import command');
  if (ster(['onboarding']).stdout.includes('already complete')) broken.push('showing the walkthrough completed first use');

  const rejected = ster(['onboarding', '--import-pairs', malformed]);
  if (rejected.status === 0 || answer(rejected).status !== 'rejected') broken.push(`a malformed pair set answered ${answer(rejected).status} (exit ${rejected.status})`);
  if (ster(['onboarding']).stdout.includes('already complete')) broken.push('a rejected import completed first use');

  const imported = ster(['onboarding', '--import-pairs', set, '--name', 'probierz']);
  const report = answer(imported);
  if (imported.status !== 0 || report.status !== 'imported') broken.push(`importing the pair set answered ${report.status} (exit ${imported.status}): ${imported.stderr.trim()}`);
  if (report.pair_count !== 2) broken.push(`the import counted ${report.pair_count} pairs, not 2`);
  if (!report.path || !existsSync(report.path) || JSON.parse(readFileSync(report.path, 'utf8')).pairs?.length !== 2) broken.push('the workspace holds no copy of the imported set');
  if (!imported.stdout.includes('first-use journey complete')) broken.push('the import does not say first use is complete');

  const again = answer(ster(['onboarding', '--import-pairs', set, '--name', 'probierz']));
  if (again.status !== 'unchanged') broken.push(`the same set again answered ${again.status}`);
  const clash = ster(['onboarding', '--import-pairs', changed, '--name', 'probierz']);
  if (clash.status === 0 || answer(clash).status !== 'conflicting') broken.push(`a different set under the same name answered ${answer(clash).status}`);
  if (report.path && JSON.parse(readFileSync(report.path, 'utf8')).pairs?.length !== 2) broken.push('a conflicting import replaced the set already there');

  if (!ster(['onboarding']).stdout.includes('already complete')) broken.push('onboarding does not say first use is complete after the import');
} finally {
  writeFileSync(join(own, 'transcript.json'), `${JSON.stringify(transcript, null, 2)}\n`);
}

if (broken.length > 0) {
  process.stderr.write(`${broken.join('\n')}\n`);
  process.exit(1);
}
process.stdout.write('Ster first use completed only when a real pair set was imported and made active\n');
