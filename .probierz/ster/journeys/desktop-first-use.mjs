#!/usr/bin/env node
// desktop-first-use: Ster Desktop's first use (2026-09-05.1,
// contrastive_pair_set_imported). With its onboarding cleared from the app's
// defaults (exported first, imported back at the end) and a Ster workspace
// and state of the journey's own (XDG_DATA_HOME, XDG_STATE_HOME), the
// walkthrough shows its three screens as the app ships them; a pair set
// that is not one Ster reads, chosen in the open panel, is refused and
// leaves first use open; a real pair set is imported and made active, which
// alone finishes it.
//
// The run names PROBIERZ_TOOLKIT_ROOT, CUA_APP_EXECUTABLE (Ster.app's
// executable) and WISENT_WORKSPACE (holding ster-desktop).
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

function required(name) {
  const value = process.env[name]?.trim();
  if (!value) {
    process.stderr.write(`${name} is not set: run this journey through \`probierz run desktop:cua --app ster\`\n`);
    // https://www.gnu.org/software/bash/manual/html_node/Exit-Status.html
    process.exit(2);
  }
  return value;
}

const { openApp, holds, freshOnboarding } = await import(pathToFileURL(join(required('PROBIERZ_TOOLKIT_ROOT'), 'packages/desktop-cua/journey.mjs')));
const executable = required('CUA_APP_EXECUTABLE');
const journey = JSON.parse(readFileSync(join(required('WISENT_WORKSPACE'), 'ster-desktop/Sources/SterDesktop/Resources/ster-desktop-first-use.json'), 'utf8'));
const titles = journey.screens.map((screen) => screen.presentation.title);
const [firstTitle, ...laterTitles] = titles;
const own = join(required('PROBIERZ_ARTIFACTS'), 'ster-desktop');
const env = { XDG_DATA_HOME: join(own, 'data'), XDG_STATE_HOME: join(own, 'state') };
for (const directory of Object.values(env)) mkdirSync(directory, { recursive: true });
const pairs = join(own, 'pairs.json');
const malformed = join(own, 'malformed.json');
writeFileSync(pairs, `${JSON.stringify({ trait_name: 'plain-spoken', pairs: [
  { positive: 'The release ships on Monday.', negative: 'We are thrilled to unveil an unprecedented release journey this Monday!' },
  { positive: 'It failed because the key expired.', negative: 'Various unforeseen circumstances may have contributed to the failure.' },
] })}\n`);
writeFileSync(malformed, `${JSON.stringify({ trait_name: 'x', pairs: [{ positive: 'only one side' }], extra: true })}\n`);
const restore = freshOnboarding({ bundle: 'com.wisent.ster.desktop', prefix: 'com.wisent.ster.onboarding' });

// What the window says once the app has answered an import the
// walkthrough's last screen started. A refusal names that step:
// ster-desktop's Onboarding/Walkthrough.swift (importFailed) words it
// `.during("importing the first pair set")`, which Failure/SterFailure.swift
// (headline) shows as `while importing the first pair set: `. An import the
// workspace accepted says `Imported: <path>` (App/Import/ImportResults.swift).
const REFUSED = 'while importing the first pair set: ';
const IMPORTED = 'Imported: ';

const broken = [];
try {
  const app = await openApp({ executable, env });
  try {
    await app.shows(firstTitle);
    app.screen('walkthrough');
    for (const title of laterTitles) {
      await app.click('Continue', { exact: true });
      await app.shows(title);
    }
    await app.click('Import Pair Set…', { exact: true });
    await app.choosePath(malformed);
    // The app answers the import with its refusal or, if it took the
    // malformed set, with `Imported: `; the journey waits for whichever it
    // says, so an import fails the journey instead of leaving it waiting on
    // a refusal that never comes. `shows` takes one string, so each answer
    // is its own wait; the toolkit's own refusal ends both when neither
    // comes.
    const answered = await Promise.any([app.shows(REFUSED), app.shows(IMPORTED)]);
    app.screen('refused');
    if (answered.tree.includes(IMPORTED)) broken.push('a malformed pair set was imported');
    else if (!answered.tree.includes(titles.at(-1))) broken.push('a malformed pair set finished first use');

    await app.click('Import Pair Set…', { exact: true });
    await app.choosePath(pairs);
    await app.gone(titles.at(-1));
    const imported = app.screen('imported');
    if (!imported.tree.includes(IMPORTED)) broken.push('the app does not say the pair set was imported');
  } finally {
    app.close();
  }
} finally {
  restore();
}
holds(broken);
