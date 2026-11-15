#!/usr/bin/env node
// desktop-first-use: Ster Desktop's first use (2026-09-05.1,
// contrastive_pair_set_imported). With its onboarding cleared from the app's
// defaults (exported first, imported back at the end) and a Ster workspace
// and state of the journey's own (XDG_DATA_HOME, XDG_STATE_HOME), the
// walkthrough shows its three screens as the app ships them; a pair set
// that is not one Ster reads, chosen in the open panel, is refused and
// leaves first use open; a real pair set is imported and made active, which
// alone finishes it. Reopened, the walkthrough is gone.
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

const broken = [];
try {
  let app = await openApp({ executable, env });
  try {
    await app.shows(firstTitle, 60000);
    app.screen('walkthrough');
    for (const title of laterTitles) {
      await app.click('Continue', { exact: true });
      await app.shows(title, 15000);
    }
    await app.click('Import Pair Set…', { exact: true });
    await app.choosePath(malformed);
    await new Promise((resolve) => setTimeout(resolve, 5000));
    if (!app.snapshot().tree.includes(titles.at(-1))) broken.push('a malformed pair set finished first use');
    app.screen('refused');

    await app.click('Import Pair Set…', { exact: true });
    await app.choosePath(pairs);
    await app.gone(titles.at(-1), 60000);
    const imported = app.screen('imported');
    if (!/Imported: /.test(imported.tree)) broken.push('the app does not say the pair set was imported');
  } finally {
    app.close();
  }

  app = await openApp({ executable, env });
  try {
    await new Promise((resolve) => setTimeout(resolve, 5000));
    const reopened = app.snapshot();
    app.screen('reopened');
    if (reopened.tree.includes(firstTitle) && reopened.tree.includes('Continue')) broken.push('the walkthrough opens again after the import');
  } finally {
    app.close();
  }
} finally {
  restore();
}
holds(broken);
