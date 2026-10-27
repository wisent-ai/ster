#!/usr/bin/env node
// Says when a committed revision lacks a construct a repair must have added,
// for a defect register's revision-bound completion probe of a capability
// (a refusal, a verb, a catalog entry) rather than of a construct that must
// go:
//
//   node tests/missing.mjs <revision> <extended-regex> <path>...
//
// Standard output carries one `missing at <revision>: <pattern> in <path>...`
// line when nothing at that revision matches, and is empty when the construct
// is present. A revision git cannot read, or a pattern git refuses, fails the
// command with git's reason.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
// argv holds node and this script before the caller's words.
const [, , revision, pattern, ...paths] = process.argv;
assert.ok(
  revision && pattern && paths.length,
  'usage: node tests/missing.mjs <revision> <extended-regex> <path>...',
);

const found = spawnSync(
  'git',
  ['-C', root, 'grep', '-q', '-E', '-e', pattern, revision, '--', ...paths],
  {
    encoding: 'utf8',
  },
);
// git grep exits zero when it matched and nonzero both when nothing matched
// and when it failed (https://git-scm.com/docs/git-grep#_exit_status); only a
// failure says why on standard error, so that complaint, a launch error or a
// signal is the failure and a quiet nonzero exit is the missing construct.
assert.ok(
  !found.error && found.signal === null && !found.stderr,
  `git grep ${pattern} at ${revision} failed (status ${found.status}, signal ${found.signal}${found.error ? `, ${found.error.message}` : ''}): ${found.stderr}`,
);
if (found.status)
  process.stdout.write(
    `missing at ${revision}: ${pattern} in ${paths.join(' ')}\n`,
  );
