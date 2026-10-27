#!/usr/bin/env node
// Lists the source lines of one committed revision that match a pattern, for
// a defect register's revision-bound completion probe of a construct that
// must not remain:
//
//   node tests/inventory.mjs <revision> <extended-regex> <path>...
//
// Standard output carries one `revision:path:line:text` per remaining
// instance and is empty when none remains. A revision git cannot read, or a
// pattern git refuses, fails the command with git's reason; git grep's own
// "nothing matched" is the empty answer, not a failure.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
// argv holds node and this script before the caller's words.
const [, , revision, pattern, ...paths] = process.argv;
assert.ok(
  revision && pattern && paths.length,
  'usage: node tests/inventory.mjs <revision> <extended-regex> <path>...',
);

// git's answer goes straight to this process's standard output, so an
// inventory of any size is passed on whole instead of filling a buffer that
// ends the child; only git's own complaint is kept to report a failure.
const found = spawnSync(
  'git',
  ['-C', root, 'grep', '-n', '-E', '-e', pattern, revision, '--', ...paths],
  {
    encoding: 'utf8',
    stdio: ['ignore', 'inherit', 'pipe'],
  },
);
// git grep exits nonzero both when nothing matched and when it failed
// (https://git-scm.com/docs/git-grep#_exit_status); only a failure says why
// on standard error, so that complaint, a launch error or a signal is the
// failure and a quiet nonzero exit is the empty inventory.
assert.ok(
  !found.error && found.signal === null && !found.stderr,
  `git grep ${pattern} at ${revision} failed (status ${found.status}, signal ${found.signal}${found.error ? `, ${found.error.message}` : ''}): ${found.stderr}`,
);
