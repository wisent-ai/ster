# Pair sets in the workspace

`ster workspace import <file> [--name <id>]` (and
`ster onboarding --import-pairs <file>`) takes a canonical pair-set document
into the workspace under `$XDG_DATA_HOME/ster` (or `~/.local/share/ster`) and
makes it the active set. The whole document is validated before anything is
written, and a rejected import leaves no partial state.

## The set's id

- With `--name`, the id is that name. It must start with an ASCII letter or
  digit and hold only letters, digits, dots, underscores and hyphens.
- Without `--name`, the id is the file name's letters and digits, lowercased,
  with each run of other characters becoming one hyphen: `My Pairs.json`
  becomes `my-pairs`.
- A file name that holds no letter or digit, or is not UTF-8, gives no id. The
  import is `rejected`, naming the file and asking for `--name`; no id is made
  up for it.

## Recognising a set again

A set is identified by its content: the BLAKE2b-512 digest of its canonical
document, of which the first 16 bytes are kept, written as 32 hex characters.
This is the `digest` each entry of the workspace state records.

| Report status | When |
|---|---|
| `imported` | New content under a new id |
| `unchanged` | The same content was imported before, under any id; that set becomes active |
| `conflicting` | The id is taken by different content, or the destination file exists outside the workspace index; nothing is overwritten |
| `rejected` | The document is invalid, the `--name` is not a valid id, or the file name gives no id |

## Lifecycle qualification

`tests/workspace/lifecycle.mjs` runs the real CLI and native request interface.
It imports caller-supplied canonical pairs, checks the durable copy and active
manifest, repeats the import, refuses conflicting and malformed imports without
changing that state, selects imported sets, and removes them while preserving
the original source. Rejected request imports retain their `ImportReport` and
fail instead of returning a successful process status.

Provide the candidate inputs in an environment file under the ignored build
directory:

```dotenv
STER=<absolute path to the candidate ster executable>
STER_TEST_SOURCE_REVISION=<full source revision of that artifact>
STER_TEST_BINARY_SHA256=<artifact SHA-256 from the qualification build>
STER_TEST_PAIRS=<absolute path to canonical pairs with an explicit trait_name>
```

With Node's environment-file support, run:

```sh
node --env-file=.build/qualification.env tests/workspace/lifecycle.mjs
```

The runner refuses a mismatched executable digest or a dirty or different
source checkout. Each run owns a unique `.build/workspace/` directory and
explicitly isolates `HOME`, `XDG_DATA_HOME` and `XDG_STATE_HOME`, including
first-use progress. It retains `report.json` with the actual commands, full
output, process statuses, executable identity and persisted-state snapshots,
then removes only its own fixture. This journey covers the CLI and native API;
graphical Desktop qualification remains a separate surface.
