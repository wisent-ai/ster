# Native tokenizer creation

`ster model tokenizer create` compiles an explicit WordLevel declaration into a
native tokenizer JSON. It writes a new file, persists and reads its bytes back,
then loads that saved document through the native tokenizer library and
encodes and decodes the declaration's verification text. Success therefore
includes an actual saved-document consumer, not just JSON generation.

## Offline reference

From this repository, with the candidate `ster` executable on your path:

```sh
ster model tokenizer create --from docs/examples/model/vocabulary.json --output .build/reference-tokenizer.json
```

The vocabulary is caller-selected input, not a built-in program classifier or
a default of the command. Native token strings, pipeline tags and output field
identities remain tokenizer protocol data. The tokenizer library owns native
serialization, including its format version and generated token IDs; Ster
does not hand-encode a tokenizer document.

## Declaration

| Field | Contract |
|---|---|
| `unknownToken` | Required content of an entry in `modelTokens`; it is the model's native unknown identity. |
| `modelTokens` | Required nonempty ordered records, each with nonempty `content`. Duplicate content is refused. Their positions determine native model IDs. |
| `addedTokens` | Explicit array of native AddedToken records. An empty array adds none. Records carry `content`, `single_word`, `lstrip`, `rstrip`, `normalized` and `special`. |
| `pipeline.normalizer` | Native normalizer object, or explicit `null`. |
| `pipeline.preTokenizer` | Native pre-tokenizer object, or explicit `null`. |
| `pipeline.postProcessor` | Native post-processor object, or explicit `null`. |
| `pipeline.decoder` | Native decoder object, or explicit `null`. |
| `verification.text` | Required nonblank text exercised through the saved tokenizer. |
| `verification.addSpecialTokens` | Required encoding choice; no value is inferred. |
| `verification.skipSpecialTokens` | Required decoding choice; no value is inferred. |

Pipeline objects use the native library's schema. For example, the reference
selects the native `Whitespace` pre-tokenizer. Omitting a pipeline stage is not
the same as explicitly choosing `null`: omission is refused by field name.
Unsupported native stage data retains the underlying deserialization cause.

Added-token metadata is distinct from model vocabulary. Declaring a special
token already present in the model does not create another identity. The
reported `vocabulary_size` is the native unique vocabulary, including additions,
not the sum of overlapping collections. Ster also checks every encoded preview
token against the saved vocabulary's actual identity mapping before decoding;
a post-processor that emits an unmapped identity cannot return success.

## Output and refusals

`--from` and `--output` are required. Missing output parents are created. The
output file is opened exclusively: existing files, directories and symlinks
are refused rather than replaced. These are emitted artifacts, not entries
in a separately managed model catalog.

The result names the output path, measured vocabulary size and actual preview:
input text, token IDs, token strings and decoded text. The decoder's omission
of a declared special token is controlled by `skipSpecialTokens`, independently
of the encoded IDs.

A missing or malformed declaration, unknown declaration field, empty or duplicate
model token, missing unknown identity, omitted pipeline stage, invalid native
pipeline, failed write/readback, identity mismatch or consumer failure is an
error. Added-token records require every native flag, unique nonempty content
and no extra fields; supplying an ID is refused rather than silently discarded.
The producer checks both the native admission count and stored identities, so
duplicate or conflicting metadata cannot disappear into native deduplication.
Diagnostics retain the failed operation, actual path or token and the
native cause. Failed creation removes only its owned output file; a cleanup
failure names that operation as well. Created parent directories are not
claimed to be rolled back.

## Compose it inside a checkpoint

A fixture declaration's `tokenizer` may name an existing native JSON file or
an object whose `from` field names this tokenizer declaration. The latter
compiles through the same reusable library producer without writing an
intermediate tokenizer file. Paths resolve relative to the fixture declaration.
The fixture result retains the actual tokenizer preview, then separately
proves that the emitted checkpoint loads and forwards its model prompt.

```sh
ster model fixture create --from docs/examples/model/fixture.json --output .build/reference-model
```

See [Native model fixtures](fixtures.md) for the complete checkpoint contract.

## Native request and Desktop

`ster request model/tokenizer/create` reads a JSON object with `from` and
`output`. It calls the same library operation as the CLI and returns its result
through the normal request event and exit-status protocol. Unknown or empty
request fields are refused before execution.

In Ster Desktop, select **Tokenizer** under **Model artifacts**, choose the
declaration and new JSON output, then run. The same form also exposes
**Model fixture** with its own declaration path. The forms supply no token,
architecture or initialization presets. Actual results and logs are displayed
separately, including failed request documents and native picker diagnostics.

## Real qualification

Provide candidate inputs in an ignored environment file:

```dotenv
STER=<absolute candidate executable path>
STER_TEST_SOURCE_REVISION=<full artifact source revision>
STER_TEST_BINARY_SHA256=<artifact SHA-256 from the qualification build>
STER_TEST_TOKENIZER_SPEC=<absolute tokenizer declaration path>
```

Then run with Node's environment-file support:

```sh
node --env-file=.build/qualification.env tests/tokenizer/creation.mjs
```

This journey executes the actual CLI and native API, reads the persisted native
vocabulary, verifies stable model identities and repeated output, exercises
special-token decoding with both explicit choices, and refuses protected-output,
malformed-input, missing-stage and blocked-parent cases. Its declaration must
include a special token to cover that decoding transition. It protects the
original declaration and all previously accepted output.

The shared runner requires the exact clean source and executable digest,
isolates home and XDG state under its own ignored build fixture, and retains
actual command/stream/status receipts and artifact snapshots under
`.build/tokenizer/`. This is CLI/API qualification, not a graphical Desktop
journey; source declarations alone do not qualify any surface.
