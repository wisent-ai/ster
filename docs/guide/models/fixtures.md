# Native model fixtures

`ster model fixture create` creates an untrained local Llama checkpoint from
an explicit declaration. It writes native config, tokenizer and safetensors
files, loads the emitted checkpoint on the CPU in full precision, and forwards
the declared prompt at the requested layers before returning success. This
exercises model mechanics; random weights do not establish language quality.

## Offline reference

From this repository, with the candidate `ster` executable on your path:

```sh
ster model fixture create --from docs/examples/model/fixture.json --output .build/reference-model
```

The reference declaration selects its adjacent config and vocabulary. Ster
compiles that vocabulary through the same native producer exposed by
`ster model tokenizer create`; no intermediate tokenizer, account, download
or GPU is required. The example's architecture and initialization are explicit
input data, not defaults of the command. Select another declaration with
`--from` to change them. Source annotations in the reference explain its values.

The output directory must be new. Missing parents are created. An existing
file, directory or symlink is refused without replacing its contents. If
writing or decoder verification fails, Ster removes this invocation's output
files and directory. A cleanup failure retains the original operation's cause
and names the cleanup operation that also failed; newly created parent
directories are not claimed to have been removed.

## Declaration

All paths resolve relative to the declaration file, not the caller's current
working directory.

| Field | Contract |
|---|---|
| `config` | Path to the native Llama config. Its bytes, including extra metadata, are preserved. |
| `tokenizer` | Path to a native tokenizer JSON, or an object whose `from` field names a tokenizer declaration to compile. |
| `initialization.seed` | Required unsigned seed for the actual native random generator. |
| `initialization.embedding` | Required finite `mean` and `scale` for embedding and untied language-head weights. |
| `initialization.projection` | Required finite `mean` and `scale` for attention and feed-forward weights. |
| `initialization.normalization` | Required finite scalar for normalization weights. |
| `initialization.bias` | Finite `mean` and `scale`, required when the config enables attention biases. |
| `verification.prompt` | Required nonblank prompt forwarded through the saved checkpoint. |
| `verification.layers` | Required nonempty list of native layer indices within the declared model. |

`seedWhy`, `normalizationWhy`, `meanWhy`, `scaleWhy` and `layersWhy` are optional
source annotations beside the corresponding declared values. They do not
supply missing values or change initialization. The accepted declaration,
including its annotations, is returned in the result.

The config must state `model_type=llama`, `hidden_size`, `intermediate_size`,
`vocab_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
`head_dim`, `max_position_embeddings`, `rms_norm_eps`, `rope_theta`,
`tie_word_embeddings`, `attention_bias` and `mlp_bias`. Dimensions must be
nonzero, query heads must divide into whole groups of key/value heads, and
projection dimensions must fit the host's native representation. Token IDs
must fit the declared embedding vocabulary. Ster refuses `mlp_bias=true`
because its Llama consumer does not apply feed-forward biases; emitting
unused bias tensors would not prove that setting worked.

Unknown declaration fields, a missing seed, an absent input file, malformed
native data, an out-of-range verification layer, unsupported layout and
non-finite stored weights or observed activations are failures, not guessed
presets. Diagnostics name the failed operation, its paths or observed values,
and the underlying cause. A tokenizer declared with `from` also returns its
actual encoding/decoding preview under `tokenizer_verification`.

Repeatability is for the same declaration and software build. The producer
uses native random sampling and safetensors serialization; historical output
captures from a different generator are not expected checkpoint bytes.

## Native request and Desktop

`ster request model/fixture/create` reads a JSON object with `from` and `output`
and uses this same library operation. It returns the structured fixture report
through the normal request event protocol. Unknown or empty request fields are
refused before creation. Operational failures preserve their full cause and
failed process status.

In Ster Desktop, select **Model fixture**, choose its declaration and new
output directory, and run it. The declaration supplies the architecture,
initialization and tokenizer source; the form invents none. Results and logs
remain separately visible, including a failed operation's result document.
Native picker cancellation preserves previous input, while acceptance without
a file URL is reported as a selection failure.

## Real qualification

`tests/fixture/creation.mjs` runs the real CLI and native request API. It reads
actual checkpoint headers and float payloads, checks repeated native outputs,
protects existing output and original inputs, exercises declaration and
request refusals, and verifies failed output cleanup at a singular decoder
normalization boundary. This is not a graphical Desktop journey.

Provide the candidate inputs in an ignored environment file:

```dotenv
STER=<absolute candidate executable path>
STER_TEST_SOURCE_REVISION=<full artifact source revision>
STER_TEST_BINARY_SHA256=<artifact SHA-256 from the qualification build>
STER_TEST_MODEL_SPEC=<absolute fixture declaration path>
```

Run with Node's environment-file support:

```sh
node --env-file=.build/qualification.env tests/fixture/creation.mjs
```

The runner requires the exact clean source candidate and matching artifact
digest, isolates home and XDG data/state inside its own ignored build fixture,
and retains `report.json` under `.build/fixture/`. Receipts contain actual
commands, input and output, native statuses and signals, source and executable
identity, observed artifact state and cleanup outcome. No source inspection,
healthy process or declaration alone qualifies the feature.
