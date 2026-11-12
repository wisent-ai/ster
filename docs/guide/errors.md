# How Ster fails

Every Ster functionality declares each way it can fail, in `src/failures`,
with the [wisent-errors](https://github.com/wisent-ai/wisent-errors)
catalogue. A declared cause states what failed, why, the state the refusal
shows, and what fixes it, so a refusal reads:

```text
Error: <what failed>: <cause>; observed: <state>; fix: <remedy>
```

with the actual paths, ids and values filled in. A step that adds context to
a declared refusal (an import that was undone, say) prints its sentence first
and the declared refusal under `Caused by:`. A functionality lists the
functionalities it uses; their causes are met through it and declared once,
where they belong.

## Exit status

| Status | When |
|---|---|
| `0` | The command did what it promised. |
| `1` | A declared cause refused it and retrying unchanged will not help, or a failure no cause declares yet. |
| `2` | A usage refusal: an argument or flag is wrong, including clap's own refusals. |
| `69` | A declared cause retrying unchanged can help (an unreachable or throttling dependency, a timeout). |

`ster request <operation>` keeps its own statuses (`0`, `1`, `2`, see
[desktop requests](desktop-requests.md)); when a declared cause refused the
operation, its result's `json.cause` carries the `functionality` that declares
the cause (the id `ster errors show` takes), the cause's `id`, `docs` (its line
on the area page), `code`, `exit` (the status the same CLI command exits
with), `usage`, `retryable` and `remedy`. A rejected or conflicting workspace
import also carries `cause` in
its report on both surfaces.

## The catalogue, by area

| Page | Functionalities |
|---|---|
| [errors/catalogue.md](errors/catalogue.md) | `ster.errors` |
| [errors/chat.md](errors/chat.md) | `ster.chat.template` |
| [errors/decide.md](errors/decide.md) | `ster.decide.request`, `ster.decide` |
| [errors/decide-calibration.md](errors/decide-calibration.md) | `ster.decide.calibration`, `.examples`, `.calibrate` |
| [errors/decide-pipeline.md](errors/decide-pipeline.md) | `ster.decisions.question`, `.fetch`, `.import`, `.import.rows`, `.synthesize`, `.split`, `.benchmark` |
| [errors/file.md](errors/file.md) | `ster.file`, the layer every document read or write uses |
| [errors/inference.md](errors/inference.md) | `ster.inference` |
| [errors/lora.md](errors/lora.md) | `ster.lora.spec`, `.artifact` |
| [errors/model.md](errors/model.md) | `ster.model.checkpoint`, `.decoder`, `.weights` |
| [errors/model-config.md](errors/model-config.md) | `ster.model.config` |
| [errors/model-fixture.md](errors/model-fixture.md) | `ster.model.fixture` |
| [errors/model-load.md](errors/model-load.md) | `ster.model.tokenizer`, `ster.model.load`, `ster.model.adapters` |
| [errors/onboarding.md](errors/onboarding.md) | `ster.onboarding` |
| [errors/pairs.md](errors/pairs.md) | `ster.pairs.file`, `.index`, `.side`, `.add`, `.remove`, `.edit`, `.save`, `.merge`, `.dedupe`, `.inspect` |
| [errors/pairs-import.md](errors/pairs-import.md) | `ster.pairs.import`, `.import.rows` |
| [errors/pairs-export.md](errors/pairs-export.md) | `ster.pairs.export` |
| [errors/pairs-synthesize.md](errors/pairs-synthesize.md) | `ster.pairs.synthesize` |
| [errors/request.md](errors/request.md) | `ster.request` |
| [errors/runtime.md](errors/runtime.md) | `ster.runtime.device`, `.encode`, `.layers`, `.steer`, `.generate`; `ster.generate` |
| [errors/steering.md](errors/steering.md) | `ster.steering.artifact` |
| [errors/support.md](errors/support.md) | `ster.support` |
| [errors/telemetry.md](errors/telemetry.md) | `ster.telemetry` |
| [errors/tensor.md](errors/tensor.md) | `ster.tensor`, the layer every forward pass, training step and tensor read or write uses |
| [errors/tokenizer.md](errors/tokenizer.md) | `ster.tokenizer.declaration`, `.create` |
| [errors/tune.md](errors/tune.md) | `ster.tune.settings`, `.objective`, `.examples`, `.run`, `.latest`, `.save`, `.resume`, `.reward`, `.rollout`, `.strength`, `.fit`, `.write`, `.merge`, `.export`, `.ablate`, `.sft`, `.preference`, `.grpo`, `.decide`, `.evaluate` |
| [errors/workflow.md](errors/workflow.md) | `ster.prompts`, `ster.vector.layers`, `.direction`, `.holdout`, `.train`, `.optimize`, `.curve`, `.evaluate`, `.compare`, `.project`, `.score`; `ster.extract`, `ster.parity` |
| [errors/workspace.md](errors/workspace.md) | `ster.workspace.store`, `.set`, `.import`, `.show`, `.select`, `.remove`, `.active` |

Each page below its marker is what
`ster errors list --markdown --page docs/guide/errors/<area>.md` prints, which
is `Operation::markdown` of every functionality the page lists:
`tests/errors/pages.rs` (`kompilator test --manifest-path Cargo.toml --test error-pages`) refuses a page that
differs from it, a page no functionality names, and a functionality naming a
page this checkout does not hold, and `tests/errors/catalogue.sh` does the same
through the built binary. Regenerate a page's section with that command; never
edit it by hand.

## Reading and checking it

```sh
ster errors list                          # every functionality and its causes, as JSON
ster errors list --markdown               # every area page's sections
ster errors show ster.workspace.import    # its causes and every cause it meets
ster errors check --repository . \
  --uncatalogued tests/errors/uncatalogued.json \
  --untested tests/errors/untested.json
```

The same answers are `ster request errors/list` (no fields) and
`ster request errors/show` (`{"id": "<functionality>"}`).

`ster errors check` refuses (exit 1) a malformed declaration, a declared test
the checkout does not hold, and a command no functionality names as an entry.
Two baselines let the catalogue grow area by area, and both may only shrink:
`tests/errors/uncatalogued.json` lists the commands whose failures are not
declared yet, and `tests/errors/untested.json` the declared tests not written,
because no local state provokes their cause:

- `ster.decisions.fetch`: everything the Hugging Face datasets server
  answers (no answer, a refusal, a limit, a failure, a page that breaks off,
  is not JSON or lacks rows or features) and every column a dataset holds
  (no such column, no class label, classes of no text, too few or too many,
  a text column of no text, no row with a known label): only the real
  server and real datasets produce them, and a listener written to answer
  them would be a canned answer standing in for the server.
- `ster.decisions.synthesize`: a writing model whose every answer is empty
  or repeats an earlier one, which only a real model run produces.
- `ster.request`: standard input breaking while the body is read, which
  needs a device or stream that fails a read after it opened; a pipe or a
  file the test redirects only ends, and a directory is its own cause.
- `ster.file`: a read or a write the system breaks off although the account
  may make it (a disk fault, a file replaced mid-read), a full disk and a file
  system mounted read-only, which need a failing disk, a filled volume or a
  mounted read-only volume no test may make on the machine it runs on.
  The device refusing a safetensors file's tensors, or failing to copy them
  back for writing, needs a device out of memory, which a test cannot bring
  about without exhausting the machine it runs on.
- `ster.workspace`: a file appearing between two steps of one import, or a
  filesystem refusing a hard link; and a source with no file name or one
  that is not UTF-8, which the pair-set read refuses first on every file
  system a test runs on (a root or `..` is a directory, and APFS stores only
  UTF-8 names).
- `ster.pairs.synthesize`: what the writing model answers (an empty opposite
  trait, a refusal judgement that is not the one JSON form, every attempt
  rejected), which only a real model run produces and no fixture fixes in
  advance.
- `ster.inference`: Stado breaking the pipe mid-call, refusing a well-formed
  request from a consumer it serves, printing something other than its
  `--json` form or a document without its answer, and an alternative
  answering a completion without an assistant message — each a defect of
  another product that no local state makes it commit.
- `ster.model.checkpoint`: the Hugging Face Hub being unreachable, limiting
  this account, failing on its side or refusing a request the client built,
  its client's TLS connector failing to start, a file that arrived and could
  not be written into the Hub cache, and a directory listing the system
  breaks off, which depend on the Hub, the system's TLS store or a disk
  fault.
- `ster.model.weights`: a GGUF tensor whose bytes cannot be read, weight
  files the system does not map and weights the device does not take, which
  need a quantization Candle does not dequantize or a machine out of memory.
- `ster.model.adapters`: the families whose placement refusals need a
  checkpoint of that family (LFM2's short convolutions, DeciLM's per-layer
  plans, LongCat-Flash's shortcut experts, looped and recurrent layers,
  hyper-connected streams, Inkling, Step3's query bottleneck, Gemma 4's and
  MiMo-V2's two head shapes, K2-Horizon's value experts): the Hugging Face
  Hub publishes no tiny random checkpoint of them, and `ster model fixture
  create` writes Llama checkpoints only. GPT-2, Mixtral, DeepSeek-V3, Mamba,
  Qwen3-Next and MiniMax are tested on their tiny random checkpoints.
- `ster.model.tokenizer`: a tokenizer file that this account may read and
  that still cannot be read (removed between the directory listing and the
  read, or a disk fault), and the tokenizers library refusing a model Ster
  built only from tokens the vocabulary holds, which needs another release of
  that library.
- `ster.model.fixture`: a verified layer of another width than the config
  Ster wrote (a defect no declaration provokes), a tensor past the host's
  memory (which on an overcommitting host is granted and then exhausts the
  machine), a disk refusing a write after the output directory was created,
  and a tokenizer.json the library reads that holds no token, which only a
  hand-written file in the library's internal form provides.
- `ster.runtime.device`, `ster.runtime.encode` and `ster.runtime.generate`: a
  device this build carries whose driver does not open, a tokenizer that
  raises on a text for another reason than a missing unknown token, and a
  tokenizer that cannot spell the ids it produced, which need a broken driver
  or tokenizers library.
- `ster.lora.spec` and `ster.tune.strength`: a target named twice and an
  empty strength list, which both entries rule out before the check (repeated
  targets collapse, no strength measures none), so only a library caller
  reaches them.
- `ster.tune.run`: a device that cannot allocate a tensor or compute a step or
  a score, which a test cannot bring about without exhausting the memory of
  the machine it runs on, and a data file removed between being read and
  being fingerprinted. A step planned without units needs the planner to cut
  an empty slice of the epoch's order, which no --batch-size or
  --accumulation makes it do.
- `ster.tune.save`: a clock before 1970, which a test cannot set without
  changing the machine's time for every process on it.
- `ster.tune.resume`: a checkpoint tensor stored in another dtype than the
  run trains, which a resume at another `--precision` never reaches (the
  run's fingerprint refuses it first) and only a tensors file rewritten by
  another tool holds.
- `ster.tune.rollout`: an assistant turn whose first sampled token ends the
  sequence, which the fixture's random weights do not draw on demand.
- `ster.tune.reward`: what an outside scorer answers (an error status, a
  refused bearer, a rate limit, a failure on its side, an answer that breaks
  off, is not text or not JSON, or holds no number at the pointer, and
  finite numbers so far apart that the group's variance overflows, which
  the length reward and a reward artifact's f32 head cannot answer): only a
  real scorer produces it, and a listener written to answer it would be a
  canned answer standing in for one.
- `ster.tune.fit`: a layer without the projection an adapter targets, a
  projection stored fused with others (refused by an export and by an
  ablation) and one stored transposed (refused by an export), which only
  hybrid, fused-projection and GPT-2-style families have, while `ster model
  fixture create` writes a Llama checkpoint; and a device that cannot compute
  the dense update, which needs a machine out of memory.
- `ster.model.decoder`: a layer reading what an earlier layer of the same
  forward pass leaves for it (the keys and values a key-value-sharing or
  looped layer reuses, LongCat-Flash's shortcut experts' output) and finding
  none, or a layer run over a residual form (AltUp or hyper-connected
  streams, attention residuals, a per-layer input) it was not built for. The
  decoder runs every layer in order, records every call's token ids before
  a hash-routed layer reads them, and builds itself and its layers from one
  config; no checkpoint or command makes either differ. A cache continued by
  a batch of another size reaches the n-gram embedding only from a library
  caller, since `ster` decodes one sequence per cache. A token id past a
  routing table needs a checkpoint whose `tid2eid` has fewer rows than its
  embedding, which the tiny DeepSeek-V4 does not have and rewriting a
  safetensors tensor's shape would forge. Logits read from a pass whose
  readout projected none need a reader of logits that runs the reward
  readout, and every one in Ster names a readout that projects. A pass over
  no tokens, activations read at other layers than asked and per-token
  states of another count need the decoder or an encode to break what each
  of them guarantees (encoding refuses a text of no tokens, the decoder
  keeps exactly the layers it is asked for, one state per token).
- `ster.model.t5`: every cause of the T5 encoder-decoder, which no loader
  builds yet: choosing the decoder refuses `model_type: "t5"` before any
  weight is read, so no command reaches the network.
- `ster.vector.layers`, `ster.vector.direction` and `ster.parity`: a model
  of no layers (`ster model fixture create` refuses a config of none, and
  every checkpoint family names one); sides of unequal counts, reads without
  values or of two widths, and a direction of another width than its reads,
  since every caller reads both sides of every pair through the runtime,
  which returns one hidden-size state per layer it is asked for, and `vector
  evaluate` checks an artifact's width against the model first; and a
  reference tensor of no positions, which no Ster command writes and only
  another tool's safetensors file holds.
- `ster.tokenizer.declaration` and `ster.tokenizer.create`: the tokenizers
  library failing to write, read back, encode with or decode with a
  tokenizer it built from a declaration it accepted, which needs another
  release of that library; more model tokens than 32-bit ids, a declaration
  of over four billion entries; and bytes read back that differ from the
  bytes written and synced, which needs another process or a disk fault.
- `ster.star`: the onboarding runtime refusing the star-request journey it
  carries, its offline transport not answering, and a journey with no
  progress right after it was opened, which need another release of
  `wisent-onboarding-client`; a state file the runtime cannot keep is
  tested.
- `ster.tensor`: what Candle refuses while computing: an accelerator out of
  memory or failing a kernel needs a machine whose device memory a test
  exhausts; a backend the build lacks is refused first by the device check
  (`ster.runtime.device`), before any tensor is made on it; a kernel
  without the weights' dtype needs a device and dtype
  pair Ster's own precision check does not already refuse; tensors that do
  not fit, a view a kernel does not take, a gradient through an inference
  kernel and an error Candle raises as a message are defects of Ster's own
  tensor code, which no input provokes; and a tensor file Candle cannot read
  is refused first by `ster.file`, which reads every such file before Candle
  does.
- `ster.telemetry`: wisent-errors' telemetry handing back a refusal naming
  no cause it declares, which needs another release of wisent-errors, and
  the system not naming the running executable, which needs the binary
  removed while it runs.
- `ster.support`: wisent-errors' support client handing back a refusal
  naming no cause it declares, which needs another release of wisent-errors.

An entry that is no longer needed is a finding too.
