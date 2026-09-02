<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="ster by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/ster) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/ster/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.com) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/ster)
[![License](https://img.shields.io/github/license/wisent-ai/ster)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54)

# Ster

Ster is a native Rust toolkit for representation reading and activation steering
in open-weight language models. It reads hidden states from selected transformer
layers, learns directions from contrastive examples, evaluates whether those
directions separate the requested trait, and applies them during generation. It
also trains the weights themselves: LoRA adapters under a supervised, a
preference, a reward-modelling or a policy-gradient objective, and the tools to
merge, score and inspect what comes out.

The product is **Ster**. Wisent is the company that builds it.

## Current product contract

Ster 0.13 provides one binary and one library crate. Both use the same versioned
JSON artifacts and native Candle runtime.

Included now:

- local and Hugging Face Llama-family checkpoints published as Safetensors;
- CPU execution, with compile-time Metal and CUDA backends;
- pair-set authoring and inspection for duplicates, refusals, length balance,
  and diversity, with no model loaded;
- synthetic pair generation from a trait description, written either by the
  local runtime or by a hosted model reached through Brama;
- last-token hidden-state extraction from any selected transformer layer;
- contrastive activation addition (`caa`), principal-direction (`pca`), and
  logistic-probe training;
- holdout selection across method and layer, published as the scored candidate
  table the choice was made on;
- artifact evaluation by pair-ordering accuracy and projection margin;
- additive residual-stream steering during autoregressive generation;
- LoRA supervised fine-tuning of local checkpoints from prompt and completion
  examples;
- direct preference optimization, and its IPO variant, over a contrastive pair
  set, scored against the frozen reference the same weights already carry;
- Bradley-Terry reward models: a scalar head trained with the adapters beneath
  it and written in the same artifact;
- group-relative policy optimization against a reward model or an offline
  deterministic reward, with a KL penalty to the frozen base;
- merging an adapter into the base weights as a standalone checkpoint;
- frozen adapter artifacts applied at generation time;
- deterministic JSON pair, activation, steering, and evaluation formats.

Explicit boundaries:

- The current native runtime accepts `model_type: "llama"`. Other architectures
  fail before weights are loaded rather than silently using a wrong adapter.
- Ster controls local open-weight models. Hosted model routing belongs to Brama.
- Steering reads hidden states, so it always runs on a local open-weight model.
  Writing pair text needs no activations, so `ster pairs synthesize` may take
  its generator from Brama instead. Ster holds no provider credential and
  speaks no provider API: it calls the gateway, which owns the routing.
- Fine-tuning trains low-rank adapters, and on a reward run the scalar head
  that reads them. It trains nothing else: the base weights are mapped
  read-only and never registered as trainable. One sequence goes through each
  forward pass with gradient accumulation standing in for a batch, and there is
  no distributed training and no fleet placement: that belongs to Stado.
- No objective consults a hosted model. `ster tune grpo` takes its reward from
  a reward model you trained or from a deterministic function of the
  completion; there is no judge model and no LLM-as-critic wired into any
  gradient.
- Ster does not own fleet placement, credentials, or release delivery; those
  belong to Stado and Skarbiec.
- The previous Python package and `wisent` command were removed in the Rust
  cutover. Python namespace compatibility is not part of the Ster contract.

## Install

Install the current source release from GitHub:

```bash
cargo install --git https://github.com/wisent-ai/ster --locked
```

From this source checkout:

```bash
cargo install --path . --locked
```

Metal and CUDA are build-time choices:

```bash
cargo install --git https://github.com/wisent-ai/ster --features metal --locked
cargo install --git https://github.com/wisent-ai/ster --features cuda --locked
```

The crates.io name `ster` is currently unclaimed and is not Ster's release
surface. `pip install ster` installs unrelated software from another publisher.

### Where checkpoint downloads land

A `--model` argument that is not a local directory is resolved against the
Hugging Face Hub, and the multi-gigabyte weights land in the default hub cache:

- weights, `config.json` and `tokenizer.json`: `~/.cache/huggingface/hub`;
- an optional access token, read only if that file already exists:
  `~/.cache/huggingface/token`.

Ster builds the hub client with hf-hub's bare `Api::new` (`src/runtime.rs:989`)
rather than its environment-aware builder, so `HF_HOME` and `HF_ENDPOINT` do not
move that cache in 0.13. To download somewhere else — another disk, a shared
volume — fetch the repository yourself and pass the directory as `--model`. Ster
runs no free-space preflight, so size the destination before a first fetch.


## First steering workflow

Create `pairs.json`:

```json
{
  "trait_name": "truthful",
  "pairs": [
    {
      "positive": "Question: What evidence supports this claim? Answer: I do not have enough evidence to confirm it.",
      "negative": "Question: What evidence supports this claim? Answer: It is definitely true because it sounds plausible."
    },
    {
      "positive": "Question: Is this citation real? Answer: I cannot verify that citation from the available context.",
      "negative": "Question: Is this citation real? Answer: Yes, the citation is unquestionably real."
    }
  ]
}
```

A set can also be produced without a text editor. `ster pairs add` appends one
pair at a time and creates the file, and its parent directory, when it does not
exist yet:

```bash
ster pairs add \
  --pairs pairs.json \
  --trait truthful \
  --positive "Question: Did the study replicate? Answer: I have not seen a replication, so I cannot claim it did." \
  --negative "Question: Did the study replicate? Answer: Of course it replicated; results that clean always hold."
```

`ster pairs synthesize` writes a whole set from a one-sentence trait
description, generating both sides of every pair with the local runtime by
default:

```bash
ster pairs synthesize \
  --model meta-llama/Llama-3.2-1B \
  --trait "answers only from verifiable evidence and says so when it cannot" \
  --count 20 \
  --output pairs.json
```

Run `ster pairs inspect --pairs pairs.json` before training: it finds duplicate
and near-duplicate pairs, sides that read as refusals, lopsided pairs where one
side is far longer than the other, and how much the set repeats itself.

Train a direction for layers 12 through 19:

```bash
ster train \
  --model meta-llama/Llama-3.2-1B \
  --pairs pairs.json \
  --layers 12..20 \
  --method caa \
  --output truthful.ster.json
```

Generate with that direction:

```bash
ster generate \
  --model meta-llama/Llama-3.2-1B \
  --vector truthful.ster.json \
  --strength 1.0 \
  --prompt "Explain the result and cite only evidence you can verify."
```

Use an immutable Hugging Face commit with `--revision <sha>` when the artifact
must remain reproducible across model updates. A local directory may be passed
to `--model` when it contains `config.json`, `tokenizer.json`, and one or more
Safetensors weight files.

## CLI

```text
ster train      learn one vector per selected layer
ster optimize   select method and layer on an 80/20 holdout
ster evaluate   measure a vector on a contrastive pair set
ster generate   run normal or steered autoregressive generation
ster extract    export hidden states for an arbitrary prompt set
ster inspect    summarize and validate a steering artifact
ster pairs      author, inspect, and synthesize contrastive pair sets
ster tune       train, merge, score, and inspect LoRA adapters
```

Run `ster <command> --help` for exact arguments. Commands return non-zero on
invalid model architecture, missing files, mismatched artifacts, invalid layer
selection, or non-finite vectors.

## Pair sets

`ster pairs` owns the file the training and evaluation commands read. It has
four subcommands:

```text
ster pairs inspect --pairs <FILE> [--dedupe-bits 3] [--dedupe-bands 8]
                                  [--refusal-threshold 0.5]
ster pairs add --pairs <FILE> --positive <TEXT> --negative <TEXT> [--trait <NAME>]
ster pairs remove --pairs <FILE> --index <N>
ster pairs synthesize --trait <TRAIT_DESCRIPTION> --count <COUNT> --output <OUTPUT>
                      [--generator local|brama] [--generator-model <ROUTE>]
                      [--model <MODEL>] [--revision <REVISION>] [--device <DEVICE>]
                      [--chat-template auto|off] [--precision f32|f16|bf16]
                      [--trait-name <NAME>] [--opposite <TEXT>]
                      [--retry-multiplier 3] [--dedupe-bits 3]
                      [--dedupe-bands 8] [--refusal-threshold 0.5]
                      [--max-new-tokens 96] [--temperature 0.9]
                      [--top-p 0.95] [--seed 42]
```

Each subcommand prints a pretty JSON document on stdout, as the other commands
do, and each write leaves a pretty JSON pair set with a trailing newline. `add`
creates the file, and its parent directory, when it does not exist, and
`--trait` sets or replaces the trait name on the file. `remove` takes a
zero-based index and refuses one outside the set with `pair index {i} is outside the set's 0..{n-1} range`.
It also refuses to remove the last pair, because a set is validated before it is
saved: the file is left untouched and the refusal is `pair set {path} contains no pairs`.

`ster pairs inspect` loads no model; every judgement it makes is textual. For
the set it reports `trait_name`, `pair_count`, `duplicate_count`,
`refusal_count`, `unbalanced_count`, and `diversity`. For each pair it reports
`index`, both texts, `positive_chars` and `negative_chars`, `positive_words` and
`negative_words`, `duplicate`, `positive_refusal` and `negative_refusal`, and
`length_ratio`.

Duplicates are found by SimHash over the normalized positive and negative text,
with 64-bit fingerprints built from BLAKE2b feature hashes and bucketed by
banded LSH over `--dedupe-bands` bands. Two pairs are near-duplicates when their
fingerprints differ in at most `--dedupe-bits` bits. The `duplicate` field is
`{"kind":"exact","of":N}` or `{"kind":"near","of":N,"distance":B}`, naming the
earlier pair. The first occurrence wins, so pair order decides which of two
near-identical pairs is flagged: the later one.

Refusals are scored across ten weighted families — `ai_disclaimer`, `policy`,
`apology_hedge`, `unable`, `cannot_action`, `prefer_rather`, `decline_refuse`,
`no_support`, `no_ability`, and `refusal_word` — and a side is flagged at or
above `--refusal-threshold`. A flag carries the score, the family, and the text
that matched, for example
`{"score":0.9,"family":"ai_disclaimer","snippet":"As an AI language model"}`. A
refusal is a useless example because it differs from the other side along the
refusal axis rather than along the trait axis.

`length_ratio` is the longer side over the shorter side in characters, and
`unbalanced_count` counts the pairs above 3.0. A pair whose sides differ that
much in length teaches length instead of the trait, which is the confound to
remove before training rather than to discover afterwards in a flattering
margin.

`diversity` reports `unique_unigrams`, `unique_bigrams`, `avg_jaccard`,
`mean_simhash_hamming`, and `min_simhash_hamming`. Inspection measures them over
the positive sides, and above 256 texts the pairwise passes are sampled with a
seeded RNG.

`ster pairs synthesize` builds a set from a trait description, in this order:

1. The opposite trait is derived with one generation, unless `--opposite` states
   it. An empty answer falls back to `neutral and plain`.
2. Each attempt generates a question, then an answer in the trait's voice, then
   an answer in the opposite's voice. Each side is stored as
   `Question: {q}\nAnswer: {a}`, the shape the example above already uses, which
   keeps the two sides matched on everything but the trait.
3. A negative that reads as a refusal is asked again exactly once with a repair
   instruction; if it still refuses, the pair is dropped. A refusing positive is
   dropped immediately, because the trait itself is what the model declined and
   re-asking would refuse again.
4. A pair within `--dedupe-bits` of a pair already kept is dropped.
5. The attempt budget is `--count` times `--retry-multiplier`, and every attempt
   prints one `synthesizing pair 3/20 (attempt 7)` progress line on stderr.

On the local route the seed advances by one on every model call. Ster builds a
fresh sampler per call, so a fixed seed would return one identical continuation
for the whole run and the set would collapse to a single pair; advancing from
`--seed` keeps the run reproducible from the one seed the caller supplied. A
temperature of zero is refused before the first generation on either route:
`synthesis requires a temperature above zero; argmax generation repeats a single prompt`.

`--generator` chooses who writes the text: `local`, the default, uses the same
local open-weight runtime every other Ster command uses, and `brama` sends the
generation to a hosted model through the Brama gateway. `--model`,
`--revision` and `--device` belong to the local route and are not read by the
hosted one, which loads no weights, resolves no device, and downloads nothing.
The local route refuses a missing model with `pairs synthesize with --generator local requires --model`,
the hosted route refuses a missing route with `pairs synthesize with --generator brama requires --generator-model`,
and anything else is `unknown generator "cloud"; expected local or brama`.

`--chat-template` belongs to the local route for the same reason `--model`
does. `auto`, the default, asks the local generator through the model's own chat
template when the checkpoint publishes one, and `off` asks it as raw text.
Synthesis is the first step of the funnel and everything downstream inherits
what it writes, which makes this the one place the mistake is expensive twice:
an instruct checkpoint addressed without its markers reads a request for pair
text as a document to continue, and answers with meta-instructional debris —
`Step 3: Make sure your emojis are visually appealing` — instead of the answer
that was asked for. `--generator brama` ignores it, because the gateway request
carries messages with roles rather than a rendered string: a chat API is
already a chat API.

The hosted route reads Brama's own documented client variables: `BRAMA_URL`,
the gateway base, defaulting to `https://brama.wisent.com`, and
`BRAMA_BEARER`, the caller's bearer. The launcher that owns the bearer supplies
it, because Ster never reads a vault itself. An empty bearer is refused with
`BRAMA_BEARER is unset or empty; the launcher supplies the Brama bearer`, and a
base that is neither https nor explicit loopback with
`BRAMA_URL must be an https:// base or an explicit http:// loopback address, because Brama answers plain http elsewhere with 426 secure_transport_required`.

`--generator-model` may name exactly four things: a declared alias, such as
Wisent's own chat alias `wisent-backend/chat/primary` or its sibling
`wisent-backend/chat/fallback`; the delegation alias `best`; a canonical
`provider/model` route, such as `anthropic/claude-3-5-haiku-latest`; or a
selector, `any` or `task:<name>`. `best` and the selectors additionally require
an agent-signed request, which Ster does not construct — it sends a bearer and
nothing more — so a Ster run uses a declared alias or a canonical route.
Anything outside that vocabulary is refused with
`the generator model must be a Brama alias, a canonical provider/model route, or a selector`.

The request to the gateway carries exactly `model`, `messages` with one user
message, `max_tokens` and `temperature`, because Brama refuses unknown fields
by name. Neither `--seed` nor `--top-p` therefore travels, and a hosted run is
not seed-reproducible: the provider owns its sampler, and the running
deduplicator is what suppresses repeated draws. Ster duplicates Brama's two
documented bounds locally to save a round trip per pair, in the gateway's own
words: `max_tokens must be between one and 32768` and
`temperature must be finite and between zero and 2`. A gateway refusal is
surfaced in Brama's own words, as `brama refused the completion: 401 unauthorized`
— the status followed by the `message` from Brama's `{"error":{...}}` envelope.
A body without that envelope becomes
`brama refused the completion with {status} and a body that is not its error envelope: {excerpt}`.

The run reports `generator`, which records which of the two wrote the set as
`local:<model id>` or `brama:<route>`, then `trait_name`,
`trait_description`, `opposite`, `requested`, `attempts`, `kept`,
`rejected_empty`, `rejected_refusals`, `rejected_duplicates`,
`refusal_retries`, and `diversity`, so a short set is
explained by the counts rather than guessed at.

The same three operations are jobs on the loopback HTTP/JSON backend that
`ster serve` exposes, streamed as NDJSON like its six existing ones.
`POST /v1/pairs/inspect` returns the inspection document for a path,
`POST /v1/pairs/save` writes a set from `traitName` and `entries` and returns
the path and pair count, and `POST /v1/pairs/synthesize` runs the loop and
returns the written path and the report. The synthesize job takes `generator`,
defaulting to `"local"`, and `generatorModel`, takes `chatTemplate` and
`precision` on the local route, and requires `model` only on that route; it
refuses a hosted run without a route with
`pairs synthesize with the brama generator requires generatorModel`, an
unrecognized value with `unknown generator; expected local or brama`, and a
local run without a model with `pairs synthesize requires a model`. This is how
Ster Desktop offers pair authoring, inspection, and synthesis beside its six
workflows.

## Steering

The steering half of Ster reads hidden states, fits directions from them,
scores those directions, and adds them during generation. Six commands:

```text
ster train --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
           [--revision <REVISION>] [--device cpu] [--layers all]
           [--method caa|pca|logistic] [--chat-template auto|off]
           [--precision f32|f16|bf16]
ster optimize --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
              [--revision <REVISION>] [--device cpu] [--layers all]
              [--chat-template auto|off] [--precision f32|f16|bf16]
ster evaluate --model <MODEL> --pairs <PAIRS> --vector <VECTOR>
              [--revision <REVISION>] [--device cpu]
              [--chat-template auto|off] [--precision f32|f16|bf16]
ster generate --model <MODEL> --prompt <PROMPT> [--vector <VECTOR>]
              [--adapter <ADAPTER>] [--revision <REVISION>] [--device cpu]
              [--chat-template auto|off] [--precision f32|f16|bf16]
              [--strength 1.0] [--max-new-tokens 128] [--temperature 0.0]
              [--top-p <TOP_P>] [--seed 42]
ster extract --model <MODEL> --input <INPUT> --output <OUTPUT>
             [--revision <REVISION>] [--device cpu] [--layers all]
             [--chat-template auto|off] [--precision f32|f16|bf16]
ster inspect <ARTIFACT>
```

`--layers` takes `all`, a comma list, or a half-open range such as `8..16`,
exactly as it does under `ster tune`, and `--method` names the estimator
`train` fits: contrastive activation addition, the leading principal
direction, or a logistic probe. `--chat-template` and `--precision` are on
every one of these commands except `inspect`, which loads no model. Each
command prints a pretty JSON document on stdout, and each is also a streamed
NDJSON job on the `ster serve` backend — `POST /v1/train`, `POST /v1/optimize`,
`POST /v1/evaluate`, `POST /v1/generate`, `POST /v1/extract` and
`POST /v1/inspect` — where every flag above is a camelCase field of the request
body, `chatTemplate` and `precision` included, each defaulting to what the CLI
defaults to.

### Selection

`ster optimize` fits every layer-and-method combination on part of the pair
set, ranks the candidates on pairs none of them were fitted on, and writes the
winner. What is new is that it publishes the ranking. It used to print the
choice — layer 9, method pca — which is a result with no evidence attached, and
a chooser that publishes only its choice is asking to be trusted.

The document is the artifact summary plus a `selection` object holding
`holdout`, with `fit_pairs` and `holdout_pairs`, and `candidates`, one row per
layer and method carrying `layer`, `method`, `holdout_accuracy`,
`holdout_margin` and `selected`. Exactly one row has `selected` true. The rows
stay in the order the search walked them rather than sorted by score, so two
runs over the same layers diff line for line. The scores cost nothing to carry:
they were computed to make the decision.

The split is reported rather than assumed, because "80/20" is a ratio and what
decides whether the ranking means anything is the two counts it produced. The
run says them before it starts —
`fitting each candidate on 3 pairs and ranking on a 1-pair holdout` — and when
the holdout comes out at a single pair it says what that costs:

```text
a one-pair holdout scores every candidate 0 or 1, so this ranking separates almost nothing; add pairs to make the choice mean something
```

That is a fact about the input rather than a defect, so it is stated rather
than refused: four pairs is the smallest set `optimize` accepts, and four pairs
yield a one-pair holdout. Ranking prefers accuracy and breaks ties on margin,
so on a holdout of one pair the tiebreak is doing all of the work.

The published direction is then refitted on every pair, holdout included, and
the artifact's `metadata` records that in one sentence:
`chosen over 66 candidates on a 1-pair holdout, then refitted on all 4 pairs`.
The split existed to rank candidates, and once the ranking is done, throwing
away a fifth of the evidence would be paying for the measurement twice. It also
means the `train_accuracy` and `train_margin` the artifact carries are the
refit's numbers over the whole set rather than the holdout scores in the table:
the table is the evidence for the choice, and the artifact's own numbers
describe the direction that was written.

### Inspection

`ster inspect` validates an artifact and prints a summary of it. It used to
serialize the artifact itself, which on a twenty-two-layer 2048-wide checkpoint
is forty-five thousand floats down a terminal, while `ster tune inspect` beside
it printed tensor names and shapes. A steering vector's content is not readable
and its shape and length are, so `inspect` now prints the same document `train`
and `optimize` print: `schema_version`, `product`, `model`, `model_revision`,
`trait_name`, `method`, `hidden_size`, `precision`, `chat_template`,
`metadata`, and a `layers` array carrying, per layer, `layer`, `width`, `norm`,
`train_accuracy` and `train_margin`. Nothing was removed from the artifact; the
numbers are still on disk for anything that wants them.

`norm` is the Euclidean length of the direction, accumulated in `f64` because a
two-thousand-term sum of squares in `f32` loses its low bits. Every direction
Ster writes is unit-normalized, so a norm that is not 1.0 to within rounding is
the fastest available sign that a file was written by something other than
Ster.

### Provenance

A steering artifact records the precision and the chat-template decision of the
run that fitted it, and `ster evaluate` and `ster generate` check them against
the run that is consuming it. Both call the same helper the tune half has used
for adapters, so a direction read in a space it was not fitted in says so on
the progress stream:

```text
warning: this direction was trained with chat template off and this run encodes applied, so the number below describes a format it was not trained in
warning: this direction was trained at precision f32 and this run maps the base weights at f16, so it is being read in a different space than it was fitted in
```

A warning and not a refusal, deliberately. Both mismatches are things an
operator may want on purpose — measuring how far a direction transfers out of
the format it was fitted in is a real question, and the measurement below is
exactly that experiment — and a refusal would make it impossible rather than
merely deliberate. Only an unnoticed mismatch is a defect. An artifact written
before these fields existed carries `null` in both and warns about neither: an
absent record is not a disagreement.

### What fitting out of format costs

[Chat templates](#chat-templates) covers the flag itself and its three
outcomes. The steering half has one further consequence, and it is the sharper
one: the hidden-state read behind `train`, `optimize`, `evaluate` and `extract`
encodes every prompt through the template, so a direction is fitted in the
space it will be applied in. That was not true before this release — pair text
went to the model raw while `generate` rendered its prompt through the template
— and the difference is not a rounding question. A direction is a displacement
between two points in the residual stream, and where those points sit depends
on the markers around the text that produced them: under a template the model
is answering a user turn, without one it is continuing a document.

Measured on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, at `--precision f32`, over
all 22 layers, with the four-pair set at `~/.stado/work/loop/pairs.json` for
the trait `calm and measured, never alarmed`. One direction was fitted with
`--chat-template off` and another with `auto`, and each was evaluated under
both. The two columns are the mean over the 22 layers of the per-layer accuracy
and margin `ster evaluate` reports:

| fitted | evaluated | mean accuracy | mean margin |
| --- | --- | --- | --- |
| `off` | `off` | 1.0000 | +4.9451 |
| `auto` | `auto` | 1.0000 | +1.0917 |
| `off` | `auto` | 0.7841 | +0.0168 |
| `auto` | `off` | 0.7614 | +0.1160 |

The two matched runs separate all four pairs at every one of the 22 layers. The
two crossed runs do not: accuracy falls below 1.0 at 12 of the 22 layers for
off evaluated as auto, the mean lands near 0.78 whichever way the crossing
runs, and the margin collapses by roughly two orders of magnitude. Layer 21 is
the clearest single reading: +20.6697 fitted and evaluated `off`, and -0.0486
for that same direction evaluated `auto`.

The sign is the part worth stopping on. In the three deepest layers — 19, 20
and 21, in both crossings — the crossed margin is negative, which is not a weak
direction but a wrong one: the projection orders the two sides of a pair
backwards, so adding that direction during generation pushes toward the side
the operator labelled negative. A direction fitted out of format does not
merely lose resolution in the layers where steering is usually applied. It
points the other way.

That is why the read encodes through the template by default rather than
offering it as something to remember. The two matched rows also show what the
flag is not: `off` and `auto` each separate the set perfectly in their own
space, and the larger raw margin of the `off` run is a property of raw-text
geometry rather than a better direction. A margin is only comparable with
another margin taken in the same format, which is exactly what the artifact now
records so that `evaluate` can say when it is not.

## Fine-tuning

`ster tune` owns adapters: four objectives that train one, and three utilities
that read one. It has seven subcommands:

```text
ster tune sft --model <MODEL> --examples <EXAMPLES> --output <OUTPUT>
              [--revision <REVISION>] [--device cpu] [--rank 8] [--alpha 16]
              [--targets query,value] [--layers all] [--epochs 1]
              [--learning-rate 0.0001] [--accumulation 8] [--warmup-steps 0]
              [--max-sequence 512] [--chat-template auto|off] [--batch-size 1]
              [--precision f32|f16|bf16] [--seed 42]
ster tune dpo --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
              [--revision <REVISION>] [--device cpu] [--rank 8] [--alpha 16]
              [--targets query,value] [--layers all] [--beta 0.1]
              [--loss dpo|ipo] [--epochs 1] [--learning-rate 0.0001]
              [--accumulation 8] [--warmup-steps 0] [--max-sequence 512]
              [--chat-template auto|off] [--batch-size 1]
              [--precision f32|f16|bf16] [--seed 42]
ster tune reward --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
                 [--revision <REVISION>] [--device cpu] [--rank 8] [--alpha 16]
                 [--targets query,value] [--layers all] [--epochs 1]
                 [--learning-rate 0.0001] [--accumulation 8] [--warmup-steps 0]
                 [--max-sequence 512] [--chat-template auto|off]
                 [--batch-size 1] [--precision f32|f16|bf16] [--seed 42]
ster tune grpo --model <MODEL> --prompts <PROMPTS> --output <OUTPUT>
               [--revision <REVISION>] [--device cpu] [--reward length]
               [--group 4] [--iterations 1] [--beta 0.04] [--rank 8]
               [--alpha 16] [--targets query,value] [--layers all]
               [--learning-rate 0.0001] [--accumulation 1] [--warmup-steps 0]
               [--max-new-tokens 64] [--temperature 0.9] [--top-p 0.95]
               [--max-sequence 512] [--chat-template auto|off]
               [--precision f32|f16|bf16] [--seed 42]
ster tune merge --model <MODEL> --adapter <ADAPTER> --output <DIR>
                [--revision <REVISION>] [--device cpu]
ster tune evaluate --model <MODEL> --examples <EXAMPLES>
                   [--revision <REVISION>] [--device cpu] [--adapter <ADAPTER>]
                   [--max-sequence 512] [--chat-template auto|off]
                   [--batch-size 1] [--precision f32|f16|bf16]
ster tune inspect <ARTIFACT>
```

### Chat templates

`--chat-template` decides what shape the text is encoded in, and it defaults to
`auto`. It is on every command that turns prose into tokens — `train`,
`optimize`, `evaluate`, `generate`, `extract`, `pairs synthesize`, and the five
`ster tune` subcommands that load a model — and mirrored as `chatTemplate` on
each matching `/v1` endpoint. `tune merge`, `tune inspect`, `inspect` and the
other `pairs` subcommands take none, for the same reason they take no
`--precision`: none of them encodes any text.

An instruct checkpoint was not post-trained on bare text. It was post-trained on
a conversation wrapped in special markers, and the exact wrapping is published
with the model: a Jinja template in `tokenizer_config.json` under
`chat_template`, or beside it in `chat_template.jinja`. Fine-tuning such a model
on untemplated prompts and completions teaches it a format it will never be
prompted in, and generating from it without the template is what produces the
rambling continuations that make people think a checkpoint is broken. A base
model is the opposite case: it has no template, and raw text is exactly right.

So `auto` applies the model's own template when the checkpoint publishes one and
encodes raw text when it does not, and says which in one sentence on the
progress stream —
`applying the model's own chat template to every prompt and completion`,
`this model publishes no chat template, so prompts and completions are encoded as raw text`,
or, for `off`,
`chat template off, so prompts and completions are encoded as raw text`. The
same decision lands in the run's report as `chat_template`, one of `applied`,
`absent` or `off`, and that report is folded into the adapter artifact, so an
adapter carries the shape it was trained in. There is no `on`: a checkpoint with
no template cannot be forced into one. `off` is the raw-text encoding every
release before this one used, byte for byte.

Under a template a supervised example is rendered twice — the prompt alone with
the marker that opens the assistant's turn, then the whole conversation — and
the completion half is the difference between them. That is what keeps the loss
window exact: a boundary measured on the untemplated prompt is short by the
length of every marker, so the model would be scored on producing its own
header. The tail includes the turn's end marker, which is right, because the
model has to learn to stop. Neither half is tokenized with the tokenizer's own
special tokens, since the rendered string already spells every marker; adding
them again would put a second begin-of-sequence in the middle of the sequence.
A preference pair has no prompt — both sides are complete responses — so each
side is rendered as an assistant turn, which is why `dpo` and `reward` compare
the same shape `sft` trains.

Rendering is done by [`minijinja`](https://crates.io/crates/minijinja), plus the
two globals Hugging Face's own renderer adds, `raise_exception` and
`strftime_now`, and the Python string and mapping methods templates call as
methods rather than filters. A template that uses something outside that is an
error naming the template's own line, not a quiet mis-render, because wrongly
marked training data is worse than none. A template that writes `bos_token` when
the tokenizer config declares no such token is refused for the same reason.

The toy checkpoint has no chat template, which makes it the test for the
fallback: under `auto` it reports `absent` and its ids are identical to `off`.
To exercise the applied path offline, copy
[`docs/examples/chat-template/tokenizer_config.json`](docs/examples/chat-template/tokenizer_config.json)
beside a copy of the toy model — it carries a template written in the toy's own
vocabulary, wrapping a user turn as `question : …` and an assistant turn as
`answer : … </s>` — and train on
[`docs/examples/chat-examples.json`](docs/examples/chat-examples.json), whose
prompts and completions are bare, so the markers come from the template rather
than from the data:

```bash
cp -R path/to/toy-model toy-chat-model
cp docs/examples/chat-template/tokenizer_config.json toy-chat-model/
ster tune evaluate --model toy-chat-model --examples docs/examples/chat-examples.json
```

The templated run reports one more completion token per example than the same
run with `--chat-template off`: the assistant turn's `</s>`, which the loss now
covers and previously could not.

### Precision

`--precision` names the dtype the frozen base weights are mapped at — `f32`,
`f16` or `bf16` — and defaults to `f32`, so every run recorded before this flag
existed is unchanged. It is on every command that maps a checkpoint — `train`,
`optimize`, `evaluate`, `generate`, `extract`, `pairs synthesize`, and the five
`ster tune` subcommands that load a model — and mirrored as `precision` on each
matching `/v1` endpoint. `tune merge` and `tune inspect` take none: neither
builds a decoder.

It names the base weights and nothing else. Adapters, a reward run's scalar
head, and every `AdamW` moment stay in F32 whatever it says, and that split is
the whole of mixed precision rather than a detail of it: a low-rank update is
small relative to the weight it corrects, and an update below that weight's own
ulp rounds to nothing in half precision, so the adapter would train while the
model did not move. Candle makes the split nearly free — `AdamW` builds both
moments at each variable's own dtype, `lora::Adapter::forward` already casts
each factor to the activation's dtype, and `to_dtype` is differentiable with a
backward that casts the gradient back — so an F32 adapter stays F32 through a
half-precision forward without anything being arranged.

Three things in the forward pass are held at F32 regardless, because they are
the places where half precision is wrong rather than merely cheaper: attention
promotes queries, keys and values before the score matmul and casts only the
output back; the rotary tables and the rotation itself run in F32, because a
position is an absolute index rather than a weight and a half mantissa there
makes neighbouring late positions round to the same angle — a phase error that
reads as a slightly different sentence and never as a numerical fault; and every
loss is summed in F32, because a log-softmax adds tens of thousands of terms
into one accumulator and in half the smallest of them stop changing it. The
key-value cache and the residual stream still store half-width values, so the
memory saving survives all three.

The rotary one has a number behind it. Scoring two held-out examples of about
1860 tokens on TinyLlama-1.1B-Chat, against the same model's own `f32` loss of
1.795495907465617: rotating in F32 costs 0.07% and rotating in half costs 1.1%,
so the promotion removes roughly fifteen sixteenths of what half precision
would otherwise take. At 128 tokens the two are indistinguishable — no position
is far enough out for a half mantissa to collide two angles — which is the
honest limit of the short fixtures and the reason the long run was worth doing.

A steering vector is cast to the model's dtype on the way in, and it survives
the cast as the same direction. Each vector in an artifact is unit-normalized
F32; at `f16` the mean relative error per component is 1.8e-4, the largest
absolute error 2.9e-5, and the cosine similarity with the original is 1.0 to
within F32's own rounding. Two to four components in 2048 fall below half
precision's smallest normal and become subnormal; none flush to zero, and they
are the near-zero components that carry none of the direction. Generating from
TinyLlama-1.1B-Chat with the same artifact, strength, seed and greedy sampling,
`f32` and `f16` produce character-identical text at strengths 1 and 2 over 32
and 96 tokens on two prompts, steered and unsteered. At strength 4 both collapse
to an immediate end-of-sequence, which is over-steering rather than a precision
effect.

`bf16` on the `cpu` device is refused, at load, before a weight is mapped:

```text
bf16 has no CPU matmul kernel in this Candle build; use --precision f16 for half precision on cpu, or --device metal for bf16
```

That is a fact about Candle rather than a policy. `cpu_backend`'s matmul accepts
F16, F32 and F64 and returns `unsupported dtype BF16 for op matmul` for anything
else, so a bf16 CPU run does not run slowly — it downloads and maps the whole
checkpoint and then dies at the first projection. `f16` on the CPU is real half
arithmetic on an Apple-silicon class machine: `gemm` selects a native
`neonfp16` microkernel on aarch64 when the hardware reports the `fp16` feature.
On `metal` all three work.

Measured on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, CPU, revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, scoring the eight checked-in
examples with `ster tune evaluate --max-sequence 128`:

| | peak resident | user CPU | loss | perplexity |
| --- | --- | --- | --- | --- |
| `--precision f32` | 6.46 GiB | 15.2 s | 4.18851804357814 | 65.92502052501293 |
| `--precision f16` | 4.29 GiB | 10.2 s | 4.17449491605984 | 65.00699737728682 |

Two runs of each, peak resident stable to 0.02% and user CPU to 4%; wall clock
is not reported because the machine was shared and the same run varied between
3.9 s and 10.8 s while its CPU time did not move. The 2.32 GiB saved is the
checkpoint's own weight count at two bytes instead of four. The loss differs by
0.33%, which is what half precision cost in accuracy here, and each precision
reproduces itself exactly.

Training is where it decides whether a run happens at all. The same command as
an SFT run over all 22 layers —
`ster tune sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --examples docs/examples/examples.json --epochs 1 --max-sequence 128 --seed 42`
— trains 88 adapter tensors and 1126400 parameters, and asks for a peak memory
footprint of 102.8 GB at `f32` against 55.6 GB at `f16`, a factor of 1.85. On a
quiet machine both finish; on a loaded one the `f32` run was killed by the
operating system partway through the first epoch — not a Ster refusal and not a
Candle allocation failure, just a signal — while `f16` completed at a final loss
of 4.200512409210205. The same `f32` command completed on a second, quieter
machine, at 76.9 s of user time against 105.6 s of system time and 2.4 million
involuntary context switches, which is the shape of a box compressing memory to
stay alive.

Footprint is quoted rather than peak resident for the training runs, because
resident moved between 23.9 and 34.6 GiB across machines while the footprint
held: resident reflects what the operating system let the process keep, and the
footprint reflects what it asked for. And most of that footprint is the
autograd tape over 22 differentiable layers rather than the weights. The tape is
not what `--precision` controls; the way to shrink it is fewer differentiable
layers, not a narrower dtype. What halving the base weights buys is the margin
that decides whether a loaded machine finishes the run.

### Reproducibility

A Ster run reproduces itself: the same command, at the same `--seed`, against
the same checkpoint, writes the same numbers and the same adapter bytes, and
`--batch-size 1` reproduces every run recorded before batching existed byte for
byte. Changing `--batch-size` changes the run, for two independent reasons: a
larger batch puts different examples in the same optimizer step, and the
floating-point sums inside a batched forward associate in a different order.
Neither is a defect and neither is a loss of reproducibility — every batch size
reproduces itself.

`--batch-size` counts rows per forward: examples for `sft` and `evaluate`, pairs
for `dpo` and `reward`, where a pair is two rows. A batch of one is one row per
forward whatever the unit, so a preference pair's two sides go through the model
as two separate one-row passes at the default — which is what makes the byte
identity above hold. `--accumulation` keeps counting forwards, so a step sees up
to `batch-size * accumulation` rows.

`--precision f32` is byte-identical too, across both the batching and the
precision work: `sft`, `dpo` and `reward` on the toy checkpoint at
`--seed 7 --epochs 2 --layers all`, run through the pre-precision binary and
through the current one at `--precision f32 --batch-size 1`, produce adapters
with identical SHA-256 digests.

### What trains, and what does not

Only the adapters train, in every objective below. Base weights arrive through
`VarBuilder::from_mmaped_safetensors` and are never registered in a `VarMap`, so
the optimizer is handed `varmap.all_vars()` and there is structurally nothing
else it could reach; a reward run adds its scalar head to that same map, and
that head is the only non-adapter weight Ster ever creates. Each adapter is a
pair of factors: `A` is drawn from a normal with mean zero and standard
deviation `1/rank`, and `B` is zeros, so the low-rank update is exactly zero
before the first step. A fresh adapter is the identity, and training starts from
the base model's own behaviour rather than from noise injected into every
projection.

That identity is load-bearing rather than cosmetic, because it makes the frozen
reference free. Two objectives need to compare the policy against the model it
started as — the preference losses against a reference log-probability, the
policy gradient against a KL — and since `B` starts at zero the base weights
*are* that model. Ster reaches it by skipping the low-rank update at every
projection for one pass, which costs one enum comparison per projection instead
of a second multi-gigabyte checkpoint. It also gives each objective a free
correctness check: with an identity adapter the reference and the policy agree
exactly, so the first step's loss is a known constant, and each section below
says which one.

Three mechanics are shared and each is deliberate rather than unfinished.
`--batch-size` sequences go through each forward pass and `--accumulation` of
those forwards are folded into one `AdamW` step from their scaled losses, so a
step sees up to `batch-size * accumulation` rows; the default of one row per
forward is what reproduces every run recorded before batching existed. A
batched forward right-pads its rows and masks every padded key out of every
real query, which is what makes stacking sequences of different lengths safe —
before that mask existed it would have trained the adapter on whatever filler
the shorter rows carried, silently, under a loss that still looked reasonable.
The KV cache is off while training, because the whole sequence goes through in
one pass, because a cache would keep the previous sequence's keys and values
inside this one's autograd graph, and because one cache cannot hold rows that
end in different places. The learning rate ramps linearly over
`--warmup-steps` steps and then decays on a cosine to a tenth of the base rate.

`--targets` names the projections that carry an adapter — `query`, `key`,
`value`, `output`, `gate`, `up`, and `down` — and accepts either the short name
or the Hugging Face spelling, `q_proj` and `k_proj` and the rest. `--layers`
takes `all`, a comma list, or a half-open range such as `8..16`, exactly as
everywhere else in Ster. `--seed` fixes a whole run: it seeds both the
per-traversal shuffle and the draw that fills every `A`, so the same command
writes a byte-identical adapter twice. The draw is taken from Ster's own
generator rather than Candle's initialiser, because the CPU device refuses to be
seeded at all and an adapter nobody can reproduce is not an artifact. A sequence
longer than `--max-sequence` is skipped rather than truncated, with one progress
line naming it, in every objective: a cut sequence is a different sequence.

Every objective writes the same pair of files, `<name>.safetensors` holding the
factors under the names `layers.{layer}.{target}.a` and
`layers.{layer}.{target}.b`, and `<name>.json` beside it carrying
`schema_version`, `product`, `kind`, `model`, `model_revision`, `rank`, `alpha`,
`targets`, `layers`, `hidden_size`, and the run's own report — so a trained
adapter always carries the run that produced it. `ster tune inspect` prints that
document with every tensor name and shape beside it, and loads no model to do
it.

An artifact says what it is, and applying it where it does not belong is a
refusal rather than a wrong answer. `ster generate --adapter <FILE>` attaches a
frozen adapter while the weights are mapped, so every token is generated
through the adapted projections; an adapter trained for another checkpoint is
refused with `adapter was trained for model "…", current model is "…"`, a width
mismatch with `adapter width {a} does not match model width {b}`, a reward
artifact with `adapter artifact is a reward model, not a generation adapter`,
and a path that is not there with `failed to read adapter <path>`. The same
four checks guard `tune merge` and `tune evaluate`, and `tune grpo --reward`
runs them in the other direction, refusing a generation adapter with
`adapter artifact is a generation adapter, not a reward model`.

### Supervised fine-tuning

`--examples` reads `{"examples": [{"prompt": "…", "completion": "…"}]}`, with an
optional `name`. Eight examples written in the toy checkpoint's own vocabulary
are checked in as [`docs/examples/examples.json`](docs/examples/examples.json),
so a first run needs no download. An example longer than `--max-sequence`
tokens is skipped because a cut completion would teach the model to stop early.

The objective is next-token cross-entropy over the completion tokens only. For a
joined sequence of `n` tokens whose completion begins at `boundary`, the scored
logits are `narrow(1, boundary - 1, n - boundary)` against the targets
`ids[boundary..]`: the distribution that predicts a token sits one position to
its left, and the prompt is never a target, because an operator writing a prompt
and a completion is not asking the model to learn to reproduce the prompt.

The report records
`examples`, `trained_examples`, `skipped_long`, `epochs`, `steps`,
`trainable_tensors`, `trainable_parameters`, `first_loss`, `final_loss`,
`mean_final_epoch_loss`, `rank`, `alpha`, `targets`, `layers`,
`learning_rate`, `accumulation`, `batch`, `chat_template`, and `precision`.

A learning rate that is not a finite number above zero is refused before the
first forward pass with
`supervised fine-tuning requires a finite learning rate above zero`, and a set
in which nothing fits the limit with
`every example is longer than the sequence limit, so there is nothing to train on`.

### Preference optimization

`ster tune dpo` trains the same adapters against a preference instead of a
target. It takes `--pairs`, the contrastive pair set `ster train` already
reads: the positive side is the chosen response and the negative side the
rejected one, so a set written for steering trains a preference without being
rewritten and [`docs/examples/pairs.json`](docs/examples/pairs.json) runs
offline on the toy checkpoint.

The objective needs the frozen reference model's log-probabilities of the same
two sequences. Ster gets them from the model it already has, by skipping the
low-rank update at every projection for that pass. That is exact rather than
approximate: `B` is zeros before the first step, so the base weights *are* the
model the policy started as, and a second copy of the checkpoint would double
the resident set to compute numbers these weights already hold. It is also why
the first step's loss is exactly `ln 2` — an identity adapter has a log-ratio
of zero — which is the cheapest available check that the two passes agree. The
reference is scored once for the whole run rather than once per epoch, because
a frozen model's log-probability of a fixed sequence is a constant.

A pair is scored whole rather than split into prompt and completion, because a
pair set carries two complete texts and no prompt field. Nothing is lost: when
the two sides share a leading prefix — exactly what `ster pairs synthesize`
writes as `Question: …\nAnswer: …` — that prefix sits at the same positions in
both sequences, so its log-probability is the same expression on both sides of
the margin and cancels out of the value and the gradient alike.

`--loss` selects the objective. `dpo` is the sigmoid loss the DPO paper
derives, `-log σ(β·margin)`, evaluated through the softplus form that does not
overflow once the policy is confident. `ipo` is the same machinery with the
squared error of equation 17, `(margin - 1/(2β))²`, over log-probabilities
divided by their token count: IPO's target is a fixed number, and a target a
long sequence reaches by length alone is not a preference signal. Anything else
is refused with `unknown preference loss "kto"; expected dpo or ipo`, and a
non-positive `--beta` with
`direct preference optimization requires a finite beta above zero`. A pair with
a side over `--max-sequence` is skipped whole, because half a pair states no
preference, and a set in which nothing fits is refused with
`every pair is longer than the sequence limit, so there is nothing to train on`.

The report records `loss`, `beta`, `pairs`, `trained_pairs`, `skipped_long`,
`epochs`, `steps`, `trainable_tensors`, `trainable_parameters`, `first_loss`,
`final_loss`, `mean_final_epoch_loss`, `accuracy`, `mean_reward_margin`,
`mean_chosen_reward`, `mean_rejected_reward`, `rank`, `alpha`, `targets`,
`layers`, `learning_rate`, `accumulation`, `batch`, `chat_template`, and
`precision`. The implicit reward is
`β·(log π - log π_ref)`, `accuracy` is the share of pairs whose chosen side
already earns the larger one, and both are measured over the final epoch, so
they describe the adapter that was written rather than an average over a policy
that was still moving.

### Reward modeling

`ster tune reward` trains a model that judges text rather than one that writes
it: one scalar per sequence, higher for the response the operator preferred. It
takes the same `--pairs` file, under the Bradley-Terry objective
`-log σ(r_chosen - r_rejected)`.

The head is a single row of `hidden_size` weights applied to the last
position's residual state, which is the only position that has attended to the
whole sequence. It has no bias: a bias is added to both scores and cancels in
the difference, so it would be a parameter with an identically zero gradient.
It is initialized to zeros rather than drawn, because one output row has no
symmetry for a draw to break — which also means a fresh head scores everything
zero and the first loss is exactly `ln 2`, the same identity check `tune dpo`
gives. Only differences are identified by the objective, so the scores in the
report are meaningful against each other and against no external unit.

The head trains together with the adapters beneath it, in one `VarMap` and
under one optimizer, and the run's tensor count is one higher than an adapter
run's because of it. Both are written to one safetensors file: a head reads a
residual stream the adapters shaped, so pairing one with adapters it never saw
would produce scores that mean nothing, and the artifact does not offer that as
a possibility. The sidecar carries a `kind` of `reward` rather than `adapter`,
and the file holds one extra tensor named `reward.head`, shaped
`[1, hidden_size]`. `kind` defaults to `adapter`, so every sidecar written
before it existed still loads and still means what it meant; that is why the
schema version does not move. `ster generate --adapter` refuses a reward
artifact with
`adapter artifact is a reward model, not a generation adapter`, because
attaching its adapters and dropping its head would decode a model nobody
trained.

The report records `pairs`, `trained_pairs`, `skipped_long`, `epochs`, `steps`,
`trainable_tensors`, `trainable_parameters`, `first_loss`, `final_loss`,
`mean_final_epoch_loss`, `accuracy`, `tied_pairs`, `mean_chosen_score`,
`mean_rejected_score`, `mean_score_margin`, `rank`, `alpha`, `targets`,
`layers`, `learning_rate`, `accumulation`, `batch`, `chat_template`, and
`precision`, with `accuracy`, `tied_pairs` and the scores measured over the
final epoch.

One reading of that report is worth stating, because it looks alarming and is
not. A one-epoch run reports `accuracy` 0.0 and `mean_score_margin` 0.0000.
That is not a head that ranked every pair backwards; it is a head that has not
moved. It starts at zeros, so every pair scores an exact tie, and a tie is not a
strict win. `tied_pairs` is what tells the two apart from the document alone:
a head that never moved ties every pair it trained on, so `tied_pairs` equals
`trained_pairs`, while a head that learned the order backwards ties none of
them and reports a negative `mean_score_margin` beside the same accuracy. It
counts exact equality rather than closeness, because the tie it exists to name
is two sides of one pair run through identical weights, not two scores a
trained head happens to find similar. Read the tie count first and the accuracy
second, and give the run more than one epoch before either number means
anything.

### Policy optimization

`ster tune grpo` is the only trainer that learns from text the model writes
rather than text someone wrote for it. `--prompts` reads
`{"prompts": ["…"]}`, the shape `ster extract` already takes; six prompts in
the toy checkpoint's vocabulary are checked in as
[`docs/examples/grpo-prompts.json`](docs/examples/grpo-prompts.json). For each
prompt it samples `--group` completions, scores them, subtracts the group's own
mean, and steps the policy toward the ones that beat it.

The group *is* the baseline. Classic policy gradient needs a second network to
say whether a reward was good; sampling several completions for one prompt and
using their mean removes that network entirely and makes the advantage
scale-free. What it costs is `--group` generations per prompt, which is the
dominant cost of the loop. Two is the smallest group that means anything, and a
smaller one is refused with
`group-relative policy optimization requires a group of at least two completions, because the group is the baseline`.
`--temperature` must exceed zero for the same reason `pairs synthesize`
requires it: argmax would draw one identical completion per group.

`--reward` names where a completion's score comes from, and the two sources
exist for different reasons. `length`, the default, counts the tokens the
policy emitted — a deterministic function with no model behind it, which is
what makes the loop runnable and checkable with no judge, no artifact and no
download. If reward does not rise under it, the bug is in the loop. Anything
else is a path to a reward artifact from `ster tune reward`, loaded frozen
beside the policy; a generation adapter passed there is refused with
`adapter artifact is a generation adapter, not a reward model`, and a path that
is neither with
`reward source "nope" is neither the keyword length nor a file that exists`.
The reward source is resolved before the policy is loaded, so a mismatch is
refused before an operator waits out a policy load to hear it.

Three details are decisions rather than defaults. A group whose completions all
scored the same has advantages of exactly zero and contributes only its KL
term; that falls out of the arithmetic rather than being special-cased, and it
is why there is no epsilon in the denominator — a floor there would turn "no
signal" into "amplify the rounding". The importance ratio `π_θ/π_old` is
exactly one, because with one gradient step per sampling round `π_old` *is*
`π_θ` at the moment of the step; it is written as `exp(logp - logp.detach())`
anyway, which needs no second forward pass to recover `logp_old` and keeps the
reported loss on the published scale, `-A + β·KL`. The KL is the k3 estimator,
`exp(d) - d - 1` for `d = log π_ref - log π_θ`, which is non-negative for every
sample and unbiased for the divergence where the naive `-d` is neither; `π_ref`
is the frozen base, reached by skipping the adapters, exactly as the preference
losses reach it.

Together those three make the first step's loss exactly zero, which is this
objective's identity check: a fresh adapter is the reference, so every KL term
is zero, and the advantages are mean-centred, so the policy term is zero too.

The report records `reward`, `prompts`, `trained_prompts`, `skipped_long`,
`group`, `iterations`, `steps`, `beta`, `trainable_tensors`,
`trainable_parameters`, `first_loss`, `final_loss`, `mean_reward`, `mean_kl`,
`policy_loss`, `max_new_tokens`, `temperature`, `top_p`, `seed`, `rank`,
`alpha`, `targets`, `layers`, `learning_rate`, `accumulation`,
`chat_template`, `precision`, and `history` —
one entry per iteration carrying `iteration`, `groups`, `completions`,
`mean_reward`, `reward_spread`, `mean_kl`, `policy_loss` and
`mean_completion_tokens`. The history is the point: a single mean over a policy
that moved the whole time hides exactly the trend the run exists to show.

### Merging

`ster tune merge` folds an adapter into the base weights and writes an ordinary
checkpoint directory: `model.safetensors` beside the source's own `config.json`
and `tokenizer.json`, which is exactly what `--model` accepts. An adapter is the
right shape while it is being trained and while it is one of several a caller
might swap between, and the wrong shape once it is finished and permanent — it
costs two extra matmuls per adapted projection per token forever, and it means
the model cannot be handed to anything that does not know what a Ster artifact
is. The output is deliberately not a Ster format; a merge that produced
something only Ster could read would have converted a portable adapter into an
unportable model.

No decoder is built. Merging rewrites tensors and never runs the model, so it
resolves the same files through the same Hub path and the same architecture
refusal, and maps nothing. The delta `(alpha / rank) * B @ A` is accumulated in
F32 and cast back to whatever the source weight was, so a BF16 checkpoint merges
to a BF16 checkpoint of the same size; accumulating in the source dtype would
round twice and, at BF16's eight bits of mantissa, would quietly discard small
updates. A sharded source whose shards name the same tensor twice is refused
rather than half-merged.

An adapter for another checkpoint is refused with
`adapter was trained for model "…", current model is "…"`, a width mismatch with
`adapter width {a} does not match model width {b}`, and a reward artifact with
`adapter artifact is a reward model, not a generation adapter` — baking a reward
model's adapters into a checkpoint and dropping its head produces a model that
generates, trained by an objective that never asked it to.

The report records `model`, `model_revision`, `adapter`, `output`, `rank`,
`alpha`, `scale`, `targets`, `layers`, `hidden_size`, `merged_tensors`,
`copied_tensors`, `total_tensors`, `parameters`, `dtype`, and `files`.


### Evaluation

`ster tune evaluate` scores a checkpoint on held-out examples and writes
nothing. The absence of an optimizer is the point: the number is meaningful
precisely because nothing about the run could have moved to produce it, where a
training loss is measured on the data that produced the gradient and falls
whether or not the model learned anything transferable. `--adapter` attaches a
frozen adapter exactly as `generate --adapter` does, so the score is the score
of the model an operator would actually run; omitting it scores the bare
checkpoint, which is the run an adapter is compared against. It takes the fused
kernels, because no gradient is wanted and paying for the composed forms would
buy an autograd tape that is discarded.

Two aggregates are reported because they answer different questions. `loss` is
total negative log-likelihood over total completion tokens, so long examples
count for more and `perplexity`, its exponential, is comparable with corpus
perplexity anywhere else. `mean_example_loss` weighs each example equally
whatever its length, which is usually what an operator comparing two adapters on
a curated set means. Reporting one and calling it "the" loss would silently pick
a side. The report is `f64` throughout: perplexity is the exponential of a loss
and overflows `f32` at a loss of about 89, which a broken adapter can reach, and
a measurement that reports infinity as a JSON null is worse than useless.

The report records `model`, `model_revision`, `adapter`, `name`, `examples`,
`evaluated`, `skipped_long`, `completion_tokens`, `loss`, `perplexity`,
`mean_example_loss`, `mean_example_perplexity`, `chat_template`, `precision`,
and `entries` — one per example with `index`, `prompt`, `completion`,
`completion_tokens`, `loss` and `perplexity`, so the worst example is a sort
rather than a second run.

Scoring runs none of the trainers' preflight — there is no learning rate, no
epoch count and no optimizer to check — so the two argument checks it does make
sit in `evaluate` itself:
`max_sequence must be at least two tokens, so that one token can predict another`,
and a batch of zero with
`evaluation requires a batch of at least one example`. Both refuse on
`POST /v1/tune/evaluate` as well, because the check is in the function rather
than in the flag.

### What fine-tuning does not do

Ster never trains a full weight. The only tensors any objective creates are the
low-rank adapter factors and, on a reward run, the scalar head that reads them.
`ster tune merge` does write full weights, but it folds a finished adapter into
them rather than training them.

Training runs where the rest of Ster runs: it loads no gateway, spends no quota,
touches no credential, and writes nothing but the artifact it was asked for. The
reward and policy loops are as local and as small as the rest — same single
process, same read-only base, same one sequence per forward — and no training is
hosted. There is no distributed training, no fleet placement, and no release
delivery; Brama owns hosted inference, Stado owns fleet placement, and neither is
called from a gradient. There is no judge model and no LLM-as-critic
anywhere in the loop: `tune grpo` takes its reward from a reward model you
trained or from a deterministic function, and if you want a hosted model's
opinion, that is `pairs synthesize --generator brama` producing training data,
not a grader wired into a gradient. Batching is bounded in the same spirit:
`--batch-size` folds rows into one padded forward, and the padding mask is what
keeps that honest, but nothing here grows into a distributed trainer.

All seven operations are jobs on the `ster serve` backend —
`POST /v1/tune/sft`, `POST /v1/tune/dpo`, `POST /v1/tune/reward`,
`POST /v1/tune/grpo`, `POST /v1/tune/merge`, `POST /v1/tune/evaluate` and
`POST /v1/tune/inspect` — streamed as NDJSON like every other job, and
`POST /v1/generate` takes the same `adapter` field. That is how Ster Desktop
offers the whole stack on its own screen, and how a finished run leaves the
adapter it wrote in the field Generate reads.

## Artifact contract

A steering artifact records:

- schema version and product identity;
- model id and resolved model revision;
- the precision the base weights were mapped at, and whether the pairs were
  read through the model's own chat template;
- trait, training method, and hidden width;
- layer-indexed normalized directions;
- training accuracy and projection margin;
- a `metadata` map, which is where `optimize` records how it chose.

`precision` is `"f32"`, `"f16"` or `"bf16"`, spelled exactly as `--precision`
spells it. `chat_template` is `"applied"`, `"absent"` or `"off"`, the same three
words an adapter sidecar records at `train.chat_template`, so one comparison
reads both kinds of artifact. Both are top-level fields, additive and
defaulted, which is why the schema version does not move: an artifact written
before they existed loads with `null` in both, means exactly what it meant, and
disagrees with nothing. Neither is a `metadata` entry, deliberately —
provenance the product wrote has to stay distinguishable from notes the
operator wrote.

Ster refuses an artifact trained for a different model, vector width, schema, or
product. This prevents a plausible-looking vector from being applied to the
wrong residual stream.

It also refuses the other product's document, in both directions. `generate`
takes `--vector` and `--adapter` side by side, and crossing them used to escape
as serde's own message — "missing field `rank` at line 1 column 207003", which
names a field of the type that failed to parse, a byte offset into a file nobody
will open, and nothing about the mistake that was actually made. Each loader now
recognises the other's document before parsing its own:

```text
direction.json is a LoRA adapter sidecar, not a steering artifact: it carries rank and targets where a steering artifact carries trait_name and vectors
direction.json is a steering artifact, not a LoRA adapter sidecar: it carries trait_name and vectors where an adapter sidecar carries rank and targets
```

The first comes from the steering loader, the second from the adapter loader,
and the leading token is the path that was read. Recognition costs one
`serde_json::Value` parse on a path that was about to parse the same bytes
anyway, and it is deliberately narrow: a steering artifact is a `trait_name`
and a set of `vectors`, an adapter sidecar is a `rank` and a set of `targets`,
neither has ever carried the other's pair, and a document carrying neither goes
on to the real loader and gets the real parse error.

## Architecture

The runtime uses Candle directly. Ster owns its Llama decoder loop so every
transformer block exposes two exact operations that generic inference APIs do
not: capture the final-token residual state after a block and add a selected
steering direction before the next block. The same decoder can also run a
differentiable pass, which is what makes fine-tuning possible at all: Candle's
fused `rotary_emb::rope`, `ops::softmax_last_dim`, and `ops::rms_norm` kernels
have no backward pass, so training selects composed equivalents at exactly those
three call sites while inference keeps the fused ones. The same pass also
chooses whether the attached adapters apply, which is what lets preference
optimization score the frozen reference without a second copy of the weights.
Tokenization, Safetensors loading, attention, KV caching, sampling, and device
kernels remain native Rust.

## Documentation and support

- Product documentation: https://ster.wisent.com/docs
- Source and defects: https://github.com/wisent-ai/ster
- Community: https://discord.gg/qRjpkthq54
- Private vulnerabilities: GitHub Security Advisories for this repository

Ster is pre-1.0. Artifact schema changes and supported-model expansion remain
subject to the repository's versioned release contract.

## License

MIT — see [LICENSE](LICENSE).
