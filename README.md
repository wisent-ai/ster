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
directions separate the requested trait, and applies them during generation.

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
- holdout selection across method and layer;
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
- Fine-tuning trains adapters and nothing else. The base weights are mapped
  read-only and never registered as trainable, one example goes through each
  forward pass with gradient accumulation standing in for a batch, and there is
  no distributed training and no fleet placement: that belongs to Stado.
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

Run `ster pairs inspect pairs.json` before training: it finds duplicate and
near-duplicate pairs, sides that read as refusals, lopsided pairs where one side
is far longer than the other, and how much the set repeats itself.

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
ster inspect    validate and print a steering artifact
ster pairs      author, inspect, and synthesize contrastive pair sets
ster tune       train adapters by supervision or preference, and inspect them
```

Run `ster <command> --help` for exact arguments. Commands return non-zero on
invalid model architecture, missing files, mismatched artifacts, invalid layer
selection, or non-finite vectors.

## Pair sets

`ster pairs` owns the file the training and evaluation commands read. It has
four subcommands:

```text
ster pairs inspect <FILE> [--dedupe-bits 3] [--dedupe-bands 8]
                          [--refusal-threshold 0.5]
ster pairs add --pairs <FILE> --positive <TEXT> --negative <TEXT> [--trait <NAME>]
ster pairs remove --pairs <FILE> --index <N>
ster pairs synthesize --trait <TRAIT_DESCRIPTION> --count <COUNT> --output <OUTPUT>
                      [--generator local|brama] [--generator-model <ROUTE>]
                      [--model <MODEL>] [--revision <REVISION>] [--device <DEVICE>]
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
defaulting to `"local"`, and `generatorModel`, and requires `model` only on the
local route; it refuses a hosted run without a route with
`pairs synthesize with the brama generator requires generatorModel`, an
unrecognized value with `unknown generator; expected local or brama`, and a
local run without a model with `pairs synthesize requires a model`. This is how
Ster Desktop offers pair authoring, inspection, and synthesis beside its six
workflows.

## Fine-tuning

`ster tune` owns adapters: four objectives that train one, and two
utilities that read one. It has six subcommands:

```text
ster tune sft --model <MODEL> --examples <EXAMPLES> --output <OUTPUT>
              [--revision <REVISION>] [--device cpu] [--rank 8] [--alpha 16]
              [--targets query,value] [--layers all] [--epochs 1]
              [--learning-rate 0.0001] [--accumulation 8] [--warmup-steps 0]
              [--max-sequence 512] [--seed 42]
ster tune dpo --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
              [--revision <REVISION>] [--device cpu] [--rank 8] [--alpha 16]
              [--targets query,value] [--layers all] [--beta 0.1]
              [--loss dpo|ipo] [--epochs 1] [--learning-rate 0.0001]
              [--accumulation 8] [--warmup-steps 0] [--max-sequence 512]
              [--seed 42]
ster tune reward --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
                 [--revision <REVISION>] [--device cpu] [--rank 8] [--alpha 16]
                 [--targets query,value] [--layers all] [--epochs 1]
                 [--learning-rate 0.0001] [--accumulation 8] [--warmup-steps 0]
                 [--max-sequence 512] [--seed 42]
ster tune grpo --model <MODEL> --prompts <PROMPTS> --output <OUTPUT>
               [--revision <REVISION>] [--device cpu] [--reward length]
               [--group 4] [--iterations 1] [--beta 0.04] [--rank 8]
               [--alpha 16] [--targets query,value] [--layers all]
               [--learning-rate 0.0001] [--accumulation 1] [--warmup-steps 0]
               [--max-new-tokens 64] [--temperature 0.9] [--top-p 0.95]
               [--max-sequence 512] [--seed 42]
ster tune merge --model <MODEL> --adapter <ADAPTER> --output <DIR>
                [--revision <REVISION>] [--device cpu]
ster tune inspect <ARTIFACT>
```

`--examples` reads `{"examples": [{"prompt": "…", "completion": "…"}]}`, with an
optional `name`. Eight examples written in the toy checkpoint's own vocabulary
are checked in as [`docs/examples/examples.json`](docs/examples/examples.json),
so a first run needs no download. `--targets` names the projections that carry
an adapter — `query`, `key`, `value`, `output`, `gate`, `up`, and `down` — and
accepts either the short name or the Hugging Face spelling, `q_proj` and
`k_proj` and the rest. `--layers` takes `all`, a comma list, or a half-open
range such as `8..16`, exactly as everywhere else in Ster. An example longer
than `--max-sequence` tokens is skipped rather than truncated, with one progress
line naming it, because a cut completion would teach the model to stop early.

Only the adapters train. Base weights arrive through
`VarBuilder::from_mmaped_safetensors` and are never registered in a `VarMap`, so
the optimizer is handed the adapter tensors and there is structurally nothing
else it could reach. Each adapter is a pair of factors: `A` is initialized
`Randn { mean: 0.0, stdev: 1/rank }` and `B` is zeros, so the low-rank update is
exactly zero before the first step. A fresh adapter is therefore the identity,
and training starts from the base model's own behaviour rather than from noise
injected into every projection.

The objective is next-token cross-entropy over the completion tokens only. For a
joined sequence of `n` tokens whose completion begins at `boundary`, the scored
logits are `narrow(1, boundary - 1, n - boundary)` against the targets
`ids[boundary..]`: the distribution that predicts a token sits one position to
its left, and the prompt is never a target, because an operator writing a prompt
and a completion is not asking the model to learn to reproduce the prompt.

One example goes through each forward pass, and `--accumulation` examples are
folded into one `AdamW` step from their scaled losses. That is deliberate rather
than unfinished: Ster's decoder has no padding token and no attention mask for a
batched sequence, so stacking examples of different lengths would train the
adapter on whatever filler the shorter rows were padded with — silently, under a
loss that still looks reasonable. The KV cache is off while training, because
the whole sequence goes through in one pass and there is nothing to reuse. The
learning rate ramps linearly over `--warmup-steps` steps and then decays on a
cosine to a tenth of the base rate. `--seed` fixes the whole run: it seeds both
the per-epoch shuffle of the example order and the draw that fills every `A`
factor, so the same command writes a byte-identical adapter twice. The draw is
taken from Ster's own generator rather than Candle's initialiser, because the
CPU device refuses to be seeded at all and an adapter nobody can reproduce is
not an artifact.

A run writes two files that travel together: `<name>.lora.safetensors`, holding
the factors under the names `layers.{layer}.{target}.a` and
`layers.{layer}.{target}.b`, and `<name>.lora.json` beside it, carrying
`schema_version`, `product`, `model`, `model_revision`, `rank`, `alpha`,
`targets`, `layers`, `hidden_size`, and the training report. The report records
`examples`, `trained_examples`, `skipped_long`, `epochs`, `steps`,
`trainable_tensors`, `trainable_parameters`, `first_loss`, `final_loss`,
`mean_final_epoch_loss`, `rank`, `alpha`, `targets`, `layers`,
`learning_rate`, and `accumulation`, so a trained adapter always carries the run
that produced it. `ster tune inspect` prints that document with every tensor
name and shape beside it, and loads no model to do it.

`ster generate --adapter <FILE>` attaches a frozen adapter while the weights are
mapped, so every token is generated through the adapted projections. An adapter
trained for another checkpoint is refused with
`adapter was trained for model "…", current model is "…"`, and a path that is
not there with `failed to read adapter <path>`. A learning rate that is not a
finite number above zero is refused before the first forward pass with
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
`layers`, `learning_rate`, and `accumulation`. The implicit reward is
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
`mean_final_epoch_loss`, `accuracy`, `mean_chosen_score`,
`mean_rejected_score`, `mean_score_margin`, `rank`, `alpha`, `targets`,
`layers`, `learning_rate`, and `accumulation`, with `accuracy` and the scores
measured over the final epoch.

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
`alpha`, `targets`, `layers`, `learning_rate`, `accumulation`, and `history` —
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

Training runs where the rest of Ster runs: it loads no gateway, spends no quota,
and writes nothing but the artifact pair. The same six operations are jobs on
the `ster serve` backend, `POST /v1/tune/sft`, `POST /v1/tune/dpo`,
`POST /v1/tune/reward`, `POST /v1/tune/grpo`, `POST /v1/tune/merge` and
`POST /v1/tune/inspect`, and `POST /v1/generate` takes the same `adapter` field.
That is how Ster Desktop offers fine-tuning on its own screen, and how a
finished run leaves the adapter it wrote in the field Generate reads.

## Artifact contract

A steering artifact records:

- schema version and product identity;
- model id and resolved model revision;
- trait, training method, and hidden width;
- layer-indexed normalized directions;
- training accuracy and projection margin.

Ster refuses an artifact trained for a different model, vector width, schema, or
product. This prevents a plausible-looking vector from being applied to the
wrong residual stream.

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
