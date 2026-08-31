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
- synthetic pair generation from a trait description on the local runtime;
- last-token hidden-state extraction from any selected transformer layer;
- contrastive activation addition (`caa`), principal-direction (`pca`), and
  logistic-probe training;
- holdout selection across method and layer;
- artifact evaluation by pair-ordering accuracy and projection margin;
- additive residual-stream steering during autoregressive generation;
- deterministic JSON pair, activation, steering, and evaluation formats.

Explicit boundaries:

- The current native runtime accepts `model_type: "llama"`. Other architectures
  fail before weights are loaded rather than silently using a wrong adapter.
- Ster controls local open-weight models. Hosted model routing belongs to Brama.
- Pair synthesis generates text with the same local open-weight runtime as every
  other Ster command. Ster still calls no hosted model to author a pair set;
  hosted routing belongs to Brama.
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
description, generating both sides of every pair with the loaded model:

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
ster pairs synthesize --model <MODEL> --trait <TRAIT_DESCRIPTION> --count <COUNT>
                      --output <OUTPUT> [--revision <REVISION>] [--device cpu]
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

`ster pairs synthesize` builds a set from a trait description on the same local
runtime every other Ster command uses, in this order:

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

The seed advances by one on every model call. Ster builds a fresh sampler per
call, so a fixed seed would return one identical continuation for the whole run
and the set would collapse to a single pair; advancing from `--seed` keeps the
run reproducible from the one seed the caller supplied. A temperature of zero is
refused before the first generation: `synthesis requires a temperature above zero; argmax generation repeats a single prompt`.

The run reports `trait_name`, `trait_description`, `opposite`, `requested`,
`attempts`, `kept`, `rejected_empty`, `rejected_refusals`,
`rejected_duplicates`, `refusal_retries`, and `diversity`, so a short set is
explained by the counts rather than guessed at.

The same three operations are jobs on the loopback HTTP/JSON backend that
`ster serve` exposes, streamed as NDJSON like its six existing ones.
`POST /v1/pairs/inspect` returns the inspection document for a path,
`POST /v1/pairs/save` writes a set from `traitName` and `entries` and returns
the path and pair count, and `POST /v1/pairs/synthesize` runs the loop and
returns the written path and the report. This is how Ster Desktop offers pair
authoring, inspection, and synthesis beside its six workflows.

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
steering direction before the next block. Tokenization, Safetensors loading,
attention, KV caching, sampling, and device kernels remain native Rust.

## Documentation and support

- Product documentation: https://ster.wisent.com/docs
- Source and defects: https://github.com/wisent-ai/ster
- Community: https://discord.gg/qRjpkthq54
- Private vulnerabilities: GitHub Security Advisories for this repository

Ster is pre-1.0. Artifact schema changes and supported-model expansion remain
subject to the repository's versioned release contract.

## License

MIT — see [LICENSE](LICENSE).
