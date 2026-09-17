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
merge, score and inspect what comes out. And it reads decisions: typed
questions about a state answered from one forward pass, with a probability for
every option and no text generated.

The product is **Ster**. Wisent is the company that builds it.

## Current product contract

Ster 0.14 provides one binary and one library crate. Both use the same versioned
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
- typed decisions — a choice from named options, a score on ordered levels,
  or a yes/no — read from the next-token distribution over answer letters
  after one cached pass over the state, every option shown under every
  letter so the model's letter preference cancels, in TypeSafe's System One
  request and answer shapes;
- temperature calibration of those decisions on labelled examples, reported
  with negative log-likelihood, expected calibration error, accuracy, and a
  shuffled-state control;

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
  no distributed training and no fleet placement. Ster does not place work on
  another machine, and nothing places Ster: it trains on the machine you start
  it on.
- No objective consults a hosted model. `ster tune grpo` takes its reward from
  a reward model you trained or from a deterministic function of the
  completion; there is no judge model and no LLM-as-critic wired into any
  gradient.
- Ster does not place work on a fleet, and manages no credentials: it has no
  credential store, no credential lifecycle, and reads what it needs from the
  environment it was started in. Neither responsibility is delegated to another
  product. The compute registry declares no Ster placement profile, no Ster OS
  unit, and no Ster workload; `stado fleet list` places Ster nowhere; and
  Skarbiec has no record of Ster in its source, README or documentation. If you
  need a run on a bigger machine, you start Ster on that machine yourself.
- Release delivery is the one thing another product does do for Ster: Stado
  installs the built binary from the `.wisent-release.json` this repository
  ships. See [Installing through Stado](#installing-through-stado).
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

Delivery through Stado and the checkpoint cache are covered in
[installing and where downloads land](docs/guide/install.md).


## First steering workflow

If you already have a canonical Ster pair set, import it instead of creating
starter rows:

```bash
ster workspace import-pairs ./my-pairs.json
ster workspace show
```

Ster validates the whole document before writing anything, keeps a canonical
copy under `$XDG_DATA_HOME/ster` (or `~/.local/share/ster`), and makes it the
active set. Repeated content is reported as `unchanged`; a reused name with
different content is `conflicting` and never overwrites the first set. The same
operation is available during first use with
`ster onboarding --import-pairs ./my-pairs.json`.

If you do not have an existing set, create `pairs.json`:


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

Train a direction for layers 12 through 19. The explicit `--pairs` below works
for the manually created file; omit it to use the active imported set:

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
ster onboarding import or replay first use
ster decide     answer typed questions about a state from one forward pass
ster calibrate  fit the temperature that makes decision probabilities honest
ster workspace  import, activate, and inspect persistent pair sets
ster serve      loopback HTTP/JSON backend for desktop apps
```

Run `ster <command> --help` for exact arguments. Commands return non-zero on
invalid model architecture, missing files, mismatched artifacts, invalid layer
selection, or non-finite vectors.

Each command family has its own page in this repository:

- [Pair sets](docs/guide/pair-sets.md) — the file every command reads, and
  `ster pairs`, which authors it.
- [Steering](docs/guide/steering.md) — choosing a direction, reading one, and
  what fitting one out of format costs.
- [Fine-tuning](docs/guide/fine-tuning.md) — what `ster tune` trains, what a
  run needs before it starts, and what it records, with
  [objectives](docs/guide/tuning/objectives.md),
  [adapters](docs/guide/tuning/adapters.md),
  [chat templates](docs/guide/tuning/chat-templates.md) and
  [precision](docs/guide/tuning/precision.md) beside it.
- [Decisions](docs/guide/decisions.md) — `ster decide`: what a decision
  is, the request and response documents, and reading one with `--explain`;
  with [calibration](docs/guide/decisions/calibration.md) beside it.

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
