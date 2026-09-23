# Working with a trained adapter

Folding one into the base weights, scoring a checkpoint with it, reading what
it declares, and the contract every artifact this product writes satisfies.

## Merging

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


## Evaluation

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
`ster request tune/evaluate` as well, because the check is in the function
rather than in the flag.

## What fine-tuning does not do

Ster never trains a full weight. The only tensors any objective creates are the
low-rank adapter factors and, on a reward run, the scalar head that reads them.
`ster tune merge` does write full weights, but it folds a finished adapter into
them rather than training them.

Training runs where the rest of Ster runs: it loads no gateway, spends no quota,
touches no credential, and writes nothing but the artifact it was asked for. The
reward and policy loops are as local and as small as the rest — same single
process, same read-only base, same one sequence per forward — and no training is
hosted. There is no distributed training and no fleet placement: a gradient
never leaves this process, and no other product places one for Ster. There is
no judge model and no LLM-as-critic
anywhere in the loop: `tune grpo` takes its reward from a reward model you
trained or from a deterministic function, and if you want a hosted model's
opinion, that is `pairs synthesize --generator brama` producing training data,
not a grader wired into a gradient. Batching is bounded in the same spirit:
`--batch-size` folds rows into one padded forward, and the padding mask is what
keeps that honest, but nothing here grows into a distributed trainer.

All seven operations are operations of `ster request`: `tune/sft`, `tune/dpo`,
`tune/reward`, `tune/grpo`, `tune/merge`, `tune/evaluate` and `tune/inspect`,
each reading its JSON body on stdin and printing NDJSON events like every
other operation ([desktop requests](../desktop-requests.md)), and
`ster request generate` takes the same `adapter` field. That is how Ster
Desktop offers the whole stack on its own screen, and how a finished run
leaves the adapter it wrote in the field Generate reads.

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

