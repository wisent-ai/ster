# Fine-tuning

What `ster tune` trains, what a run needs before it starts, and what it
records about itself. The four objectives are in
[objectives](objectives.md); reading and folding a trained adapter is in
[adapters](adapters.md). The [README](../../../README.md) links here from its
command list.



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

## A run that is interrupted is lost

There is one adapter writer in the product, `lora::Artifact::save`, and every
objective reaches it exactly once — after `tune::sft`, `tune::dpo`,
`tune::reward` or `tune::grpo` has returned its finished report. Nothing is
written before that: no periodic checkpoint, no snapshot per epoch, no partial
adapter. Ster installs no signal handler either, so `Ctrl-C`, a closed terminal,
an OOM kill or a lost machine ends the process with nothing on disk. A run
killed in its sixth hour leaves the sixth hour and the five before it
unrecoverable, and there is no `--resume`: the only way to continue is to start
again from the base checkpoint.

Size a run to what you are willing to lose. Fewer `--epochs` over a smaller
`--examples` file, written to separate outputs, is recoverable work; one long
run is not.

## Reproducibility

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

## What trains, and what does not

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

