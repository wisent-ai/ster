# Fine-tuning

What `ster tune` trains, what a run needs before it starts, and what it
records about itself. The training objectives are in
[objectives](objectives.md); reading and folding a trained adapter is in
[adapters](adapters.md). The [README](../../../README.md) links here from its
command list.



`ster tune` declares the training primitives and the operations on their
artifacts. Each leaf has its own required inputs:

```text
ster tune sft --model <MODEL> --examples <EXAMPLES> --output <OUTPUT>
              --rank <R> --alpha <A> --epochs <N> --learning-rate <LR>
              --accumulation <N> --warmup-steps <N> --max-sequence <TOKENS> --batch-size <N> --seed <SEED>
              [--revision <REVISION>] [--device cpu] [--targets query,value]
              [--layers all] [--chat-template auto|off]
              [--precision f32|f16|bf16]
              [--checkpoint <DIR> --checkpoint-every <STEPS>] [--resume <DIR>]
ster tune dpo --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
              --rank <R> --alpha <A> --beta <B> --epochs <N> --learning-rate <LR>
              --accumulation <N> --warmup-steps <N> --max-sequence <TOKENS> --batch-size <N> --seed <SEED>
              [--revision <REVISION>] [--device cpu] [--targets query,value]
              [--layers all] [--loss dpo|ipo]
              [--chat-template auto|off] [--precision f32|f16|bf16]
              [--checkpoint <DIR> --checkpoint-every <STEPS>] [--resume <DIR>]
ster tune bipo --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
               --layer <LAYER> --strength <STRENGTH> --beta <B> --epochs <N>
               --learning-rate <LR> --accumulation <N> --warmup-steps <N>
               --max-sequence <TOKENS> --batch-size <N> --seed <SEED>
               [--revision <REVISION>] [--device cpu] [--loss dpo|ipo]
               [--chat-template auto|off] [--precision f32|f16|bf16]
               [--checkpoint <DIR> --checkpoint-every <STEPS>] [--resume <DIR>]
ster tune reward --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
                 --rank <R> --alpha <A> --epochs <N> --learning-rate <LR>
                 --accumulation <N> --warmup-steps <N> --max-sequence <TOKENS> --batch-size <N> --seed <SEED>
                 [--revision <REVISION>] [--device cpu] [--targets query,value]
                 [--layers all] [--chat-template auto|off]
                 [--precision f32|f16|bf16]
                 [--checkpoint <DIR> --checkpoint-every <STEPS>] [--resume <DIR>]
ster tune grpo --model <MODEL> --prompts <PROMPTS> --output <OUTPUT>
               --group <N> --iterations <N> --beta <B> --rank <R> --alpha <A>
               --learning-rate <LR> --accumulation <N> --warmup-steps <N> --max-new-tokens <N>
               --temperature <T> --top-p <P> --max-sequence <TOKENS> --seed <SEED> --turns <N>
               [--revision <REVISION>] [--device cpu] [--reward length]
               [--user-model <MODEL>] [--user-revision <REVISION>]
               [--targets query,value] [--layers all]
               [--chat-template auto|off] [--precision f32|f16|bf16]
               [--checkpoint <DIR> --checkpoint-every <STEPS>] [--resume <DIR>]
ster tune decide --model <MODEL> --examples <EXAMPLES> --output <OUTPUT>
                 --rank <R> --alpha <A> --epochs <N> --permutations <N>
                 --learning-rate <LR> --accumulation <N> --warmup-steps <N>
                 --max-sequence <TOKENS> --batch-size <N> --seed <SEED>
                 [--revision <REVISION>] [--device cpu] [--targets query,value]
                 [--layers all] [--chat-template auto|off] [--precision f32|f16|bf16]
                 [--checkpoint <DIR> --checkpoint-every <STEPS>] [--resume <DIR>]
ster tune merge --model <MODEL> --adapter <ADAPTER> --output <DIR>
                [--revision <REVISION>] [--device cpu]
ster tune export --model <MODEL> --adapter <ADAPTER> --format peft --output <DIR>
                 [--revision <REVISION>] [--device cpu]
ster tune evaluate --model <MODEL> --examples <EXAMPLES>
                   --max-sequence <TOKENS> --batch-size <N>
                   [--revision <REVISION>] [--device cpu] [--adapter <ADAPTER>]
                   [--chat-template auto|off] [--precision f32|f16|bf16]
ster tune inspect <ARTIFACT>
```

Every training number that applies to the selected objective is the caller's:
rank, alpha, epochs, learning rate, accumulation, warmup steps, sequence limit,
batch size, seed, beta, layer, strength, group size, iterations, token budget,
temperature, nucleus mass, assistant turns and decision permutations. Omitting
one is a usage error naming the flag, and the native body refuses the same
missing field before execution. A zero warmup or a request for every cyclic
answer order is still an explicit choice, not an omitted numeric field.
Each report records the values the run used.

## Checkpoints and resuming a run

All trainers take `--checkpoint <DIR> --checkpoint-every <STEPS>`, and write
their state into
`DIR` after every `STEPS`-th optimizer step: the trained tensors (a reward run's
head among them), both AdamW moments of each, the learning rate of every step
taken, where the run stands (epoch or iteration, steps taken, steps taken in
this pass, the running loss totals — for `dpo` and `reward` their accuracy, tie
and score totals too; for `grpo` the count of completions drawn, which seeds
the next draw, the totals of the iteration in progress and every finished
iteration) and a fingerprint of what the run was started from — the model and
its revision, a digest of the examples, pairs or prompts file, the
chat-template choice, the precision, every training option and the seed, and
for `grpo` the reward source and the model playing the user. Each checkpoint is
written under a unique temporary attempt name and renamed into its completed
directory, `DIR/<checkpoint>`. `DIR/latest` selects that directory; only after
publishing it does Ster remove the checkpoint it replaces. A process stop
before publication leaves the previous checkpoint selected and whole.
Interrupted temporary or unpublished directories can remain, but cannot
collide with a later attempt to save the same step. Directory names are not
step identifiers: read `state.json` for the saved position. Giving one of the
two flags without the other is a usage error naming the missing one (exit 2),
and `--checkpoint-every 0` is refused as no step count.

`--resume <DIR>` continues the stopped run from `DIR/latest`. Ster owns the
optimizer for this (`tune::checkpoint::optimizer`, the same AdamW update as
Candle's, whose moments Candle keeps private), so the resumed run steps with
the moments the stopped one had: it replays the shuffles of the epochs it
skips, skips the steps already taken, and ends with the same adapter and the
same losses as a run that never stopped. A resumed `grpo` run draws exactly the
completions the stopped run would have drawn next, because the draw count is
carried; a reward that answers differently on a second call (an outside scorer
whose service changed) is the one thing a checkpoint cannot replay. The run
being started must be the
run that wrote the checkpoint; otherwise it is refused before a step is
taken:

- `checkpoint <DIR>/<checkpoint> belongs to another run, so continuing it would train an adapter no single run produced: <field> was <old> and is <new>; …`, naming every field that differs;
- `--resume <DIR> holds no checkpoint: <DIR>/latest could not be read`;
- `<DIR>/latest names invalid checkpoint "<name>"; expected one step directory inside <DIR>` when the pointer escapes the checkpoint directory or does not name a checkpoint;
- `<DIR>/<checkpoint>/state.json is not a Ster checkpoint state`, or `… says it is "<kind>", not a ster tune checkpoint`;
- `<DIR>/<checkpoint>/tensors.safetensors holds no <part>/<tensor>, which this run trains`, or `… holds <name>, which this run does not train`.

The report of a resumed run carries `resumed_from` (the step it continued
from) and every report carries `checkpoints` (the steps after which one was
written).

Decision training adds its permutation setting and labelled-decision digest
to the fingerprint. [BiPO](../steering/bipo.md#checkpoints-and-resuming-a-run)
records its direction and corresponding optimizer state.

The request API uses the same `checkpoint`, `checkpointEvery` and `resume`
fields on every trainer. A directory without its step count, or a step count
without its directory, is refused before execution. Ster Desktop's Checkpoints
section sends those fields for every Tune objective. A run started from one
surface resumes through the other because both deserialize the same inputs
and execute the same training implementation.

The real lifecycle journey is
`node --env-file=.build/qualification.env tests/tuning/checkpoints.mjs`.
It trains every objective from explicit declarations, observes atomic
checkpoint-pointer publication, interrupts the actual writer with SIGTERM and
resumes through the native request frontend. Final tensors and loss must match
uninterrupted training. A completed process or a checkpoint at the final step
does not qualify as interruption evidence.

The journey also protects a valid unreferenced older-format attempt, checks
mismatched-run, incomplete-setting and missing-state refusals, and exercises
an escaping pointer that actually resolves to a valid owned checkpoint outside
the checkpoint root. Receipts, persisted state and final comparisons remain
under `.build/tuning/checkpoints/`; see the
[required inputs and interpretation](qualification/consumers.md#checkpoint-lifecycles).

## Reproducibility

A Ster run reproduces itself: the same command, at the same `--seed`, against
the same checkpoint, writes the same numbers and the same adapter bytes, and
`--batch-size 1` reproduces every run recorded before batching existed byte for
byte. Changing `--batch-size` changes the run, for two independent reasons: a
larger batch puts different examples in the same optimizer step, and the
floating-point sums inside a batched forward associate in a different order.
Neither is a defect and neither is a loss of reproducibility — every batch size
reproduces itself.

`--batch-size` counts examples for `sft` and `evaluate`, pairs for `dpo`,
`bipo` and `reward`, and expanded answer-order rows for `decide`. A pair is two
sequences. A batch of one is one row per
forward whatever the unit, so a preference pair's two sides go through the model
as two separate one-row passes — which is what makes the byte
identity above hold. `--accumulation` keeps counting forwards, so a step sees up
to `batch-size * accumulation` rows.

`--precision f32` is byte-identical too, across both the batching and the
precision work: `sft`, `dpo` and `reward` on the toy checkpoint at
`--seed 7 --epochs 2 --layers all`, run through the pre-precision binary and
through the current one at `--precision f32 --batch-size 1`, produce adapters
with identical SHA-256 digests. Those comparisons predate the learning-rate
schedule decaying to zero: runs recorded before it decayed to a tenth of the
base rate, so an adapter trained today at the same settings differs from one
of them whenever the run had steps after its warmup.

## What trains, and what does not

Weight objectives train adapters, with a scoring head added for reward training.
BiPO instead trains a direction at one residual-stream layer; its base remains
frozen too. Base weights are mapped read-only and are not registered as optimizer
variables. The optimizer receives the explicitly registered trainable tensors;
a reward run adds its scalar head to that set. Each weight adapter is a
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
step sees up to `batch-size * accumulation` rows; a batch size of one row per
forward is what reproduces every run recorded before batching existed. A
batched forward right-pads its rows and masks every padded key out of every
real query, which is what makes stacking sequences of different lengths safe —
before that mask existed it would have trained the adapter on whatever filler
the shorter rows carried, silently, under a loss that still looked reasonable.
The KV cache is off while training, because the whole sequence goes through in
one pass, because a cache would keep the previous sequence's keys and values
inside this one's autograd graph, and because one cache cannot hold rows that
end in different places. The learning rate ramps linearly over
`--warmup-steps` steps and then decays on a cosine to zero, as SGDR (Loshchilov
and Hutter, 2017) and Hugging Face's `get_cosine_schedule_with_warmup` do; the
progress counts steps taken, so the last step still learns.

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

Each weight-training objective writes `<name>.safetensors` holding the
factors under the names `layers.{layer}.{target}.a` and
`layers.{layer}.{target}.b`, and `<name>.json` beside it carrying
`schema_version`, `product`, `kind`, `model`, `model_revision`, `rank`, `alpha`,
`targets`, `layers`, `hidden_size`, and the run's own report — so a trained
adapter always carries the run that produced it. `ster tune inspect` prints that
document with every tensor name and shape beside it, and loads no model to do
it.

BiPO writes its learned direction as a steering JSON artifact instead; it has
neither LoRA rank nor projection factors. Use it as a steering direction, not
as `--adapter`.


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


## Consumer qualification

[Tuning consumer qualification](qualification/consumers.md) documents the
revision-bound BiPO and portable PEFT export journeys, explicit experiment
inputs, real vLLM dependency and retained evidence.
