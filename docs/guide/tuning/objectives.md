# Training objectives

Supervised fine-tuning, preference optimization, reward modelling and policy
optimization: what each one reads, what it writes, and how a run is read back.
[Fine-tuning](fine-tuning.md) covers what they share.

## Supervised fine-tuning

`--examples` reads `{"examples": [{"prompt": "…", "completion": "…"}]}`, with an
optional `name`. An example may also carry `"system": "…"`, the system turn the model
will be served with: the chat template renders it before the user turn, it belongs to
the prompt side and is never a target, and it is what lets a model that is served
under an instruction be trained under that same instruction. A blank `system` is
refused with `example set <label> contains an empty system prompt`, and an example with
a `system` on a run whose chat template is `absent` or `off` is refused with
`training example has a system prompt, which only a chat template can place, and this
run's chat template is <absent|off>`, because raw text has no place for a system turn.
`ster tune evaluate` reads the same field and echoes it in each scored entry.
Eight examples written in the toy checkpoint's own vocabulary
are checked in as [`docs/examples/tuning/examples.json`](../../examples/tuning/examples.json),
so a first run needs no download: `ster toy-model <DIR>` writes that checkpoint
(a four-layer grouped-query Llama shape with seeded random weights and a
WordLevel tokenizer over the same vocabulary). An example longer than `--max-sequence`
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

## Preference optimization

`ster tune dpo` trains the same adapters against a preference instead of a
target. It takes `--pairs`, the contrastive pair set `ster vector train` already
reads: the positive side is the chosen response and the negative side the
rejected one, so a set written for steering trains a preference without being
rewritten and [`docs/examples/pairs.json`](../../examples/pairs.json) runs
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

## Steering-vector preference optimization (BiPO)

`ster tune bipo` learns one steering vector instead of an adapter, by the
bi-directional preference optimization of Cao et al. (arXiv 2406.00045). The
policy is the frozen model with `d · strength · v` added to the residual stream
at `--layer`; the reference is the same model with nothing added, scored once up
front exactly as `ster tune dpo` scores it. Each forward draws `d` from {+, −}:
with `+v` the loss asks the model to prefer each pair's positive side, with `−v`
its negative side, which is the same `--loss` (`dpo` or `ipo`) with `--beta`
negated. One vector therefore learns to move the behaviour both ways, and
`ster generate --vector <artifact> --strength s` turns it up, down or past zero
afterwards. `v` starts at zero, so the first policy is the reference itself, and
it is the only variable: no adapter is registered and no base weight moves.

Every setting is the operator's: `--layer`, `--strength` (the multiple added while
training), `--beta`, `--epochs`, `--learning-rate`, `--accumulation`,
`--warmup-steps`, `--max-sequence`, `--batch-size` and `--seed`. A layer outside
the model is refused with `layer <n> is outside the model's <layers> layers`; a
beta, strength or learning rate that is not a finite number above zero with
`bi-directional preference optimization requires a finite <name> above zero, not
<value>`; epochs, accumulation, batch size or sequence limit of zero with
`bi-directional preference optimization requires <name> of at least one`. Pairs
are encoded and skipped as in `ster tune dpo`. The output is a steering artifact
(`method: "bipo"`, one layer vector whose `train_margin` and `train_accuracy`
are the final epoch's mean reward margin and accuracy), and the report records
`layer`, `strength`, `loss`, `beta`, `pairs`, `trained_pairs`, `skipped_long`,
`epochs`, `steps`, `first_loss`, `final_loss`, `mean_final_epoch_loss`,
`accuracy`, `mean_reward_margin`, `vector_norm`, `learning_rate`,
`accumulation`, `batch`, `chat_template` and `precision`.

```bash
ster toy-model toy-model
ster tune bipo --model toy-model --pairs docs/examples/pairs.json --output calm.bipo.json \
  --layer 2 --strength 1 --beta 0.1 --epochs 2 --learning-rate 0.01 --accumulation 1 \
  --warmup-steps 0 --max-sequence 256 --batch-size 4 --seed 7
ster generate --model toy-model --vector calm.bipo.json --strength 2 --prompt "describe the sea ."
```

## Reward modeling

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

## Policy optimization

`ster tune grpo` is the only trainer that learns from text the model writes
rather than text someone wrote for it. `--prompts` reads
`{"prompts": ["…"]}`, the shape `ster extract` already takes; six prompts in
the toy checkpoint's vocabulary are checked in as
[`docs/examples/tuning/grpo-prompts.json`](../../examples/tuning/grpo-prompts.json). For each
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

`--reward` names where a completion's score comes from, and the three sources
exist for different reasons. `length`, the default, counts the tokens the
policy emitted — a deterministic function with no model behind it, which is
what makes the loop runnable and checkable with no judge, no artifact and no
download. If reward does not rise under it, the bug is in the loop. An
`http://` or `https://` address is an outside scorer, for a quality Ster holds
no model of (an AI-text detector, a house judge): every completion is posted as
`{"text": "<completion>"}`, with `STER_REWARD_BEARER` as the bearer when it is
set, and the reward is the number at the JSON pointer the address's fragment
names — `--reward 'https://detector.example/score#/ai_probability'`; the
fragment is not sent. An address without a fragment is refused with
`reward scorer "…" names no field`, and a scorer that answers an HTTP error,
something other than JSON, or no number at the pointer stops the run with
what it answered, because a missing reward is not a zero. Anything else is a
path to a reward artifact from `ster tune reward`, loaded frozen
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

### Conversations with a simulated user

`--turns N` above one samples whole conversations instead of single
completions. The prompt is the user's first turn; the policy answers it, and
`--user-model` writes the user's next turn by answering the same conversation
with every role swapped, at the run's own `--max-new-tokens`, `--temperature`
and `--top-p`. That repeats until the policy has written `N` turns or the
simulated user replies with nothing, which ends the conversation early. Both
models must render conversations through a chat template; a policy whose
template is absent or turned off is refused with `a conversation of several
turns needs the model's chat template, and this run's chat template is absent`,
and a user model without one with `the simulated user <MODEL> publishes no chat
template, so it cannot write a user's turn in a conversation`.

Each assistant turn is scored by `--reward` with the conversation before it in
front, and a conversation's reward is the sum over its assistant turns. A
conversation the simulated user ends early has fewer turns to earn from, so the
conversation's length in turns is part of the reward without a separate weight.
The advantage is the conversation's, centred on its group, and every assistant
turn in it carries that advantage; the loss averages each turn over its own
tokens and then over the conversation's turns. A refusal is whatever the reward
model scores low, never a list of phrases.

`--turns 1` (the default) with `--user-model` is refused with `a simulated user
writes turns between assistant turns, so it needs turns above one; this run
asked for one`, and `--turns` above one without it with `a conversation of <N>
assistant turns needs a simulated user model to write the turns in between`. A
turn whose conversation plus `--max-new-tokens` would pass `--max-sequence` is
refused with the turn number and both token counts rather than cut.

`ster converse` plays the same conversations with no gradient, for
evaluation: `--model` (with an optional `--adapter`) answers, `--user-model`
plays the user, and each prompt in `--prompts` opens one conversation of at
most `--turns` assistant turns. Every conversation is written to `--output` as
one JSONL line, `{"prompt_index", "messages": [{role, content}, ...]}`, and the
command prints the count of conversations and assistant turns.
`--max-new-tokens`, `--max-sequence`, `--temperature` and `--seed` have no
default: they decide what the evaluation measured, so the run names them.
`--turns` below two is refused with `a conversation with a simulated user needs
--turns of at least 2; one turn is ster generate`.

The report records `reward`, `prompts`, `trained_prompts`, `skipped_long`,
`group`, `turns`, `user_model`, `iterations`, `steps`, `beta`,
`trainable_tensors`, `trainable_parameters`, `first_loss`, `final_loss`,
`mean_reward`, `mean_kl`, `policy_loss`, `max_new_tokens`, `temperature`,
`top_p`, `seed`, `rank`, `alpha`, `targets`, `layers`, `learning_rate`,
`accumulation`, `chat_template`, `precision`, and `history` — one entry per
iteration carrying `iteration`, `groups`, `completions`, `mean_reward`,
`reward_spread`, `mean_kl`, `policy_loss`, `mean_turns` and
`mean_completion_tokens` (per assistant turn). The history is the point: a
single mean over a policy that moved the whole time hides exactly the trend the
run exists to show.

