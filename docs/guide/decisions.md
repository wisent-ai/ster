# Decisions

Answering typed questions about a state without generating text. The
[README](../../README.md) links here from its command list, and
[calibration](decisions/calibration.md) — making the probabilities honest —
has its own page beside this one.

```text
ster decide --model <MODEL> --request <REQUEST>
            [--revision <REVISION>] [--device cpu|metal|cuda|cuda:N]
            [--chat-template auto|off] [--precision f32|f16|bf16] [--permutations 0]
            [--calibration <CALIBRATION>] [--output <OUTPUT>] [--explain]
            [--vector <ARTIFACT> --strength <S>]...
```

It prints a pretty JSON document on stdout and is also the operation
`ster request decide`, whose JSON body on stdin carries every flag above but
`--vector` and `--strength` as a camelCase field, with the request document
itself inline under `request` rather than as a path, because the desktop
composes it ([desktop requests](desktop-requests.md)).

## What a decision is

A language model asked a question writes its answer one token at a time, and
everything that makes generated text hard for software to depend on comes from
that: the parsing, the refusals, the option that was never offered, the missing
confidence. A decision needs none of it. When the options are fixed in
advance, the whole answer is already in the distribution the model holds at
one position. Ster shows the model the state, the question and the options
under letters, ends the prompt where the answer would begin, and reads how much
mass the next-token distribution puts on `A`, `B`, `C` and so on. That is one
forward pass with no sampling. It cannot produce an option that was not
offered, and it hands back a probability for every option that was.

The request and answer shapes are the ones TypeSafe's System One API uses for
its hosted model Jev, so a workflow written against that API runs against a
local open-weight checkpoint here. Ster makes no claim about matching that
model's quality; it makes the mechanism available on a checkpoint you hold.

Three things keep the number honest rather than merely available.

**The state is read once.** Every question about one state begins with the
same text, so the tokens all the renderings share go through the model once
with the key-value cache on, and each rendering then costs only its own
suffix: the question, the options, the answer prompt. On the recorded run
below that was 84 shared tokens and 380 across eight renderings. The shared
run is found on the tokens rather than the text, because a tokenizer may merge
across the point where two renderings diverge.

**Every option is shown under every letter.** A model prefers some letters
whatever they label — TinyLlama-1.1B-Chat answers `B` to almost anything —
and one rendering cannot tell that preference from a reading of the content.
So a question with *n* options is rendered in all *n* cyclic orders, each
rendering's letter distribution is normalized on its own, and the
log-probabilities are averaged across the orders. Every option wears the
favoured letter exactly once, so the favour lands on each of them equally and
cancels; what the model read from the content survives in every order and adds
up. `--permutations` caps the number of orders; `1` is a single pass with no
correction and `0`, the default, is all of them.

**The temperature can be fitted.** A raw model is usually too sure.
[`ster calibrate`](decisions/calibration.md) divides the averaged log-scores
by one number fitted on labelled decisions, which sharpens or flattens every
distribution without ever changing which option wins.

## The request

A state and a map of questions. The state is a string, or any JSON value,
which is rendered as JSON. Question ids are yours; the model never sees them
and the answer comes back under the same id. A `model` field is accepted and
ignored, because the model is the checkpoint on the command line.

```json
{
  "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10? I paid already and the invoice was fine.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "returns": "Exchanges, refunds, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language or threatening to leave"]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this message convey urgency?"
    }
  }
}
```

- A **choice** picks one option from a named set. `criteria` maps each option
  name to a description, or `null` when the name says enough. Options are
  shown in name order.
- A **score** places the state on an ordered scale. `criteria` is the levels
  in order, and level *i* is the *i*th entry.
- A **noul** is a yes/no judgement. `criteria` may carry `"true"` and
  `"false"` descriptions of what each side means.

`instructions` and every description may be a string, an object or an array;
anything that is not a string is rendered as JSON for the model to read. Every
option is answered with one letter, so a question carries at most 26 options
or levels.

`--request -` reads the document from standard input.

## The response

This is what the request above produced on `HuggingFaceTB/SmolLM2-1.7B-Instruct`
at revision `31b70e2e869a7173562077fd711b654946d38674`, CPU, `f32`, in 8.1
seconds including the checkpoint load, with the explain block cut for space:

```json
{
  "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "revision": "31b70e2e869a7173562077fd711b654946d38674",
  "chat_template": "applied",
  "precision": "f32",
  "temperature": 1.0,
  "calibration": null,
  "permutations": 0,
  "answers": {
    "department": {
      "type": "choice",
      "choice": "returns",
      "probabilities": { "billing": 0.0998, "returns": 0.8411, "shipping": 0.0590 },
      "confidence": 0.5061
    },
    "frustration": {
      "type": "score",
      "score": 0.9432,
      "legend": { "0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language or threatening to leave" },
      "probabilities": { "0": 0.2082, "1": 0.6404, "2": 0.1514 },
      "confidence": 0.1827
    },
    "is_urgent": { "type": "noul", "noul": 0.6120 }
  },
  "usage": { "input_tokens": 464, "forward_passes": 9, "output_tokens": 0 }
}
```

- `choice` is the option with the most probability; `probabilities` sums to
  one over the options.
- `score` is the probability-weighted level, `Σ level × p(level)`, so it can
  land between levels; `legend` maps each level number back to its text.
- `noul` is the probability of yes. It carries no confidence, because with two
  options the number already says how sure the model is.
- `confidence` on a choice or a score is `1 − entropy / ln(n)`: `1` when all
  the mass is on one option, `0` when it is spread evenly. It describes the
  distribution, not whether the answer is right.
- `usage.input_tokens` counts the tokens the model actually processed — the
  shared state once, then every suffix. `forward_passes` is the prefix pass
  plus one per rendering. `output_tokens` is always zero.

`--output <OUTPUT>` writes the same document to a file as well.

### Reading an answer with `--explain`

`--explain` adds a block per question with the option texts in canonical order
and, for each order the question was shown in, the letter every option wore
and the probability the model put on it in that order alone. This is what to
read when an answer looks flat: it says whether the content or the letter
decided it.

```json
"department": {
  "options": ["billing: Charges, invoices, payment problems", "returns: Exchanges, refunds, wrong or damaged items", "shipping: Delivery status, delays, lost packages"],
  "orders": [
    { "letters": ["A", "B", "C"], "probabilities": [0.1112, 0.8733, 0.0155] },
    { "letters": ["C", "A", "B"], "probabilities": [0.0890, 0.7223, 0.1887] },
    { "letters": ["B", "C", "A"], "probabilities": [0.0903, 0.8467, 0.0631] }
  ]
}
```

Here `returns` wins in every order, whatever letter it wears, which is what a
reading of the content looks like. On TinyLlama-1.1B-Chat the same request
gives `returns` 0.78 in the order where it wears `B`, and 0.11 and 0.08 in the
other two; the averaged answer is 0.34 / 0.32 / 0.34 with confidence 0.0002,
which is the honest result for a model that has not read the message.

## Steered decisions

`--vector <ARTIFACT> --strength <S>`, repeated as `ster generate` takes it,
reads every question with the steering artifacts added to the residual
stream: at each layer any artifact carries, the sum of each vector times its
strength is added at every position the model reads — the shared state and
every rendering of every question — exactly as generation adds it to a
prompt. The answer is the steered model's distribution over the same
letters, and the response names what it was read under:

```json
"steering": [{ "vector": "calm.ster.json", "trait_name": "calm", "strength": 4.0 }]
```

An unsteered response carries no `steering` field. Every artifact is read
before the checkpoint loads; one `--vector` without its `--strength` is
refused (`ster.runtime.vectors.strengths-unpaired`, exit 2), and an artifact
fitted for another model or width, a strength that is not finite and
strengths that are all zero are refused as `ster generate` refuses them
([errors/runtime.md](errors/runtime.md)). `ster decisions benchmark` takes
the same flags; see [training a decision model](#training-a-decision-model).
A calibration is fitted on the unsteered model, so a temperature applied to
a steered read is the one the unsteered model earned.

## Refusals

Every refusal is a declared cause listed with its fix and its test on
[errors/decide.md](errors/decide.md) and
[errors/decide-calibration.md](errors/decide-calibration.md). The request
document is refused before the checkpoint loads and exits `1` with `Error:`
and the sentence; `ster request decide` refuses the same document with a
single result event of status `2`, before anything loads.

- `checking the decision request: its state is empty, …` (`state-empty`),
  `… it asks no question` (`questions-none`), `… a question's id is empty, …`
  (`question-id-empty`)
- `checking question '<id>': its instructions are empty, …`
  (`instructions-empty`)
- `checking choice question '<id>': it offers <n> options, and a decision
  needs at least two to choose between` (`options-too-few`; `score question`
  and `levels` for a score)
- `checking choice question '<id>': it offers <n> options, and each is
  answered under one of the 26 letters A to Z read at one token position`
  (`options-too-many`)
- `… one of its options has an empty name, …` (`option-name-empty`) and
  `checking score question '<id>': its level <i> has an empty description, …`
  (`level-empty`)
- `reading decision request <path>: no file is there: …` (`request-missing`);
  a file this account may not read, a directory, bytes that are not JSON or
  JSON of another form are the [file layer's](errors/file.md) refusals,
  `reading the decision request <path>: …`, with `/dev/stdin` as the path for
  `--request -`
- `reading calibration <path>: …` and `applying calibration <path>: it was
  fitted for model <fitted>, …`, on the [calibration page](decisions/calibration.md)

After the load, a question whose prompt does not fit the model's context is
refused as `rendering question '<id>': its prompt is <n> tokens, and this
model reads at most <max> at once`; a tokenizer with no single token for an
answer letter as `finding the token of answer label "<label>": …`; one that
spells a letter with a token the weights do not score as `reading the answer
under letter <letter>: …` (`letter-outside`); and an `--output` that cannot
be written as `writing the decision to <path>: …`.

## Training a decision model

Four commands turn labelled decisions into a measured adapter, each reading
and writing the one labelled-decision document `ster calibrate` already reads:

- `ster decisions fetch` reads rows of a Hugging Face classification dataset
  into labelled decisions. Its configuration, split, text column and label
  column are named by the caller; none is assumed, because a guessed column
  trains on the wrong text in silence. A row whose text cell is null or blank, or whose
  label is not one of the column's class names, is skipped and counted under
  `skipped`; a text column a row lacks, a text cell that is not a string, a
  page without `rows`, or a label column without string class names is
  refused by name.

  A multiple-choice dataset, whose options differ from row to row (ARC,
  MMLU, TruthfulQA MC1, HellaSwag), is read instead of `--text-field` and
  `--label-field` with the row flags `ster pairs import --benchmark choices`
  takes, each a JSON pointer into the row and checked the same way before
  any row is asked for: `--question` (the state), `--choices` (the
  options), `--answer` and `--answer-form index|label|text`, with `--labels`
  for `label`:

  ```text
  ster decisions fetch --dataset allenai/ai2_arc --config ARC-Challenge --split test \
    --question /question --choices /choices/text --answer /answerKey \
    --answer-form label --labels /choices/label \
    --question-id answer --instructions "Which answer is correct?" --count 500 --output arc.json
  ```

  Each row becomes one choice question whose options are its choices (two
  of the same text are one option) and whose label is its correct choice.
  A row that cannot become one is listed under `report.refused` with its
  `row`, its `reason` and its `cause` of `ster.pairs.import.rows` (or
  `ster.decide.request.options-too-many` for more than 26 choices), and
  `label_counts` counts the position of the correct choice among the row's,
  from 0, so an answer key that favours one position is visible. A fetch
  whose every row is refused is refused with the first refused row's cause;
  a split with no row from `--offset` with `rows-none`. The two kinds of
  flags exclude each other (a usage error, exit 2).
- `ster decisions import` reads a JSONL file of `context`, `options` and
  `label` rows; the question's id and wording are the caller's. A line that
  cannot become a labelled decision is listed under `report.rejected` with
  its `line`, its `reason` (the declared refusal, with what the line held)
  and its `cause` of `ster.decisions.import.rows`: a line that is not JSON
  or not an object, a missing field, a context, options or label of another
  type, a blank context, fewer than two or more than 26 options, a blank or
  repeated option, or a label past the options. An export whose every line
  is refused is refused with its first refused line's cause and fix, and how
  many lines were refused under each cause.
- `ster decisions synthesize` writes labelled decisions with a hosted model
  through Stado's inference alternatives from a schema — a domain and the
  questions — one label per state by construction. `--per-option` and
  `--retry-multiplier` are required; `--temperature` and `--max-tokens` are
  sent when given, and without them the answering alternative's own apply;
  the report records the temperature and token budget stated (null for one
  left out).
- `ster decisions split` holds out a seeded fraction of examples, both
  `--holdout` and `--seed` stated by the caller; `ster tune decide` trains
  LoRA adapters on the training side; `ster decisions benchmark` measures the
  base model and the adapter on the held-out side over the stated
  `--ece-bins`.

`ster decisions benchmark` writes, beside its aggregate `metrics`,
`control` and `by_type`, one entry of `rows` per labelled question: the
`example` it belongs to (from 0), the `question` id, its `options` as a
label names them (a choice's option names, a score's levels, a noul's `true`
and `false`), the `gold` label, and a `base` read holding the `predicted`
option, whether it is `correct` and the `probabilities` of every option in
the order of `options`. With `--vector <ARTIFACT> --strength <S>` (repeated,
as `ster decide` takes them) every question is read a second time under the
steering artifacts: each row gains a `steered` read, and the report a
`steered` block with the `vectors` and strengths, its own `metrics` and
`by_type`, and how many questions the steering turned from wrong to right
(`improved`) and from right to wrong (`worsened`). One run therefore gives
base accuracy (`metrics.accuracy`), steered accuracy
(`steered.metrics.accuracy`) and both counts on the same rows; latency, input
tokens and the shuffled-state control are the unsteered read's.

```text
ster decisions benchmark --model <MODEL> --examples <HELD_OUT> --ece-bins <N> --output <REPORT>
                         [--adapter <ADAPTER>] [--calibration <CALIBRATION>]
                         [--chat-template auto|off] [--precision f32|f16|bf16]
                         [--permutations 0] [--device cpu|metal|cuda|cuda:N]
                         [--vector <ARTIFACT> --strength <S>]...
```

Each has its own page on the documentation site under CLI commands.
