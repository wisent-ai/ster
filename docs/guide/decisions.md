# Decisions

Answering typed questions about a state without generating text. The
[README](../../README.md) links here from its command list, and
[calibration](decisions/calibration.md) — making the probabilities honest —
has its own page beside this one.

```text
ster decide --model <MODEL> --request <REQUEST>
            [--revision <REVISION>] [--device cpu] [--chat-template auto|off]
            [--precision f32|f16|bf16] [--permutations 0]
            [--calibration <CALIBRATION>] [--output <OUTPUT>] [--explain]
```

It prints a pretty JSON document on stdout and is also a streamed NDJSON job
on the `ster serve` backend, `POST /v1/decide`, where every flag above is a
camelCase field of the request body and the request document itself travels
inline under `request` rather than as a path, because the desktop composes it.

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

## Refusals

All of these are decided before the checkpoint loads and exit `1` with
`Error:` and the sentence. The serve backend returns the same sentence as a
`400` body before streaming.

- `a decide request has an empty state`
- `a decide request needs at least one question`
- `a decide request has a question with an empty id`
- `question '<id>' has empty instructions`
- `choice question '<id>' needs at least two options`
- `choice question '<id>' has <n> options; Ster labels at most 26`
- `choice question '<id>' has an option with an empty name "<name>"`
- `score question '<id>' needs at least two levels`
- `score question '<id>' has <n> levels; Ster labels at most 26`
- `score question '<id>' has an empty description at level <i>`
- `failed to read <path>`, `invalid decide request <path>` or
  `invalid decide request on standard input`, with the reader's own cause
  beneath.
- `calibration <path> was fitted for model '<fitted>', not '<loaded>'`, and
  the other artifact refusals on the [calibration page](decisions/calibration.md).

After the load, a question whose rendering does not fit the model's context is
refused as `question '<id>' renders to <n> tokens, past the <max> this model
was built for`, and a tokenizer that has no single token for an answer letter
as `this tokenizer has no single token for the answer label "<label>", so it
cannot be read from one position`.

## Tests

[`tests/decide/main.rs`](../../tests/decide/main.rs) drives the built binary
against the real `HuggingFaceTB/SmolLM2-1.7B-Instruct` checkpoint in the
Hugging Face cache: the request above is answered and its response read back
from disk, every order is explained, a calibration is fitted and then read by
a decision and refused by another model, and every refusal above that needs no
model is checked for its exact sentence.

```bash
cargo test --release --test decide
```
