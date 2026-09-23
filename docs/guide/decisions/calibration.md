# Calibration

Making the probabilities a [decision](../decisions.md) hands back honest: a
model that is right nine times in ten should say `0.9`, and a raw language
model rarely does. `ster calibrate` fits one temperature on labelled
decisions; `ster decide --calibration` applies it.

```text
ster calibrate --model <MODEL> --examples <EXAMPLES> --output <OUTPUT>
               [--revision <REVISION>] [--device cpu] [--chat-template auto|off]
               [--precision f32|f16|bf16] [--permutations 0]
```

It writes the artifact to `--output`, prints it, and is also the operation
`ster request calibrate`, with `examples` and `output` as paths in the request
body ([desktop requests](../desktop-requests.md)).

## What is fitted

Temperature scaling is the smallest correction that fixes overconfidence: one
number the averaged log-scores of every question are divided by before the
softmax. It sharpens or flattens every distribution at once and cannot change
which option wins, so it cannot trade accuracy for calibration. The value is
the one that minimizes the mean negative log-likelihood of the labels, found
by golden-section search over `e^-3` to `e^3`; a result on either bound is
worth reading as a warning that the labels and the model disagree badly.

## The labelled set

Requests with the correct answer to some of their questions:

```json
{
  "examples": [
    {
      "state": "I was charged twice for order 4471 and the second invoice is wrong. Fix this today or I am disputing it with my bank.",
      "questions": {
        "department": { "type": "choice", "instructions": "Which team should handle this?", "criteria": { "returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems" } },
        "frustration": { "type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language or threatening to leave"] },
        "is_urgent": { "type": "noul", "instructions": "Does this message convey urgency?" }
      },
      "answers": { "department": "billing", "frustration": 2, "is_urgent": true }
    }
  ]
}
```

A choice is labelled with its option name, a score with its level index, a
noul with `true` or `false`. Not every question needs a label. The full set
the recorded run used is
[`tests/decide/fixtures/examples.json`](../../../tests/decide/fixtures/examples.json):
four support messages, eleven labels.

## The artifact

Written on `HuggingFaceTB/SmolLM2-1.7B-Instruct`, CPU, `f32`, in 48 seconds:

```json
{
  "schema": "ster-calibration/1",
  "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "revision": "31b70e2e869a7173562077fd711b654946d38674",
  "chat_template": "applied",
  "precision": "f32",
  "temperature": 0.8367,
  "examples": 4,
  "questions": 11,
  "before":  { "temperature": 1.0,    "nll": 0.6469, "ece": 0.3095, "accuracy": 0.5455 },
  "after":   { "temperature": 0.8367, "nll": 0.6415, "ece": 0.2943, "accuracy": 0.5455 },
  "control": { "temperature": 0.8367, "nll": 1.5052, "ece": 0.4362, "accuracy": 0.2727 }
}
```

- `before` and `after` are the same labels scored at temperature `1.0` and at
  the fitted one: mean negative log-likelihood of the correct option, expected
  calibration error over ten confidence bins, and how often the winning option
  was the labelled one. Accuracy is identical in both, because scaling never
  changes the winner; the fit can only move `nll` and `ece`.
- `control` is the same labels with every question judged against another
  example's state — each example's questions paired with the next example's
  state. A model that reads the state scores its labels better on the real
  pairing than on the shuffled one; a model answering from the options and
  the letters alone scores the same on both, and no temperature can make that
  honest. Here 0.55 against 0.27. The control needs two examples to exist.
- A temperature below one sharpens the distributions; above one flattens
  them.

Eleven labels is enough to demonstrate the mechanism and far too few to trust
the number; fit on the decisions your workflow actually makes.

## Using it

`ster decide --calibration <CALIBRATION>` divides every question's averaged
log-scores by the fitted temperature and records the path and the value in
the response's `calibration` and `temperature` fields. The artifact names the
model it was fitted on, and a decision on a different model is refused before
a weight is mapped:

```text
Error: calibration calibration.json was fitted for model 'HuggingFaceTB/SmolLM2-1.7B-Instruct', not 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
```

## Refusals

All exit `1` with `Error:` and the sentence, before the checkpoint loads;
`ster request calibrate` ends with the same sentence as its last log event and
a status-`1` result.

A labelled set:

- `calibration needs at least one labelled example`
- `calibration needs at least one labelled question`
- `calibration example <i> is not a valid request`, over the request refusal
- `example <i> labels question '<id>', which it does not ask`
- `example <i> answers question '<id>' with <label>, which is not one of its options`
- `invalid calibration examples <path>`, `failed to read <path>`

An artifact handed to `ster decide --calibration`:

- `invalid calibration <path>`
- `calibration <path> has schema "<schema>"; this Ster reads ster-calibration/1`
- `calibration <path> has a temperature that is not a positive number`
- `calibration <path> was fitted for model '<fitted>', not '<loaded>'`
