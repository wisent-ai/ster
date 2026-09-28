# Chat templates

The conversation format a checkpoint was post-trained in, and what applying
or skipping it does to a run. Every report and every artifact records which
of the two happened.



`--chat-template` decides what shape the text is encoded in, and it defaults to
`auto`. It is on every command that turns prose into tokens — `train`,
`optimize`, `evaluate`, `generate`, `extract`, `pairs synthesize`, and the five
`ster tune` subcommands that load a model — and mirrored as `chatTemplate` on
each matching `/v1` endpoint. `tune merge`, `tune inspect`, `inspect` and the
other `pairs` subcommands take none, for the same reason they take no
`--precision`: none of them encodes any text.

An instruct checkpoint was not post-trained on bare text. It was post-trained on
a conversation wrapped in special markers, and the exact wrapping is published
with the model: a Jinja template in `tokenizer_config.json` under
`chat_template`, or beside it in `chat_template.jinja`. Fine-tuning such a model
on untemplated prompts and completions teaches it a format it will never be
prompted in, and generating from it without the template is what produces the
rambling continuations that make people think a checkpoint is broken. A base
model is the opposite case: it has no template, and raw text is exactly right.

So `auto` applies the model's own template when the checkpoint publishes one and
encodes raw text when it does not, and says which in one sentence on the
progress stream —
`applying the model's own chat template to every prompt and completion`,
`this model publishes no chat template, so prompts and completions are encoded as raw text`,
or, for `off`,
`chat template off, so prompts and completions are encoded as raw text`. The
same decision lands in the run's report as `chat_template`, one of `applied`,
`absent` or `off`, and that report is folded into the adapter artifact, so an
adapter carries the shape it was trained in. There is no `on`: a checkpoint with
no template cannot be forced into one. `off` is the raw-text encoding every
release before this one used, byte for byte.

Under a template a supervised example is rendered twice — the prompt alone with
the marker that opens the assistant's turn, then the whole conversation — and
the completion half is the difference between them. That is what keeps the loss
window exact: a boundary measured on the untemplated prompt is short by the
length of every marker, so the model would be scored on producing its own
header. The tail includes the turn's end marker, which is right, because the
model has to learn to stop. Neither half is tokenized with the tokenizer's own
special tokens, since the rendered string already spells every marker; adding
them again would put a second begin-of-sequence in the middle of the sequence.
A preference pair has no prompt — both sides are complete responses — so each
side is rendered as an assistant turn, which is why `dpo` and `reward` compare
the same shape `sft` trains.

Rendering is done by [`minijinja`](https://crates.io/crates/minijinja), plus the
two globals Hugging Face's own renderer adds, `raise_exception` and
`strftime_now`, and the Python string and mapping methods templates call as
methods rather than filters. A template that uses something outside that is an
error naming the template's own line, not a quiet mis-render, because wrongly
marked training data is worse than none. A template that writes `bos_token` when
the tokenizer config declares no such token is refused for the same reason.

The toy checkpoint has no chat template, which makes it the test for the
fallback: under `auto` it reports `absent` and its ids are identical to `off`.
To exercise the applied path offline, copy
[`docs/examples/chat-template/tokenizer_config.json`](../../examples/chat-template/tokenizer_config.json)
beside a copy of the toy model — it carries a template written in the toy's own
vocabulary, wrapping a user turn as `question : …` and an assistant turn as
`answer : … </s>` — and train on
[`docs/examples/tuning/chat-examples.json`](../../examples/tuning/chat-examples.json), whose
prompts and completions are bare, so the markers come from the template rather
than from the data:

```bash
ster toy-model toy-chat-model
cp docs/examples/chat-template/tokenizer_config.json toy-chat-model/
ster tune evaluate --model toy-chat-model --examples docs/examples/tuning/chat-examples.json
```

The templated run reports one more completion token per example than the same
run with `--chat-template off`: the assistant turn's `</s>`, which the loss now
covers and previously could not.

