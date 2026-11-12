# Unreleased: chat templates, encoding, devices, generation and steering

Part of the [changelog](../../../CHANGELOG.md).

- `ster.chat.template` splits what used to be one sentence per site: a
  template list without a default (naming the listed names), an entry
  without template text, a template that refuses the turns
  (`raise_exception`, `render-refused`) apart from one calling a method the
  renderer lacks (`render-failed`, with the error kind and line), and a
  template that does not parse (with its line).
- Encoding text declares how it fails (`ster.runtime.encode`,
  `docs/guide/errors/runtime.md`): a blank prompt (now exit 2), text the
  tokenizer turns into no tokens or refuses, a system prompt without a chat
  template, a conversation without one, and an answer label the vocabulary
  has no token for. The unreachable empty-completion and two-token checks of
  `encode_example` are gone. `tests/runtime/encode.sh` provokes each.
- Choosing the device and precision (`ster.runtime.device`): an unknown word
  and bf16 on the cpu exit 2, a device this build lacks exits 1;
  steering (`ster.runtime.steer`, `.layers`): an artifact for another model
  or width, a layer past the model, a strength not finite or all zero;
  sampling (`ster.runtime.generate`): a zero token budget (exit 2), a prompt
  past the context; and `ster generate` (`ster.generate`): a --vector
  without its --strength, now checked before any artifact is read, and a
  system turn or an answers file that cannot be read or written.
  `tests/runtime/device.sh`, `steer.sh` and `generate.sh` provoke them.
  The steering half of `GenerationOptions` and the artifact argument of
  `Runtime::generate`, `sample` and `sample_tokens`, which no caller
  passed, are gone; the steering plan moves to `runtime/infer/steer.rs`.
- A steering artifact declares how it fails (`ster.steering.artifact`,
  `docs/guide/errors/steering.md`): missing, unreadable, an adapter sidecar
  in its place, not an artifact, another schema or product, no vectors, a
  vector of another width or not finite, and an output that cannot be
  written; `ster vector inspect` and `compare` no longer rewrap them.
  `tests/steering/artifact.sh` provokes each. `src/artifact.rs` is split
  into `artifact/mod.rs` and `artifact/steering.rs`.
- Invariants of the decoder and the runtime that no input reaches (a
  batched forward's shape, a forward without logits, activations at layers
  not asked for) are assertions instead of refusals a run could not cause.
- A checkpoint's chat template declares how it fails (`ster.chat.template`,
  `docs/guide/errors/chat.md`): an unknown `--chat-template` word (exit 2),
  a template file that cannot be read, a `chat_template` of no known form, a
  template writing a token the config does not declare, one that does not
  parse or raises while rendering, and one whose rendering hides where the
  completion starts. A template refused at load used to reach the run that
  applied it as a sentence only; it now reaches it as its declared cause.
  `tests/chat/template.sh` provokes each.
- A computation Candle refuses is declared by the variant of Candle's error,
  never its words (`ster.tensor`, `docs/guide/errors/tensor.md`): the
  accelerator refusing it (out of memory or a driver failing, exit 69), a
  kernel without the weights' dtype, tensors Ster built that do not fit, a
  view a kernel does not take, a gradient through an inference kernel, a
  tensor file Candle cannot read, and an error Candle raises as a message.
  A site that knows its step names it; a `?` no site declared is declared
  when the command ends, its step the context the error carries, so no
  Candle error reaches the reader undeclared.
- `ster vector score`, the decision read and the whole-sequence passes
  refuse a passed row of no tokens, activations read at other layers than
  asked and per-token states of another count through declared causes
  (`ster.model.decoder.rows-empty`, `activations-unrequested`,
  `positions-unkept`), where they stopped the run on an assertion.
- `--precision bf16 --device cpu` names the model it would have read.
