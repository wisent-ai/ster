# Unreleased: loading models, configs, tokenizers and fixtures

Part of the [changelog](../../../../CHANGELOG.md).

- Main compiles again: the T5 block's clamp names `half::f16::MAX`, and
  `half` is now a declared dependency (candle-core does not re-export it);
  LongCat-Flash's loader passes the config path to the `zero_expert_type`
  read in its refusal, and a latent scale whose switch cannot be read now
  refuses the load instead of failing to compile. Stado's updates of the
  installed ster had failed on these three errors since 66805b5.

- `ster model fixture create` and `ster request model/fixture/create`
  refuse through declared causes of `ster.model.fixture`
  (`docs/guide/models/fixtures.md`, errors page `model-fixture`): each of the
  declaration, config and tokenizer absent (exit 2, where it used to exit 1),
  unreadable (with mode and owner), not JSON or of another shape (with line
  and column), and every check of what they declare, the output and the
  verification, each naming the value it found beside the one it needs. An
  output below a file is refused as such instead of as a directory Ster may
  not search. `tests/models/fixture.sh` provokes 24 of the causes;
  `model fixture create` leaves `tests/errors/uncatalogued.json`.
- Reading a tokenizer says why it failed: a file this account may not read
  (with its mode and owner), one that vanished while read, bytes that are not
  UTF-8, a tokenizer.json that is not JSON apart from JSON the tokenizers
  library does not read (naming its top-level keys), a tokenizer config that
  is not JSON (line and column), and an empty PLaMo or tiktoken vocabulary,
  which used to load and fail at the first prompt with "unk_id is missing"
  (seen live). The library refusing a model Ster built from the vocabulary
  is `ster.model.tokenizer.tokenizer-model-unbuilt` where three `expect`s
  stood. The chat template reads its tokenizer config through the same
  reader.
- `experts/layouts.rs::stored` (275 lines) is split into one function per
  expert layout (`layouts/modules.rs`, `sliced.rs`, `laid.rs`); GRPO's
  running totals move to `grpo/totals.rs`, the chat template's config reading
  to `chat/template/config.rs`, the request events to `request/events.rs`.
- `tests/models/decoder.sh` provokes `ster.model.decoder.route-outside` on a
  copy of a tiny DeepSeek-V4 whose first routing table names expert 256;
  `tests/models/hy_v4.sh` runs under its own home and Hub cache.
- The rotary tables, LFM2's short convolution, DeepSeek-V4's indexer and its
  attention loader, the delta-rule input loader and the artifact requests of
  `ster request vector/*` move to modules of their own, so none of those
  files passes 300 lines.
- Config settings the decoder cannot run are refused while the config is
  read (`ster.model.config.settings-unrunnable`) instead of failing at the
  first weight or forward pass: DeepSeek Sparse Attention's indexer without
  `q_lora_rank` or rotating more than `index_head_dim`, MiniMax-M3 index
  heads that do not divide the query heads or rotate past `sparse_index_dim`,
  Qwen3-Next and OLMo Hybrid value heads uneven among key heads, `xielu`
  outside Apertus' dense feed-forward, and Solar skip connections that blend
  before any layer keeps a state. A DeepSeek-V3.2 config without
  `q_lora_rank` used to stop on a missing `kv_a_proj_with_mqa` tensor.
  `tests/models/settings.sh` provokes the indexer refusals.
- What a forward pass meets that only a defect of Ster could cause is a
  declared `ster.model.decoder` cause (`keys-unshared`, `shortcut-unopened`,
  `layer-form-unbuilt`, `ngram-rows`, `token-ids-unset`, `xielu-unheld`,
  `entries-unmatched`), and a token id past a hash layer's routing table is
  `token-unrouted`; `SteeringLlama::forward_pass` and `forward_batch` return
  the declared refusal a layer raised. The 27 free-text `bail!`s in the model
  are gone: invariants the types can carry are restructured away (the
  indexer lives inside the latent query bottleneck it reads, the dense
  SwiGLU clamp is its own two-variant `DenseClamp`, a no-op mixer or absent
  feed-forward adds zeros, the decoder matches its residual form without an
  `expect`).
- `model/attention/mod.rs`, `model/layer/mod.rs`, `model/decoder.rs` and
  `model/layer/experts.rs` (1643, 1510, 944 and 919 lines) are split into
  modules under 300 lines; `Attention::load_reading`, `Block::load`,
  `SteeringLlama::load_with_adapters` and `Experts::load` are built from
  helpers instead of single bodies of 300 to 530 lines.
- A tokenizer config whose `break_around_repeated_chars_threshold` or
  `break_around_consecutive_spaces_threshold` asks for a split the regular
  expression engine cannot compile is refused as
  `ster.model.tokenizer.split-uncompilable`, naming the key and the value.
- A LongCat n-gram config whose tables do not split `hidden_size` evenly is
  refused while its config is read, naming the three keys.
- A tensor a family stores in another shape than its config gives it (an
  Inkling convolution or global scale, a router bias, a hash routing table,
  an xIELU scalar, `A_log`, any GGUF tensor) is refused as
  `ster.model.weights.weights-unfit` with both shapes; a GGUF tensor the
  file does not list is named as absent, one it cannot read is
  `gguf-unreadable`, and a hash routing table naming an expert its layer
  does not hold is `ster.model.decoder.route-outside`.
- Loading a model refuses through declared causes on
  `docs/guide/errors/model.md`, `model-config.md` and `model-load.md`
  (`ster.model.checkpoint`, `.weights`, `.config`, `.load`): a local path
  with no checkpoint directory is refused as such, where it used to be asked
  of the Hugging Face Hub and answered 404; a Hub repository the Hub does not
  show this account (401, 403, 404) is refused as unknown (exit 1) and a Hub
  that cannot be reached or fails mid-download exits 69; a config that is not
  one JSON object, a missing or mistyped required key, a model type Ster has
  no decoder for, a quantized checkpoint, every setting a family's decoder
  does not implement, GGUF and safetensors files that cannot be mapped,
  weights that do not fit their config, an adapter target the family lacks
  (now exit 2), a family that holds no adapters, and an artifact trained for
  another model, width or purpose each name what failed, why, the state seen
  and the fix. `tests/models/load.sh`, `adapter.sh`, `targets.sh` and
  `hub.sh` provoke them.
- A setting a family's decoder cannot run is refused through the cause of its
  reason, where one cause (`settings-unrunnable`) used to carry the site's
  sentence as its reason: `settings-unimplemented`, `-contradictory`,
  `-out-of-range`, `-incomplete` and `-malformed`, each with its own fix. The
  observed state is the site's sentence and what the config states for every
  key it names (`hidden_act: "tanh"`, `absent` for a key the config leaves
  out), where it used to print a fixed sentence about them.
  `tests/models/settings.sh` provokes the four reasons `load.sh` does not.
  The other causes whose observed state was a fixed sentence now carry what
  they saw too: weights that do not fit
  their config name the tensor, the shape the config gives it and the shape
  the checkpoint stores; a family that holds no adapters names the layers
  that rule them out; a Hub client that cannot start shows `HF_HOME`; Stado
  that cannot start shows `PATH`; a broken call to Stado shows the system's
  error kind; Stado's refusals show its exit status and sentence; and
  `ster errors check` lists its findings in the refusal itself.
- `src/model/mod.rs` is split into `architecture/`, `names/`, `specs/`,
  `mode.rs` and `plan.rs`, and Mistral's `params.json` blocks into
  `runtime/load/mistral/blocks.rs`, so no file of either passes 300 lines.
- Every key of a checkpoint's config is read through one reader,
  `src/model/keys`: the decoder families, the Llama keys Candle needs, the
  rotation (`rope_scaling`, `rope_parameters`), DBRX's, ChatGLM's, Step's,
  MiniMax-M3's and Nemotron Puzzle's key copies, the Mistral `params.json`
  conversion and its tokenizer's end-of-sequence lookup and the multimodal
  `text_config`. Only a key the config leaves out, or
  states as `null`, takes the default its config class gives it; a key
  stated with another type (`"attention_bias": "yes"`, a `rope_type` written
  as a number, a `block_configs` entry's `ffn_mult` written as text) is
  refused as `ster.model.config.key-mistyped`, naming the key — under the
  section it sits in — the value and the type expected. It used to be read
  as if it had been left out, so a mistyped switch silently loaded the model
  with the default.
- The T5 encoder-decoder network (`src/model/t5`,
  `runtime/infer/encoder_decoder.rs`, `model/network`, `model/input`)
  declares how it fails (`ster.model.t5`): inputs Ster's own caller builds
  that disagree, each under its own cause (`t5-cross-attention-misplaced`,
  `t5-encoder-memory-missing`, `-held`, `t5-cache-empty`, `-uneven`,
  `-layers-mismatch`, `-unsupported`, `t5-position-mismatch`,
  `t5-last-position-padded`, `t5-capture-batched`, `-outside`,
  `t5-steering-width`, `t5-lengths-without-tokens`, `t5-lengths-count`,
  `t5-rows-mismatch`, `t5-rows-empty`, `t5-padding-none`, `-outside`) where
  one `t5-inputs-inconsistent` joined every check, sizes past the platform's,
  a table the host has no memory for, a prompt past `n_positions`, a
  checkpoint stating no `n_positions`, and a whole-sequence pass asked of it;
  an encoder's padding lengths are named as the encoder's, where they were
  named the decoder's; its config is read through the
  shared key reader, so a key of another type is `key-mistyped` and one left
  out is `key-absent`, and its weights refuse as `ster.model.weights` does.
  No loader builds it yet: choosing the decoder refuses `model_type: "t5"`
  before any weight is read (`ster.model.config.family-unknown`).
- A Hugging Face Hub refusal names the request the Hub answered (`GET
  https://huggingface.co/api/models/<model>/revision/<revision>` or the
  file's `resolve` address, as the Hub client built it) beside its status
  and the client's words. `tests/models/hub.sh` asserts it.
- Reading a checkpoint's tokenizer declares how it fails
  (`ster.model.tokenizer`): a tokenizer.json the library cannot read, an
  unreadable or non-JSON tokenizer_config.json or vocabulary, a vocabulary row
  out of form, a tiktoken vocabulary whose class is unnamed or unsupported,
  a BOS the config does not name or the vocabulary lacks, and special tokens
  out of range or out of form; `tests/models/tokenizer.sh` provokes each. A
  row out of form is refused through the cause of its reason (a PLaMo row
  that is not JSON or holds no scored piece, a tiktoken row without a
  space, a token that is not base64, a rank that is not a token id), and so
  is an `added_tokens_decoder` out of form (not an object, a key that is
  not an id, an entry without its content, each naming the tokenizer
  config), where two causes carried the site's sentence as their reason.
- Placing LoRA adapters is its own functionality, `ster.model.adapters`,
  which loading, merging and exporting use: a family that holds no adapter
  on every layer is refused through the cause of its reason (short
  convolutions, per-layer plans, shortcut experts, repeated layers,
  hyper-connected streams, Inkling's projections, lightning attention,
  gated delta rules, state-space mixers) and a target the family does not
  store through the cause of its reason (a bottlenecked query, two head
  shapes, value experts, latent attention, experts, no gate), each naming
  the targets asked for, where two causes carried the reason as text.
  `tests/models/targets.sh` provokes six on tiny Hub checkpoints.
- A tokenizer.json or tokenizer_config.json that ends inside its JSON value
  is `ster.model.tokenizer.tokenizer-file-truncated` (cut short, fetch it
  again), where it was refused as not JSON; `tests/models/tokenizer.sh`
  provokes it.
