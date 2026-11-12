//! How a checkpoint's decoder differs from the plain Llama block.
//!
//! Every family Ster loads is a rotary, grouped-query decoder; what changes
//! between them is a handful of switches and tensor names, each read from the
//! checkpoint's own `config.json` rather than assumed:
//!
//! * **Head width** — `head_dim` when the config states it (Qwen3, Gemma),
//!   otherwise `hidden_size / num_attention_heads`. Attention may then be
//!   wider or narrower than the residual stream.
//! * **Query and key norms** — Qwen3 and Gemma 3 normalise each head of the
//!   query and key (`self_attn.q_norm`, `self_attn.k_norm` over `head_dim`);
//!   OLMo 2 normalises the whole query and key projection at once. Both come
//!   after the projection and before the rotary embedding.
//! * **Projection bias** — Qwen2 adds a bias to query, key and value; a Llama
//!   or Mistral config with `attention_bias` adds one to the output too.
//! * **Sliding-window attention** — a query on a local layer sees only the
//!   `sliding_window` keys behind it. Which layers are local comes from the
//!   config's `layer_types` when it lists them, otherwise from the family:
//!   every layer for Mistral, layers from `max_window_layers` on for Qwen2 and
//!   Qwen3 with `use_sliding_window`, every even layer for Gemma 2.
//! * **Fused projections** — Phi-3 stores query, key and value as one
//!   `qkv_proj` and the gate and up projections as one `gate_up_proj`; each
//!   is split back into the separate projections at load.
//! * **Gemma's conventions** — every RMS norm scales by `1 + weight`, the
//!   embedding is multiplied by `sqrt(hidden_size)`, the feed-forward gate is
//!   the tanh-approximated GELU, and the word embeddings are tied.
//! * **Gemma 2's additions** — a norm after attention and another after the
//!   feed-forward, scores scaled by `query_pre_attn_scalar` instead of the head
//!   width, and `tanh` soft-capping of attention scores and final logits.
//! * **Post-norm blocks** — OLMo 2 has no norm before attention or the
//!   feed-forward; it normalises each sublayer's output before the residual
//!   add instead.
//! * **Granite's scales** — `embedding_multiplier` on the embedding,
//!   `residual_multiplier` on each sublayer's output, `attention_multiplier`
//!   in place of `1 / sqrt(head_dim)`, and final logits divided by
//!   `logits_scaling`.
//! * **Rotary per layer** — SmolLM3 skips the rotary embedding on the layers
//!   its `no_rope_layers` marks; Gemma 3 rotates local layers with
//!   `rope_local_base_freq` and global ones with `rope_theta`.
//! * **Partial rotation** — `partial_rotary_factor` rotates only the first
//!   share of each head (Phi-4-mini); the rest passes through.
//! * **Rotary scaling** — `linear` divides every angle by a factor; Phi-3's
//!   `longrope` rescales each frequency by a stated factor, a short list
//!   inside the original context and a long one beyond it.
//! * **Norm kind** — RMS for most families; LayerNorm, with a bias
//!   (StableLM, Starcoder2, Phi-2, Nemotron) or without (Cohere), for others.
//! * **Feed-forward kind** — a gated feed-forward for most; a plain
//!   up-activation-down one for Starcoder2, Phi-2 and Nemotron, with its own
//!   tensor names and optional bias.
//! * **Parallel blocks** — Cohere, Phi-2 and StableLM with
//!   `use_parallel_residual` feed one normalised input to attention and the
//!   feed-forward and add both outputs to the residual at once.
//! * **Interleaved rotation** — Cohere rotates adjacent pairs rather than
//!   halves.

mod fields;
mod layers;
mod llama;
#[path = "placing/placement.rs"]
mod placement;
#[path = "placing/targets.rs"]
mod targets;

pub use fields::Architecture;
pub use placement::Placement;
