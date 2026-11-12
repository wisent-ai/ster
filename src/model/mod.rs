//! model.rs — the decoder Ster reads, steers and trains through.
//!
//! # Where the casts are, and the one rule that puts them there
//!
//! The base weights are mapped at whatever dtype the loader chose — F32, F16
//! or BF16 — and the adapters, any head, and every optimizer moment are F32
//! regardless, because a low-rank correction smaller than the weight's own ulp
//! rounds to nothing in half. Inside the forward the rule is narrower than
//! "half everywhere" and can be stated in one line:
//!
//! **A tensor may be held at the weights' dtype. A sum over many terms is
//! taken in F32.**
//!
//! Half precision costs a mantissa, and a mantissa only matters where error
//! accumulates. A projection is one dot product against a weight that is
//! itself half — nothing is gained by widening it. A softmax, a norm, a rotary
//! angle and a log-probability all accumulate across a sequence, and each of
//! them is where a half mantissa turns into a wrong answer rather than a
//! slightly rounded one. So, site by site, with the reason each one is where
//! it is:
//!
//! * **Both masks are `u8` and therefore dtype-free.** [`Cache::mask`] and
//!   [`padded_causal_mask`] mark a key as hidden with a one, and
//!   [`masked_fill`] turns that into a negative infinity in the dtype the
//!   scores are already in. A mask has no precision to lose, and building one
//!   per dtype would be a way to get it wrong.
//! * **Attention scores and the softmax are F32, always.** Query, key and
//!   value are promoted before the score matmul and the result is cast back
//!   after the value matmul. The softmax sums an exponential over the whole
//!   key axis, which is the longest reduction in the pass and the one that
//!   grows with context; in F16 its accumulator saturates while the true value
//!   is still finite.
//! * **Rotary is F32 in and F32 through, and casts back.** The angles are
//!   computed F32 whatever the weights are, because they come from absolute
//!   position rather than from a weight: in F16 two neighbouring late
//!   positions round to the same angle, which rotates two different tokens
//!   identically. [`apply_rotary`] returns the rotated query and key at the
//!   weights' dtype, so the key-value cache still stores half-precision keys
//!   and half precision is still a memory saving. Measured, because the reason
//!   above is only a reason: TinyLlama-1.1B-Chat at F16, two held-out examples
//!   1864 and 1860 tokens long, scored by a build holding the tables and the
//!   rotation at F16 and by this one. The F32 rotation gives a corpus loss of
//!   1.7967694600423176 against the F32 model's own 1.795495907465617 — 0.07%
//!   — while rotating at F16 gives 1.81551726659139, or 1.1%, so this cast
//!   removes about fifteen sixteenths of what half precision otherwise costs
//!   at that length. Both builds agree to the last digit at `--precision f32`.
//!   At 128 tokens the two are indistinguishable, which is the honest limit of
//!   the claim: this matters as the context grows, and a short set cannot see
//!   it.
//! * **Norms promote themselves.** Both spellings — the fused
//!   `ops::rms_norm` and the composed `LayerNorm::forward` that training takes
//!   — cast F16 and BF16 up to F32 for the sum of squares and back afterwards
//!   (candle-nn-0.11.0/src/layer_norm.rs:123-138). This file adds nothing, and
//!   should not: a second promotion around a function that already promotes is
//!   a cast that reads as a safeguard and is really just a copy.
//! * **The readout leaves in F32.** Both the vocabulary projection and the
//!   residual stream are cast to F32 before they are returned, so every loss
//!   in `tune` is F32 arithmetic over an F32 input whatever the checkpoint was
//!   mapped at, and no objective has to know which precision it is training
//!   against.
//!
//! What stays at the weights' dtype is everything whose error does not
//! compound: the embedding lookup, the four attention projections, the three
//! feed-forward projections, the residual stream between layers, and the
//! key-value cache. That is where the bytes are, which is why mapping them
//! half is worth doing at all.
//!
//! Every cast above short-circuits to a handle clone when the dtypes already
//! agree (candle-core-0.11.0/src/tensor.rs:2453), so an F32 run records the
//! same ops it recorded before any of this existed — which is a claim the
//! product is expected to demonstrate by writing a byte-identical adapter, not
//! merely to assert here.

mod architecture;
mod attention;
mod cache;
mod decoder;
mod deepseek4;
mod depth;
mod inkling;
pub(crate) mod input;
pub(crate) mod keys;
mod layer;
#[path = "modes/misfit.rs"]
mod misfit;
#[path = "modes/mode.rs"]
mod mode;
mod names;
mod network;
mod plan;
mod specs;
pub(crate) mod t5;

pub use architecture::{Architecture, Placement};
pub use cache::Cache;
pub use decoder::SteeringLlama;
pub use mode::{Mode, Pass, Readout, Route};
pub use names::{Names, StoredNames};
pub(crate) use network::Network;
pub use plan::{ForwardOutput, SteeringPlan};
pub use specs::*;

pub(crate) use attention::attend::ALIBI_SPAN;
pub(crate) use deepseek4::names::NativeNames;
pub(crate) use misfit::misshapen;
