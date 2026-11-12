//! The rotary embedding a query and key turn by: Candle's kernels on an
//! inference pass, the same arithmetic composed of differentiable ops on a
//! training pass.

use candle_core::{DType, Tensor};

use super::super::Pass;

/// Rotates `input` by `cos` and `sin`, the angles of exactly its positions,
/// in F32, and returns the result at `weights`.
///
/// The rotation is a multiply-add against angles of absolute position, so it
/// is the one place in the forward where a half-precision
/// mantissa is spent on something other than a weight: at F16 the angles
/// themselves collide late in the context, and the phase error it introduces
/// looks like a slightly different sentence rather than like a numerical
/// fault. F32 in, F32 through, `weights` out — the cast back is what keeps a
/// half-precision key-value cache half-precision, since the key this returns
/// is the key the cache stores.
///
/// At F32 every cast here short-circuits to a handle clone
/// (candle-core-0.11.0/src/tensor.rs:2453), so the F32 path is unchanged down
/// to the op it records.
///
/// The angles are `rotary_dim / 2` wide. When that is narrower than the head
/// (a config's `partial_rotary_factor`), only the head's first `rotary_dim`
/// components rotate and the rest pass through, as Phi-4-mini, GPT-NeoX and
/// StableLM do. `interleaved` rotates adjacent pairs `(2i, 2i + 1)` instead
/// of halves `d / 2` apart, as Cohere does.
pub(in crate::model) fn apply_rotary(
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    weights: DType,
    pass: Pass,
    interleaved: bool,
) -> candle_core::Result<Tensor> {
    let (_, _, _, head_dim) = input.dims4()?;
    let rotary_dim = 2 * cos.dim(1)?;
    let input = input.to_dtype(DType::F32)?;
    let (rotating, kept) = if rotary_dim < head_dim {
        (
            input.narrow(3, 0, rotary_dim)?,
            Some(input.narrow(3, rotary_dim, head_dim - rotary_dim)?),
        )
    } else {
        (input, None)
    };
    let rotated = match (pass, interleaved) {
        (Pass::Inference, false) => {
            candle_nn::rotary_emb::rope(&rotating.contiguous()?, &cos, &sin)?
        }
        (Pass::Inference, true) => {
            candle_nn::rotary_emb::rope_i(&rotating.contiguous()?, &cos, &sin)?
        }
        (Pass::Differentiable, false) => rope_composed(&rotating, &cos, &sin)?,
        (Pass::Differentiable, true) => rope_interleaved_composed(&rotating, &cos, &sin)?,
    };
    let rotated = match kept {
        Some(kept) => Tensor::cat(&[&rotated, &kept], 3)?,
        None => rotated,
    };
    rotated.to_dtype(weights)
}

/// The interleaved rotation written out of ops that have a backward pass:
/// `candle_nn::rotary_emb::rope_i` computes, for each pair `(2i, 2i + 1)`,
/// `x[2i] * cos[i] - x[2i+1] * sin[i]` and `x[2i] * sin[i] + x[2i+1] * cos[i]`,
/// and ends in a no-backward op like `rope`. The head is viewed as `d / 2`
/// pairs, the two members rotated, and the pairs laid back out.
fn rope_interleaved_composed(
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_core::Result<Tensor> {
    let (batch, heads, sequence, head_dim) = input.dims4()?;
    let pairs = input.reshape((batch, heads, sequence, head_dim / 2, 2))?;
    let even = pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let rotated_even = (even.broadcast_mul(&cos)? - odd.broadcast_mul(&sin)?)?;
    let rotated_odd = (even.broadcast_mul(&sin)? + odd.broadcast_mul(&cos)?)?;
    Tensor::stack(&[&rotated_even, &rotated_odd], 4)?.reshape((batch, heads, sequence, head_dim))
}

/// The rotary embedding written out of ops that have a backward pass.
///
/// Candle ships no differentiable rope: `candle_nn::rotary_emb::rope` ends in
/// `apply_op3_no_bwd` (candle-nn-0.11.0/src/rotary_emb.rs:580). Rather than
/// assume a convention, this reproduces the `RotaryEmb` CPU kernel in that same
/// file, which I read at lines 348-388. Passing a two-dimensional `cos`/`sin`
/// leaves the kernel's `unbatched_rope` flag false (line 349), and for every
/// batch and head it then walks `i_d` over `0..d/2` with
/// `i1 = i_t * d + i_d` (line 375), `i2 = i1 + d / 2` (line 376) and
/// `i_cs = i_t * (d / 2) + i_d` (line 377), writing:
///
/// ```text
/// dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];   // line 384
/// dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];   // line 385
/// ```
///
/// So this is the "rotate half" form: the last dimension splits into halves
/// `d / 2` apart, not adjacent interleaved pairs. `cos` and `sin` are `d / 2`
/// wide, indexed by position alone, and broadcast across batch and head. The
/// `t == 1` fast path (lines 352-367) is the same arithmetic with `i_t` pinned
/// to zero, so one expression covers both. Each output element is still exactly
/// one multiply, one multiply and one add or subtract, in that order, so in F32
/// the composed result is bit-comparable with the kernel rather than merely
/// close.
fn rope_composed(input: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, _, head_dim) = input.dims4()?;
    let half = head_dim / 2;
    let first = input.narrow(candle_core::D::Minus1, 0, half)?;
    let second = input.narrow(candle_core::D::Minus1, half, half)?;
    // `cos` and `sin` arrive as [sequence, d / 2]; two leading unit axes make
    // them broadcast over batch and head, which is what the kernel's `i_cs`
    // ignoring `bh_i` does by hand.
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let rotated_first = (first.broadcast_mul(&cos)? - second.broadcast_mul(&sin)?)?;
    let rotated_second = (first.broadcast_mul(&sin)? + second.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&rotated_first, &rotated_second], candle_core::D::Minus1)
}
