//! The parts of a layer's attention its family adds around the projections:
//! the query and key norms, IQuest-LoopCoder's loop mix, ALiBi's slopes,
//! the elementwise output gate and GPT-OSS's sinks.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::super::super::{
    Architecture, GateFunction, Positions, QkvLayout, QueryKeyNorm,
    layer::{
        norm::{Norm, NormSpec},
        projection,
    },
};
use super::super::{LoopMix, attend::alibi_slopes};
use super::Shape;

/// The query and key norms the family names, `spec` the norm they are; a
/// Gemma 4 key-value-sharing layer has no key of its own to norm.
pub(super) fn query_key_norms(
    architecture: &Architecture,
    layer: usize,
    spec: NormSpec,
    shape: Shape,
    builder: &VarBuilder<'_>,
) -> candle_core::Result<(Option<Norm>, Option<Norm>)> {
    let names = architecture.names;
    let Shape {
        heads,
        key_value_heads,
        head_dim,
        query_width,
        key_value_width,
        ..
    } = shape;
    Ok(match architecture.query_key_norm {
        QueryKeyNorm::None => (None, None),
        QueryKeyNorm::PerHead => (
            Some(spec.load(head_dim, builder.pp(names.query_norm))?),
            match architecture.key_value_source(layer) {
                Some(_) => None,
                None => Some(spec.load(head_dim, builder.pp(names.key_norm))?),
            },
        ),
        QueryKeyNorm::Full => (
            Some(spec.load(query_width, builder.pp(names.query_norm))?),
            Some(spec.load(key_value_width, builder.pp(names.key_norm))?),
        ),
        QueryKeyNorm::HeadWeights => (
            Some(spec.load_head_weights(heads, head_dim, builder.pp(names.query_norm))?),
            Some(spec.load_head_weights(key_value_heads, head_dim, builder.pp(names.key_norm))?),
        ),
        QueryKeyNorm::HeadModules => (
            Some(spec.load_head_modules(heads, head_dim, builder.pp("q_layernorm"))?),
            Some(spec.load_head_modules(key_value_heads, head_dim, builder.pp("k_layernorm"))?),
        ),
        // Llama 4 norms only the rotating layers' queries and keys.
        QueryKeyNorm::Unscaled if architecture.rotates(layer) => (
            Some(spec.unscaled(head_dim, builder)?),
            Some(spec.unscaled(head_dim, builder)?),
        ),
        QueryKeyNorm::Unscaled => (None, None),
        QueryKeyNorm::Weightless => (
            Some(spec.unscaled(head_dim, builder)?),
            Some(spec.unscaled(head_dim, builder)?),
        ),
    })
}

/// IQuest-LoopCoder's gate for `layer`, `gate_projections.{layer}` beside
/// `layers` below the model root, when its loops mix by a gate.
pub(super) fn loop_mix(
    layer_builder: &VarBuilder<'_>,
    architecture: &Architecture,
    layer: usize,
    heads: usize,
    head_dim: usize,
) -> candle_core::Result<Option<LoopMix>> {
    let Some((loops, window)) = architecture
        .loops
        .and_then(|loops| loops.gate_window.map(|window| (loops, window)))
    else {
        return Ok(None);
    };
    let root = match architecture.names.layers.rsplit_once('.') {
        Some((parent, _)) if !parent.is_empty() => layer_builder.root().pp(parent),
        _ => layer_builder.root(),
    };
    let gate = root.pp("gate_projections").pp(layer.to_string());
    Ok(Some(LoopMix {
        physical: loops.physical,
        window,
        weight: gate
            .get((heads, head_dim), "weight")?
            .to_dtype(DType::F32)?
            .reshape((1, heads, 1, head_dim))?,
        bias: gate
            .get(heads, "bias")?
            .to_dtype(DType::F32)?
            .reshape((1, heads, 1, 1))?,
    }))
}

/// ALiBi's slope per head, `[1, heads, 1, 1]`, for a family whose positions
/// are a linear (or Step1's square-root) bias on the scores. Falcon adds the
/// bias before dividing the scores, so its slopes are divided too.
pub(super) fn alibi(
    architecture: &Architecture,
    heads: usize,
    device: &Device,
) -> candle_core::Result<Option<Tensor>> {
    let scale = match architecture.positions {
        Positions::Alibi { inside_scale: true } => 1.0 / architecture.score_divisor,
        Positions::Alibi { inside_scale: false } | Positions::AlibiRoot => 1.0,
        _ => return Ok(None),
    };
    let slopes = Tensor::new(alibi_slopes(heads), device)?.reshape((1, heads, 1, 1))?;
    Ok(Some(if scale == 1.0 { slopes } else { (slopes * scale)? }))
}

/// K2-Horizon's and AFMoE's elementwise gate over attention's output, or
/// HRM-Text's, whose rows lead its fused `gqkv_proj`.
pub(super) fn elementwise_gate(
    builder: &VarBuilder<'_>,
    architecture: &Architecture,
    input: usize,
    shape: Shape,
) -> candle_core::Result<Option<(Linear, GateFunction)>> {
    let Some(function) = architecture.attention_gate else {
        return Ok(None);
    };
    let width = shape.heads * shape.value_dim;
    let gate = if architecture.qkv_layout == QkvLayout::GateStacked {
        let fused = builder.pp(architecture.names.fused_qkv);
        let rows = width + shape.query_width + shape.key_value_width + shape.value_width;
        let weight = fused.get((rows, input), "weight")?.narrow(0, 0, width)?;
        let bias = if architecture.query_key_value_bias {
            Some(fused.get(rows, "bias")?.narrow(0, 0, width)?)
        } else {
            None
        };
        Linear::new(weight, bias)
    } else {
        projection(
            input,
            width,
            false,
            architecture.conv1d,
            builder.pp(architecture.names.attention_gate),
        )?
    };
    Ok(Some((gate, function)))
}

/// GPT-OSS's learned sink per head, `[1, heads, 1, 1]` in F32, on the
/// layers that have one.
pub(super) fn sinks(
    builder: &VarBuilder<'_>,
    architecture: &Architecture,
    layer: usize,
    heads: usize,
) -> candle_core::Result<Option<Tensor>> {
    if !architecture.sinks_at(layer) {
        return Ok(None);
    }
    Ok(Some(
        builder
            .get(heads, architecture.names.sinks)?
            .to_dtype(DType::F32)?
            .reshape((1, heads, 1, 1))?,
    ))
}
