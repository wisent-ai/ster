//! Where a layer's query, key and value projections come from: one matrix
//! each, DeepSeek's latent bottlenecks, or row slices of one fused matrix in
//! a family's own layout.

use candle_nn::{Linear, VarBuilder};

use super::super::super::{
    Architecture, QkvLayout,
    layer::{norm::NormSpec, projection},
};
use super::super::{Projections, latent::Latent};
use super::Shape;
use super::fused::{grouped, paired_key_value};

/// This layer's projections, and Qwen3-Next's output gate where its query
/// rows carry one. `value_from_experts` is a K2-Horizon routed layer, whose
/// value experts stand in for `v_proj`.
pub(super) fn projections(
    builder: &VarBuilder<'_>,
    architecture: &Architecture,
    layer: usize,
    input: usize,
    shape: Shape,
    value_from_experts: bool,
) -> candle_core::Result<(Projections, Option<Linear>)> {
    let names = architecture.names;
    let bias = architecture.query_key_value_bias;
    let conv1d = architecture.conv1d;
    let Shape {
        heads,
        query_width,
        key_value_width,
        value_width,
        ..
    } = shape;
    // A Gemma 4 key-value-sharing layer projects its query alone.
    if let Some(source) = architecture.key_value_source(layer) {
        let query = projection(input, query_width, bias, conv1d, builder.pp(names.query))?;
        return Ok((Projections::QueryOnly { query, source }, None));
    }
    // Dots3-Note's sliding-window layers run their own latent attention and
    // attend densely, with no indexer.
    if let Some(own) = architecture.window_latent_at(layer) {
        let latent = Latent::load(
            builder,
            input,
            heads,
            own.latent,
            bias,
            NormSpec::of(architecture),
            own.scales,
            architecture.rotated_key_norm,
            None,
        )?;
        return Ok((Projections::Latent(latent), None));
    }
    if let Some(spec) = architecture.latent {
        // A GLM-5 `shared` layer has no indexer: it reuses the last indexed
        // layer's choice.
        let indexer = architecture
            .sparse_index
            .filter(|index| !index.shared(layer));
        let latent = Latent::load(
            builder,
            input,
            heads,
            spec,
            bias,
            NormSpec::of(architecture),
            architecture.latent_scales,
            architecture.rotated_key_norm,
            indexer,
        )?;
        return Ok((Projections::Latent(latent), None));
    }
    let (query, key, value, output_gate) = match architecture.qkv_layout {
        QkvLayout::Separate if architecture.output_gate => {
            let (query, key, value, gate) = gated_query(builder, architecture, input, shape)?;
            (query, key, Some(value), Some(gate))
        }
        QkvLayout::Separate => {
            let query = projection(
                match architecture.query_bottleneck {
                    Some(bottleneck) => bottleneck,
                    None => input,
                },
                query_width,
                bias,
                conv1d,
                builder.pp(names.query),
            )?;
            let key = projection(input, key_value_width, bias, conv1d, builder.pp(names.key))?;
            // Gemma 4's full-attention layers under `attention_k_eq_v` have
            // no `v_proj`: the value is the key projection. Nor do
            // K2-Horizon's routed layers, whose value experts give it.
            let key_is_value = architecture
                .global_at(layer)
                .is_some_and(|global| global.key_is_value);
            let value = if key_is_value || value_from_experts {
                None
            } else {
                Some(projection(input, value_width, bias, conv1d, builder.pp(names.value))?)
            };
            (query, key, value, None)
        }
        QkvLayout::Stacked | QkvLayout::GateStacked => {
            let (query, key, value) = stacked(builder, architecture, input, shape)?;
            (query, key, Some(value), None)
        }
        QkvLayout::Grouped => {
            let (query, key, value) = grouped(builder, architecture, input, shape)?;
            (query, key, Some(value), None)
        }
        QkvLayout::PairedKeyValue => {
            let (query, key, value) = paired_key_value(builder, architecture, input, shape)?;
            (query, key, Some(value), None)
        }
    };
    Ok((Projections::Standard { query, key, value }, output_gate))
}

/// Qwen3-Next's `q_proj` holds each head's query rows and then its gate
/// rows; both are gathered once at load. Query, key, value and the gate.
fn gated_query(
    builder: &VarBuilder<'_>,
    architecture: &Architecture,
    input: usize,
    shape: Shape,
) -> candle_core::Result<(Linear, Linear, Linear, Linear)> {
    let names = architecture.names;
    let bias = architecture.query_key_value_bias;
    let conv1d = architecture.conv1d;
    let Shape {
        heads,
        head_dim,
        query_width,
        key_value_width,
        ..
    } = shape;
    let rows = builder.pp(names.query);
    let weight = rows
        .get((2 * query_width, input), "weight")?
        .reshape((heads, 2, head_dim, input))?;
    let paired_bias = if bias {
        Some(rows.get(2 * query_width, "bias")?.reshape((heads, 2, head_dim))?)
    } else {
        None
    };
    let part = |slot: usize| -> candle_core::Result<Linear> {
        Ok(Linear::new(
            weight
                .narrow(1, slot, 1)?
                .contiguous()?
                .reshape((query_width, input))?,
            paired_bias
                .as_ref()
                .map(|bias| bias.narrow(1, slot, 1)?.contiguous()?.reshape(query_width))
                .transpose()?,
        ))
    };
    Ok((
        part(0)?,
        projection(input, key_value_width, bias, conv1d, builder.pp(names.key))?,
        projection(input, key_value_width, bias, conv1d, builder.pp(names.value))?,
        part(1)?,
    ))
}

/// Phi-3, GPT-2 and GPT-BigCode store query, key and value as one matrix,
/// rows in that order. Each projection is a row slice of it — a view of the
/// mapped weight, not a copy — so every adapter site stays separate. GPT-2's
/// `Conv1D` stores it transposed, so that one is laid out once at load.
/// HRM-Text's `gqkv_proj` puts its elementwise gate's rows first.
fn stacked(
    builder: &VarBuilder<'_>,
    architecture: &Architecture,
    input: usize,
    shape: Shape,
) -> candle_core::Result<(Linear, Linear, Linear)> {
    let Shape {
        heads,
        query_width,
        value_dim,
        key_value_width,
        value_width,
        ..
    } = shape;
    let gate_width = if architecture.qkv_layout == QkvLayout::GateStacked {
        heads * value_dim
    } else {
        0
    };
    let rows = gate_width + query_width + key_value_width + value_width;
    let fused = builder.pp(architecture.names.fused_qkv);
    let weight = if architecture.conv1d {
        fused.get((input, rows), "weight")?.t()?.contiguous()?
    } else {
        fused.get((rows, input), "weight")?
    };
    let bias = if architecture.query_key_value_bias {
        Some(fused.get(rows, "bias")?)
    } else {
        None
    };
    let slice = |start: usize, width: usize| -> candle_core::Result<Linear> {
        Ok(Linear::new(
            weight.narrow(0, gate_width + start, width)?,
            bias.as_ref()
                .map(|bias| bias.narrow(0, gate_width + start, width))
                .transpose()?,
        ))
    };
    Ok((
        slice(0, query_width)?,
        slice(query_width, key_value_width)?,
        slice(query_width + key_value_width, value_width)?,
    ))
}
