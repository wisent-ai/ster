//! The fused query, key and value matrices whose rows a family groups by
//! key-value head: each projection's rows are gathered, a copy made once at
//! load.

use candle_nn::{Linear, VarBuilder};

use super::super::super::{Architecture, layer::projection};
use super::Shape;

/// GPT-NeoX, BLOOM and Falcon lay the rows out by key-value group: each
/// group's query heads, then its key head, then its value head. With as many
/// groups as heads (GPT-NeoX, BLOOM) that is each head's query, key and value
/// in turn. Gathering a projection's rows is a copy, made once at load.
pub(super) fn grouped(
    builder: &VarBuilder<'_>,
    architecture: &Architecture,
    input: usize,
    shape: Shape,
) -> candle_core::Result<(Linear, Linear, Linear)> {
    let Shape {
        heads,
        key_value_heads,
        head_dim,
        query_width,
        key_value_width,
        ..
    } = shape;
    let per_group = heads / key_value_heads;
    let rows = query_width + 2 * key_value_width;
    let fused = builder.pp(architecture.names.fused_qkv);
    let weight = fused.get((rows, input), "weight")?.reshape((
        key_value_heads,
        per_group + 2,
        head_dim,
        input,
    ))?;
    let bias = if architecture.query_key_value_bias {
        Some(fused.get(rows, "bias")?.reshape((key_value_heads, per_group + 2, head_dim))?)
    } else {
        None
    };
    let part = |start: usize, count: usize| -> candle_core::Result<Linear> {
        let width = key_value_heads * count * head_dim;
        Ok(Linear::new(
            weight
                .narrow(1, start, count)?
                .contiguous()?
                .reshape((width, input))?,
            bias.as_ref()
                .map(|bias| bias.narrow(1, start, count)?.contiguous()?.reshape(width))
                .transpose()?,
        ))
    };
    Ok((part(0, per_group)?, part(per_group, 1)?, part(per_group + 1, 1)?))
}

/// TeleChat2 keeps the query apart and stacks each key-value head's key rows,
/// then its value rows, in one `key_value` matrix. Gathering the key and
/// value rows is a copy, made once at load.
pub(super) fn paired_key_value(
    builder: &VarBuilder<'_>,
    architecture: &Architecture,
    input: usize,
    shape: Shape,
) -> candle_core::Result<(Linear, Linear, Linear)> {
    let names = architecture.names;
    let bias = architecture.query_key_value_bias;
    let Shape {
        key_value_heads,
        head_dim,
        query_width,
        key_value_width,
        ..
    } = shape;
    let fused = builder.pp(names.fused_qkv);
    let paired = fused
        .get((2 * key_value_width, input), "weight")?
        .reshape((key_value_heads, 2, head_dim, input))?;
    let paired_bias = if bias {
        Some(fused.get(2 * key_value_width, "bias")?.reshape((key_value_heads, 2, head_dim))?)
    } else {
        None
    };
    let part = |slot: usize| -> candle_core::Result<Linear> {
        Ok(Linear::new(
            paired
                .narrow(1, slot, 1)?
                .contiguous()?
                .reshape((key_value_width, input))?,
            paired_bias
                .as_ref()
                .map(|bias| bias.narrow(1, slot, 1)?.contiguous()?.reshape(key_value_width))
                .transpose()?,
        ))
    };
    let query = projection(input, query_width, bias, architecture.conv1d, builder.pp(names.query))?;
    Ok((query, part(0)?, part(1)?))
}
