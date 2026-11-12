//! Layouts that stack every expert in a form a projection does not read
//! (inputs first, or columns interleaved), so each expert's projections are
//! laid out once at load.

use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder, linear_no_bias};

use crate::model::MixtureOfExperts;

use super::super::expert::Expert;

/// GPT-OSS: `mlp.experts.gate_up_proj` `[experts, hidden, 2 ·
/// intermediate]` with gate and up columns interleaved (gate at even
/// columns), `down_proj` `[experts, intermediate, hidden]`, each with a
/// bias, and the router `mlp.router`, which has one too.
pub(super) fn gpt_oss(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let (count, intermediate) = (spec.count, spec.intermediate);
    let block = builder.pp("mlp");
    let stacked = block.pp("experts");
    let gate_up = stacked.get((count, hidden, 2 * intermediate), "gate_up_proj")?;
    let gate_up_bias = stacked.get((count, 2 * intermediate), "gate_up_proj_bias")?;
    let down = stacked.get((count, intermediate, hidden), "down_proj")?;
    let down_bias = stacked.get((count, hidden), "down_proj_bias")?;
    let experts = (0..count)
        .map(|expert| -> candle_core::Result<Expert> {
            let paired = gate_up.get(expert)?.reshape((hidden, intermediate, 2))?;
            let paired_bias = gate_up_bias.get(expert)?.reshape((intermediate, 2))?;
            let column = |slot: usize| -> candle_core::Result<Linear> {
                Ok(Linear::new(
                    paired.narrow(2, slot, 1)?.squeeze(2)?.t()?.contiguous()?,
                    Some(paired_bias.narrow(1, slot, 1)?.squeeze(1)?.contiguous()?),
                ))
            };
            Ok(Expert {
                gate: Some(column(0)?),
                up: column(1)?,
                down: Some(Linear::new(
                    down.get(expert)?.t()?.contiguous()?,
                    Some(down_bias.get(expert)?),
                )),
            })
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok((candle_nn::linear(hidden, count, block.pp("router"))?, experts))
}

/// DBRX: `ffn.experts.mlp.w1` (gate) and `v1` (up) `[experts · width,
/// hidden]` as a projection stores them, and `w2` the same shape, so each
/// expert's down projection is its rows turned; the router
/// `ffn.router.layer`.
pub(super) fn dbrx(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let (count, intermediate) = (spec.count, spec.intermediate);
    let block = builder.pp("ffn");
    let mlp = block.pp("experts").pp("mlp");
    let stacked = |name: &str| mlp.get((count * intermediate, hidden), name);
    let (gate, up, down) = (stacked("w1")?, stacked("v1")?, stacked("w2")?);
    let experts = (0..count)
        .map(|expert| -> candle_core::Result<Expert> {
            let rows = |tensor: &Tensor| tensor.narrow(0, expert * intermediate, intermediate);
            Ok(Expert {
                gate: Some(Linear::new(rows(&gate)?, None)),
                up: Linear::new(rows(&up)?, None),
                down: Some(Linear::new(rows(&down)?.t()?.contiguous()?, None)),
            })
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    let router = linear_no_bias(hidden, count, block.pp("router").pp("layer"))?;
    Ok((router, experts))
}

/// Llama 4: `feed_forward.experts.gate_up_proj` `[experts, hidden, 2 ·
/// width]`, gate columns then up columns, `down_proj` `[experts, width,
/// hidden]`, and the router `feed_forward.router`.
pub(super) fn llama4(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let (count, intermediate) = (spec.count, spec.intermediate);
    let block = builder.pp("feed_forward");
    let stacked = block.pp("experts");
    let gate_up = stacked.get((count, hidden, 2 * intermediate), "gate_up_proj")?;
    let down = stacked.get((count, intermediate, hidden), "down_proj")?;
    let experts = (0..count)
        .map(|expert| -> candle_core::Result<Expert> {
            let columns = gate_up.get(expert)?;
            let half = |start: usize| -> candle_core::Result<Linear> {
                Ok(Linear::new(
                    columns.narrow(1, start, intermediate)?.t()?.contiguous()?,
                    None,
                ))
            };
            Ok(Expert {
                gate: Some(half(0)?),
                up: half(intermediate)?,
                down: Some(Linear::new(down.get(expert)?.t()?.contiguous()?, None)),
            })
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok((linear_no_bias(hidden, count, block.pp("router"))?, experts))
}
