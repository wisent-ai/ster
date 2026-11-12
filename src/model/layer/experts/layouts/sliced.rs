//! Layouts that stack every expert in a few tensors as a projection stores
//! them, so each expert is a view of its slice, not a copy.

use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder, linear_no_bias};

use crate::model::MixtureOfExperts;

use super::super::expert::Expert;

/// Every expert of a `gate_up` `[experts, 2 · intermediate, width]` holding
/// each expert's gate rows above its up rows, and a `down` `[experts, width,
/// intermediate]`: views of each expert's slice.
pub(super) fn gate_up_rows(
    gate_up: &Tensor,
    down: &Tensor,
    count: usize,
    intermediate: usize,
) -> candle_core::Result<Vec<Expert>> {
    (0..count)
        .map(|expert| -> candle_core::Result<Expert> {
            let rows = gate_up.get(expert)?;
            Ok(Expert {
                gate: Some(Linear::new(rows.narrow(0, 0, intermediate)?, None)),
                up: Linear::new(rows.narrow(0, intermediate, intermediate)?, None),
                down: Some(Linear::new(down.get(expert)?, None)),
            })
        })
        .collect()
}

/// Step3: `moe.gate_proj.weight`, `moe.up_proj.weight` `[experts,
/// intermediate, hidden]` and `moe.down_proj.weight` `[experts, hidden,
/// intermediate]`, and the router `moe.gate`.
pub(super) fn step3(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let (count, intermediate) = (spec.count, spec.intermediate);
    let block = builder.pp("moe");
    let gate = block.get((count, intermediate, hidden), "gate_proj.weight")?;
    let up = block.get((count, intermediate, hidden), "up_proj.weight")?;
    let down = block.get((count, hidden, intermediate), "down_proj.weight")?;
    let experts = (0..count)
        .map(|expert| -> candle_core::Result<Expert> {
            Ok(Expert {
                gate: Some(Linear::new(gate.get(expert)?, None)),
                up: Linear::new(up.get(expert)?, None),
                down: Some(Linear::new(down.get(expert)?, None)),
            })
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok((linear_no_bias(hidden, count, block.pp("gate"))?, experts))
}

/// GraniteMoE: `block_sparse_moe.input_linear.weight` `[experts, 2 ·
/// intermediate, hidden]`, gate rows then up rows, `output_linear.weight`
/// `[experts, hidden, intermediate]`, and the router `router.layer`.
pub(super) fn granite(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let (count, intermediate) = (spec.count, spec.intermediate);
    let block = builder.pp("block_sparse_moe");
    let input = block.get((count, 2 * intermediate, hidden), "input_linear.weight")?;
    let output = block.get((count, hidden, intermediate), "output_linear.weight")?;
    let experts = gate_up_rows(&input, &output, count, intermediate)?;
    let router = linear_no_bias(hidden, count, block.pp("router").pp("layer"))?;
    Ok((router, experts))
}

/// Gemma 4: `experts.gate_up_proj` `[experts, 2 · width, hidden]`, gate rows
/// above up rows, and `experts.down_proj` `[experts, hidden, width]`. Its
/// router multiplies the normed input by `router.scale` and by
/// `hidden^-0.5` before `router.proj`; both fold into the projection's
/// columns once at load.
pub(super) fn gemma4(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let (count, intermediate) = (spec.count, spec.intermediate);
    let block = builder.pp("experts");
    let gate_up = block.get((count, 2 * intermediate, hidden), "gate_up_proj")?;
    let down = block.get((count, hidden, intermediate), "down_proj")?;
    let experts = gate_up_rows(&gate_up, &down, count, intermediate)?;
    let router = builder.pp("router");
    let scale = (router.get(hidden, "scale")? * (hidden as f64).powf(-0.5))?;
    let projection = router.get((count, hidden), "proj.weight")?;
    let weight = projection.broadcast_mul(&scale.reshape((1, hidden))?)?;
    Ok((Linear::new(weight, None), experts))
}
