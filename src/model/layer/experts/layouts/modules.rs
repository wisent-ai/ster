//! Layouts that store one module per expert below the layer, with the
//! router beside them; the Qwen family also reads Transformers 5's stacked
//! save of the same experts.

use candle_nn::{Linear, VarBuilder, linear_no_bias};

use crate::model::{ExpertLayout, MixtureOfExperts};

use super::super::expert::Expert;
use super::sliced::gate_up_rows;
use super::width;

/// Mixtral (`block_sparse_moe`) and LFM2 (`feed_forward`): `experts.{e}`
/// of `w1`, `w3`, `w2`, and the router `gate`.
pub(super) fn mixtral(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let block = if spec.layout == ExpertLayout::Lfm2 {
        builder.pp("feed_forward")
    } else {
        builder.pp("block_sparse_moe")
    };
    let width = width(hidden, spec);
    let experts = (0..spec.count)
        .map(|expert| {
            Expert::load(
                width,
                spec.intermediate,
                Some("w1"),
                ["w3", "w2"],
                block.pp("experts").pp(expert.to_string()),
            )
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok((linear_no_bias(hidden, spec.count, block.pp("gate"))?, experts))
}

/// The block and the router's name below it each Qwen-like layout uses.
fn qwen_block<'a>(builder: &VarBuilder<'a>, layout: ExpertLayout) -> (VarBuilder<'a>, &'static str) {
    match layout {
        ExpertLayout::Jamba => (builder.pp("feed_forward"), "router"),
        ExpertLayout::HunYuan => (builder.pp("mlp"), "gate.wg"),
        ExpertLayout::NemotronH => (builder.pp("mixer"), "gate"),
        ExpertLayout::HyV3 => (builder.pp("mlp"), "router.gate"),
        ExpertLayout::LongCat => (builder.pp("mlp"), "router.classifier"),
        _ => (builder.pp("mlp"), "gate"),
    }
}

/// Qwen, Jamba, HunYuan, Nemotron-H, HY-V3, LongCat-Flash and DeepSeek-V4:
/// `experts.{e}` of a gate (none in Nemotron-H's experts, `w1` in
/// DeepSeek-V4's), an up and a down projection, or Transformers 5's stacked
/// `experts.gate_up_proj` `[experts, 2 · width, hidden]`, gate rows first,
/// and `experts.down_proj` `[experts, hidden, width]` (A.X-K1 and any
/// checkpoint it writes). LongCat-Flash's router also scores its identity
/// experts.
pub(super) fn qwen(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let (count, intermediate) = (spec.count, spec.intermediate);
    let (block, router) = qwen_block(builder, spec.layout);
    let gate = match spec.layout {
        ExpertLayout::NemotronH => None,
        ExpertLayout::DeepseekV4 => Some("w1"),
        _ => Some("gate_proj"),
    };
    let names = if spec.layout == ExpertLayout::DeepseekV4 {
        ["w3", "w2"]
    } else {
        ["up_proj", "down_proj"]
    };
    let width = width(hidden, spec);
    let stacked = block.pp("experts");
    let experts = if gate.is_some() && stacked.contains_tensor("gate_up_proj") {
        let gate_up = stacked.get((count, 2 * intermediate, width), "gate_up_proj")?;
        let down = stacked.get((count, width, intermediate), "down_proj")?;
        gate_up_rows(&gate_up, &down, count, intermediate)?
    } else {
        (0..count)
            .map(|expert| {
                Expert::load(width, intermediate, gate, names, stacked.pp(expert.to_string()))
            })
            .collect::<candle_core::Result<Vec<_>>>()?
    };
    let router = linear_no_bias(hidden, count + spec.identity_experts, block.pp(router))?;
    Ok((router, experts))
}

/// K2-Horizon's value experts: `self_attn.v_router` and one projection per
/// expert, `self_attn.v_experts.{e}`, whose SiLU output is the value.
pub(super) fn mova(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    let block = builder.pp("self_attn");
    let experts = (0..spec.count)
        .map(|expert| -> candle_core::Result<Expert> {
            Ok(Expert {
                gate: None,
                up: linear_no_bias(
                    hidden,
                    spec.intermediate,
                    block.pp("v_experts").pp(expert.to_string()),
                )?,
                down: None,
            })
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok((linear_no_bias(hidden, spec.count, block.pp("v_router"))?, experts))
}
