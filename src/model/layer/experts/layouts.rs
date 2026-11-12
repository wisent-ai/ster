//! Where each family stores its experts and their router: one module per
//! expert ([`modules`]), every expert stacked with each expert a view of its
//! slice ([`sliced`]), or stacked and laid out once at load ([`laid`]).

mod laid;
mod modules;
mod sliced;

use candle_nn::{Linear, VarBuilder};

use crate::model::{ExpertLayout, MixtureOfExperts};

use super::expert::Expert;

/// The router and every stored expert of `spec`'s layout, below the
/// layer's `builder`.
pub(super) fn stored(
    builder: &VarBuilder<'_>,
    hidden: usize,
    spec: &MixtureOfExperts,
) -> candle_core::Result<(Linear, Vec<Expert>)> {
    match spec.layout {
        ExpertLayout::Mixtral | ExpertLayout::Lfm2 => modules::mixtral(builder, hidden, spec),
        ExpertLayout::Qwen
        | ExpertLayout::Jamba
        | ExpertLayout::HunYuan
        | ExpertLayout::NemotronH
        | ExpertLayout::HyV3
        | ExpertLayout::LongCat
        | ExpertLayout::DeepseekV4 => modules::qwen(builder, hidden, spec),
        ExpertLayout::Mova => modules::mova(builder, hidden, spec),
        ExpertLayout::Step3 => sliced::step3(builder, hidden, spec),
        ExpertLayout::Granite => sliced::granite(builder, hidden, spec),
        ExpertLayout::Gemma4 => sliced::gemma4(builder, hidden, spec),
        ExpertLayout::GptOss => laid::gpt_oss(builder, hidden, spec),
        ExpertLayout::Dbrx => laid::dbrx(builder, hidden, spec),
        ExpertLayout::Llama4 => laid::llama4(builder, hidden, spec),
    }
}

/// The width `spec`'s experts work on: the layer's `hidden`, or the latent
/// width Kimi-K3 (`routed_expert_hidden_size`) and Nemotron-H
/// (`moe_latent_size`) project into.
fn width(hidden: usize, spec: &MixtureOfExperts) -> usize {
    spec.latent.as_ref().map_or(hidden, |latent| latent.width)
}
