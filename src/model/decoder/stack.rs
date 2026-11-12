//! The decoder's stack of layers, built once per stored layer.

use candle_nn::VarBuilder;
use candle_transformers::models::llama::Config;

use crate::lora::Adapters;

use super::super::{
    Architecture,
    layer::{DecoderLayer, shared::SharedBlock},
};

/// The stack of `num_hidden_layers` layers. A looped model (Nanbeige's
/// `num_loops`) runs its stored layers more than once, each later pass
/// reusing the first pass's tensors; LongCat-Flash's stored layers are two
/// Ster layers each, read under their halves' names; HRM-Text's two stacks
/// are stored apart, each below its own path. Zamba2's shared blocks are
/// mapped once, from the first hybrid layers that use them, and every later
/// use shares their tensors.
pub(super) fn stack(
    builder: &VarBuilder<'_>,
    config: &Config,
    architecture: &Architecture,
    adapters: &Adapters,
) -> candle_core::Result<Vec<DecoderLayer>> {
    let names = architecture.names;
    let shared = match architecture.shared_blocks {
        Some(blocks) => blocks
            .layers()
            .take(blocks.blocks)
            .map(|owner| {
                let block_builder =
                    builder.pp(format!("{}.{owner}.shared_transformer", names.layers));
                let block =
                    SharedBlock::load(&block_builder, config, architecture, owner, blocks)?;
                Ok((block, block_builder))
            })
            .collect::<candle_core::Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    let stored = match (architecture.recurrence, architecture.loops) {
        (Some(recurrence), _) => 2 * recurrence.per_stack,
        (None, Some(loops)) => loops.physical,
        (None, None) => config.num_hidden_layers,
    };
    let physical = (0..stored)
        .map(|index| {
            let block = architecture
                .shared_blocks
                .filter(|blocks| {
                    index < u128::BITS as usize && blocks.hybrid_layers & (1u128 << index) != 0
                })
                .and_then(|blocks| shared.get(blocks.slot(index) % blocks.blocks))
                .map(|(block, block_builder)| (block, block_builder));
            let (source, layer_names) = architecture.stored_layer(index);
            let half;
            let layer_architecture = if layer_names == architecture.names {
                architecture
            } else {
                half = Architecture {
                    names: layer_names,
                    ..architecture.clone()
                };
                &half
            };
            let parent = if architecture.recurrence.is_some() {
                layer_names.layers
            } else {
                names.layers
            };
            DecoderLayer::load(
                builder.pp(format!("{parent}.{source}")),
                config,
                layer_architecture,
                index,
                adapters,
                block,
            )
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok((0..config.num_hidden_layers)
        .map(|index| match architecture.recurrence {
            Some(recurrence) => physical[recurrence.stored(index)].clone(),
            None => physical[index % stored].clone(),
        })
        .collect())
}
