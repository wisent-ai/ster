//! Building one decoder layer and running it over the residual stream in
//! each form a family keeps it: one hidden state, Gemma 3n's AltUp streams,
//! DeepSeek-V4's hyper-connected streams, or Kimi-K3's attention residuals.

use candle_core::Tensor;
use candle_nn::{Module, VarBuilder, linear_no_bias};
use candle_transformers::models::llama::Config;

use crate::failures::{model::decoder::LAYER_FORM_UNBUILT, refuse_candle};
use crate::lora::Adapters;

use super::super::{
    Architecture, Cache, Mode,
    deepseek4::HyperConnection,
    depth::{DepthMix, DepthMixes},
};
use super::block::Block;
use super::norm::NormSpec;
use super::shared::SharedBlock;
use super::{DecoderLayer, LayerInputs, PerLayerInput, altup};

/// The refusal of `layer` asked to run a residual form, `form`, it was not
/// built for.
fn unbuilt(layer: usize, form: &str) -> candle_core::Error {
    refuse_candle(
        &LAYER_FORM_UNBUILT,
        &[("layer", &layer.to_string()), ("form", form)],
    )
}

impl DecoderLayer {
    /// `shared` is, on a Zamba2 hybrid layer, the shared block it uses and
    /// that block's builder (`model.layers.{owner}.shared_transformer`).
    pub(in crate::model) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
        shared: Option<(&SharedBlock, &VarBuilder<'_>)>,
    ) -> candle_core::Result<Self> {
        let per_layer_input = match architecture.per_layer_input {
            Some(spec) => Some(PerLayerInput {
                gate: linear_no_bias(
                    config.hidden_size,
                    spec.width,
                    builder.pp("per_layer_input_gate"),
                )?,
                projection: linear_no_bias(
                    spec.width,
                    config.hidden_size,
                    builder.pp("per_layer_projection"),
                )?,
                norm: NormSpec::of(architecture)
                    .load(config.hidden_size, builder.pp("post_per_layer_input_norm"))?,
                activation: architecture.activation,
            }),
            None => None,
        };
        let scalar = if architecture.layer_scalar {
            Some(builder.get(1, "layer_scalar")?)
        } else {
            None
        };
        let altup = match architecture.altup_streams {
            Some(streams) => Some(altup::AltUp::load(
                builder.pp("altup"),
                config.hidden_size,
                streams,
                NormSpec::of(architecture),
            )?),
            None => None,
        };
        let depth = match architecture.depth_block_size {
            Some(block_size) => {
                let mix = |norm: &str, projection: &str| {
                    DepthMix::load(
                        &builder,
                        config.hidden_size,
                        norm,
                        projection,
                        architecture.norm_eps,
                    )
                };
                Some(DepthMixes {
                    attention: mix("self_attention_res_norm", "self_attention_res_proj")?,
                    feed_forward: mix("mlp_res_norm", "mlp_res_proj")?,
                    block_size,
                })
            }
            None => None,
        };
        let hyper = match architecture.hyper_connections {
            Some(spec) => {
                let connection = |kind: &str| {
                    HyperConnection::load(
                        &builder,
                        kind,
                        config.hidden_size,
                        spec.streams,
                        spec.form,
                        spec.eps,
                        architecture.norm_eps,
                    )
                };
                Some((connection("attn")?, connection("ffn")?))
            }
            None => None,
        };
        Ok(Self {
            block: Block::load(builder, config, architecture, layer, adapters, shared)?,
            per_layer_input,
            scalar,
            altup,
            depth,
            hyper,
        })
    }

    /// Kimi-K3's layer under attention residuals: `partial` is the running
    /// sum of the current block of layers, which this layer returns with its
    /// two sublayers' outputs added, and `blocks` the sums of the finished
    /// blocks before it, which this layer extends when it opens a block.
    /// Each sublayer reads the mix of `blocks` and the running sum.
    pub(in crate::model) fn forward_depth(
        &self,
        partial: &Tensor,
        blocks: &mut Vec<Tensor>,
        inputs: &LayerInputs<'_>,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let Some(depth) = &self.depth else {
            return Err(unbuilt(layer, "Kimi-K3's attention residuals"));
        };
        let input = if blocks.is_empty() {
            partial.clone()
        } else {
            depth.attention.mix(blocks, partial)?
        };
        let mut running = Some(partial.clone());
        if layer % depth.block_size == 0 {
            blocks.extend(running.take());
        }
        let attended = self.block.mixer_output(
            &input,
            inputs.embedded,
            index_pos,
            layer,
            cache,
            mask,
            mode,
        )?;
        let running = match running {
            Some(running) => (running + attended)?,
            None => attended,
        };
        let input = depth.feed_forward.mix(blocks, &running)?;
        running + self.block.feed_forward_output(&input, layer, cache, mode)?
    }

    /// DeepSeek-V4's layer over its residual streams `[batch, sequence, N,
    /// hidden]`: each sublayer reads the streams collapsed by its
    /// hyper-connection, and its output is spread back over them beside the
    /// streams' own mix.
    pub(in crate::model) fn forward_hyper(
        &self,
        streams: &Tensor,
        inputs: &LayerInputs<'_>,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let Some((attention, feed_forward)) = &self.hyper else {
            return Err(unbuilt(layer, "DeepSeek-V4's hyper-connected streams"));
        };
        let mixing = attention.mixing(streams)?;
        let attended = self.block.mixer_output(
            &mixing.collapsed,
            inputs.embedded,
            index_pos,
            layer,
            cache,
            mask,
            mode,
        )?;
        let streams = mixing.spread(&attended, streams)?;
        let mixing = feed_forward.mixing(&streams)?;
        let output = self
            .block
            .feed_forward_output(&mixing.collapsed, layer, cache, mode)?;
        mixing.spread(&output, &streams)
    }

    /// The window this layer's attention looks through, if any.
    pub(in crate::model) fn window(&self) -> Option<usize> {
        self.block.window()
    }

    pub(in crate::model) fn forward(
        &self,
        hidden: &Tensor,
        inputs: &LayerInputs<'_>,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let output =
            self.block
                .forward(hidden, inputs.embedded, index_pos, layer, cache, mask, mode)?;
        let output = match (&self.per_layer_input, &inputs.per_layer) {
            (Some(per_layer), Some(input)) => {
                let gated = (per_layer
                    .activation
                    .apply(&per_layer.gate.forward(&output)?)?
                    * input)?;
                let added = per_layer
                    .norm
                    .forward(&per_layer.projection.forward(&gated)?, mode.pass)?;
                (output + added)?
            }
            (Some(_), None) => return Err(unbuilt(layer, "Gemma 4's per-layer input")),
            (None, _) => output,
        };
        match &self.scalar {
            Some(scalar) => output.broadcast_mul(scalar),
            None => Ok(output),
        }
    }

    /// Gemma 3n's layer over its AltUp streams, the first being the hidden
    /// state the decoder steers and reads: every stream predicted from all
    /// of them, the block run on the first prediction, every stream
    /// corrected by what the block did, and the per-layer input — gated by
    /// the first corrected stream times `correct_output_scale` — added to
    /// every stream but the first.
    pub(in crate::model) fn forward_streams(
        &self,
        streams: &[Tensor],
        inputs: &LayerInputs<'_>,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Vec<Tensor>> {
        let Some(altup) = &self.altup else {
            return Err(unbuilt(layer, "Gemma 3n's AltUp streams"));
        };
        let predictions = altup.predict(streams, mode.pass)?;
        let activated = self.block.forward(
            &predictions[0],
            inputs.embedded,
            index_pos,
            layer,
            cache,
            mask,
            mode,
        )?;
        let mut corrected = altup.correct(&predictions, &activated, mode.pass)?;
        if let (Some(per_layer), Some(input)) = (&self.per_layer_input, &inputs.per_layer) {
            let first = altup.scaled(&corrected[0])?;
            let gated = (per_layer
                .activation
                .apply(&per_layer.gate.forward(&first)?)?
                * input)?;
            let added = per_layer
                .norm
                .forward(&per_layer.projection.forward(&gated)?, mode.pass)?;
            for stream in corrected.iter_mut().skip(1) {
                *stream = (&*stream + &added)?;
            }
        }
        Ok(corrected)
    }
}

