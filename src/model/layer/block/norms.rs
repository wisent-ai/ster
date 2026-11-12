//! The family's attention block: which norms it has, what each is called,
//! and the attention and feed-forward between them.

use super::super::super::{
    attention::relative::RelativeAttention, deepseek4::attention::CompressedAttention,
};
use super::super::mixer::Mixer;
use super::super::norm::{Norm, NormSpec};
use super::load::Loader;
use super::{Block, BlockScales, altup};

/// The norms around an attention block's two halves.
pub(super) struct BlockNorms {
    pub(super) attention: Option<Norm>,
    pub(super) attention_output: Option<Norm>,
    pub(super) feed_forward: Option<Norm>,
    pub(super) feed_forward_output: Option<Norm>,
}

impl Loader<'_> {
    /// Which norms a block has comes from the architecture; what each is
    /// called comes from the family's names (Llama's
    /// `post_attention_layernorm` before the feed-forward, Gemma 2's
    /// `pre_feedforward_layernorm`, GLM-4's `post_self_attn_layernorm` over
    /// attention's output). A parallel block (Cohere, Phi-2, GPT-J, StableLM's
    /// `use_parallel_residual`) feeds one normalised input to both halves,
    /// unless the feed-forward has its own norm (GPT-NeoX); OLMo 2 has no norm
    /// before either sublayer.
    pub(super) fn norms(&self) -> candle_core::Result<BlockNorms> {
        let architecture = self.architecture;
        let names = architecture.names;
        let norm = |name: &str| self.norm(name);
        Ok(
            match (
                architecture.parallel,
                architecture.pre_norms,
                architecture.output_norms,
            ) {
                (true, _, _) => BlockNorms {
                    attention: Some(norm(names.attention_norm)?),
                    attention_output: None,
                    feed_forward: if architecture.parallel_norms {
                        Some(norm(names.feed_forward_norm)?)
                    } else {
                        None
                    },
                    feed_forward_output: None,
                },
                (false, true, false) => BlockNorms {
                    attention: Some(norm(names.attention_norm)?),
                    attention_output: None,
                    feed_forward: Some(norm(names.feed_forward_norm)?),
                    feed_forward_output: None,
                },
                (false, true, true) => BlockNorms {
                    attention: Some(norm(names.attention_norm)?),
                    attention_output: Some(self.output_norm(names.attention_output_norm, 0)?),
                    feed_forward: Some(norm(names.feed_forward_norm)?),
                    feed_forward_output: Some(
                        self.output_norm(names.feed_forward_output_norm, 1)?,
                    ),
                },
                (false, false, _) => BlockNorms {
                    attention: None,
                    attention_output: Some(norm(names.attention_output_norm)?),
                    feed_forward: None,
                    feed_forward_output: Some(norm(names.feed_forward_output_norm)?),
                },
            },
        )
    }

    /// A norm over a sublayer's output, `half` 0 for attention's and 1 for
    /// the feed-forward's. MuseGlimmer's output norms keep their own epsilon
    /// (`post_norm_eps`); PLaMo 3's add their own shift for each half to the
    /// stored scale in place of the block's offset. Without the shifts the
    /// stored scale is read as it is.
    fn output_norm(&self, name: &str, half: usize) -> candle_core::Result<Norm> {
        let architecture = self.architecture;
        let spec = self.spec;
        let output_spec = NormSpec {
            eps: match architecture.output_norm_eps {
                Some(eps) => eps,
                None => spec.eps,
            },
            offset: spec.offset && architecture.output_norm_shifts.is_none(),
            ..spec
        };
        let norm = output_spec.load(self.config.hidden_size, self.builder.pp(name))?;
        match architecture.output_norm_shifts {
            Some((attention, feed_forward)) => {
                norm.shifted(if half == 0 { attention } else { feed_forward })
            }
            None => Ok(norm),
        }
    }

    /// The family's attention (Inkling's relative attention, DeepSeek-V4's
    /// compressed one) and feed-forward between the norms it names.
    pub(super) fn standard(&self) -> candle_core::Result<Block> {
        let (config, architecture, layer) = (self.config, self.architecture, self.layer);
        let names = architecture.names;
        let norms = self.norms()?;
        let builder = &self.builder;
        let mixer = match (&architecture.inkling, &architecture.compressed) {
            (Some(spec), _) => Mixer::Relative(Box::new(RelativeAttention::load(
                builder,
                config.hidden_size,
                spec,
                architecture.norm_eps,
                layer,
            )?)),
            (None, Some(spec)) => Mixer::Compressed(Box::new(CompressedAttention::load(
                builder,
                config.hidden_size,
                spec,
                architecture.norm_eps,
                layer,
            )?)),
            (None, None) => Mixer::Attention(self.attention()?),
        };
        let routed_output_norm = architecture.routed_output_norm
            && architecture.experts.is_some()
            && architecture.routed(layer);
        Ok(Block {
            attention_norm: norms.attention,
            mixer,
            attention_output_norm: norms.attention_output,
            // Gemma 4's paired block norms the residual itself.
            feed_forward_norm: if architecture.side_experts.is_some() {
                None
            } else {
                norms.feed_forward
            },
            feed_forward: Some(self.feed_forward()?),
            feed_forward_output_norm: match norms.feed_forward_output {
                Some(existing) => Some(existing),
                None if routed_output_norm => Some(self.norm(names.feed_forward_output_norm)?),
                None => None,
            },
            residual_multiplier: architecture.residual_multiplier,
            parallel: architecture.parallel,
            scales: architecture.scaled_residuals.map(|scales| BlockScales {
                from_normed: scales.from_normed,
                mixer: scales.full_attention,
                feed_forward: scales.feed_forward,
            }),
            laurel: match architecture.laurel_rank {
                Some(rank) => Some(altup::Laurel::load(
                    builder.pp("laurel"),
                    config.hidden_size,
                    rank,
                    self.spec,
                )?),
                None => None,
            },
        })
    }
}
