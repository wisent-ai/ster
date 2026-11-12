//! Building the whole decoder from a checkpoint's weights: where the family
//! keeps them, the embeddings, the stack of layers, the final norm and the
//! head.

use anyhow::Result;
use candle_nn::{Linear, VarBuilder, embedding};
use candle_transformers::models::llama::Config;

use crate::lora::Adapters;

use super::super::{
    Architecture, Names, Positions,
    deepseek4::HyperHead,
    depth::DepthMix,
    layer::{altup::StreamProjections, norm::NormSpec, projection},
};
use super::SteeringLlama;
use super::embeddings::{NgramEmbeddings, PerLayerEmbeddings};
use super::stack::stack;

impl SteeringLlama {
    pub fn load(
        builder: VarBuilder<'_>,
        config: Config,
        architecture: Architecture,
    ) -> Result<Self> {
        Ok(Self::load_with_adapters(
            builder,
            config,
            architecture,
            Adapters::default(),
        )?)
    }

    /// Loads the frozen base and attaches `adapters`.
    ///
    /// The base weights come from `builder`, which Ster maps read-only out of
    /// safetensors; only the adapter factors were ever registered in a `VarMap`.
    /// Attaching them here therefore cannot make a base weight trainable, which
    /// is the property the whole training path rests on.
    pub fn load_with_adapters(
        builder: VarBuilder<'_>,
        config: Config,
        mut architecture: Architecture,
        adapters: Adapters,
    ) -> candle_core::Result<Self> {
        // A multimodal checkpoint keeps the language model below a wrapper
        // (Gemma 3's `language_model`); the vision tower beside it is never
        // mapped. The head may sit outside it, at the checkpoint's root
        // (Qwen3.5's and MuseGlimmer's `lm_head`).
        let outer = builder.clone();
        let builder = if architecture.names.wrapper.is_empty() {
            builder
        } else {
            builder.pp(architecture.names.wrapper)
        };
        locate_names(&builder, &mut architecture);
        let names = architecture.names;
        let root = if names.root.is_empty() {
            builder.clone()
        } else {
            builder.pp(names.root)
        };
        let hidden = config.hidden_size;
        let embeddings = embedding(config.vocab_size, hidden, builder.pp(names.embeddings))?;
        let positions = match architecture.positions {
            Positions::Learned { offset } => Some((
                embedding(
                    config.max_position_embeddings + offset,
                    hidden,
                    builder.pp(names.positions),
                )?,
                offset,
            )),
            _ => None,
        };
        let spec = NormSpec::of(&architecture);
        // BLOOM's embedding norm is stored; MuseGlimmer's is weightless and
        // names no tensor.
        let embedding_norm = match (architecture.embedding_norm, names.embedding_norm) {
            (false, _) => None,
            (true, "") => Some(spec.unscaled(hidden, &builder)?),
            (true, name) => Some(spec.load(hidden, builder.pp(name))?),
        };
        let lm_head = if config.tie_word_embeddings {
            let bias = if architecture.lm_head_bias {
                Some(builder.pp(names.lm_head).get(config.vocab_size, "bias")?)
            } else {
                None
            };
            Linear::new(embeddings.embeddings().clone(), bias)
        } else {
            untied_head(&builder, &outer, &config, &architecture)?
        };
        let final_spec = NormSpec {
            offset: spec.offset && !architecture.plain_final_norm,
            ..spec
        };
        let final_norm = final_spec.load(hidden, builder.pp(names.final_norm))?;
        let per_layer = match architecture.per_layer_input {
            Some(per_layer) => {
                let packed = config.num_hidden_layers * per_layer.width;
                Some(PerLayerEmbeddings {
                    embeddings: embedding(
                        per_layer.vocab,
                        packed,
                        root.pp("embed_tokens_per_layer"),
                    )?,
                    projection: candle_nn::linear_no_bias(
                        hidden,
                        packed,
                        root.pp("per_layer_model_projection"),
                    )?,
                    norm: spec.load(per_layer.width, root.pp("per_layer_projection_norm"))?,
                    width: per_layer.width,
                })
            }
            None => None,
        };
        let ngram = match architecture.ngram {
            Some(ngram) => Some(NgramEmbeddings::load(
                ngram,
                config.vocab_size,
                hidden,
                root.pp("ngram_embeddings"),
            )?),
            None => None,
        };
        let streams = match architecture.altup_streams {
            Some(count) => Some(StreamProjections::load(&root, hidden, count)?),
            None => None,
        };
        let layers = stack(&builder, &config, &architecture, &adapters)?;
        let low_start = match architecture.recurrence {
            Some(_) => Some(builder.get(hidden, "model.z_L_init")?),
            None => None,
        };
        // Kimi-K3's last mix, of every block sum and the last running sum,
        // below the model's root.
        let depth_output = match architecture.depth_block_size {
            Some(_) => Some(DepthMix::load(
                &root,
                hidden,
                "output_attn_res_norm",
                "output_attn_res_proj",
                architecture.norm_eps,
            )?),
            None => None,
        };
        // DeepSeek-V4's collapse of its last streams, below the model's root;
        // GLM-5-Next averages them and stores nothing.
        let hyper_head = match architecture.hyper_connections {
            Some(spec) if spec.learned_head => Some(HyperHead::load(
                &root,
                hidden,
                spec.streams,
                spec.eps,
                architecture.norm_eps,
            )?),
            _ => None,
        };
        Ok(Self {
            embeddings,
            positions,
            embedding_norm,
            layers,
            per_layer,
            ngram,
            streams,
            low_start,
            depth_output,
            hyper_head,
            final_norm,
            lm_head,
            config,
            architecture,
            adapters,
        })
    }
}

/// The names this checkpoint keeps its tensors under. A checkpoint saved
/// from the base model drops the family's root (`wte` rather than
/// `transformer.wte`), which shows in whether the embedding is where the root
/// says. Transformers saves Kimi-K2.5's decoder layers as `model.blocks`
/// where Moonshot's release keeps `model.layers`, which shows in where the
/// first layer's norm is.
fn locate_names(builder: &VarBuilder<'_>, architecture: &mut Architecture) {
    if !builder.contains_tensor(&format!("{}.weight", architecture.names.embeddings)) {
        architecture.names = architecture.names.without_root();
    }
    let first_norm =
        |layers: &str| format!("{layers}.0.{}.weight", architecture.names.attention_norm);
    if architecture.names.layers == Names::LLAMA.layers
        && !builder.contains_tensor(&first_norm(architecture.names.layers))
        && builder.contains_tensor(&first_norm(Names::BLOCKS_LAYERS))
    {
        architecture.names.layers = Names::BLOCKS_LAYERS;
    }
}

/// The head of a checkpoint that does not tie it to the embedding, inside
/// the wrapper or at the checkpoint's root `outer`; DeepSeek-V4's own
/// checkpoints keep it as `head`.
fn untied_head(
    builder: &VarBuilder<'_>,
    outer: &VarBuilder<'_>,
    config: &Config,
    architecture: &Architecture,
) -> candle_core::Result<Linear> {
    let names = architecture.names;
    let stored =
        |builder: &VarBuilder<'_>, name: &str| builder.contains_tensor(&format!("{name}.weight"));
    let name = if !stored(builder, names.lm_head)
        && !stored(outer, names.lm_head)
        && stored(builder, "head")
    {
        "head"
    } else {
        names.lm_head
    };
    let head = if stored(builder, name) { builder } else { outer };
    projection(
        config.hidden_size,
        config.vocab_size,
        architecture.lm_head_bias,
        false,
        head.pp(name),
    )
}
