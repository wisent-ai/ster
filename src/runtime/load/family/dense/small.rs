//! Llama-shaped families that add a key or two to Llama's block.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Activation, Architecture, FeedForwardKind, Names, NormKind, Positions, QueryKeyNorm,
    SkipConnections,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::deepseek::deepseek_experts;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, number, numbers, text, whole};

/// What the families this file reads add to the Llama block; [`super::super::family`]
/// sends only their `model_type` values here.
pub(in crate::runtime::load::family) fn configure(
    at: &At,
    architecture: &mut Architecture,
) -> Result<()> {
    let At {
        model_type,
        raw,
        llama,
        layers,
        path,
        ..
    } = *at;
    match model_type {
        "exaone_moe" => {
            // K-EXAONE: EXAONE 4's attention (per-head query and key norms,
            // and no rotation on its full-attention layers beside
            // sliding-window ones) in a pre-norm block, with DeepSeek-V3's
            // sigmoid router and `num_shared_experts` shared experts on the
            // layers `mlp_layer_types` calls `sparse`.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.sliding_window = whole(raw, "sliding_window");
            architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
        }
        "internlm3" => {
            architecture.query_key_value_bias = flag(raw, "qkv_bias");
            architecture.output_bias = flag(raw, "bias");
        }
        "TeleFLM" => {
            // Tele-FLM: Llama's block; under `use_mup` the embeddings are
            // multiplied by `input_mult` and the logits by `output_mult /
            // mup_scale_factor`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            if flag(raw, "use_mup") {
                let (Some(input), Some(output), Some(scale)) = (
                    number(raw, "input_mult"),
                    number(raw, "output_mult"),
                    number(raw, "mup_scale_factor").filter(|scale| *scale != 0.0),
                ) else {
                    bail!(
                        "{} declares use_mup without input_mult, output_mult and a nonzero mup_scale_factor",
                        path.display()
                    );
                };
                architecture.embedding_multiplier = Some(input);
                architecture.logits_multiplier = Some(output / scale);
            }
        }
        "seed_oss" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = flag(raw, "attention_out_bias");
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
        }
        "solar" => {
            // Solar Pro: Llama's block with block skip connections — before
            // the layers `bskcn_1` and `bskcn_2` list the hidden state is
            // kept, and before those `bskcn_3` and `bskcn_4` list it is
            // blended with the first and second kept one by the inference
            // weight, the second of `bskcn_tv`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            fits(layers, path)?;
            let listed = |key: &str| -> Result<u128> {
                let Some(list) = raw.get(key).and_then(Value::as_array) else {
                    bail!("{} declares a Solar model without {key}", path.display());
                };
                list.iter().try_fold(0u128, |set, entry| {
                    match entry
                        .as_u64()
                        .map(|layer| layer as usize)
                        .filter(|layer| *layer < layers)
                    {
                        Some(layer) => Ok(set | (1u128 << layer)),
                        None => bail!(
                            "{} lists {entry} in {key}, which is not one of its {layers} layers",
                            path.display()
                        ),
                    }
                })
            };
            let [_, inference] = numbers::<2>(raw, "bskcn_tv", path)?;
            architecture.skip_connections = Some(SkipConnections {
                save: [listed("bskcn_1")?, listed("bskcn_2")?],
                blend: [listed("bskcn_3")?, listed("bskcn_4")?],
                weight: inference,
            });
        }
        "PanguEmbedded" => {
            // openPangu-Embedded: Llama's block with `bias` on every
            // attention projection (`qkv_bias`, when stated, for query, key
            // and value), `mlp_bias`, and under `sandwich_norm` norms over
            // each sublayer's output.
            let bias = flag(raw, "attention_bias") || flag(raw, "bias");
            architecture.output_bias = bias;
            architecture.query_key_value_bias =
                raw.get("qkv_bias").and_then(Value::as_bool).unwrap_or(bias);
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            if flag(raw, "sandwich_norm") {
                architecture.output_norms = true;
                architecture.names = Names::PANGU_SANDWICH;
            }
        }
        "telechat3" => {
            // TeleChat3: Llama's block with `attention_bias` on every
            // attention projection and `mlp_bias`; its `telechat3-yarn`
            // rotation is read with the other scalings.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
        }
        "step1" => {
            // Step1 (Step-Audio's language model): Llama's block with
            // `num_attention_groups` key-value heads and no rotation;
            // positions enter as `-slope · sqrt(distance)` on the scores,
            // with ALiBi's slopes, as its `build_alibi_cache` builds them.
            architecture.positions = Positions::AlibiRoot;
        }
        "iquestcoder" => {
            // IQuest-Coder: Llama with `mlp_bias`, OLMo's `clip_qkv`, and
            // Qwen2's sliding window on the layers from `max_window_layers`
            // under `use_sliding_window`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.clip_qkv = number(raw, "clip_qkv");
            if flag(raw, "use_sliding_window") {
                architecture.sliding_window = whole(raw, "sliding_window");
                let from = whole(raw, "max_window_layers").unwrap_or(0);
                architecture.sliding_layers =
                    every_layer(layers, path)? & !every_layer(from.min(layers), path)?;
            }
        }
        "arcee" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::UP_DOWN;
        }
        "jais2" => {
            // Jais 2: Arcee's plain `up_proj`/`down_proj` feed-forward under
            // LayerNorms with bias; `attention_bias` and `mlp_bias` default
            // to true and `hidden_act` to squared ReLU, as Transformers'
            // `Jais2Config` defines them.
            let stated = |key: &str| raw.get(key).and_then(Value::as_bool).unwrap_or(true);
            architecture.query_key_value_bias = stated("attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = stated("mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.names = Names::UP_DOWN;
            if text(raw, "hidden_act").is_none() {
                architecture.activation = Activation::Relu2;
            }
        }
        "minicpm" => {
            // MiniCPM's muP scales: the embedding by `scale_emb`, each
            // sublayer by `scale_depth / sqrt(layers)`, and the logits by
            // `dim_model_base / hidden_size`.
            architecture.embedding_multiplier = number(raw, "scale_emb");
            architecture.residual_multiplier =
                number(raw, "scale_depth").map(|depth| depth / (layers as f64).sqrt());
            architecture.logits_multiplier =
                number(raw, "dim_model_base").map(|base| base / llama.hidden_size as f64);
        }
        "orion" => {
            architecture.norm = NormKind::Layer { bias: true };
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
