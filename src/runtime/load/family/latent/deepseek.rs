//! DeepSeek-V2 and V3 and the families that run their latent attention and router.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{Architecture, Names, QueryKeyNorm};
use crate::runtime::load::family::At;
use crate::runtime::load::family::custom::llama4::llama4_temperature;
use crate::runtime::load::family::experts::deepseek::deepseek_experts;
use crate::runtime::load::family::experts::index::sparse_index;
use crate::runtime::load::family::experts::step::step_experts;
use crate::runtime::load::family::latent::dots3::dots3_note;
use crate::runtime::load::family::read::{flag, number, whole};

/// What the families this file reads add to the Llama block; [`super::super::family`]
/// sends only their `model_type` values here.
pub(in crate::runtime::load::family) fn configure(
    at: &At,
    architecture: &mut Architecture,
) -> Result<()> {
    let At {
        model_type,
        raw,
        scaling,
        llama,
        layers,
        path,
    } = *at;
    match model_type {
        "glm4_moe" => {
            // GLM-4.5's MoE: biased query, key and value, optional per-head
            // query and key norms, rotation over `partial_rotary_factor` by
            // halves, and DeepSeek-V3's router and shared experts.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
            }
            architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
        }
        // GLM-4.7-Flash (`glm4_moe_lite`) is DeepSeek-V3's latent attention
        // and router under GLM's name.
        // DeepSeek-V3.2 and GLM-5 (`glm_moe_dsa`) add DeepSeek Sparse
        // Attention's indexer.
        // openPangu-Ultra-MoE is DeepSeek-V3's latent attention and a sigmoid
        // router without selection bias, under its own key names (read as
        // DeepSeek's), with sandwich norms.
        "deepseek_v2" | "deepseek_v3" | "minicpm3" | "glm4_moe_lite" | "deepseek_v32"
        | "glm_moe_dsa" | "axk1" | "pangu_ultra_moe" | "dots3_note" => {
            if architecture.latent.is_none() {
                bail!(
                    "{} declares no kv_lora_rank; Ster implements {model_type} with its latent attention",
                    path.display()
                );
            }
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            if model_type == "minicpm3" {
                // MiniCPM's muP scales; its rotation is by halves.
                architecture.embedding_multiplier = number(raw, "scale_emb");
                architecture.residual_multiplier =
                    number(raw, "scale_depth").map(|depth| depth / (layers as f64).sqrt());
                architecture.logits_multiplier =
                    number(raw, "dim_model_base").map(|base| base / llama.hidden_size as f64);
            } else {
                // DeepSeek rotates adjacent pairs (`rope_interleave`, on by
                // default).
                architecture.interleaved_rotary = raw
                    .get("rope_interleave")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
            }
            if raw
                .get("n_routed_experts")
                .is_some_and(|count| !count.is_null())
            {
                architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
            }
            // Mistral Large 3 (read from Mistral's own format) adds Llama
            // 4's query temperature (`llama_4_scaling`).
            architecture.query_temperature = llama4_temperature(raw, scaling, layers, path)?;
            if matches!(model_type, "deepseek_v32" | "glm_moe_dsa" | "dots3_note") {
                architecture.sparse_index = Some(sparse_index(raw, model_type, layers, path)?);
            }
            if model_type == "dots3_note" {
                dots3_note(raw, llama, architecture, path)?;
            }
            // A.X-K1 normalises each routed layer's feed-forward output by
            // `post_mlp_layernorm` before the residual add.
            if model_type == "axk1" {
                architecture.names = Names::AXK1;
                architecture.routed_output_norm = true;
            }
            // openPangu-Ultra-MoE's `sandwich_norm`: norms over each
            // sublayer's output, `pre_mlp_layernorm` before the feed-forward.
            if flag(raw, "sandwich_norm") {
                architecture.output_norms = true;
                architecture.names = Names::PANGU_SANDWICH;
            }
        }
        "step3_text" => {
            // Step3 (its text decoder, `text_config` of `step3_vl`): the
            // query narrows through `q_proj` to `share_q_dim`, passes the RMS
            // norm `inter_norm` and widens through `wq`, beside
            // `num_attention_groups` key-value heads, and Step's experts.
            let Some(width) = whole(raw, "share_q_dim").filter(|width| *width > 0) else {
                bail!(
                    "{} declares a Step3 model without share_q_dim",
                    path.display()
                );
            };
            architecture.names = Names::STEP3;
            architecture.query_bottleneck = Some(width);
            architecture.experts = Some(step_experts(raw, layers, path)?);
        }
        "deepseek" => {
            // DeepSeek-MoE (v1): Llama's attention, rotating by halves, and
            // DeepSeek's experts — `n_routed_experts` scored by softmax and
            // `n_shared_experts` shared ones, dense before
            // `first_k_dense_replace` and off `moe_layer_freq`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
