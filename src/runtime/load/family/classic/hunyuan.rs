//! ERNIE 4.5, HunYuan, HY V3 and GPT-OSS.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, Names, QueryKeyNorm, Scoring, SharedExpert, SharedForm, SwigluLimit,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::{experts, qwen_dense_layers};
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, number, uniform, whole};

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
        "hy_v3" => {
            // HY V3 (Hy3): per-head query and key norms, a dense
            // feed-forward on the layers `mlp_layer_types` calls `dense` (or
            // the first `first_k_dense_replace`), and elsewhere `num_experts`
            // experts chosen by sigmoid score plus `mlp.expert_bias`,
            // renormalised under `route_norm`, scaled by
            // `router_scaling_factor`, beside `num_shared_experts` shared
            // experts in `mlp.shared_mlp`.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            fits(layers, path)?;
            let dense = if raw.get("mlp_layer_types").is_some_and(Value::is_array) {
                qwen_dense_layers(raw, layers, path)?
            } else {
                let first = whole(raw, "first_k_dense_replace").unwrap_or(0).min(layers);
                (0..first).fold(0u128, |set, layer| set | (1u128 << layer))
            };
            let normalize = raw
                .get("route_norm")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let mut routed = experts(
                raw,
                "num_experts",
                "moe_intermediate_size",
                normalize,
                ExpertLayout::HyV3,
                dense,
                path,
            )?;
            routed.scoring = Scoring::Sigmoid;
            routed.selection_bias = Some("mlp.expert_bias");
            routed.routed_scale = number(raw, "router_scaling_factor");
            routed.shared = whole(raw, "num_shared_experts")
                .filter(|shared| *shared > 0)
                .map(|shared| SharedExpert {
                    intermediate: shared * routed.intermediate,
                    module: "mlp.shared_mlp",
                    gated: false,
                    form: SharedForm::GateUpDown,
                });
            architecture.experts = Some(routed);
        }
        "ernie4_5" | "ernie4_5_moe" => {
            let bias = flag(raw, "use_bias");
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.interleaved_rotary = true;
            if model_type == "ernie4_5_moe" {
                // ERNIE 4.5's experts cover the layers from
                // `moe_layer_start_index` to `moe_layer_end_index` (the last
                // when negative) every `moe_layer_interval`; scores are a
                // softmax, chosen with `moe_statics.e_score_correction_bias`
                // added and renormalised, beside `moe_num_shared_experts`
                // shared experts.
                fits(layers, path)?;
                let start = whole(raw, "moe_layer_start_index").unwrap_or(0);
                let end = raw
                    .get("moe_layer_end_index")
                    .and_then(Value::as_i64)
                    .filter(|end| *end >= 0)
                    .map_or(layers.saturating_sub(1), |end| end as usize);
                let interval = whole(raw, "moe_layer_interval").unwrap_or(1).max(1);
                let dense = (0..layers)
                    .filter(|layer| {
                        *layer < start || *layer > end || (layer - start) % interval != 0
                    })
                    .fold(0u128, |set, layer| set | (1u128 << layer));
                let mut routed = experts(
                    raw,
                    "moe_num_experts",
                    "moe_intermediate_size",
                    true,
                    ExpertLayout::Qwen,
                    dense,
                    path,
                )?;
                routed.shared = whole(raw, "moe_num_shared_experts")
                    .filter(|shared| *shared > 0)
                    .map(|shared| SharedExpert {
                        intermediate: shared * routed.intermediate,
                        module: "mlp.shared_experts",
                        gated: false,
                        form: SharedForm::GateUpDown,
                    });
                routed.selection_bias = Some("mlp.moe_statics.e_score_correction_bias");
                architecture.experts = Some(routed);
            }
        }
        "gpt_oss" => {
            // GPT-OSS: biased attention projections with a learned sink per
            // head, sliding-window layers from `layer_types`, and a mixture
            // of `num_local_experts` biased experts behind a biased router,
            // the top `num_experts_per_tok` renormalised, each with the
            // clamped gate `swiglu_limit` bounds.
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.attention_sinks = every_layer(layers, path)?;
            architecture.sliding_window = whole(raw, "sliding_window");
            let mut routed = experts(
                raw,
                "num_local_experts",
                "intermediate_size",
                true,
                ExpertLayout::GptOss,
                0,
                path,
            )?;
            routed.swiglu_limit = number(raw, "swiglu_limit").map(SwigluLimit::GptOss);
            architecture.experts = Some(routed);
        }
        "hunyuan_v1_dense" | "hunyuan_v1_moe" => {
            // HunYuan: per-head query and key norms named `query_layernorm`
            // and `key_layernorm`, applied after the rotation.
            architecture.names = Names::HUNYUAN;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.norm_after_rotary = true;
            }
            if model_type == "hunyuan_v1_moe" {
                if flag(raw, "use_cla") || flag(raw, "use_mla") {
                    bail!(
                        "{} shares attention across layers (use_cla) or compresses it (use_mla); Ster implements HunYuan's own attention on every layer",
                        path.display()
                    );
                }
                // HunYuan-MoE: `num_experts` experts under `mlp.gate.wg`,
                // the top `moe_topk` weighted by their renormalised softmax,
                // beside `mlp.shared_mlp`, `num_shared_expert` times
                // `intermediate_size` wide, unless `use_mixed_mlp_moe` is
                // false; the first `moe_layer_num_skipped` layers stay dense.
                let width = if raw.get("moe_intermediate_size").is_some() {
                    "moe_intermediate_size"
                } else {
                    "intermediate_size"
                };
                fits(layers, path)?;
                let skipped = whole(raw, "moe_layer_num_skipped").unwrap_or(0).min(layers);
                let dense = (0..skipped).fold(0u128, |set, layer| set | (1u128 << layer));
                let mut routed = experts(
                    raw,
                    "num_experts",
                    width,
                    true,
                    ExpertLayout::HunYuan,
                    dense,
                    path,
                )?;
                if raw.get("use_mixed_mlp_moe").and_then(Value::as_bool) != Some(false) {
                    let shared = uniform(raw, "num_shared_expert", path)?.unwrap_or(1);
                    routed.shared = (shared > 0).then_some(SharedExpert {
                        intermediate: shared * llama.intermediate_size,
                        module: "mlp.shared_mlp",
                        gated: false,
                        form: SharedForm::GateUpDown,
                    });
                }
                architecture.experts = Some(routed);
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
