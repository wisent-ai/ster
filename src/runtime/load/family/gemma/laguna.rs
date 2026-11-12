//! Laguna.

use std::path::Path;

use anyhow::Result;
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, GateFunction, GlobalAttention, Names, QueryKeyNorm, Scoring,
    SharedExpert, SharedForm,
};
use crate::runtime::load::family::experts::{experts, qwen_dense_layers};
use crate::runtime::load::family::read::layers::listed_layers;
use crate::runtime::load::family::read::{
    describe, flag, list, number, number_or, whole, within_or,
};
use crate::runtime::load::family::rope::scaling::rope_scaling;

/// Laguna (`laguna`): Llama's block with per-head query and key norms and a
/// per-head gate (`g_proj` through softplus under `gating` `per-head`; any
/// other gating is refused). Its layers
/// alternate full and `sliding_window` attention as `layer_types` lists,
/// each kind with its own head count (`num_attention_heads_per_layer`) and
/// rotation (`rope_parameters` per layer type: the full layers' YaRN over
/// their `partial_rotary_factor` share, the sliding ones' plain rotation
/// over theirs), the sliding layers with a learned sink per head
/// (`self_attn.sink`) under `swa_attention_sink_enabled`. The layers
/// `mlp_layer_types` calls `sparse` route over `num_experts` experts by
/// sigmoid scores that `mlp.experts.e_score_correction_bias` moves to
/// choose, renormalised and scaled by `moe_routed_scaling_factor`, the
/// weights applied to the experts' input under
/// `moe_apply_router_weight_on_input`, beside a `shared_expert`.
pub(in crate::runtime::load::family) fn laguna(
    raw: &Value,
    scaling: Option<&Value>,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(types) = list(raw, "layer_types", path)? else {
        refused!(
            Incomplete,
            in raw,
            ["layer_types"],
            "{} declares a Laguna model without layer_types",
            path.display()
        );
    };
    let windowed = listed_layers(types, layers, path)?;
    if number(raw, "moe_router_logit_softcapping", path)?.is_some_and(|cap| cap > 0.0) {
        refused!(
            Unimplemented,
            in raw,
            ["moe_router_logit_softcapping"],
            "{} caps its router logits (moe_router_logit_softcapping); Ster implements Laguna's router without a cap",
            path.display()
        );
    }
    architecture.sliding_window = whole(raw, "sliding_window", path)?;
    architecture.sliding_layers = windowed;
    architecture.query_key_norm = QueryKeyNorm::PerHead;
    architecture.query_key_value_bias = flag(raw, "attention_bias", path)?;
    architecture.output_bias = architecture.query_key_value_bias;
    match raw.get("gating") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => architecture.head_gate = Some(GateFunction::NaturalSoftplus),
        Some(Value::String(kind)) if kind == "per-head" => {
            architecture.head_gate = Some(GateFunction::NaturalSoftplus)
        }
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["gating"],
            "{} declares gating {other}; Ster implements Laguna's per-head gate",
            path.display()
        ),
    }
    if flag(raw, "swa_attention_sink_enabled", path)? {
        architecture.names = Names::LAGUNA;
        architecture.attention_sinks = windowed;
    }
    // Each kind's own head count, as `num_attention_heads_per_layer` lists
    // it; the sliding-window count is the base one.
    let full_heads = match list(raw, "num_attention_heads_per_layer", path)? {
        Some(counts) => {
            if counts.len() != layers {
                refused!(
                    Contradictory,
                    in raw,
                    ["num_attention_heads_per_layer", "num_hidden_layers"],
                    "{} lists {} num_attention_heads_per_layer for {layers} layers",
                    path.display(),
                    counts.len()
                );
            }
            let mut kinds = [None, None];
            for (layer, count) in counts.iter().enumerate() {
                let count = count.as_u64().map(|count| count as usize);
                let slot = usize::from(windowed & (1u128 << layer) != 0);
                match (kinds[slot], count) {
                    (_, None) => refused!(
                        Malformed,
                        in raw,
                        ["num_attention_heads_per_layer"],
                        "{} lists a head count for layer {layer} that is not a whole number",
                        path.display()
                    ),
                    (Some(seen), Some(count)) if seen != count => refused!(
                        Unimplemented,
                        in raw,
                        ["num_attention_heads_per_layer", "layer_types"],
                        "{} gives its {} layers more than one head count; Ster gives each attention kind one",
                        path.display(),
                        if slot == 1 {
                            "sliding-window"
                        } else {
                            "full-attention"
                        }
                    ),
                    (_, count) => kinds[slot] = count,
                }
            }
            if kinds[1].is_some_and(|count| count != llama.num_attention_heads) {
                refused!(
                    Contradictory,
                    in raw,
                    ["num_attention_heads", "num_attention_heads_per_layer"],
                    "{} states num_attention_heads {} unlike its sliding-window layers' {:?}",
                    path.display(),
                    llama.num_attention_heads,
                    kinds[1]
                );
            }
            kinds[0]
        }
        None => None,
    };
    let head_dim = architecture.head_dim;
    let share = |value: f64| (head_dim as f64 * value) as usize;
    let global_share = match scaling {
        Some(scaling) => {
            within_or(scaling, "rope_parameters", "partial_rotary_factor", 1.0, path)?
        }
        None => 1.0,
    };
    let global_rotary = share(global_share);
    let local_rotary = share(number_or(raw, "local_partial_rotary_factor", 1.0, path)?);
    if [global_rotary, local_rotary]
        .iter()
        .any(|rotated| *rotated == 0 || rotated % 2 != 0 || *rotated > head_dim)
    {
        refused!(
            OutOfRange,
            [
                "head_dim" => head_dim,
                "rope_parameters.partial_rotary_factor" => describe(
                    scaling.and_then(|scaling| scaling.get("partial_rotary_factor"))
                ),
                "local_partial_rotary_factor" => describe(raw.get("local_partial_rotary_factor")),
            ],
            "{} rotates {global_rotary} and {local_rotary} of {head_dim} components per head; each must be even, above zero and at most the head",
            path.display()
        );
    }
    architecture.rotary_dim = local_rotary;
    architecture.rope_scaling = rope_scaling(scaling, global_rotary, raw, llama, path)?;
    let base = f64::from(llama.rope_theta);
    architecture.local_rope_theta =
        Some(number_or(raw, "rope_local_base_freq", base, path)? as f32);
    architecture.global_attention = Some(GlobalAttention {
        heads: full_heads.filter(|heads| *heads != llama.num_attention_heads),
        head_dim,
        key_value_heads: None,
        rotary_dim: global_rotary,
        key_is_value: false,
    });
    let mut routed = experts(
        raw,
        "num_experts",
        "moe_intermediate_size",
        true,
        ExpertLayout::Qwen,
        qwen_dense_layers(raw, layers, path)?,
        path,
    )?;
    routed.scoring = Scoring::Sigmoid;
    routed.selection_bias = Some("mlp.experts.e_score_correction_bias");
    routed.routed_scale = number(raw, "moe_routed_scaling_factor", path)?;
    routed.weight_input = flag(raw, "moe_apply_router_weight_on_input", path)?;
    routed.shared = whole(raw, "shared_expert_intermediate_size", path)?
        .filter(|width| *width > 0)
        .map(|intermediate| SharedExpert {
            intermediate,
            module: "mlp.shared_expert",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    architecture.experts = Some(routed);
    Ok(())
}
