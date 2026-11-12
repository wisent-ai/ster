//! DeepSeek-V4.

use std::path::Path;

use serde_json::Value;
use {crate::failures::model::Stated, anyhow::Result};

use crate::model::{
    Architecture, CompressYarn, CompressedSpec, ExpertLayout, Scoring, SharedExpert, SharedForm,
    SwigluLimit,
};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{
    list, number, stated, text, whole, whole_or, within, within_or,
};
use crate::runtime::load::family::sparse::hy_v4::hyper_connections;

/// DeepSeek-V4's compression rates by layer kind when its config states
/// none (`DeepseekV4Config.default_compress_rates` in
/// `configuration_deepseek_v4.py`), and the per-layer `compress_ratios` of
/// older configs, read through the same table
/// (`_COMPRESS_RATIO_TO_LAYER_TYPE`).
pub(in crate::runtime::load::family) const DEEPSEEK_V4_SPARSE_RATE: usize = 4;

pub(in crate::runtime::load::family) const DEEPSEEK_V4_HEAVY_RATE: usize = 128;

/// How many leading layers route by token id when the config lists no
/// `mlp_layer_types` and no `num_hash_layers`
/// (`DeepseekV4Config.default_num_hash_layers`).
pub(in crate::runtime::load::family) const DEEPSEEK_V4_HASH_LAYERS: usize = 3;

/// DeepSeek-V4 (`deepseek_v4`), in Transformers' tensor layout: see
/// [`CompressedSpec`]. `layer_types` (else the older `compress_ratios`, 0
/// sliding, 4 sparse, 128 heavy, else two heavy layers then sparse and heavy
/// in turn) says which layers compress; `mlp_layer_types` (else the first
/// `num_hash_layers`) which route by token id. Every layer's feed-forward is
/// routed — experts scored `sqrtsoftplus`, chosen with
/// `mlp.gate.e_score_correction_bias` added, renormalised and scaled by
/// `routed_scaling_factor`, beside one shared expert — and every SwiGLU is
/// clamped by `swiglu_limit`.
pub(in crate::runtime::load::family) fn deepseek_v4(
    raw: &Value,
    scaling: Option<&Value>,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    let size = |key: &str| -> Result<usize> {
        whole(raw, key, path)?.filter(|size| *size > 0).or_refuse(raw, &[key], || {
            format!(
                "{} declares a DeepSeek-V4 model without {key}",
                path.display()
            )
        })
    };
    stated::<&serde_json::Map<String, Value>>(raw, "compress_rates", path)?;
    let rate = |kind: &str, default: usize| {
        whole_or(raw, &format!("compress_rates.{kind}"), default, path)
    };
    let (sparse_rate, heavy_rate) = (
        rate("compressed_sparse_attention", DEEPSEEK_V4_SPARSE_RATE)?,
        rate("heavily_compressed_attention", DEEPSEEK_V4_HEAVY_RATE)?,
    );
    let kinds: Vec<String> = match (
        list(raw, "layer_types", path)?,
        list(raw, "compress_ratios", path)?,
    ) {
        (Some(types), _) => types
            .iter()
            .map(|kind| kind.as_str().map_or_else(|| kind.to_string(), str::to_owned))
            .collect(),
        (None, Some(ratios)) => ratios
            .iter()
            .map(|ratio| match ratio.as_u64() {
                Some(0) => Ok("sliding_attention".to_owned()),
                Some(ratio) if ratio as usize == DEEPSEEK_V4_SPARSE_RATE => {
                    Ok("compressed_sparse_attention".to_owned())
                }
                Some(ratio) if ratio as usize == DEEPSEEK_V4_HEAVY_RATE => {
                    Ok("heavily_compressed_attention".to_owned())
                }
                _ => refused!(
                    Unimplemented,
                    in raw,
                    ["compress_ratios"],
                    "{} lists compress_ratios entry {ratio}; DeepSeek-V4 reads 0, 4 and 128",
                    path.display()
                ),
            })
            .collect::<Result<_>>()?,
        (None, None) => (0..layers)
            .map(|layer| {
                if layer >= 2 && layer % 2 == 1 {
                    "compressed_sparse_attention"
                } else {
                    "heavily_compressed_attention"
                }
            })
            .map(str::to_owned)
            .collect(),
    };
    if kinds.len() < layers {
        refused!(
            Contradictory,
            in raw,
            ["layer_types", "compress_ratios", "num_hidden_layers"],
            "{} lists {} layer kinds for {layers} layers",
            path.display(),
            kinds.len()
        );
    }
    let (mut sparse_layers, mut heavy_layers) = (0u128, 0u128);
    for (layer, kind) in kinds.iter().take(layers).enumerate() {
        match kind.as_str() {
            "sliding_attention" => {}
            "compressed_sparse_attention" => sparse_layers |= 1u128 << layer,
            "heavily_compressed_attention" => heavy_layers |= 1u128 << layer,
            other => refused!(
                Unimplemented,
                in raw,
                ["layer_types", "compress_ratios"],
                "{} declares layer {layer} as {other:?}; Ster implements DeepSeek-V4's sliding_attention, compressed_sparse_attention and heavily_compressed_attention",
                path.display()
            ),
        }
    }
    let hash_layers = match list(raw, "mlp_layer_types", path)? {
        Some(types) => {
            let mut set = 0u128;
            for (layer, kind) in types.iter().take(layers).enumerate() {
                match kind.as_str() {
                    Some("hash_moe") => set |= 1u128 << layer,
                    Some("moe") => {}
                    other => refused!(
                        Malformed,
                        in raw,
                        ["mlp_layer_types"],
                        "{} declares feed-forward {layer} as {other:?}; DeepSeek-V4's are hash_moe and moe",
                        path.display()
                    ),
                }
            }
            set
        }
        None => {
            let count = whole_or(raw, "num_hash_layers", DEEPSEEK_V4_HASH_LAYERS, path)?
                .min(layers);
            (0..count).fold(0u128, |set, layer| set | 1u128 << layer)
        }
    };
    let head_dim = size("head_dim")?;
    let rotary_dim = match (
        number(raw, "partial_rotary_factor", path)?,
        whole(raw, "qk_rope_head_dim", path)?,
    ) {
        (Some(factor), _) => (head_dim as f64 * factor) as usize,
        (None, Some(rotary)) => rotary,
        (None, None) => refused!(
            Incomplete,
            in raw,
            ["partial_rotary_factor", "qk_rope_head_dim"],
            "{} declares neither partial_rotary_factor nor qk_rope_head_dim",
            path.display()
        ),
    };
    if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        refused!(
            OutOfRange,
            in raw,
            ["head_dim", "partial_rotary_factor", "qk_rope_head_dim"],
            "{} rotates {rotary_dim} of each head's {head_dim} channels; it must be even and within the head",
            path.display()
        );
    }
    let heads = size("num_attention_heads")?;
    let output_groups = size("o_groups")?;
    if (heads * head_dim) % output_groups != 0 {
        refused!(
            Contradictory,
            in raw,
            ["num_attention_heads", "head_dim", "o_groups"],
            "{} splits {heads} heads of {head_dim} into {output_groups} output groups unevenly",
            path.display()
        );
    }
    let yarn = match scaling.filter(|scaling| scaling.is_object()) {
        Some(scaling) => {
            let kind = match within::<&str>(scaling, "rope_scaling", "type", path)? {
                Some(kind) => Some(kind),
                None => within(scaling, "rope_scaling", "rope_type", path)?,
            };
            (kind == Some("yarn")).then_some(scaling)
        }
        None => None,
    };
    let compress_yarn = match yarn {
        Some(scaling) => {
            let field = |key: &str| -> Result<f64> {
                within(scaling, "rope_scaling", key, path)?.or_refuse(scaling, &[key], || {
                    format!("{} states a YaRN rotation without {key}", path.display())
                })
            };
            Some(CompressYarn {
                factor: field("factor")? as f32,
                original: field("original_max_position_embeddings")? as usize,
                beta_fast: field("beta_fast")? as f32,
                beta_slow: field("beta_slow")? as f32,
                truncate: within_or(scaling, "rope_scaling", "truncate", true, path)?,
            })
        }
        None => None,
    };
    let Some(main_theta) = number(raw, "rope_theta", path)? else {
        refused!(
            Incomplete,
            in raw,
            ["rope_theta"],
            "{} declares a DeepSeek-V4 model without rope_theta",
            path.display()
        );
    };
    let Some(compress_theta) = number(raw, "compress_rope_theta", path)? else {
        refused!(
            Incomplete,
            in raw,
            ["compress_rope_theta"],
            "{} declares a DeepSeek-V4 model without compress_rope_theta",
            path.display()
        );
    };
    architecture.hyper_connections = Some(hyper_connections(raw, true, path)?);
    architecture.compressed = Some(CompressedSpec {
        heads,
        head_dim,
        rotary_dim,
        query_rank: size("q_lora_rank")?,
        output_groups,
        output_rank: size("o_lora_rank")?,
        window: size("sliding_window")?,
        sparse_layers,
        heavy_layers,
        sparse_rate,
        heavy_rate,
        index_heads: size("index_n_heads")?,
        index_head_dim: size("index_head_dim")?,
        index_top_k: size("index_topk")?,
        main_theta: main_theta as f32,
        compress_theta: compress_theta as f32,
        compress_yarn,
    });
    if whole_or(raw, "n_shared_experts", 1, path)? != 1 {
        refused!(
            Unimplemented,
            in raw,
            ["n_shared_experts"],
            "{} declares n_shared_experts other than one; DeepSeek-V4 has one shared expert",
            path.display()
        );
    }
    let mut routed = experts(
        raw,
        "n_routed_experts",
        "moe_intermediate_size",
        true,
        ExpertLayout::DeepseekV4,
        0,
        path,
    )?;
    routed.scoring = match text(raw, "scoring_func", path)? {
        Some("sqrtsoftplus") => Scoring::SqrtSoftplus,
        Some("sigmoid") => Scoring::Sigmoid,
        Some("softmax") => Scoring::Softmax,
        other => refused!(
            Unimplemented,
            in raw,
            ["scoring_func"],
            "{} declares scoring_func {other:?}; Ster implements sqrtsoftplus, sigmoid and softmax",
            path.display()
        ),
    };
    routed.selection_bias = Some("mlp.gate.e_score_correction_bias");
    routed.routed_scale = number(raw, "routed_scaling_factor", path)?;
    routed.swiglu_limit = number(raw, "swiglu_limit", path)?.map(SwigluLimit::Inner);
    routed.hash_layers = hash_layers;
    routed.shared = Some(SharedExpert {
        intermediate: routed.intermediate,
        module: "mlp.shared_experts",
        gated: false,
        form: SharedForm::GateUpDown,
    });
    architecture.experts = Some(routed);
    Ok(())
}
