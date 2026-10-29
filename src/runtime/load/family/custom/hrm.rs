//! HRM.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{Architecture, GateFunction, Names, NormKind, QkvLayout, Recurrence};
use crate::runtime::load::family::read::{flag, number, whole};

/// HRM-Text (`hrm_text`): two stacks of `num_layers_per_stack` (else
/// `num_hidden_layers`) Llama layers — `attn.gqkv_proj` holding the
/// elementwise attention gate's rows, then query, key and value, the gate's
/// sigmoid multiplying attention's output before `attn.o_proj`, a fused
/// `mlp.gate_up_proj`, every norm a scale-free RMS norm — run as
/// [`Recurrence`] lays out over `H_cycles` and `L_cycles`, the embedding
/// multiplied by `embedding_scale`. Under `prefix_lm` (on by default, as the
/// config class sets it) a prompt read whole from position zero is one
/// bidirectional block, as the model card asks for with `token_type_ids` of
/// ones; decode steps and passes that score every position stay causal.
pub(in crate::runtime::load::family) fn hrm_text(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let (Some(high_cycles), Some(low_cycles)) = (whole(raw, "H_cycles"), whole(raw, "L_cycles"))
    else {
        bail!(
            "{} declares an HRM-Text model without H_cycles and L_cycles",
            path.display()
        );
    };
    let per_stack = whole(raw, "num_layers_per_stack").unwrap_or(layers);
    let recurrence = Recurrence {
        per_stack,
        high_cycles,
        low_cycles,
    };
    if per_stack == 0 || high_cycles == 0 || recurrence.layers() > u128::BITS as usize {
        bail!(
            "{} runs {high_cycles} high cycles of {low_cycles} low passes over stacks of {per_stack} layers; Ster runs at least one layer and one cycle, and at most {} layers in all",
            path.display(),
            u128::BITS
        );
    }
    architecture.names = Names::HRM;
    architecture.norm = NormKind::BareRms;
    architecture.qkv_layout = QkvLayout::GateStacked;
    architecture.attention_gate = Some(GateFunction::Sigmoid);
    architecture.fused_feed_forward = true;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    architecture.feed_forward_bias = flag(raw, "mlp_bias");
    architecture.embedding_multiplier = number(raw, "embedding_scale");
    architecture.recurrence = Some(recurrence);
    architecture.prefix_lm = raw
        .get("prefix_lm")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    Ok(())
}
