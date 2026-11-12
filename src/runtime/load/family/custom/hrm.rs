//! HRM.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::model::{Architecture, GateFunction, Names, NormKind, QkvLayout, Recurrence};
use crate::runtime::load::family::read::{flag, flag_or, number, whole, whole_or};

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
    let (Some(high_cycles), Some(low_cycles)) = (whole(raw, "H_cycles", path)?, whole(raw, "L_cycles", path)?)
    else {
        refused!(
            Incomplete,
            in raw,
            ["H_cycles", "L_cycles"],
            "{} declares an HRM-Text model without H_cycles and L_cycles",
            path.display()
        );
    };
    let per_stack = whole_or(raw, "num_layers_per_stack", layers, path)?;
    let recurrence = Recurrence {
        per_stack,
        high_cycles,
        low_cycles,
    };
    if per_stack == 0 || high_cycles == 0 || recurrence.layers() > u128::BITS as usize {
        refused!(
            OutOfRange,
            in raw,
            ["H_cycles", "L_cycles", "num_layers_per_stack", "num_hidden_layers"],
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
    architecture.query_key_value_bias = flag(raw, "attention_bias", path)?;
    architecture.output_bias = architecture.query_key_value_bias;
    architecture.feed_forward_bias = flag(raw, "mlp_bias", path)?;
    architecture.embedding_multiplier = number(raw, "embedding_scale", path)?;
    architecture.recurrence = Some(recurrence);
    architecture.prefix_lm = flag_or(raw, "prefix_lm", true, path)?;
    Ok(())
}
