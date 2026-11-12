//! Refusing the adapter targets an [`Architecture`] has no single projection
//! for.

use anyhow::Result;
use candle_transformers::models::llama::Config;
use wisent_errors::Cause;

use crate::failures::{model::weights::adapters as causes, refuse};
use crate::lora::Target;

use super::Architecture;

impl Architecture {
    /// Refuses adapter targets this family has no single projection for, so
    /// an adapter is not created and then never trained; `model` names the
    /// checkpoint in the refusal.
    pub fn check_targets(&self, targets: &[Target], config: &Config, model: &str) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }
        let asked: Vec<&str> = targets.iter().map(|target| target.name()).collect();
        let asked = asked.join(", ");
        if let Some((cause, feature)) = self.unadaptable() {
            return Err(refuse(
                cause,
                &[("model", model), ("feature", &feature), ("targets", &asked)],
            ));
        }
        if let Some(target) = targets
            .iter()
            .find(|target| self.placement(**target, 0, config).is_none())
        {
            let choices: Vec<&str> = Target::ALL
                .into_iter()
                .filter(|choice| self.placement(*choice, 0, config).is_some())
                .map(Target::name)
                .collect();
            let choices = choices.join(", ");
            return Err(refuse(
                self.absent(*target),
                &[
                    ("model", model),
                    ("target", target.name()),
                    ("choices", &choices),
                    ("targets", &asked),
                ],
            ));
        }
        Ok(())
    }

    /// Why this family stores no single tensor for `target`, as the cause
    /// of that reason.
    fn absent(&self, target: Target) -> &'static Cause {
        if self.query_bottleneck.is_some() && target == Target::Query {
            &causes::QUERY_BOTTLENECKED
        } else if (self.global_attention.is_some() || self.window_latent.is_some())
            && !feed_forward_target(target)
        {
            &causes::HEADS_DIFFER
        } else if self.value_experts.is_some() && target == Target::Value {
            &causes::VALUE_EXPERTS
        } else if self.latent.is_some() && !feed_forward_target(target) {
            &causes::ATTENTION_LATENT
        } else if self.experts.is_some() {
            &causes::FEED_FORWARD_EXPERTS
        } else {
            &causes::FEED_FORWARD_UNGATED
        }
    }

    /// Why no adapter attaches to this family's layers, as the cause of that
    /// reason, and the layers the config gave it that say so; `None` for a
    /// family adapters attach to.
    fn unadaptable(&self) -> Option<(&'static Cause, String)> {
        if let Some(spec) = &self.short_convolution {
            return Some((
                &causes::SHORT_CONVOLUTION,
                format!("short-convolution layers: {}", listed(spec.layers)),
            ));
        }
        if let Some(plans) = &self.layer_plans {
            return Some((
                &causes::LAYER_PLANS,
                format!("block_configs: {} per-layer plans", plans.len()),
            ));
        }
        if let Some(experts) = &self.shortcut_experts {
            return Some((
                &causes::SHORTCUT_EXPERTS,
                format!("shortcut-connected experts: {} routed", experts.count),
            ));
        }
        if let Some(loops) = self.loops {
            return Some((
                &causes::LAYERS_REPEATED,
                format!("loops: {} stored layers run {} times", loops.physical, loops.count),
            ));
        }
        if let Some(recurrence) = self.recurrence {
            return Some((
                &causes::LAYERS_REPEATED,
                format!(
                    "recurrence: {} high cycles of {} low passes over stacks of {} layers",
                    recurrence.high_cycles, recurrence.low_cycles, recurrence.per_stack
                ),
            ));
        }
        if self.compressed.is_some() || self.hyper_connections.is_some() {
            return Some((
                &causes::STREAMS_HYPER_CONNECTED,
                format!(
                    "hyper-connected residual streams: {}",
                    self.hyper_connections.map_or(1, |streams| streams.streams)
                ),
            ));
        }
        if let Some(spec) = self.inkling {
            return Some((
                &causes::INKLING_PROJECTIONS,
                format!("Inkling sliding-window layers: {}", listed(spec.sliding_layers)),
            ));
        }
        if let Some(spec) = &self.lightning {
            return Some((
                &causes::LIGHTNING_ATTENTION,
                format!("lightning-attention layers: {}", listed(spec.layers)),
            ));
        }
        if let Some(spec) = &self.delta_rule {
            return Some((
                &causes::DELTA_RULE,
                format!("gated delta-rule layers: {}", listed(spec.layers)),
            ));
        }
        if let Some(spec) = &self.state_space {
            return Some((
                &causes::STATE_SPACE,
                format!("state-space layers: {}", listed(spec.layers)),
            ));
        }
        None
    }
}

fn feed_forward_target(target: Target) -> bool {
    matches!(target, Target::Gate | Target::Up | Target::Down)
}

/// The layers a per-layer bit set marks, in order.
fn listed(layers: u128) -> String {
    (0..u128::BITS as usize)
        .filter(|layer| layers & (1u128 << layer) != 0)
        .map(|layer| layer.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
