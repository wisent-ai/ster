//! What an [`Architecture`] runs at one layer.

use crate::model::{
    DeltaRuleSpec, GlobalAttention, LayerPlan, LightningSpec, MixtureOfExperts, Names, Positions,
    ShortConvolution, StateSpaceSpec, WindowLatent,
};

use super::Architecture;

impl Architecture {
    /// The window layer `layer` attends through, or `None` for full attention.
    pub fn window(&self, layer: usize) -> Option<usize> {
        let sliding = layer < 128 && self.sliding_layers & (1u128 << layer) != 0;
        self.sliding_window.filter(|_| sliding)
    }

    /// The stored layer Ster layer `layer` reads its tensors from, and the
    /// names it reads them by: the half of a LongCat-Flash layer
    /// (`self_attn.0`, `mlps.0`, … then `.1`), or the layer itself.
    pub fn stored_layer(&self, layer: usize) -> (usize, Names) {
        if let Some(recurrence) = self.recurrence {
            return if layer < recurrence.per_stack {
                (
                    layer,
                    Names {
                        layers: Names::HRM_LOW_LAYERS,
                        ..self.names
                    },
                )
            } else {
                (
                    layer - recurrence.per_stack,
                    Names {
                        layers: Names::HRM_HIGH_LAYERS,
                        ..self.names
                    },
                )
            };
        }
        match self.shortcut_experts {
            Some(_) if layer % 2 == 0 => (layer / 2, Names::LONGCAT_FIRST),
            Some(_) => (layer / 2, Names::LONGCAT_SECOND),
            None => (layer, self.names),
        }
    }

    /// DeciLM's plan for `layer`, when the family states one per layer.
    pub fn layer_plan(&self, layer: usize) -> Option<LayerPlan> {
        self.layer_plans
            .as_ref()
            .and_then(|plans| plans.get(layer).copied())
    }

    /// The full-attention spec layer `layer` uses: Gemma 4's or MiMo-V2's
    /// on a layer that attends past any window, `None` otherwise.
    pub fn global_at(&self, layer: usize) -> Option<GlobalAttention> {
        self.global_attention
            .filter(|_| self.window(layer).is_none())
    }

    /// The sliding-window latent attention layer `layer` runs, when the
    /// family gives its windowed layers their own (Dots3-Note).
    pub fn window_latent_at(&self, layer: usize) -> Option<WindowLatent> {
        self.window_latent.filter(|_| self.window(layer).is_some())
    }

    /// Whether layer `layer`'s softmax holds a learned sink logit per head.
    pub fn sinks_at(&self, layer: usize) -> bool {
        layer < 128 && self.attention_sinks & (1u128 << layer) != 0
    }

    /// How many components the global rotation spans: the full-attention
    /// layers' own rotated width under a global attention spec,
    /// `rotary_dim` otherwise.
    pub fn global_rotary_dim(&self) -> usize {
        match self.global_attention {
            Some(global) => global.rotary_dim,
            None => self.rotary_dim,
        }
    }

    /// The layer whose keys and values a Gemma 4 key-value-sharing layer
    /// reuses: the last layer before the sharing ones whose attention is of
    /// the same kind (sliding-window or full).
    pub fn key_value_source(&self, layer: usize) -> Option<usize> {
        let first = self.shared_key_values.filter(|first| layer >= *first)?;
        let kind = self.window(layer).is_some();
        (0..first)
            .rev()
            .find(|earlier| self.window(*earlier).is_some() == kind)
    }

    /// Whether `layer` is the one whose keys and values the sharing layers of
    /// its kind reuse.
    pub fn stores_key_values(&self, layer: usize) -> bool {
        let Some(first) = self.shared_key_values.filter(|first| layer < *first) else {
            return false;
        };
        let kind = self.window(layer).is_some();
        (layer + 1..first).all(|later| self.window(later).is_some() != kind)
    }

    /// Whether layer `layer` applies the rotary embedding: never for a family
    /// whose positions are learned or ALiBi.
    pub fn rotates(&self, layer: usize) -> bool {
        self.positions == Positions::Rotary
            && (layer >= 128 || self.unrotated_layers & (1u128 << layer) == 0)
    }

    /// Width of the query projection and of the attention output.
    pub fn attention_width(&self, heads: usize) -> usize {
        heads * self.head_dim
    }

    /// The short convolution layer `layer` uses in place of attention, if
    /// any.
    pub fn short_convolution_at(&self, layer: usize) -> Option<&ShortConvolution> {
        self.short_convolution
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// The state-space mixer layer `layer` uses in place of attention, if
    /// any.
    pub fn state_space_at(&self, layer: usize) -> Option<&StateSpaceSpec> {
        self.state_space
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// The lightning attention layer `layer` uses in place of attention, if
    /// any.
    pub fn lightning_at(&self, layer: usize) -> Option<&LightningSpec> {
        self.lightning
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// The gated delta-rule mixer layer `layer` uses in place of attention,
    /// if any.
    pub fn delta_rule_at(&self, layer: usize) -> Option<&DeltaRuleSpec> {
        self.delta_rule
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// Layer `layer`'s mixture of experts, with the layer's own count of
    /// chosen experts and width where the family states one.
    pub fn experts_at(&self, layer: usize) -> Option<MixtureOfExperts> {
        let mut experts = self.experts.clone()?;
        if let Some(Some((top_k, intermediate))) = self
            .expert_overrides
            .as_ref()
            .and_then(|all| all.get(layer))
        {
            experts.top_k = *top_k;
            experts.intermediate = *intermediate;
        }
        Some(experts)
    }

    /// Whether layer `layer`'s feed-forward is the mixture of experts.
    pub fn routed(&self, layer: usize) -> bool {
        self.experts
            .as_ref()
            .is_some_and(|experts| layer >= 128 || experts.dense_layers & (1u128 << layer) == 0)
    }

    /// Width of one head's value: narrower than the query in latent
    /// attention and under MiMo-V2's `v_head_dim`, the head width otherwise.
    pub fn value_dim(&self) -> usize {
        match (self.latent, self.value_head_dim) {
            (Some(latent), _) => latent.value,
            (None, Some(value_head_dim)) => value_head_dim,
            (None, None) => self.head_dim,
        }
    }
}
