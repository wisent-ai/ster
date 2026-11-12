//! Where an adapter target's weight sits in a checkpoint.

use candle_transformers::models::llama::Config;

use crate::lora::Target;
use crate::model::QkvLayout;

use super::Architecture;

impl Architecture {
    /// Where `target`'s weight lives at `layer`, or `None` when this family
    /// has no single such projection: the gate of a plain feed-forward, any
    /// feed-forward projection of a mixture of experts, or the key, value and
    /// bottlenecked query of latent attention.
    ///
    /// A fused tensor holds several projections; the placement then names the
    /// row blocks this projection owns, so merging adds the update to those
    /// rows alone. Phi-3 stacks query, key and value; GPT-NeoX interleaves
    /// them head by head; Phi-3 and GLM stack gate above up.
    pub fn placement(&self, target: Target, layer: usize, config: &Config) -> Option<Placement> {
        let feed_forward = matches!(target, Target::Gate | Target::Up | Target::Down);
        if feed_forward && self.experts.is_some() {
            return None;
        }
        let names = self.names;
        let in_layer = |leaf: &str| format!("{}.{layer}.{leaf}.weight", names.layers);
        let in_attention = |leaf: &str| in_layer(&format!("{}.{leaf}", names.attention));
        // A Mamba block has no attention or feed-forward projection.
        if self.state_space.is_some() || self.delta_rule.is_some() || self.lightning.is_some() {
            return None;
        }
        // Step3's query passes a bottleneck and a norm before `wq`, so `wq`
        // reads no hidden state an adapter could share.
        if self.query_bottleneck.is_some() && target == Target::Query {
            return None;
        }
        // Gemma 4's full-attention heads are wider than its sliding-window
        // ones, MiMo-V2's have fewer key-value heads, and Dots3-Note's
        // sliding-window layers run a latent attention of their own, so no
        // attention projection has one shape on every layer.
        if (self.global_attention.is_some() || self.window_latent.is_some()) && !feed_forward {
            return None;
        }
        // K2-Horizon's routed layers take their value from value experts,
        // with no `v_proj`.
        if self.value_experts.is_some() && target == Target::Value {
            return None;
        }
        if let Some(latent) = self.latent {
            match target {
                Target::Query if latent.query_rank.is_none() => {
                    return Some(Placement::whole(in_attention(names.query)));
                }
                Target::Query | Target::Key | Target::Value => return None,
                _ => {}
            }
        }
        let head_dim = self.head_dim;
        let query = self.attention_width(config.num_attention_heads);
        let key_value = config.num_key_value_heads * head_dim;
        let attention_offset = match target {
            Target::Query => Some(0),
            Target::Key => Some(1),
            Target::Value => Some(2),
            _ => None,
        };
        let intermediate = config.intermediate_size;
        let placement = match (attention_offset, self.qkv_layout) {
            // Each head's query rows sit before its gate rows.
            (Some(0), QkvLayout::Separate) if self.output_gate => {
                let blocks = (0..config.num_attention_heads)
                    .map(|head| (head * 2 * head_dim, head * head_dim, head_dim))
                    .collect();
                Placement::blocks(in_attention(names.query), blocks)
            }
            (Some(slot), QkvLayout::Separate) => Placement::whole(in_attention(match slot {
                0 => names.query,
                1 => names.key,
                _ => names.value,
            })),
            (Some(slot), QkvLayout::Stacked) => {
                let (start, rows) = match slot {
                    0 => (0, query),
                    1 => (query, key_value),
                    _ => (query + key_value, key_value),
                };
                Placement::blocks(in_attention(names.fused_qkv), vec![(start, 0, rows)])
            }
            (Some(slot), QkvLayout::Grouped) => {
                // Each key-value group holds its query heads, then its key
                // head, then its value head.
                let groups = config.num_key_value_heads;
                let per_group = config.num_attention_heads / groups;
                let stride = (per_group + 2) * head_dim;
                let blocks = (0..groups)
                    .flat_map(|group| {
                        let (first, count, own) = match slot {
                            0 => (0, per_group, group * per_group),
                            1 => (per_group, 1, group),
                            _ => (per_group + 1, 1, group),
                        };
                        (0..count).map(move |head| {
                            (
                                group * stride + (first + head) * head_dim,
                                (own + head) * head_dim,
                                head_dim,
                            )
                        })
                    })
                    .collect();
                Placement::blocks(in_attention(names.fused_qkv), blocks)
            }
            (Some(0), QkvLayout::PairedKeyValue) => Placement::whole(in_attention(names.query)),
            (Some(slot), QkvLayout::PairedKeyValue) => {
                // Each key-value head holds its key rows, then its value rows.
                let first = if slot == 1 { 0 } else { head_dim };
                let blocks = (0..config.num_key_value_heads)
                    .map(|head| (head * 2 * head_dim + first, head * head_dim, head_dim))
                    .collect();
                Placement::blocks(in_attention(names.fused_qkv), blocks)
            }
            // HRM-Text refuses adapters before a placement is asked for.
            (Some(_), QkvLayout::GateStacked) => return None,
            (None, _) => match target {
                Target::Output => Placement::whole(in_layer(names.output)),
                Target::Gate if self.fused_feed_forward => {
                    Placement::blocks(in_layer(names.fused_gate_up), vec![(0, 0, intermediate)])
                }
                Target::Up if self.fused_feed_forward => Placement::blocks(
                    in_layer(names.fused_gate_up),
                    vec![(intermediate, 0, intermediate)],
                ),
                Target::Gate => Placement::whole(in_layer(names.gate?)),
                Target::Up => Placement::whole(in_layer(names.up)),
                _ => Placement::whole(in_layer(names.down)),
            },
        };
        // A multimodal checkpoint keeps the language model below a wrapper
        // (Gemma 3's `language_model`).
        let tensor = if names.wrapper.is_empty() {
            placement.tensor
        } else {
            format!("{}.{}", names.wrapper, placement.tensor)
        };
        Some(Placement {
            tensor,
            transposed: self.conv1d,
            root: names.root,
            ..placement
        })
    }
}

/// Where one projection's weight sits in the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub tensor: String,
    /// `None` when the projection is the whole tensor; otherwise the row
    /// blocks it owns, as `(row in the tensor, row in the projection,
    /// rows)`, rows counted in `[outputs, inputs]` orientation.
    pub blocks: Option<Vec<(usize, usize, usize)>>,
    /// The tensor is stored `[inputs, outputs]` (GPT-2's `Conv1D`).
    pub transposed: bool,
    /// The family's root prefix, which a checkpoint saved from the base
    /// model leaves off every name.
    pub root: &'static str,
}

impl Placement {
    fn whole(tensor: String) -> Self {
        Self {
            tensor,
            blocks: None,
            transposed: false,
            root: "",
        }
    }

    fn blocks(tensor: String, blocks: Vec<(usize, usize, usize)>) -> Self {
        Self {
            blocks: Some(blocks),
            ..Self::whole(tensor)
        }
    }

    /// The tensor's name in a checkpoint saved without the root prefix, when
    /// the family has one.
    pub fn without_root(&self) -> Option<String> {
        if self.root.is_empty() {
            return None;
        }
        self.tensor
            .strip_prefix(self.root)
            .and_then(|rest| rest.strip_prefix('.'))
            .map(str::to_owned)
    }
}
