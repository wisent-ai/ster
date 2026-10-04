//! A mixture-of-experts feed-forward: a router scores every expert for every
//! token, the top `k` run on that token, and their outputs are summed by the
//! router's weights.
//!
//! Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE and GraniteMoE all route this way and
//! differ in tensor names, in whether the chosen weights are renormalised to
//! sum to one, and in whether a shared expert runs on every token beside the
//! routed ones (Qwen2-MoE).

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{
    Activation, ExpertGroups, ExpertLayout, MixtureOfExperts, Scoring, SharedExpert,
};

#[derive(Debug, Clone)]
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    fn forward(&self, input: &Tensor, activation: Activation) -> candle_core::Result<Tensor> {
        let gated = (activation.apply(&self.gate.forward(input)?)? * self.up.forward(input)?)?;
        self.down.forward(&gated)
    }

    fn load(
        hidden: usize,
        intermediate: usize,
        names: [&str; 3],
        builder: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        let [gate, up, down] = names;
        Ok(Self {
            gate: linear_no_bias(hidden, intermediate, builder.pp(gate))?,
            up: linear_no_bias(hidden, intermediate, builder.pp(up))?,
            down: linear_no_bias(intermediate, hidden, builder.pp(down))?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct Experts {
    router: Linear,
    /// DeepSeek-V3's `e_score_correction_bias`, `[experts]`, read on the
    /// host: it moves which experts are chosen, never how much they weigh.
    selection_bias: Option<Vec<f32>>,
    experts: Vec<Expert>,
    /// The expert every token goes through, and Qwen2-MoE's sigmoid gate
    /// that scales it (DeepSeek's shared experts are added as they are).
    shared: Option<(Expert, Option<Linear>)>,
    spec: MixtureOfExperts,
    activation: Activation,
}

impl Experts {
    /// `builder` is the layer's.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: &MixtureOfExperts,
        activation: Activation,
    ) -> candle_core::Result<Self> {
        let count = spec.count;
        let intermediate = spec.intermediate;
        let (router, experts) = match spec.layout {
            ExpertLayout::Mixtral => {
                let block = builder.pp("block_sparse_moe");
                let experts = (0..count)
                    .map(|expert| {
                        Expert::load(
                            hidden,
                            intermediate,
                            ["w1", "w3", "w2"],
                            block.pp("experts").pp(expert.to_string()),
                        )
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("gate"))?, experts)
            }
            ExpertLayout::Qwen | ExpertLayout::Jamba => {
                let (block, router) = if spec.layout == ExpertLayout::Jamba {
                    (builder.pp("feed_forward"), "router")
                } else {
                    (builder.pp("mlp"), "gate")
                };
                let experts = (0..count)
                    .map(|expert| {
                        Expert::load(
                            hidden,
                            intermediate,
                            ["gate_proj", "up_proj", "down_proj"],
                            block.pp("experts").pp(expert.to_string()),
                        )
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp(router))?, experts)
            }
            // GraniteMoE stacks every expert into two tensors: `input_linear`
            // `[experts, 2 * intermediate, hidden]`, gate rows then up rows,
            // and `output_linear` `[experts, hidden, intermediate]`. Each
            // expert is a view of its slice, not a copy.
            ExpertLayout::Granite => {
                let block = builder.pp("block_sparse_moe");
                let input = block.get((count, 2 * intermediate, hidden), "input_linear.weight")?;
                let output = block.get((count, hidden, intermediate), "output_linear.weight")?;
                let experts = (0..count)
                    .map(|expert| -> candle_core::Result<Expert> {
                        let rows = input.get(expert)?;
                        Ok(Expert {
                            gate: Linear::new(rows.narrow(0, 0, intermediate)?, None),
                            up: Linear::new(rows.narrow(0, intermediate, intermediate)?, None),
                            down: Linear::new(output.get(expert)?, None),
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("router").pp("layer"))?, experts)
            }
        };
        let shared = match spec.shared {
            Some(SharedExpert { intermediate, gated }) => {
                let block = builder.pp("mlp");
                let name = if gated { "shared_expert" } else { "shared_experts" };
                Some((
                    Expert::load(
                        hidden,
                        intermediate,
                        ["gate_proj", "up_proj", "down_proj"],
                        block.pp(name),
                    )?,
                    if gated {
                        Some(linear_no_bias(hidden, 1, block.pp("shared_expert_gate"))?)
                    } else {
                        None
                    },
                ))
            }
            None => None,
        };
        let selection_bias = if spec.selection_bias {
            Some(
                builder
                    .pp("mlp")
                    .pp("gate")
                    .get(count, "e_score_correction_bias")?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
            )
        } else {
            None
        };
        Ok(Self {
            router,
            selection_bias,
            experts,
            shared,
            spec: spec.clone(),
            activation,
        })
    }

    /// The experts one token goes to, from its scores: the `top_k` highest
    /// once the selection bias is added, among the best groups' experts when
    /// routing is group-limited. Ties keep expert order, so the choice is
    /// deterministic.
    fn choose(&self, scores: &[f32]) -> Vec<usize> {
        let count = scores.len();
        let ranked: Vec<f32> = match &self.selection_bias {
            Some(bias) => scores.iter().zip(bias).map(|(score, bias)| score + bias).collect(),
            None => scores.to_vec(),
        };
        let descending = |values: &[f32], items: &mut Vec<usize>| {
            items.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
        };
        let mut allowed: Vec<usize> = (0..count).collect();
        if let Some(ExpertGroups { groups, chosen_groups, rank_by_top_two }) = self.spec.groups {
            let size = count / groups;
            let group_score = |group: usize| -> f32 {
                let mut members: Vec<f32> = ranked[group * size..(group + 1) * size].to_vec();
                members.sort_by(|left, right| right.total_cmp(left));
                if rank_by_top_two {
                    members.iter().take(2).sum()
                } else {
                    members[0]
                }
            };
            let scored: Vec<f32> = (0..groups).map(group_score).collect();
            let mut order: Vec<usize> = (0..groups).collect();
            descending(&scored, &mut order);
            let kept: Vec<usize> = order.into_iter().take(chosen_groups).collect();
            allowed.retain(|expert| kept.contains(&(expert / size)));
        }
        descending(&ranked, &mut allowed);
        allowed.truncate(self.spec.top_k);
        allowed
    }

    /// Routes every token of `hidden` `[batch, sequence, width]`.
    ///
    /// The choice of experts is a discrete decision read on the host; the
    /// weights stay tensors taken from the router's softmax through a 0/1
    /// mask, so a differentiable pass still carries the router's gradient
    /// back into the residual stream. Every op here — `softmax` composed of
    /// `exp` and `sum`, `index_select`, `index_add`, the mask multiply — has a
    /// backward pass.
    pub(super) fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        let tokens = batch * sequence;
        let flat = hidden.reshape((tokens, width))?;
        let device = flat.device();
        let count = self.experts.len();
        let logits = self.router.forward(&flat)?.to_dtype(DType::F32)?;
        let scores = match self.spec.scoring {
            Scoring::Softmax => candle_nn::ops::softmax(&logits, D::Minus1)?,
            // sigmoid, composed so it has a backward pass.
            Scoring::Sigmoid => (logits.neg()?.exp()? + 1.0)?.recip()?,
        };
        let host = scores.to_vec2::<f32>()?;
        let mut mask = vec![0f32; tokens * count];
        let mut routed: Vec<Vec<u32>> = vec![Vec::new(); count];
        for (token, row) in host.iter().enumerate() {
            for expert in self.choose(row) {
                mask[token * count + expert] = 1.0;
                routed[expert].push(token as u32);
            }
        }
        let mask = Tensor::from_vec(mask, (tokens, count), device)?;
        let chosen = (scores * mask)?;
        let weights = if self.spec.normalize {
            chosen.broadcast_div(&chosen.sum_keepdim(D::Minus1)?)?
        } else {
            chosen
        };
        let weights = match self.spec.routed_scale {
            Some(scale) => (weights * scale)?,
            None => weights,
        };
        let mut output = flat.zeros_like()?;
        for (expert, tokens) in routed.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let index = Tensor::new(tokens.as_slice(), device)?;
            let inputs = flat.index_select(&index, 0)?;
            let produced = self.experts[expert].forward(&inputs, self.activation)?;
            let weight = weights
                .index_select(&index, 0)?
                .narrow(1, expert, 1)?
                .to_dtype(produced.dtype())?;
            output = output.index_add(&index, &produced.broadcast_mul(&weight)?, 0)?;
        }
        if let Some((shared, gate)) = &self.shared {
            let produced = shared.forward(&flat, self.activation)?;
            let produced = match gate {
                // sigmoid, composed so it has a backward pass.
                Some(gate) => produced
                    .broadcast_mul(&(gate.forward(&flat)?.neg()?.exp()? + 1.0)?.recip()?)?,
                None => produced,
            };
            output = (output + produced)?;
        }
        output.reshape((batch, sequence, width))
    }
}
