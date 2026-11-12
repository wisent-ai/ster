//! Routing every token to its experts: the router's scores, the experts it
//! chooses, and their outputs summed by its weights.

use candle_core::{D, DType, Tensor};
use candle_nn::Module;

use crate::failures::{
    model::decoder::{TOKEN_IDS_UNSET, TOKEN_UNROUTED},
    refuse_candle,
};
use crate::model::{ExpertGroups, ExpertLayout, Pass, Scoring};

use super::Experts;
use super::sparse_mixer::sparse_mixer;

impl Experts {
    /// The experts one token goes to, from its scores: the `top_k` highest
    /// once the selection bias is added, among the best groups' experts when
    /// routing is group-limited. Ties keep expert order, so the choice is
    /// deterministic.
    fn choose(&self, scores: &[f32]) -> Vec<usize> {
        let count = scores.len();
        let ranked: Vec<f32> = match &self.selection_bias {
            Some(bias) => scores
                .iter()
                .zip(bias)
                .map(|(score, bias)| score + bias)
                .collect(),
            None => scores.to_vec(),
        };
        let descending = |values: &[f32], items: &mut Vec<usize>| {
            items.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
        };
        let mut allowed: Vec<usize> = (0..count).collect();
        if let Some(ExpertGroups {
            groups,
            chosen_groups,
            rank_by_top_two,
        }) = self.spec.groups
        {
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

    /// The weights of the experts `choose` picks from `scores` `[tokens,
    /// experts]` — or, on a hash layer, the experts `ids`' rows of
    /// `tid2eid` name — zero elsewhere and renormalised when the family
    /// does; every chosen token is listed under its expert in `routed`.
    fn ranked(
        &self,
        scores: Tensor,
        routed: &mut [Vec<u32>],
        ids: Option<&[u32]>,
    ) -> candle_core::Result<Tensor> {
        let (tokens, count) = scores.dims2()?;
        let host = scores.to_vec2::<f32>()?;
        let mut mask = vec![0f32; tokens * count];
        for (token, row) in host.iter().enumerate() {
            let chosen: Vec<usize> = match &self.hash {
                Some(table) => {
                    let Some(&id) = ids.and_then(|ids| ids.get(token)) else {
                        return Err(refuse_candle(
                            &TOKEN_IDS_UNSET,
                            &[
                                ("given", &ids.map_or(0, <[u32]>::len).to_string()),
                                ("tokens", &tokens.to_string()),
                            ],
                        ));
                    };
                    let start = id as usize * self.spec.top_k;
                    let Some(row) = table.get(start..start + self.spec.top_k) else {
                        return Err(refuse_candle(
                            &TOKEN_UNROUTED,
                            &[
                                ("id", &id.to_string()),
                                ("rows", &(table.len() / self.spec.top_k).to_string()),
                            ],
                        ));
                    };
                    row.iter().map(|expert| *expert as usize).collect()
                }
                None => self.choose(row),
            };
            for expert in chosen {
                mask[token * count + expert] = 1.0;
                routed[expert].push(token as u32);
            }
        }
        let mask = Tensor::from_vec(mask, (tokens, count), scores.device())?;
        let chosen = (scores * mask)?;
        if self.spec.normalize {
            chosen.broadcast_div(&chosen.sum_keepdim(D::Minus1)?)
        } else {
            Ok(chosen)
        }
    }

    /// Routes every token of `hidden` `[batch, sequence, width]`.
    ///
    /// The choice of experts is a discrete decision read on the host; the
    /// weights stay tensors taken from the router's softmax through a 0/1
    /// mask, so a differentiable pass still carries the router's gradient
    /// back into the residual stream. Every op here — `softmax` composed of
    /// `exp` and `sum`, `index_select`, `index_add`, the mask multiply — has a
    /// backward pass.
    pub(in crate::model) fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        self.route(hidden, hidden, None)
    }

    /// [`Experts::forward`] with the ids of the tokens `hidden` holds, which
    /// a hash layer chooses its experts by.
    pub(in crate::model) fn forward_ids(
        &self,
        hidden: &Tensor,
        ids: Option<&[u32]>,
    ) -> candle_core::Result<Tensor> {
        self.route(hidden, hidden, ids)
    }

    /// [`Experts::forward`] with the router reading `router_input` rather
    /// than the experts' input (Gemma 4's router reads the residual through
    /// a scale-free norm, its experts through `pre_feedforward_layernorm_2`).
    pub(in crate::model::layer) fn forward_routed(
        &self,
        router_input: &Tensor,
        hidden: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.route(router_input, hidden, None)
    }

    fn route(
        &self,
        router_input: &Tensor,
        hidden: &Tensor,
        ids: Option<&[u32]>,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        let tokens = batch * sequence;
        let flat = hidden.reshape((tokens, width))?;
        let device = flat.device();
        // LongCat-Flash's identity experts follow the stored ones.
        let count = self.experts.len() + self.spec.identity_experts;
        let logits = self
            .router
            .forward(&router_input.reshape((tokens, router_input.dim(D::Minus1)?))?)?
            .to_dtype(DType::F32)?;
        let mut routed: Vec<Vec<u32>> = vec![Vec::new(); count];
        let weights = match self.spec.scoring {
            Scoring::Softmax => self.ranked(
                candle_nn::ops::softmax(&logits, D::Minus1)?,
                &mut routed,
                ids,
            )?,
            // sigmoid, composed so it has a backward pass.
            Scoring::Sigmoid => {
                self.ranked((logits.neg()?.exp()? + 1.0)?.recip()?, &mut routed, ids)?
            }
            Scoring::SparseMixer { jitter } => {
                sparse_mixer(&logits, self.spec.top_k, jitter, &mut routed)?
            }
            // softplus as `relu(x) + ln(1 + exp(-|x|))`, which no logit
            // overflows.
            Scoring::SqrtSoftplus => {
                let softplus = (logits.relu()? + (logits.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
                self.ranked(softplus.sqrt()?, &mut routed, ids)?
            }
        };
        let weights = match self.spec.routed_scale {
            Some(scale) => (weights * scale)?,
            None => weights,
        };
        let weights = match &self.expert_scales {
            Some(scales) => weights.broadcast_mul(scales)?,
            None => weights,
        };
        // K2-Horizon's value experts yield a value `intermediate` wide; every
        // other mixture yields the hidden width. Nemotron-H's latent experts
        // read and sum in `moe_latent_size`.
        let produced_width = if self.spec.layout == ExpertLayout::Mova {
            self.spec.intermediate
        } else {
            width
        };
        let routed_input = match &self.latent {
            Some((down, _, _)) => down.forward(&flat)?,
            None => flat.clone(),
        };
        let routed_width = routed_input.dim(D::Minus1)?;
        let mut output = Tensor::zeros(
            (
                tokens,
                if self.latent.is_some() {
                    routed_width
                } else {
                    produced_width
                },
            ),
            flat.dtype(),
            device,
        )?;
        for (expert, tokens) in routed.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let index = Tensor::new(tokens.as_slice(), device)?;
            let inputs = routed_input.index_select(&index, 0)?;
            let weight = weights
                .index_select(&index, 0)?
                .narrow(1, expert, 1)?
                .to_dtype(inputs.dtype())?;
            // Llama 4 weighs each expert's input; every other family its
            // output.
            let (inputs, weight) = if self.spec.weight_input {
                (inputs.broadcast_mul(&weight)?, None)
            } else {
                (inputs, Some(weight))
            };
            let produced = match self.experts.get(expert) {
                Some(stored) => stored.forward(&inputs, self.routed_activation, self.clamps.0)?,
                None => inputs,
            };
            let produced = match weight {
                Some(weight) => produced.broadcast_mul(&weight.to_dtype(produced.dtype())?)?,
                None => produced,
            };
            output = output.index_add(&index, &produced, 0)?;
        }
        if let Some((_, norm, up)) = &self.latent {
            // The composed norm, so a differentiable pass keeps its backward.
            if let Some(norm) = norm {
                output = norm.forward(&output, Pass::Differentiable)?;
            }
            output = up.forward(&output)?;
        }
        if let Some((shared, gate)) = &self.shared {
            let produced = shared.forward(&flat, self.activation, self.clamps.1)?;
            let produced = match gate {
                // sigmoid, composed so it has a backward pass.
                Some(gate) => {
                    produced.broadcast_mul(&(gate.forward(&flat)?.neg()?.exp()? + 1.0)?.recip()?)?
                }
                None => produced,
            };
            output = (output + produced)?;
            // Cohere2-MoE averages the routed and shared outputs.
            if self.spec.average_shared {
                output = (output / 2.0)?;
            }
        }
        output.reshape((batch, sequence, produced_width))
    }
}
