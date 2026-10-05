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
    Activation, ExpertGroups, ExpertLayout, MixtureOfExperts, Scoring, SharedExpert, SharedForm, SwigluLimit,
};

#[derive(Debug, Clone)]
struct Expert {
    /// `None` in Nemotron-H's experts, which are `down(act(up))`.
    gate: Option<Linear>,
    up: Linear,
    /// `None` in K2-Horizon's value experts, which are `act(up)`.
    down: Option<Linear>,
}

/// GPT-OSS's gate sharpness: the gate is `g · sigmoid(1.702 · g)`, the
/// sigmoid approximation of GELU it was trained with.
const GPT_OSS_GATE_SHARPNESS: f64 = 1.702;

/// One expert's clamped SwiGLU, resolved for its layer.
#[derive(Debug, Clone, Copy)]
enum Clamp {
    /// GPT-OSS's `(clamp(up, -limit, limit) + 1) · g · sigmoid(1.702 g)`
    /// where `g = min(gate, limit)`.
    GptOss(f64),
    /// Step 3.5's `min(act(gate), limit) · clamp(up, -limit, limit)`.
    Step(f64),
}

impl Expert {
    /// `down(act(gate) · up)`, or `down(act(up))` without a gate, with the
    /// gate and up clamped as `clamp` says.
    fn forward(
        &self,
        input: &Tensor,
        activation: Activation,
        clamp: Option<Clamp>,
    ) -> candle_core::Result<Tensor> {
        let up = self.up.forward(input)?;
        let gated = match (&self.gate, clamp) {
            (Some(gate), Some(Clamp::GptOss(limit))) => {
                let gate = gate.forward(input)?.minimum(limit)?;
                let up = (up.clamp(-limit, limit)? + 1.0)?;
                let sigmoid = ((gate.clone() * -GPT_OSS_GATE_SHARPNESS)?.exp()? + 1.0)?.recip()?;
                ((gate * sigmoid)? * up)?
            }
            (Some(gate), Some(Clamp::Step(limit))) => {
                let gate = activation.apply(&gate.forward(input)?)?.minimum(limit)?;
                (gate * up.clamp(-limit, limit)?)?
            }
            (Some(gate), None) => (activation.apply(&gate.forward(input)?)? * up)?,
            (None, _) => activation.apply(&up)?,
        };
        match &self.down {
            Some(down) => down.forward(&gated),
            None => Ok(gated),
        }
    }

    /// An expert of three unbiased projections named `[gate, up, down]`, or
    /// of `[up, down]` alone when `gate` is `None`.
    fn load(
        hidden: usize,
        intermediate: usize,
        gate: Option<&str>,
        [up, down]: [&str; 2],
        builder: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            gate: gate
                .map(|gate| linear_no_bias(hidden, intermediate, builder.pp(gate)))
                .transpose()?,
            up: linear_no_bias(hidden, intermediate, builder.pp(up))?,
            down: Some(linear_no_bias(intermediate, hidden, builder.pp(down))?),
        })
    }
}

#[derive(Debug, Clone)]
pub(in crate::model) struct Experts {
    router: Linear,
    /// DeepSeek-V3's `e_score_correction_bias`, `[experts]`, read on the
    /// host: it moves which experts are chosen, never how much they weigh.
    selection_bias: Option<Vec<f32>>,
    experts: Vec<Expert>,
    /// The expert every token goes through, and Qwen2-MoE's sigmoid gate
    /// that scales it (DeepSeek's shared experts are added as they are).
    shared: Option<(Expert, Option<Linear>)>,
    /// Gemma 4's `router.per_expert_scale`, `[1, experts]` in F32: each
    /// chosen expert's weight is multiplied by its own scale after the
    /// renormalisation.
    expert_scales: Option<Tensor>,
    spec: MixtureOfExperts,
    activation: Activation,
    /// The routed experts' and the shared expert's clamps on this layer.
    clamps: (Option<Clamp>, Option<Clamp>),
}

impl Experts {
    /// `builder` is the layer's; `layer` picks Step 3.5's clamps.
    pub(in crate::model) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: &MixtureOfExperts,
        activation: Activation,
        layer: usize,
    ) -> candle_core::Result<Self> {
        let count = spec.count;
        let intermediate = spec.intermediate;
        let (router, experts) = match spec.layout {
            ExpertLayout::Mixtral | ExpertLayout::Lfm2 => {
                let block = if spec.layout == ExpertLayout::Lfm2 {
                    builder.pp("feed_forward")
                } else {
                    builder.pp("block_sparse_moe")
                };
                let experts = (0..count)
                    .map(|expert| {
                        Expert::load(
                            hidden,
                            intermediate,
                            Some("w1"),
                            ["w3", "w2"],
                            block.pp("experts").pp(expert.to_string()),
                        )
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("gate"))?, experts)
            }
            ExpertLayout::Qwen
            | ExpertLayout::Jamba
            | ExpertLayout::HunYuan
            | ExpertLayout::NemotronH
            | ExpertLayout::HyV3 => {
                let (block, router) = match spec.layout {
                    ExpertLayout::Jamba => (builder.pp("feed_forward"), "router"),
                    ExpertLayout::HunYuan => (builder.pp("mlp"), "gate.wg"),
                    ExpertLayout::NemotronH => (builder.pp("mixer"), "gate"),
                    ExpertLayout::HyV3 => (builder.pp("mlp"), "router.gate"),
                    _ => (builder.pp("mlp"), "gate"),
                };
                // Nemotron-H's experts have no gate projection.
                let gate = (spec.layout != ExpertLayout::NemotronH).then_some("gate_proj");
                let stacked = block.pp("experts");
                // Transformers 5 saves every expert stacked: `gate_up_proj`
                // `[experts, 2 · width, hidden]`, gate rows first, and
                // `down_proj` `[experts, hidden, width]` (A.X-K1 and any
                // checkpoint it writes); each expert is a view of its slice.
                let experts = if gate.is_some() && stacked.contains_tensor("gate_up_proj") {
                    let gate_up = stacked.get((count, 2 * intermediate, hidden), "gate_up_proj")?;
                    let down = stacked.get((count, hidden, intermediate), "down_proj")?;
                    (0..count)
                        .map(|expert| -> candle_core::Result<Expert> {
                            let rows = gate_up.get(expert)?;
                            Ok(Expert {
                                gate: Some(Linear::new(rows.narrow(0, 0, intermediate)?, None)),
                                up: Linear::new(rows.narrow(0, intermediate, intermediate)?, None),
                                down: Some(Linear::new(down.get(expert)?, None)),
                            })
                        })
                        .collect::<candle_core::Result<Vec<_>>>()?
                } else {
                    (0..count)
                        .map(|expert| {
                            Expert::load(hidden, intermediate, gate, ["up_proj", "down_proj"], stacked.pp(expert.to_string()))
                        })
                        .collect::<candle_core::Result<Vec<_>>>()?
                };
                (linear_no_bias(hidden, count, block.pp(router))?, experts)
            }
            // Step3 stacks each projection of every expert in one tensor as a
            // projection stores it; each expert is a view of its slice.
            ExpertLayout::Step3 => {
                let block = builder.pp("moe");
                let gate = block.get((count, intermediate, hidden), "gate_proj.weight")?;
                let up = block.get((count, intermediate, hidden), "up_proj.weight")?;
                let down = block.get((count, hidden, intermediate), "down_proj.weight")?;
                let experts = (0..count)
                    .map(|expert| -> candle_core::Result<Expert> {
                        Ok(Expert {
                            gate: Some(Linear::new(gate.get(expert)?, None)),
                            up: Linear::new(up.get(expert)?, None),
                            down: Some(Linear::new(down.get(expert)?, None)),
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("gate"))?, experts)
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
                            gate: Some(Linear::new(rows.narrow(0, 0, intermediate)?, None)),
                            up: Linear::new(rows.narrow(0, intermediate, intermediate)?, None),
                            down: Some(Linear::new(output.get(expert)?, None)),
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("router").pp("layer"))?, experts)
            }
            // GPT-OSS stacks every expert inputs-first: `gate_up_proj`
            // `[experts, hidden, 2 * intermediate]` with gate and up columns
            // interleaved (gate at even columns), `down_proj` `[experts,
            // intermediate, hidden]`, each with a bias; the router has one
            // too. Each expert's projections are laid out once at load.
            ExpertLayout::GptOss => {
                let block = builder.pp("mlp");
                let experts_block = block.pp("experts");
                let gate_up = experts_block.get((count, hidden, 2 * intermediate), "gate_up_proj")?;
                let gate_up_bias = experts_block.get((count, 2 * intermediate), "gate_up_proj_bias")?;
                let down = experts_block.get((count, intermediate, hidden), "down_proj")?;
                let down_bias = experts_block.get((count, hidden), "down_proj_bias")?;
                let experts = (0..count)
                    .map(|expert| -> candle_core::Result<Expert> {
                        let paired = gate_up.get(expert)?.reshape((hidden, intermediate, 2))?;
                        let paired_bias = gate_up_bias.get(expert)?.reshape((intermediate, 2))?;
                        let column = |slot: usize| -> candle_core::Result<Linear> {
                            Ok(Linear::new(
                                paired.narrow(2, slot, 1)?.squeeze(2)?.t()?.contiguous()?,
                                Some(paired_bias.narrow(1, slot, 1)?.squeeze(1)?.contiguous()?),
                            ))
                        };
                        Ok(Expert {
                            gate: Some(column(0)?),
                            up: column(1)?,
                            down: Some(Linear::new(
                                down.get(expert)?.t()?.contiguous()?,
                                Some(down_bias.get(expert)?),
                            )),
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (candle_nn::linear(hidden, count, block.pp("router"))?, experts)
            }
            // DBRX stacks every expert's rows: `w1` (gate) and `v1` (up) are
            // `[experts · width, hidden]` as a projection stores them, and
            // `w2` is the same shape, so each expert's down projection is
            // its rows turned once at load.
            ExpertLayout::Dbrx => {
                let block = builder.pp("ffn");
                let mlp = block.pp("experts").pp("mlp");
                let stacked = |name: &str| mlp.get((count * intermediate, hidden), name);
                let (gate, up, down) = (stacked("w1")?, stacked("v1")?, stacked("w2")?);
                let experts = (0..count)
                    .map(|expert| -> candle_core::Result<Expert> {
                        let rows = |tensor: &Tensor| tensor.narrow(0, expert * intermediate, intermediate);
                        Ok(Expert {
                            gate: Some(Linear::new(rows(&gate)?, None)),
                            up: Linear::new(rows(&up)?, None),
                            down: Some(Linear::new(rows(&down)?.t()?.contiguous()?, None)),
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("router").pp("layer"))?, experts)
            }
            // Gemma 4 stacks every expert's gate rows above its up rows in
            // `experts.gate_up_proj` `[experts, 2 · width, hidden]` and keeps
            // `experts.down_proj` `[experts, hidden, width]`; each expert is a
            // view of its slice. Its router multiplies the normed input by
            // `router.scale` and by `hidden^-0.5` before `router.proj`; both
            // fold into the projection's columns once at load.
            ExpertLayout::Gemma4 => {
                let block = builder.pp("experts");
                let gate_up = block.get((count, 2 * intermediate, hidden), "gate_up_proj")?;
                let down = block.get((count, hidden, intermediate), "down_proj")?;
                let experts = (0..count)
                    .map(|expert| -> candle_core::Result<Expert> {
                        let rows = gate_up.get(expert)?;
                        Ok(Expert {
                            gate: Some(Linear::new(rows.narrow(0, 0, intermediate)?, None)),
                            up: Linear::new(rows.narrow(0, intermediate, intermediate)?, None),
                            down: Some(Linear::new(down.get(expert)?, None)),
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let router = builder.pp("router");
                let scale = (router.get(hidden, "scale")? * (hidden as f64).powf(-0.5))?;
                let projection = router.get((count, hidden), "proj.weight")?;
                let weight = projection.broadcast_mul(&scale.reshape((1, hidden))?)?;
                (Linear::new(weight, None), experts)
            }
            // K2-Horizon's value experts: `self_attn.v_router` and one
            // projection per expert, `self_attn.v_experts.{e}`, whose SiLU
            // output is the value.
            ExpertLayout::Mova => {
                let block = builder.pp("self_attn");
                let experts = (0..count)
                    .map(|expert| -> candle_core::Result<Expert> {
                        Ok(Expert {
                            gate: None,
                            up: linear_no_bias(hidden, intermediate, block.pp("v_experts").pp(expert.to_string()))?,
                            down: None,
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("v_router"))?, experts)
            }
        };
        let shared = match spec.shared {
            Some(SharedExpert { intermediate, module, gated, form }) => {
                let block = builder.pp(module);
                let expert = match form {
                    // Granite 4.0's `shared_mlp`: gate rows then up rows in
                    // `input_linear`, the down projection in `output_linear`.
                    SharedForm::Stacked => {
                        let input = block.get((2 * intermediate, hidden), "input_linear.weight")?;
                        Expert {
                            gate: Some(Linear::new(input.narrow(0, 0, intermediate)?, None)),
                            up: Linear::new(input.narrow(0, intermediate, intermediate)?, None),
                            down: Some(Linear::new(
                                block.get((hidden, intermediate), "output_linear.weight")?,
                                None,
                            )),
                        }
                    }
                    SharedForm::GateUpDown => Expert::load(
                        hidden,
                        intermediate,
                        Some("gate_proj"),
                        ["up_proj", "down_proj"],
                        block,
                    )?,
                    SharedForm::UpDown => {
                        Expert::load(hidden, intermediate, None, ["up_proj", "down_proj"], block)?
                    }
                };
                let gate = if gated {
                    Some(linear_no_bias(hidden, 1, builder.pp("mlp").pp("shared_expert_gate"))?)
                } else {
                    None
                };
                Some((expert, gate))
            }
            None => None,
        };
        // DeepSeek stores the bias as `[experts]`, ERNIE as `[1, experts]`;
        // either flattens to one score per expert.
        let selection_bias = match spec.selection_bias {
            Some(tensor) => {
                let bias = builder
                    .get_unchecked(tensor)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                if bias.len() != count {
                    candle_core::bail!(
                        "{tensor} holds {} scores for {count} experts",
                        bias.len()
                    );
                }
                Some(bias)
            }
            None => None,
        };
        let expert_scales = if spec.layout == ExpertLayout::Gemma4 {
            Some(
                builder
                    .pp("router")
                    .get(count, "per_expert_scale")?
                    .to_dtype(DType::F32)?
                    .reshape((1, count))?,
            )
        } else {
            None
        };
        // Step 3.5 states a limit per layer, zero meaning none.
        let at = |limits: &[f64]| limits.get(layer).copied().filter(|limit| *limit != 0.0).map(Clamp::Step);
        let clamps = match &spec.swiglu_limit {
            Some(SwigluLimit::GptOss(limit)) => (Some(Clamp::GptOss(*limit)), None),
            Some(SwigluLimit::Step { routed, shared }) => (at(routed), at(shared)),
            None => (None, None),
        };
        Ok(Self {
            router,
            selection_bias,
            experts,
            shared,
            expert_scales,
            spec: spec.clone(),
            activation,
            clamps,
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

    /// The weights of the experts `choose` picks from `scores` `[tokens,
    /// experts]`, zero elsewhere and renormalised when the family does; every
    /// chosen token is listed under its expert in `routed`.
    fn ranked(&self, scores: Tensor, routed: &mut [Vec<u32>]) -> candle_core::Result<Tensor> {
        let (tokens, count) = scores.dims2()?;
        let host = scores.to_vec2::<f32>()?;
        let mut mask = vec![0f32; tokens * count];
        for (token, row) in host.iter().enumerate() {
            for expert in self.choose(row) {
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
        self.forward_routed(hidden, hidden)
    }

    /// [`Experts::forward`] with the router reading `router_input` rather
    /// than the experts' input (Gemma 4's router reads the residual through
    /// a scale-free norm, its experts through `pre_feedforward_layernorm_2`).
    pub(super) fn forward_routed(&self, router_input: &Tensor, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        let tokens = batch * sequence;
        let flat = hidden.reshape((tokens, width))?;
        let device = flat.device();
        let count = self.experts.len();
        let logits = self
            .router
            .forward(&router_input.reshape((tokens, router_input.dim(D::Minus1)?))?)?
            .to_dtype(DType::F32)?;
        let mut routed: Vec<Vec<u32>> = vec![Vec::new(); count];
        let weights = match self.spec.scoring {
            Scoring::Softmax => self.ranked(candle_nn::ops::softmax(&logits, D::Minus1)?, &mut routed)?,
            // sigmoid, composed so it has a backward pass.
            Scoring::Sigmoid => self.ranked((logits.neg()?.exp()? + 1.0)?.recip()?, &mut routed)?,
            Scoring::SparseMixer { jitter } => {
                sparse_mixer(&logits, self.spec.top_k, jitter, &mut routed)?
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
        // other mixture yields the hidden width.
        let produced_width = if self.spec.layout == ExpertLayout::Mova { self.spec.intermediate } else { width };
        let mut output = Tensor::zeros((tokens, produced_width), flat.dtype(), device)?;
        for (expert, tokens) in routed.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let index = Tensor::new(tokens.as_slice(), device)?;
            let inputs = flat.index_select(&index, 0)?;
            let produced = self.experts[expert].forward(&inputs, self.activation, self.clamps.0)?;
            let weight = weights
                .index_select(&index, 0)?
                .narrow(1, expert, 1)?
                .to_dtype(produced.dtype())?;
            output = output.index_add(&index, &produced.broadcast_mul(&weight)?, 0)?;
        }
        if let Some((shared, gate)) = &self.shared {
            let produced = shared.forward(&flat, self.activation, self.clamps.1)?;
            let produced = match gate {
                // sigmoid, composed so it has a backward pass.
                Some(gate) => produced
                    .broadcast_mul(&(gate.forward(&flat)?.neg()?.exp()? + 1.0)?.recip()?)?,
                None => produced,
            };
            output = (output + produced)?;
        }
        output.reshape((batch, sequence, produced_width))
    }
}

/// PhiMoE's SparseMixer as Transformers' `sparsemixer` runs it outside
/// training. Each of `rounds` rounds takes the highest logit not yet chosen
/// (the first on a tie), drops the chosen experts and every expert whose
/// distance below it, divided by the larger of its own magnitude and the
/// best logit, exceeds `2 · jitter`, and weighs the chosen expert by its
/// softmax over the rest. The choice is read on the host; each weight is a
/// softmax of the logits through masks, so the router keeps its gradient.
fn sparse_mixer(
    logits: &Tensor,
    rounds: usize,
    jitter: f32,
    routed: &mut [Vec<u32>],
) -> candle_core::Result<Tensor> {
    let (tokens, count) = logits.dims2()?;
    let host = logits.to_vec2::<f32>()?;
    // Per round: an additive mask, zero for an expert kept and minus
    // infinity for one dropped, and the chosen expert marked with a one.
    let mut dropped = vec![vec![0f32; tokens * count]; rounds];
    let mut picked = vec![vec![0f32; tokens * count]; rounds];
    for (token, row) in host.iter().enumerate() {
        let mut taken = vec![false; count];
        for (drop, pick) in dropped.iter_mut().zip(picked.iter_mut()) {
            let Some(best) = (0..count)
                .filter(|expert| !taken[*expert])
                .reduce(|kept, next| if row[next] > row[kept] { next } else { kept })
            else {
                break;
            };
            let top = row[best];
            for (expert, logit) in row.iter().enumerate() {
                if taken[expert] || (top - logit) / logit.abs().max(top) > 2.0 * jitter {
                    drop[token * count + expert] = f32::NEG_INFINITY;
                }
            }
            pick[token * count + best] = 1.0;
            routed[best].push(token as u32);
            taken[best] = true;
        }
    }
    let device = logits.device();
    let mut weights = logits.zeros_like()?;
    for (drop, pick) in dropped.into_iter().zip(picked) {
        let drop = Tensor::from_vec(drop, (tokens, count), device)?;
        let pick = Tensor::from_vec(pick, (tokens, count), device)?;
        let gates = candle_nn::ops::softmax(&(logits + drop)?, D::Minus1)?;
        weights = (weights + (gates * pick)?)?;
    }
    Ok(weights)
}
