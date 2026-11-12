//! A mixture-of-experts feed-forward: a router scores every expert for every
//! token, the top `k` run on that token, and their outputs are summed by the
//! router's weights.
//!
//! Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE and GraniteMoE all route this way and
//! differ in tensor names, in whether the chosen weights are renormalised to
//! sum to one, and in whether a shared expert runs on every token beside the
//! routed ones (Qwen2-MoE).

pub(super) mod expert;
mod layouts;
mod route;
mod sparse_mixer;

use candle_core::{DType, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};

use super::norm::{Norm, NormSpec};
use crate::model::{
    Activation, ExpertLayout, LatentExperts, MixtureOfExperts, NormKind, SharedExpert, SharedForm,
    SwigluLimit,
};
use expert::{Clamp, Expert, GPT_OSS_GATE_SHARPNESS};

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
    /// The shared expert's activation, the model's `hidden_act`.
    activation: Activation,
    /// The routed experts' activation: the model's, unless the family
    /// states its own (HY-V4's SwiGLU experts).
    routed_activation: Activation,
    /// The routed experts' and the shared expert's clamps on this layer.
    clamps: (Option<Clamp>, Option<Clamp>),
    /// The latent experts' projection down, the norm over their sum, and the
    /// projection back.
    latent: Option<(Linear, Option<Norm>, Linear)>,
    /// DeepSeek-V4's hash routing on this layer: every token id's `top_k`
    /// experts, `tid2eid` flattened row by row.
    hash: Option<Vec<u32>>,
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
        let (router, experts) = layouts::stored(builder, hidden, spec)?;
        let shared = match spec.shared {
            Some(SharedExpert {
                intermediate,
                module,
                gated,
                form,
            }) => {
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
                    Some(linear_no_bias(
                        hidden,
                        1,
                        builder.pp("mlp").pp("shared_expert_gate"),
                    )?)
                } else {
                    None
                };
                Some((expert, gate))
            }
            None => None,
        };
        // DeepSeek stores the bias as `[experts]`, ERNIE as `[1, experts]`;
        // either flattens to one score per expert. A hash layer chooses by
        // token id and stores none.
        let selection_bias = match spec
            .selection_bias
            .filter(|_| spec.hash_layers >> layer & 1 == 0)
        {
            Some(tensor) => {
                let stored = builder.get_unchecked(tensor)?;
                let bias = stored
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let routes = count + spec.identity_experts;
                if bias.len() != routes {
                    return Err(crate::model::misshapen(tensor, (routes,), stored.shape()));
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
        let at = |limits: &[f64]| {
            limits
                .get(layer)
                .copied()
                .filter(|limit| *limit != 0.0)
                .map(Clamp::Step)
        };
        let clamps = match &spec.swiglu_limit {
            Some(SwigluLimit::GptOss(limit)) => (
                Some(Clamp::GptOss {
                    limit: *limit,
                    alpha: GPT_OSS_GATE_SHARPNESS,
                }),
                None,
            ),
            Some(SwigluLimit::Oai { limit, alpha }) => {
                let clamp = Clamp::GptOss {
                    limit: *limit,
                    alpha: *alpha,
                };
                (Some(clamp), Some(clamp))
            }
            Some(SwigluLimit::Step { routed, shared }) => (at(routed), at(shared)),
            Some(SwigluLimit::Inner(limit)) => {
                (Some(Clamp::Inner(*limit)), Some(Clamp::Inner(*limit)))
            }
            Some(SwigluLimit::RoutedInner(limit)) => (Some(Clamp::Inner(*limit)), None),
            None => (None, None),
        };
        // Latent projections sit beside the experts, with a bias when the
        // checkpoint stores one (Nemotron-H's `mlp_bias`).
        let latent = match spec.latent {
            Some(LatentExperts {
                width,
                down,
                up,
                norm,
            }) => {
                let projection =
                    |input: usize, output: usize, name: &str| -> candle_core::Result<Linear> {
                        let module = builder.pp(name);
                        let bias = if module.contains_tensor("bias") {
                            Some(module.get(output, "bias")?)
                        } else {
                            None
                        };
                        Ok(Linear::new(module.get((output, input), "weight")?, bias))
                    };
                let norm = match norm {
                    Some((name, eps)) => Some(
                        NormSpec {
                            kind: NormKind::Rms,
                            eps,
                            offset: false,
                            groups: 1,
                        }
                        .load(width, builder.pp(name))?,
                    ),
                    None => None,
                };
                Some((
                    projection(hidden, width, down)?,
                    norm,
                    projection(width, hidden, up)?,
                ))
            }
            None => None,
        };
        let hash = if spec.hash_layers >> layer & 1 == 1 {
            let block = builder.pp(if spec.layout == ExpertLayout::DeepseekV4 {
                "mlp"
            } else {
                "block_sparse_moe"
            });
            let table = block.pp("gate").get_unchecked("tid2eid")?;
            let (rows, width) = table.dims2()?;
            if width != spec.top_k {
                let name = format!("layer {layer}'s mlp.gate.tid2eid");
                return Err(crate::model::misshapen(
                    &name,
                    (rows, spec.top_k),
                    table.shape(),
                ));
            }
            let flat = table
                .to_dtype(DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?;
            if let Some((slot, bad)) = flat
                .iter()
                .enumerate()
                .find(|(_, expert)| !(0..count as i64).contains(*expert))
            {
                return Err(crate::failures::refuse_candle(
                    &crate::failures::model::decoder::ROUTE_OUTSIDE,
                    &[
                        ("layer", &layer.to_string()),
                        ("token", &(slot / width).to_string()),
                        ("expert", &bad.to_string()),
                        ("count", &count.to_string()),
                    ],
                ));
            }
            Some(flat.into_iter().map(|expert| expert as u32).collect())
        } else {
            None
        };
        Ok(Self {
            router,
            selection_bias,
            experts,
            shared,
            expert_scales,
            spec: spec.clone(),
            activation,
            routed_activation: match spec.routed_activation {
                Some(stated) => stated,
                None => activation,
            },
            clamps,
            latent,
            hash,
        })
    }
}
