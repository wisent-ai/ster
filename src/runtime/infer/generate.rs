//! Sampling a continuation, with a steering artifact applied or not, and the
//! exact token sequence the model saw while producing it.

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::{
    artifact::SteeringArtifact,
    model::SteeringPlan,
};

use super::super::{Runtime, validate_layers};

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    /// Scale on the steering vector. It belongs to a vector: a run without
    /// one carries none, and a run with one is refused without it.
    pub strength: Option<f64>,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
}

/// What one sampling call produced.
///
/// The two token vectors concatenate to the exact sequence the model saw, and
/// `prompt.len()` is the boundary a completion-only loss scores from — which
/// is why the prompt travels back out rather than being re-derived: a caller
/// that tokenized the prompt itself would be trusting two encodes to agree.
#[derive(Debug, Clone)]
pub struct Completion {
    pub prompt: Vec<u32>,
    pub tokens: Vec<u32>,
    pub text: String,
}

impl Runtime {
    /// One sampled continuation, decoded.
    ///
    /// Everything here is [`Runtime::sample`]; only the text survives, which
    /// is what every caller outside policy optimization wants.
    pub fn generate(
        &self,
        prompt: &str,
        artifact: Option<&SteeringArtifact>,
        options: GenerationOptions,
    ) -> Result<String> {
        Ok(self.sample(prompt, artifact, options)?.text)
    }

    /// One sampled continuation: the prompt as the sampler tokenized it, the
    /// tokens drawn after it, and their text.
    ///
    /// Policy optimization needs all three. It has to score the exact sequence
    /// the policy produced, and decoding to text and re-encoding would not
    /// reliably give that sequence back — a tokenizer is not injective over
    /// its own output. Handing back the ids the sampler actually pushed makes
    /// the scored sequence the sampled sequence by construction.
    pub fn sample(
        &self,
        prompt: &str,
        artifact: Option<&SteeringArtifact>,
        options: GenerationOptions,
    ) -> Result<Completion> {
        // The same mismatch that ruins training ruins decoding: an instruct
        // checkpoint handed a bare prompt continues the text instead of
        // answering it. Under a template the prompt becomes a user turn
        // followed by the marker that opens the assistant's, which is the
        // context the model was post-trained to answer from.
        let tokens = self.encode_prompt(prompt)?;
        self.sample_tokens(tokens, artifact, options)
    }

    /// One sampled continuation of a context that is already tokenized.
    ///
    /// A conversation of several turns is rendered and tokenized by the
    /// caller, because only the caller knows which turns it holds; the
    /// sampling and the returned split are exactly [`Runtime::sample`]'s.
    pub fn sample_tokens(
        &self,
        mut tokens: Vec<u32>,
        artifact: Option<&SteeringArtifact>,
        options: GenerationOptions,
    ) -> Result<Completion> {
        if options.max_new_tokens == 0 {
            bail!("max_new_tokens must be greater than zero");
        }
        if tokens.len() >= self.model.config().max_position_embeddings {
            bail!(
                "prompt contains {} tokens, model context allows fewer than {}",
                tokens.len(),
                self.model.config().max_position_embeddings
            );
        }
        let prompt_len = tokens.len();
        let plan = match artifact {
            Some(artifact) => {
                artifact.validate()?;
                if artifact.model != self.model_id {
                    bail!(
                        "artifact was trained for model {:?}, current model is {:?}",
                        artifact.model,
                        self.model_id
                    );
                }
                if artifact.hidden_size != self.hidden_size() {
                    bail!(
                        "artifact width {} does not match model width {}",
                        artifact.hidden_size,
                        self.hidden_size()
                    );
                }
                validate_layers(
                    &artifact
                        .vectors
                        .iter()
                        .map(|vector| vector.layer)
                        .collect::<Vec<_>>(),
                    self.layer_count(),
                )?;
                let strength = options
                    .strength
                    .context("a steering vector needs a strength; Ster assumes none")?;
                Some(SteeringPlan::new(
                    artifact
                        .vectors
                        .iter()
                        .map(|vector| (vector.layer, vector.values.clone())),
                    strength,
                    self.hidden_size(),
                    &self.device,
                    self.dtype,
                )?)
            }
            None => None,
        };
        let sampling = if options.temperature <= 0.0 {
            Sampling::ArgMax
        } else if let Some(top_p) = options.top_p {
            Sampling::TopP {
                p: top_p,
                temperature: options.temperature,
            }
        } else {
            Sampling::All {
                temperature: options.temperature,
            }
        };
        let mut sampler = LogitsProcessor::from_sampling(options.seed, sampling);
        let mut cache = self.cache(true)?;
        for step in 0..options.max_new_tokens {
            let (context, index_pos) = if step == 0 {
                (tokens.clone(), 0)
            } else {
                (
                    vec![*tokens.last().expect("tokens are non-empty")],
                    tokens.len() - 1,
                )
            };
            let input = Tensor::new(context.as_slice(), &self.device)?.unsqueeze(0)?;
            let output = self
                .model
                .forward(&input, index_pos, &mut cache, plan.as_ref(), &[])?;
            // `Mode::DECODE` always asks for the last position's logits, so
            // this is the one readout that cannot be absent.
            let logits = output
                .logits
                .context("the decode pass produced no logits")?;
            let next = sampler.sample(&logits.squeeze(0)?)?;
            tokens.push(next);
            if self.eos_tokens.contains(&next) {
                break;
            }
            if tokens.len() >= self.model.config().max_position_embeddings {
                break;
            }
        }
        let text = self
            .tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(|error| anyhow::anyhow!("failed to decode generated tokens: {error}"))?;
        let sampled = tokens.split_off(prompt_len);
        Ok(Completion {
            prompt: tokens,
            tokens: sampled,
            text,
        })
    }
}
