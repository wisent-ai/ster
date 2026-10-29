//! The runtime operations a T5 encoder-decoder model runs. The prompt is the
//! encoder's source; the decoder starts from the checkpoint's
//! `decoder_start_token_id`. The decoder state at that start token is the one
//! that writes the answer's first token, so it is where a direction is read,
//! where steering is added, and what a decision's distribution comes from.

use std::num::NonZeroUsize;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;

use crate::model::{ForwardOutput, Mode, SteeringPlan, input::Inputs, t5};

use super::super::{Runtime, requested_activations};
use super::Completion;

impl Runtime {
    /// One sampled answer to `source`: the decoder starts from its start
    /// token, reads the encoded source through cross-attention, and keeps
    /// its own keys and values between steps.
    pub(super) fn sample_encoder_decoder(
        &self,
        model: &t5::Model,
        source: Vec<u32>,
        plan: Option<&SteeringPlan>,
        max_new_tokens: usize,
        sampler: &mut LogitsProcessor,
    ) -> Result<Completion> {
        let config = model.config();
        self.check_source(model, &source)?;
        // The decoder holds its start token and every sampled token but the
        // last, one position per step.
        let steps = NonZeroUsize::new(max_new_tokens)
            .context("max_new_tokens must be greater than zero")?;
        let mut cache = t5::Cache::new(config.decoder_layers, Some(steps));
        let encoder = Tensor::new(source.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut sampled = Vec::with_capacity(max_new_tokens);
        let mut next_input = config.decoder_start;
        for step in 0..max_new_tokens {
            let decoder = Tensor::new(&[next_input], &self.device)?.unsqueeze(0)?;
            // The source is encoded once, on the first step; later steps read
            // the encoder memory the cache holds.
            let inputs = Inputs {
                encoder: (step == 0).then_some(&encoder),
                encoder_lengths: None,
                decoder: &decoder,
                decoder_lengths: None,
            };
            let output = model.forward(inputs, step, &mut cache, plan, &[], Mode::DECODE)?;
            // `Mode::DECODE` always asks for the last position's logits.
            let logits = output
                .logits
                .context("the decode pass produced no logits")?;
            let next = sampler.sample(&logits.squeeze(0)?)?;
            sampled.push(next);
            if self.eos_tokens.contains(&next) {
                break;
            }
            // The next step's input would occupy decoder position `step + 1`,
            // past the last one `n_positions` allows.
            if config.context_length.is_some_and(|limit| step + 1 >= limit) {
                break;
            }
            next_input = next;
        }
        let text = self
            .tokenizer
            .decode(&sampled, true)
            .map_err(|error| anyhow::anyhow!("failed to decode generated tokens: {error}"))?;
        Ok(Completion {
            prompt: source,
            tokens: sampled,
            text,
        })
    }

    /// The decoder's hidden states at `layers` for its start token, after
    /// reading `source`, with `steering` added when given.
    pub(super) fn activations_encoder_decoder(
        &self,
        model: &t5::Model,
        source: &[u32],
        layers: &[usize],
        steering: Option<&SteeringPlan>,
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        let output = self.first_decoder_step(model, source, steering, layers)?;
        requested_activations(output.activations, layers)
    }

    /// The first answer token's distribution after each of `prefix` followed
    /// by one of `suffixes`. T5's encoder attends in both directions, so a
    /// suffix changes how every prefix token is encoded, and each source is
    /// encoded whole rather than continuing a shared prefix's state.
    pub(super) fn next_token_logits_encoder_decoder(
        &self,
        model: &t5::Model,
        prefix: &[u32],
        suffixes: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        suffixes
            .iter()
            .map(|suffix| {
                let source: Vec<u32> = prefix.iter().chain(suffix.iter()).copied().collect();
                let logits = self
                    .first_decoder_step(model, &source, None, &[])?
                    .logits
                    .context("this forward pass was asked for no vocabulary projection")?;
                Ok(logits.squeeze(0)?.to_vec1::<f32>()?)
            })
            .collect()
    }

    /// One pass of the decoder's start token over the encoded `source`.
    fn first_decoder_step(
        &self,
        model: &t5::Model,
        source: &[u32],
        steering: Option<&SteeringPlan>,
        captures: &[usize],
    ) -> Result<ForwardOutput> {
        self.check_source(model, source)?;
        let config = model.config();
        let encoder = Tensor::new(source, &self.device)?.unsqueeze(0)?;
        let decoder = Tensor::new(&[config.decoder_start], &self.device)?.unsqueeze(0)?;
        // No later step reads this pass's decoder keys, so none are kept.
        let mut cache = t5::Cache::new(config.decoder_layers, None);
        let inputs = Inputs {
            encoder: Some(&encoder),
            encoder_lengths: None,
            decoder: &decoder,
            decoder_lengths: None,
        };
        model.forward(inputs, 0, &mut cache, steering, captures, Mode::DECODE)
    }

    /// Refuses a source the encoder cannot read: none at all, or longer than
    /// the `n_positions` the checkpoint states.
    fn check_source(&self, model: &t5::Model, source: &[u32]) -> Result<()> {
        if source.is_empty() {
            bail!("a T5 prompt needs at least one token for its encoder to read");
        }
        if let Some(limit) = model.config().context_length {
            if source.len() > limit {
                bail!(
                    "prompt contains {} tokens, past the {limit} positions (n_positions) this T5 model's encoder was built for",
                    source.len()
                );
            }
        }
        Ok(())
    }
}
