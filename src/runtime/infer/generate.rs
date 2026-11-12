//! Sampling a continuation, with steering artifacts applied or not, and the
//! exact token sequence the model saw while producing it.

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

use crate::{
    artifact::SteeringArtifact,
    failures::{
        refuse,
        runtime::generate::{NEW_TOKENS_NONE, PROMPT_TOO_LONG, UNSPELLABLE},
    },
    model::{Mode, Network, SteeringPlan},
};

use super::super::Runtime;

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
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
    /// One unsteered sampled continuation, decoded.
    ///
    /// Everything here is [`Runtime::sample`]; only the text survives, which
    /// is what every caller outside policy optimization wants.
    pub fn generate(&self, prompt: &str, options: GenerationOptions) -> Result<String> {
        Ok(self.sample(prompt, options)?.text)
    }

    /// One unsteered sampled continuation: the prompt as the sampler
    /// tokenized it, the tokens drawn after it, and their text.
    ///
    /// Policy optimization needs all three. It has to score the exact sequence
    /// the policy produced, and decoding to text and re-encoding would not
    /// reliably give that sequence back — a tokenizer is not injective over
    /// its own output. Handing back the ids the sampler actually pushed makes
    /// the scored sequence the sampled sequence by construction.
    pub fn sample(&self, prompt: &str, options: GenerationOptions) -> Result<Completion> {
        // The same mismatch that ruins training ruins decoding: an instruct
        // checkpoint handed a bare prompt continues the text instead of
        // answering it. Under a template the prompt becomes a user turn
        // followed by the marker that opens the assistant's, which is the
        // context the model was post-trained to answer from.
        let tokens = self.encode_prompt(prompt)?;
        self.sample_tokens(tokens, options)
    }

    /// One unsteered sampled continuation of a context that is already
    /// tokenized.
    ///
    /// A conversation of several turns is rendered and tokenized by the
    /// caller, because only the caller knows which turns it holds; the
    /// sampling and the returned split are exactly [`Runtime::sample`]'s.
    pub fn sample_tokens(&self, tokens: Vec<u32>, options: GenerationOptions) -> Result<Completion> {
        self.sample_planned(tokens, None, options)
    }

    /// One sampled continuation with every artifact in `parts` added at its
    /// own strength — several traits steered at once. Each artifact passes
    /// the checks a single one does. No parts samples unsteered.
    pub fn sample_tokens_mixed(
        &self,
        tokens: Vec<u32>,
        parts: &[(&SteeringArtifact, f64)],
        options: GenerationOptions,
    ) -> Result<Completion> {
        let plan = self.steering_plan_mixed(parts)?;
        self.sample_planned(tokens, plan.as_ref(), options)
    }

    /// [`Runtime::sample_tokens_mixed`] for one prompt, encoded the way this
    /// run encodes prompts.
    pub fn generate_mixed(
        &self,
        prompt: &str,
        parts: &[(&SteeringArtifact, f64)],
        options: GenerationOptions,
    ) -> Result<String> {
        let tokens = self.encode_prompt(prompt)?;
        Ok(self.sample_tokens_mixed(tokens, parts, options)?.text)
    }

    fn sample_planned(
        &self,
        mut tokens: Vec<u32>,
        plan: Option<&SteeringPlan>,
        options: GenerationOptions,
    ) -> Result<Completion> {
        if options.max_new_tokens == 0 {
            return Err(refuse(
                &NEW_TOKENS_NONE,
                &[
                    ("tokens", &options.max_new_tokens.to_string()),
                    ("prompt", &tokens.len().to_string()),
                    ("model", &self.model_id),
                ],
            ));
        }
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
        let model = match &self.model {
            Network::Causal(model) => model,
            // The prompt is T5's encoder source; the answer is its decoder's.
            Network::EncoderDecoder(model) => {
                return self.sample_encoder_decoder(
                    model,
                    tokens,
                    plan,
                    options.max_new_tokens,
                    &mut sampler,
                );
            }
        };
        let context_length = model.config().max_position_embeddings;
        if tokens.len() >= context_length {
            return Err(refuse(
                &PROMPT_TOO_LONG,
                &[
                    ("tokens", &tokens.len().to_string()),
                    ("context", &context_length.to_string()),
                ],
            ));
        }
        let prompt_len = tokens.len();
        let mut cache = self.cache(model, true)?;
        // The first step reads the whole prompt; each later one reads only
        // the token the step before it sampled, at its position.
        let (mut context, mut index_pos) = (tokens.clone(), 0);
        for _ in 0..options.max_new_tokens {
            let input = Tensor::new(context.as_slice(), &self.device)?.unsqueeze(0)?;
            let output = model.forward(&input, index_pos, &mut cache, plan, &[])?;
            // `SteeringLlama::forward` runs `Mode::DECODE`.
            let logits = output.into_logits(Mode::DECODE.readout)?;
            let next = sampler.sample(&logits.squeeze(0)?)?;
            tokens.push(next);
            (context, index_pos) = (vec![next], tokens.len() - 1);
            if self.eos_tokens.contains(&next) {
                break;
            }
            if tokens.len() >= context_length {
                break;
            }
        }
        let text = self.spell(&tokens[prompt_len..])?;
        let sampled = tokens.split_off(prompt_len);
        Ok(Completion {
            prompt: tokens,
            tokens: sampled,
            text,
        })
    }

    /// `tokens` as text, special tokens left out.
    pub(super) fn spell(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, true).map_err(|error| {
            refuse(
                &UNSPELLABLE,
                &[
                    ("tokens", &format!("{tokens:?}")),
                    ("vocabulary", &self.tokenizer.get_vocab_size(true).to_string()),
                    ("error", &error.to_string()),
                ],
            )
        })
    }
}
