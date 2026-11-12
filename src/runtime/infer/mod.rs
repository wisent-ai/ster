//! Running the model: one forward pass per shape a caller needs, and the
//! hidden states a steering direction is read from.

use anyhow::Result;
use candle_core::Tensor;

use crate::model::{Cache, ForwardOutput, Mode, Network, Route, SteeringLlama, SteeringPlan};

use super::{Runtime, requested_activations, validate_layers};

mod encoder_decoder;
mod generate;
mod positions;
mod rows;
mod steer;

pub use generate::{Completion, GenerationOptions};

/// What reaches the whole-sequence passes below, named in the refusal a T5
/// checkpoint gets from them.
const WHOLE_SEQUENCE: &str = "a whole-sequence pass (ster tune sft, dpo, grpo, reward, decide, bipo and evaluate, ster vector evaluate --strengths, and ster parity)";

impl Runtime {
    /// One differentiable forward over `ids`, returning logits `[1, n, vocab]`.
    pub fn forward_train(&self, ids: &[u32]) -> Result<Tensor> {
        self.logits(
            ids,
            Mode::TRAIN,
            "a training forward pass needs at least one token",
        )
    }

    /// One non-differentiable forward over `ids`, returning logits `[1, n, vocab]`.
    ///
    /// `route` picks which model answers. [`Route::Adapted`] is the policy;
    /// [`Route::Base`] is the frozen reference — the same mapped weights with
    /// the low-rank update skipped, which is precisely the model the adapters
    /// started as, since `B` is zeros before the first step. Preference
    /// optimization gets its reference log-probabilities this way rather than
    /// by loading a second copy of the checkpoint.
    ///
    /// Nothing here is backpropagated, so it takes the fused kernels. A caller
    /// holding a trainable runtime must not route a *policy* score through it:
    /// the adapter variables would record a graph whose rope, softmax and
    /// norm nodes have no backward pass. That caller wants `forward_train`.
    pub fn forward_scored(&self, ids: &[u32], route: Route) -> Result<Tensor> {
        self.logits(
            ids,
            Mode::score(route),
            "a scoring forward pass needs at least one token",
        )
    }

    /// One differentiable forward over `ids`, returning the residual stream
    /// after the final norm, `[1, n, hidden]`, and no vocabulary projection.
    ///
    /// This is what a reward head reads. Skipping the vocabulary matmul is not
    /// a micro-optimization on a real checkpoint: it is the widest matmul in
    /// the pass, and a reward run would compute and backpropagate all of it
    /// only to throw the result away.
    pub fn forward_hidden(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .forward_once(
                ids,
                Mode::REWARD,
                "a reward forward pass needs at least one token",
            )?
            .hidden)
    }

    /// The same residual stream with no autograd tape, for a reward model that
    /// is judging rather than being trained.
    ///
    /// A reward model inside a policy-optimization loop is frozen by
    /// definition — a moving judge is a moving target — so it takes the fused
    /// kernels and records nothing.
    pub fn forward_hidden_scored(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .forward_once(
                ids,
                Mode::JUDGE,
                "a scoring forward pass needs at least one token",
            )?
            .hidden)
    }

    /// The next-token distribution after each of several continuations of
    /// one shared prefix: `suffixes.len()` vectors of `vocab` F32 logits.
    ///
    /// This is what a decision reads — not a sampled continuation but the
    /// distribution the model holds at the end of a prompt, before any token
    /// is drawn — and it is read this way because every question about one
    /// state shares that state. The prefix goes through the model once with
    /// the key-value cache on; each suffix then runs on its own copy of that
    /// cache from position `prefix.len()`, so a question costs its own tokens
    /// and never the state's again. The copy is a refcount, not a memcpy:
    /// appending to the cache builds new tensors and leaves the prefix's
    /// untouched, which is what lets one prefix serve every suffix.
    ///
    /// An empty prefix is allowed and means every suffix is a whole prompt;
    /// an empty suffix is not, since there would be no position of its own
    /// to read.
    pub fn next_token_logits_after(
        &self,
        prefix: &[u32],
        suffixes: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        // Every caller encodes each suffix through a path that refuses a
        // text the tokenizer turns into no tokens.
        if suffixes.is_empty() || suffixes.iter().any(|suffix| suffix.is_empty()) {
            let lengths: Vec<usize> = suffixes.iter().map(|suffix| suffix.len()).collect();
            return Err(empty_rows("a decision forward pass", &lengths));
        }
        let model = match &self.model {
            Network::Causal(model) => model,
            Network::EncoderDecoder(model) => {
                return self.next_token_logits_encoder_decoder(model, prefix, suffixes);
            }
        };
        let mut shared = self.cache(model, true)?;
        if !prefix.is_empty() {
            let input = Tensor::new(prefix, &self.device)?.unsqueeze(0)?;
            model.forward(&input, 0, &mut shared, None, &[])?;
        }
        suffixes
            .iter()
            .map(|suffix| {
                let mut cache = shared.clone();
                let input = Tensor::new(*suffix, &self.device)?.unsqueeze(0)?;
                let logits = model
                    .forward(&input, prefix.len(), &mut cache, None, &[])?
                    .into_logits(Mode::DECODE.readout)?;
                Ok(logits.squeeze(0)?.to_vec1::<f32>()?)
            })
            .collect()
    }

    /// A fresh decode state for the causal `model`: its rotary frequencies,
    /// one key-value slot and one recurrent-state slot per layer.
    pub(super) fn cache(
        &self,
        model: &SteeringLlama,
        use_kv_cache: bool,
    ) -> candle_core::Result<Cache> {
        Cache::new(
            use_kv_cache,
            self.dtype,
            model.config(),
            model.architecture(),
            &self.device,
        )
    }

    /// A forward that must produce logits, unwrapped.
    fn logits(&self, ids: &[u32], mode: Mode, empty: &str) -> Result<Tensor> {
        self.forward_once(ids, mode, empty)?
            .into_logits(mode.readout)
    }

    /// The body every whole-sequence forward shares: one sequence, no KV cache.
    ///
    /// The cache stays off because the whole sequence goes through in one pass,
    /// so there is nothing to reuse, and because a cache would keep the previous
    /// sequence's keys and values alive inside this one's autograd graph — the
    /// backward pass would then walk tensors that no longer correspond to the
    /// input being scored.
    fn forward_once(&self, ids: &[u32], mode: Mode, empty: &str) -> Result<ForwardOutput> {
        // Every caller hands ids an encode produced, and encoding refuses a
        // text the tokenizer turns into no tokens.
        if ids.is_empty() {
            return Err(empty_rows(empty, &[0]));
        }
        let model = self.model.causal(WHOLE_SEQUENCE)?;
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mut cache = self.cache(model, false)?;
        Ok(model.forward_pass(&input, 0, &mut cache, None, &[], mode)?)
    }

    /// The hidden states one prompt produces at the requested layers.
    ///
    /// This is the read behind `train`, `optimize`, `evaluate` and `extract`,
    /// and it encodes through [`Runtime::encode_prompt`] for a reason that is
    /// not symmetry. A direction is a displacement between two points in the
    /// residual stream, and where those points sit depends on the markers
    /// around the text that produced them: under a template the model is
    /// answering a user turn, without one it is continuing a document. Fitting
    /// a direction from raw pair text and then adding it during a templated
    /// decode measures one space and steers another.
    pub fn activations(&self, prompt: &str, layers: &[usize]) -> Result<Vec<(usize, Vec<f32>)>> {
        self.activations_steered(prompt, layers, None)
    }

    /// [`Runtime::activations`] read while `steering` is added to the
    /// residual stream, so a caller can see where a steered run's states
    /// land beside the unsteered ones. `None` is the unsteered read itself.
    pub fn activations_steered(
        &self,
        prompt: &str,
        layers: &[usize],
        steering: Option<&SteeringPlan>,
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        validate_layers(layers, self.layer_count())?;
        let ids = self.encode_prompt(prompt)?;
        let model = match &self.model {
            Network::Causal(model) => model,
            Network::EncoderDecoder(model) => {
                return self.activations_encoder_decoder(model, &ids, layers, steering);
            }
        };
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut cache = self.cache(model, false)?;
        let output = model.forward(&input, 0, &mut cache, steering, layers)?;
        requested_activations(output.activations, layers)
    }

    /// One prompt, tokenized the way this run has decided prompts are
    /// tokenized: as a user turn the assistant is about to answer, or as raw
    /// text when there is no template or the operator turned it off.
    ///
    /// The rendered string already spells every marker the model expects, so
    /// the tokenizer is asked not to add its own — a second begin-of-sequence
    /// would be a token no inference path ever produces.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        match self.applied_template() {
            Some(template) => self.tokenize(&template.prompt(prompt)?, false, "prompt"),
            None => self.encode(prompt),
        }
    }
}

/// The refusal of `pass` handed rows of `lengths`, one of them empty or none
/// at all, which no encode produces.
fn empty_rows(pass: &str, lengths: &[usize]) -> anyhow::Error {
    crate::failures::refuse(
        &crate::failures::model::decoder::ROWS_EMPTY,
        &[("pass", pass), ("lengths", &format!("{lengths:?}"))],
    )
}
