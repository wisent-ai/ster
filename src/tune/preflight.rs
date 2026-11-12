//! What every trainer checks and builds before its first step: the runtime
//! and its adapters, the optimizer over exactly what the run created, and the
//! sentences an operator sees when a request cannot be trained.

use anyhow::Result;
use candle_core::{Device, Tensor, Var};
use candle_nn::VarMap;
use wisent_errors::Cause;

use crate::failures::{
    refuse,
    tune::settings::{
        ACCUMULATION_NONE, BATCH_NONE, LEARNING_RATE_UNUSABLE, PASSES_NONE, SEQUENCE_SHORT,
    },
};
use crate::{lora, runtime::Runtime, workflow};

// MARK: - Shared trainer setup

/// Everything an adapter trainer checks before it touches a model, and the
/// four words it says it in.
///
/// Four objectives run the same five checks against the same numbers. Copying
/// them would let one copy gain a check the others silently lacked, so they
/// live here once, and the vocabulary is what makes one set of checks refuse
/// in four voices:
///
/// * `subject` opens every sentence.
/// * `unit` is one item of the training set — an example, a pair, a prompt.
/// * `pass` is one traversal of it. Three objectives call that an epoch;
///   policy optimization calls it an iteration, because it re-samples rather
///   than re-reading and is not seeing the same data twice.
/// * `noun` names the tensors in the opening progress line. Most runs train
///   adapters and nothing else, but a reward run also trains a scalar head,
///   and calling that an adapter would be wrong.
///
/// The supervised sentences are the ones the runbook quotes and this
/// vocabulary keeps them byte-identical.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Preflight<'a> {
    pub subject: &'a str,
    pub unit: &'a str,
    pub pass: &'a str,
    pub noun: &'a str,
    pub epochs: usize,
    pub accumulation: usize,
    /// Sequences folded into one forward pass, as the caller stated it; a
    /// batch changes only how many sequences share a kernel launch, never
    /// what the loss is. `None` for an objective that folds no sequences
    /// together: policy optimization samples and scores one completion at a
    /// time, so it has no batch to refuse.
    pub batch: Option<usize>,
    pub learning_rate: f64,
    pub max_sequence: usize,
}

/// What a passed preflight hands the trainer.
pub(crate) struct Trainable {
    /// The spec with `layers` pinned to this model's real indices.
    pub spec: lora::Spec,
    pub vars: Vec<Var>,
    pub tensors: usize,
    pub parameters: usize,
    /// The longest sequence this run will score: the operator's limit, clamped
    /// to the context the model was configured for.
    pub limit: usize,
}

impl Preflight<'_> {
    /// Refuses the numbers no run can train with, before any model, data or
    /// checkpoint is read; [`Self::open`] checks them again for a caller
    /// that went straight to the trainer.
    pub(crate) fn check(&self) -> Result<()> {
        let refused = |cause: &'static Cause, value: &str| -> Result<()> {
            Err(refuse(
                cause,
                &[
                    ("subject", self.subject),
                    ("pass", self.pass),
                    ("unit", self.unit),
                    ("value", value),
                ],
            ))
        };
        if self.epochs == 0 {
            return refused(&PASSES_NONE, "0");
        }
        if self.accumulation == 0 {
            return refused(&ACCUMULATION_NONE, "0");
        }
        if self.batch == Some(0) {
            return refused(&BATCH_NONE, "0");
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return refused(&LEARNING_RATE_UNUSABLE, &self.learning_rate.to_string());
        }
        if self.max_sequence < 2 {
            return refused(&SEQUENCE_SHORT, &self.max_sequence.to_string());
        }
        Ok(())
    }

    pub(crate) fn open(
        &self,
        runtime: &Runtime,
        varmap: &VarMap,
        spec: &lora::Spec,
    ) -> Result<Trainable> {
        self.check()?;
        let layer_count = runtime.layer_count();
        spec.validate(layer_count)?;
        let spec = spec.resolved(layer_count);

        // Nothing in the map is a base weight: the base was mapped read-only
        // and never registered, so the map holds the adapters and, on a reward
        // run, the scalar head. Handing the whole map to AdamW is therefore the
        // same statement as "train only what this run created".
        // `Runtime::load_trainable` refuses a spec without targets and layers,
        // so the map it returns always holds adapters.
        let vars = varmap.all_vars();
        assert!(
            !vars.is_empty(),
            "a trainer is handed the VarMap Runtime::load_trainable filled"
        );
        let tensors = vars.len();
        let parameters: usize = vars.iter().map(|var| var.elem_count()).sum();
        workflow::progress(format!(
            "training {tensors} {} ({parameters} parameters) at rank {} across {} layers",
            self.noun,
            spec.rank,
            spec.layers.len()
        ));
        let limit = self.max_sequence.min(runtime.context_length()?);
        Ok(Trainable {
            spec,
            vars,
            tensors,
            parameters,
            limit,
        })
    }
}

/// The log-probability the model assigns to each of `ids[start..]`, as a
/// `[len(ids) - start]` vector.
///
/// `logits` is `[1, n, vocab]` and the distribution that predicts token `t`
/// sits at index `t - 1`, so the scored window is the logit rows
/// `[start - 1, n - 1)` against the targets `ids[start..]`. `start` is
/// therefore at least one, and it always is: every tokenizer path in Ster
/// prepends a begin-of-sequence marker, so position zero is never a token
/// anyone asked the model to predict.
///
/// Per token rather than summed, because the objectives disagree about what to
/// do with them: a preference loss wants the sum, a policy-gradient loss wants
/// a per-token ratio and a per-token divergence. Summing is the caller's one
/// extra line; unsumming is impossible.
pub(crate) fn token_logprobs(
    logits: &Tensor,
    ids: &[u32],
    start: usize,
    device: &Device,
) -> Result<Tensor> {
    // Every caller encodes a context token before the scored window and at
    // least one token in it, and refuses an input that encodes without
    // either before its first forward pass.
    assert!(
        start > 0 && start < ids.len(),
        "a scored sequence holds a context token and a predicted token"
    );
    let scored = ids.len() - start;
    let (_, positions, vocab) = logits.dims3()?;
    assert_eq!(
        positions,
        ids.len(),
        "the forward pass answers one position per token"
    );
    // The window is promoted to F32 before the log-softmax whatever the base
    // weights were mapped at. A log-softmax is a max, an exponential, a sum
    // across the whole vocabulary and a logarithm; in half precision the sum
    // is where tens of thousands of small terms are added into one
    // accumulator and the smallest of them stop changing it. The forward pass
    // may run in half, but nothing that becomes a loss is summed in half.
    let window = logits
        .narrow(1, start - 1, scored)?
        .reshape((scored, vocab))?
        .to_dtype(candle_core::DType::F32)?;
    let log_probabilities = candle_nn::ops::log_softmax(&window, candle_core::D::Minus1)?;
    let targets = Tensor::new(&ids[start..], device)?.reshape((scored, 1))?;
    Ok(log_probabilities.gather(&targets, 1)?.squeeze(1)?)
}

/// The log-probability of the whole window, as one scalar.
pub(crate) fn sequence_logprob(
    logits: &Tensor,
    ids: &[u32],
    start: usize,
    device: &Device,
) -> Result<Tensor> {
    Ok(token_logprobs(logits, ids, start, device)?.sum_all()?)
}

/// `log(1 + exp(x))`, computed the way that does not overflow.
///
/// Every preference objective in Ster ends in `-log sigmoid(z)`, which is
/// `softplus(-z)`. The direct form is infinite in F32 from about x = 89, while
/// the true value there is 89. Pulling the positive part out through `relu`
/// leaves `log(1 + exp(-|x|))`, whose argument is always in (1, 2], and every
/// op in the expression records a backward node.
pub(crate) fn softplus(x: &Tensor) -> Result<Tensor> {
    let tail = ((x.abs()?.neg()?.exp()? + 1.0)?).log()?;
    Ok((x.relu()? + tail)?)
}
