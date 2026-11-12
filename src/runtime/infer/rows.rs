//! The batched forwards: many right-padded sequences in one pass, for
//! training and scoring several examples at once.

use anyhow::Result;
use candle_core::Tensor;

use crate::model::{ForwardOutput, Mode, Route, SteeringPlan};

use super::super::Runtime;

impl Runtime {
    /// One differentiable forward over several sequences at once, returning
    /// logits `[batch, width, vocab]` where `width` is the longest row.
    ///
    /// Row `r`'s real logits are its first `rows[r].len()` positions: the
    /// padding is on the right, so every row is sliced from the front to its
    /// own length, which is the same offset convention a single-sequence
    /// forward already hands back.
    pub fn forward_train_rows(&self, rows: &[&[u32]]) -> Result<Tensor> {
        self.row_logits(
            rows,
            Mode::TRAIN,
            "a training forward pass needs at least one token",
        )
    }

    /// The same differentiable batched pass with `steering` added to the
    /// residual stream, so a loss over the logits reaches a steering vector
    /// the plan was built over (`SteeringPlan::from_tensors`).
    pub fn forward_steered_rows(&self, rows: &[&[u32]], steering: &SteeringPlan) -> Result<Tensor> {
        self.forward_rows(
            rows,
            Some(steering),
            Mode::TRAIN,
            "a training forward pass needs at least one token",
        )?
        .into_logits(Mode::TRAIN.readout)
    }

    /// The same batched pass with no autograd tape, routed through the policy
    /// or the frozen reference exactly as `forward_scored` routes one.
    pub fn forward_scored_rows(&self, rows: &[&[u32]], route: Route) -> Result<Tensor> {
        self.row_logits(
            rows,
            Mode::score(route),
            "a scoring forward pass needs at least one token",
        )
    }

    /// The batched scoring pass with `steering` added to the residual stream
    /// of the frozen model and no autograd tape: how far a fixed artifact
    /// moves the model's log-probabilities, read without training anything.
    pub fn forward_steered_scored_rows(
        &self,
        rows: &[&[u32]],
        steering: &SteeringPlan,
    ) -> Result<Tensor> {
        let mode = Mode::score(Route::Base);
        self.forward_rows(
            rows,
            Some(steering),
            mode,
            "a scoring forward pass needs at least one token",
        )?
        .into_logits(mode.readout)
    }

    /// The batched residual stream a reward head reads, `[batch, width,
    /// hidden]`, with no vocabulary projection.
    pub fn forward_hidden_rows(&self, rows: &[&[u32]]) -> Result<Tensor> {
        Ok(self
            .forward_rows(
                rows,
                None,
                Mode::REWARD,
                "a reward forward pass needs at least one token",
            )?
            .hidden)
    }

    /// The batched equivalent of `logits`, which must produce logits too.
    fn row_logits(&self, rows: &[&[u32]], mode: Mode, empty: &str) -> Result<Tensor> {
        self.forward_rows(rows, None, mode, empty)?.into_logits(mode.readout)
    }

    /// The body every batched forward shares: many right-padded sequences, no
    /// KV cache.
    ///
    /// The cache is off for the reason it is off in `forward_once`, and here
    /// it is also structural: one cache cannot hold rows that end in different
    /// places. Filler is token zero, which is never read — `forward_batch`
    /// masks every padded key out of every real query — and is chosen only
    /// because it is the one id every vocabulary has.
    fn forward_rows(
        &self,
        rows: &[&[u32]],
        steering: Option<&SteeringPlan>,
        mode: Mode,
        empty: &str,
    ) -> Result<ForwardOutput> {
        let lengths: Vec<usize> = rows.iter().map(|row| row.len()).collect();
        // Every row is an encode's output, and encoding refuses a text the
        // tokenizer turns into no tokens.
        let Some(width) = lengths
            .iter()
            .copied()
            .max()
            .filter(|_| !lengths.contains(&0))
        else {
            return Err(crate::failures::refuse(
                &crate::failures::model::decoder::ROWS_EMPTY,
                &[("pass", empty), ("lengths", &format!("{lengths:?}"))],
            ));
        };
        let model = self.model.causal(super::WHOLE_SEQUENCE)?;
        let mut flat = Vec::with_capacity(rows.len() * width);
        for row in rows {
            flat.extend_from_slice(row);
            flat.resize(flat.len() + width - row.len(), 0);
        }
        let input = Tensor::from_vec(flat, (rows.len(), width), &self.device)?;
        let mut cache = self.cache(model, false)?;
        Ok(model.forward_batch(&input, &lengths, &mut cache, steering, mode)?)
    }
}
