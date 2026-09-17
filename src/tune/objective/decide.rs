//! decide.rs — calibrated decision training, Ster's RLCD.
//!
//! A decision is read from the model's next-token distribution over answer
//! letters at the end of a prompt ([`crate::decide`]). This objective trains
//! LoRA adapters so that distribution is right *and honest*: the loss is the
//! negative log-probability of the correct letter after the distribution is
//! restricted to the letters on offer. That is a proper scoring rule — the
//! only way to minimize it in expectation is to report the true probability —
//! so a model trained this way is pushed toward calibration rather than
//! toward confidence, which is the whole difference between a decision and a
//! generated answer.
//!
//! Three things about the rows are decisions rather than defaults:
//!
//! * **Every cyclic order is a training row.** A question with *n* options
//!   contributes *n* rows, each with the options under different letters and
//!   the label moved accordingly. The letter is therefore never predictive,
//!   and the model cannot lower the loss by learning a letter prior — it has
//!   to read the state. Reading uses the same orders, so training and reading
//!   see the same distribution of prompts.
//! * **The distribution is restricted before the loss.** Mass the model puts
//!   on any token that is not an answer letter is ignored, so the objective
//!   never teaches the model to stop writing prose in general — only to put
//!   its answer mass in the right place relative to the other letters. Every
//!   spelling a letter has in the vocabulary is summed, as the reader sums it.
//! * **An over-long rendering is skipped, not truncated.** A cut state is a
//!   different state, and a label for a state the model did not see is noise.

use anyhow::{bail, Context, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap, ops::log_softmax};
use rand::{seq::SliceRandom, rngs::StdRng, SeedableRng};
use serde::Serialize;

use super::super::{
    batch,
    preflight::{Preflight, Trainable},
    schedule,
};
use crate::{
    decide::{render_rows, ExampleSet, Request},
    lora,
    runtime::Runtime,
    workflow,
};

#[derive(Debug, Clone)]
pub struct DecideOptions {
    pub spec: lora::Spec,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
    /// Rows folded into one forward pass.
    pub batch: usize,
    pub warmup_steps: usize,
    pub max_sequence: usize,
    /// Option orders per question: `0` for every cyclic shift.
    pub permutations: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecideReport {
    pub examples: usize,
    /// Labelled questions across the examples.
    pub questions: usize,
    /// Training rows: one per labelled question per option order.
    pub rows: usize,
    pub trained_rows: usize,
    pub skipped_long: usize,
    pub epochs: usize,
    pub steps: usize,
    pub trainable_tensors: usize,
    pub trainable_parameters: usize,
    /// Mean negative log-probability of the correct option, at the first
    /// step and the last, and over the final epoch.
    pub first_loss: f32,
    pub final_loss: f32,
    pub mean_final_epoch_loss: f32,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub layers: Vec<usize>,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
    pub permutations: usize,
}

/// One training row: the tokens, which letter position holds the correct
/// option, and the token ids each letter position may be spelled with.
struct TrainingRow {
    ids: Vec<u32>,
    truth: usize,
    labels: Vec<Vec<u32>>,
}

/// Trains the adapters `varmap` owns against the labelled questions in
/// `examples`.
pub fn decide(
    runtime: &Runtime,
    varmap: &VarMap,
    examples: &ExampleSet,
    options: &DecideOptions,
) -> Result<DecideReport> {
    examples.validate()?;
    let Trainable { spec, vars, tensors: trainable_tensors, parameters: trainable_parameters, limit } =
        Preflight {
            subject: "calibrated decision training",
            unit: "row",
            pass: "epoch",
            noun: "adapter tensors",
            epochs: options.epochs,
            accumulation: options.accumulation,
            batch: options.batch,
            learning_rate: options.learning_rate,
            max_sequence: options.max_sequence,
        }
        .open(runtime, varmap, &options.spec)?;

    let mut rows: Vec<TrainingRow> = Vec::new();
    let mut questions = 0usize;
    let mut skipped_long = 0usize;
    let mut total_rows = 0usize;
    for (index, example) in examples.examples.iter().enumerate() {
        let request = Request {
            state: example.request.state.clone(),
            model: None,
            questions: example
                .request
                .questions
                .iter()
                .filter(|(id, _)| example.answers.contains_key(*id))
                .map(|(id, question)| (id.clone(), question.clone()))
                .collect(),
        };
        if request.questions.is_empty() {
            continue;
        }
        questions += request.questions.len();
        let truths: Vec<usize> = request
            .questions
            .iter()
            .map(|(id, question)| question.truth_index(&example.answers[id]).expect("validated label"))
            .collect();
        let (rendered, labels) = render_rows(runtime, &request, options.permutations)
            .with_context(|| format!("example {index} could not be rendered"))?;
        for row in rendered {
            total_rows += 1;
            if row.ids.len() > limit {
                skipped_long += 1;
                workflow::progress(format!(
                    "skipping a rendering of example {index}: {} tokens exceed the {limit} token limit",
                    row.ids.len()
                ));
                continue;
            }
            let truth = row
                .order
                .iter()
                .position(|&option| option == truths[row.question])
                .expect("every option is shown once");
            rows.push(TrainingRow {
                ids: row.ids,
                truth,
                labels: (0..row.order.len()).map(|position| labels[&position].clone()).collect(),
            });
        }
    }
    if rows.is_empty() {
        bail!("every rendering is longer than the sequence limit, so there is nothing to train on");
    }

    let mut optimizer = AdamW::new(vars, ParamsAdamW { lr: options.learning_rate, ..Default::default() })
        .context("failed to initialize the AdamW optimizer")?;

    let lengths: Vec<usize> = rows.iter().map(|row| row.ids.len()).collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let steps_per_epoch = batch::steps_per_epoch(rows.len(), options.batch, options.accumulation);
    let total_steps = steps_per_epoch * options.epochs;
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let mut step = 0usize;
    let mut first_loss: Option<f32> = None;
    let mut final_loss = 0f32;
    let mut mean_final_epoch_loss = 0f32;

    for epoch in 0..options.epochs {
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let mut epoch_loss = 0f64;
        let mut epoch_rows = 0usize;

        for plan in batch::plan(&order, &lengths, options.batch, options.accumulation) {
            optimizer.set_learning_rate(schedule(options.learning_rate, step, total_steps, options.warmup_steps));
            let mut summed: Option<Tensor> = None;
            let mut group_loss = 0f64;
            for forward in &plan.forwards {
                let ids: Vec<&[u32]> = forward.iter().map(|&slot| rows[slot].ids.as_slice()).collect();
                let read = batch::read_rows(&ids, options.batch, 1, |pass| runtime.forward_train_rows(pass))?;
                for (position, &slot) in forward.iter().enumerate() {
                    let value = decision_loss(&read[position], &rows[slot], runtime)?;
                    group_loss += value.to_scalar::<f32>()? as f64;
                    let scaled = (value / scale)?;
                    summed = Some(match summed {
                        Some(total) => (total + scaled)?,
                        None => scaled,
                    });
                }
            }
            let Some(summed) = summed else {
                bail!("an accumulation group contained no rows");
            };
            optimizer.backward_step(&summed).context("failed to backpropagate the accumulated loss")?;
            let step_loss = (group_loss / plan.units as f64) as f32;
            epoch_loss += group_loss;
            epoch_rows += plan.units;
            step += 1;
            if first_loss.is_none() {
                first_loss = Some(step_loss);
            }
            final_loss = step_loss;
            workflow::progress(format!(
                "epoch {}/{} step {step}/{total_steps} row {epoch_rows}/{} loss {step_loss:.4}",
                epoch + 1,
                options.epochs,
                rows.len()
            ));
        }
        let epoch_mean = (epoch_loss / epoch_rows.max(1) as f64) as f32;
        if epoch + 1 == options.epochs {
            mean_final_epoch_loss = epoch_mean;
        }
        workflow::progress(format!("epoch {}/{} mean loss {epoch_mean:.4}", epoch + 1, options.epochs));
    }

    Ok(DecideReport {
        examples: examples.examples.len(),
        questions,
        rows: total_rows,
        trained_rows: rows.len(),
        skipped_long,
        epochs: options.epochs,
        steps: step,
        trainable_tensors,
        trainable_parameters,
        first_loss: first_loss.unwrap_or(final_loss),
        final_loss,
        mean_final_epoch_loss,
        rank: spec.rank,
        alpha: spec.alpha,
        targets: spec.targets.iter().map(|target| target.name().to_owned()).collect(),
        layers: spec.layers.clone(),
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        batch: options.batch,
        permutations: options.permutations,
    })
}

/// The negative log-probability of the correct letter, over the letters on
/// offer only.
///
/// `logits` is `[1, n, vocab]`; the distribution that answers sits at the
/// last position. Each letter's logit is the log-sum-exp over its spellings,
/// exactly as the reader takes it, and the log-softmax runs over those letter
/// logits alone. Everything stays on the autograd tape: the gather, the
/// log-sum-exp and the normalization are tensor ops, so the gradient reaches
/// the adapters through the same arithmetic the reader uses.
fn decision_loss(logits: &Tensor, row: &TrainingRow, runtime: &Runtime) -> Result<Tensor> {
    let (_, positions, _) = logits.dims3()?;
    if positions != row.ids.len() {
        bail!("the forward pass returned {positions} positions for {} tokens", row.ids.len());
    }
    let last = logits.i((0, positions - 1, ..))?.to_dtype(DType::F32)?;
    let mut letters = Vec::with_capacity(row.labels.len());
    for spellings in &row.labels {
        let picked = last.index_select(&Tensor::new(spellings.as_slice(), runtime.device())?, 0)?;
        let max = picked.max(0)?;
        let summed = picked.broadcast_sub(&max)?.exp()?.sum(0)?.log()?;
        letters.push((summed + max)?);
    }
    let letters = Tensor::stack(&letters, 0)?;
    let log_probabilities = log_softmax(&letters, 0)?;
    Ok(log_probabilities.i(row.truth)?.neg()?)
}
