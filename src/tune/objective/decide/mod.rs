//! decide — calibrated decision training, Ster's RLCD.
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
//!
//! A run given a checkpoint directory writes its state there every
//! `--checkpoint-every` steps, and a run given `--resume` continues from the
//! newest one exactly where it stopped, as supervised fine-tuning does.

mod options;
mod rows;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{ParamsAdamW, VarMap};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

pub use options::{DecideOptions, DecideReport};

use super::super::{
    batch,
    checkpoint::{Position, optimizer::AdamW},
    preflight::{Preflight, Trainable},
    schedule,
};
use crate::{decide::ExampleSet, runtime::Runtime, workflow};
use options::Running;
use rows::{Rows, decision_loss, training_rows};

/// Trains the adapters `varmap` owns against the labelled questions in
/// `examples`.
pub fn decide(runtime: &Runtime, varmap: &VarMap, examples: &ExampleSet, options: &DecideOptions) -> Result<DecideReport> {
    examples.validate()?;
    let Trainable { spec, vars: _, tensors: trainable_tensors, parameters: trainable_parameters, limit } = Preflight {
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

    let Rows { rows, questions, rendered, skipped_long } =
        training_rows(runtime, examples, options.permutations, limit)?;
    if rows.is_empty() {
        bail!("every rendering is longer than the sequence limit, so there is nothing to train on");
    }

    let fingerprint = options.fingerprint();
    let mut optimizer = AdamW::over(varmap, ParamsAdamW { lr: options.learning_rate, ..Default::default() })
        .context("failed to initialize the AdamW optimizer")?;
    let resumed = options.durable.load(&mut optimizer, &fingerprint)?;
    let mut running: Running = match &resumed {
        Some(position) => serde_json::from_value(position.running.clone())
            .context("the checkpoint's running report is not one calibrated decision training wrote")?,
        None => Running::default(),
    };
    let mut checkpoints = Vec::new();

    let lengths: Vec<usize> = rows.iter().map(|row| row.ids.len()).collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let total_steps = batch::steps_per_epoch(rows.len(), options.batch, options.accumulation) * options.epochs;
    let mut order: Vec<usize> = rows.iter().enumerate().map(|(slot, _)| slot).collect();

    for (epoch, _) in std::iter::repeat_n((), options.epochs).enumerate() {
        // Reseeded per epoch so a run is reproducible from `seed` alone; a
        // resumed run replays the shuffles of the epochs it skips.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let plans = batch::plan(&order, &lengths, options.batch, options.accumulation);
        let remaining = match &resumed {
            Some(position) if position.pass > epoch => continue,
            Some(position) if position.pass == epoch => plans.get(position.within..).with_context(|| {
                format!("the checkpoint stands {} steps into epoch {epoch}, which has {} steps", position.within, plans.len())
            })?,
            _ => {
                running.epoch_losses.clear();
                running.epoch_units.clear();
                &plans[..]
            }
        };

        for plan in remaining {
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                optimizer.rates.len(),
                total_steps,
                options.warmup_steps,
            ));
            let mut summed: Option<Tensor> = None;
            let mut losses: Vec<f64> = Vec::with_capacity(plan.units);
            for forward in &plan.forwards {
                let ids: Vec<&[u32]> = forward.iter().map(|&slot| rows[slot].ids.as_slice()).collect();
                // One sequence is one row.
                let read =
                    batch::read_rows(&ids, options.batch, ids.len() / forward.len(), |pass| runtime.forward_train_rows(pass))?;
                for (logits, &slot) in read.iter().zip(forward) {
                    let value = decision_loss(logits, &rows[slot], runtime)?;
                    losses.push(value.to_scalar::<f32>()? as f64);
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

            let group_loss: f64 = losses.iter().sum();
            let step_loss = (group_loss / plan.units as f64) as f32;
            running.epoch_losses.push(group_loss);
            running.epoch_units.push(plan.units);
            running.first_loss.get_or_insert(step_loss);
            running.final_loss = Some(step_loss);
            let step = optimizer.rates.len();
            let seen: usize = running.epoch_units.iter().sum();
            workflow::progress(format!(
                "epochs done {}/{} step {step}/{total_steps} row {seen}/{} loss {step_loss:.4}",
                running.epoch_means.len(),
                options.epochs,
                rows.len()
            ));
            if options.durable.due(step) {
                let position = Position {
                    pass: epoch,
                    step,
                    within: running.epoch_units.len(),
                    running: serde_json::to_value(&running)?,
                };
                options.durable.save(&optimizer, &fingerprint, &position)?;
                checkpoints.push(step);
            }
        }

        if running.epoch_units.is_empty() {
            bail!("epoch {epoch} took no step, so it has no mean loss");
        }
        let seen: usize = running.epoch_units.iter().sum();
        let epoch_mean = (running.epoch_losses.iter().sum::<f64>() / seen as f64) as f32;
        running.epoch_means.push(epoch_mean);
        workflow::progress(format!(
            "epochs done {}/{} mean loss {epoch_mean:.4}",
            running.epoch_means.len(),
            options.epochs
        ));
    }

    let (Some(first_loss), Some(final_loss), Some(&mean_final_epoch_loss)) =
        (running.first_loss, running.final_loss, running.epoch_means.last())
    else {
        bail!("the run took no step, so it has no loss to report");
    };
    Ok(DecideReport {
        examples: examples.examples.len(),
        questions,
        rows: rendered,
        trained_rows: rows.len(),
        skipped_long,
        epochs: options.epochs,
        steps: optimizer.rates.len(),
        trainable_tensors,
        trainable_parameters,
        first_loss,
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
        resumed_from: resumed.map(|position| position.step),
        checkpoints,
    })
}
