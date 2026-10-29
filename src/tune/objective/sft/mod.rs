//! sft — supervised fine-tuning: masked next-token cross-entropy.
//!
//! The plainest objective Ster has, and the only one that learns from a
//! completion someone already wrote. Everything about it that is a decision
//! rather than a default is one of two things:
//!
//! * **The prompt is never a target.** The loss window starts at the
//!   completion boundary ([`loss`]). An operator writing `{"prompt": …,
//!   "completion": …}` is not asking the model to learn to reproduce the
//!   prompt.
//! * **An over-long example is skipped, not truncated.** A cut completion is a
//!   different completion, and one that ends mid-sentence teaches the model to
//!   stop early.
//!
//! A run given a checkpoint directory writes its state there every
//! `--checkpoint-every` steps, and a run given `--resume` continues from the
//! newest one exactly where it stopped: the same shuffles are replayed, the
//! steps already taken are skipped, and the optimizer's moments and steps are
//! the ones the stopped run had.

mod loss;
mod options;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{ParamsAdamW, VarMap};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

pub use options::{SftOptions, SftReport};

use super::super::{
    batch,
    checkpoint::{Position, optimizer::AdamW},
    examples::ExampleSet,
    preflight::{Preflight, Trainable},
    schedule,
};
use crate::{runtime::Runtime, workflow};
use options::Running;

/// Trains the adapters `varmap` owns against `examples`.
///
/// `runtime` must be the one `Runtime::load_trainable` returned alongside
/// `varmap`; the two are a pair, and passing a plain `Runtime` here would
/// produce a run whose optimizer has no variables to step.
pub fn sft(
    runtime: &Runtime,
    varmap: &VarMap,
    examples: &ExampleSet,
    options: &SftOptions,
) -> Result<SftReport> {
    examples.validate(&examples.label())?;
    let Trainable {
        spec,
        vars: _,
        tensors: trainable_tensors,
        parameters: trainable_parameters,
        limit,
    } = Preflight {
        subject: "supervised fine-tuning",
        unit: "example",
        pass: "epoch",
        noun: "adapter tensors",
        epochs: options.epochs,
        accumulation: options.accumulation,
        batch: Some(options.batch),
        learning_rate: options.learning_rate,
        max_sequence: options.max_sequence,
    }
    .open(runtime, varmap, &options.spec)?;

    // Tokenized once, up front. Encoding is deterministic, so repeating it per
    // epoch would buy nothing, and doing it here means the skip report is
    // complete before the first gradient is taken.
    let mut encoded: Vec<(usize, Vec<u32>, usize)> = Vec::with_capacity(examples.examples.len());
    for (index, example) in examples.examples.iter().enumerate() {
        let (ids, boundary) = runtime
            .encode_example(
                example.system.as_deref(),
                &example.prompt,
                &example.completion,
            )
            .with_context(|| format!("example {index} could not be encoded"))?;
        if ids.len() > limit {
            workflow::progress(format!(
                "skipping example {index}: {} tokens exceed the {limit} token limit",
                ids.len()
            ));
            continue;
        }
        encoded.push((index, ids, boundary));
    }
    if encoded.is_empty() {
        bail!("every example is longer than the sequence limit, so there is nothing to train on");
    }

    let fingerprint = options.fingerprint();
    let mut optimizer = AdamW::over(
        varmap,
        ParamsAdamW {
            lr: options.learning_rate,
            ..Default::default()
        },
    )
    .context("failed to initialize the AdamW optimizer")?;
    let resumed = options.durable.load(&mut optimizer, &fingerprint)?;
    let mut running: Running = match &resumed {
        Some(position) => serde_json::from_value(position.running.clone())
            .context("the checkpoint's running report is not one supervised fine-tuning wrote")?,
        None => Running::default(),
    };
    let mut checkpoints = Vec::new();

    let lengths: Vec<usize> = encoded.iter().map(|(_, ids, _)| ids.len()).collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let total_steps =
        batch::steps_per_epoch(encoded.len(), options.batch, options.accumulation) * options.epochs;
    let mut order: Vec<usize> = encoded.iter().enumerate().map(|(slot, _)| slot).collect();

    for (epoch, _) in std::iter::repeat_n((), options.epochs).enumerate() {
        // Reseeded per epoch rather than carried across epochs so that a run
        // is reproducible from `seed` alone; a resumed run replays the
        // shuffles of the epochs it skips, because each one shuffles the
        // order the one before left.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let plans = batch::plan(&order, &lengths, options.batch, options.accumulation);
        let remaining = match &resumed {
            Some(position) if position.pass > epoch => continue,
            Some(position) if position.pass == epoch => {
                plans.get(position.within..).with_context(|| {
                    format!(
                        "the checkpoint stands {} steps into epoch {epoch}, which has {} steps",
                        position.within,
                        plans.len()
                    )
                })?
            }
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
            // `batch` examples per forward, `accumulation` forwards per
            // optimizer step, each forward right-padded with a mask that hides
            // the future and the filler. The per-example loss is divided by the
            // constant `accumulation * batch`, so a short tail step moves the
            // weights proportionally less than a full one.
            let mut summed: Option<Tensor> = None;
            let mut losses: Vec<f64> = Vec::with_capacity(plan.units);
            for forward in &plan.forwards {
                let rows: Vec<&[u32]> = forward
                    .iter()
                    .map(|&slot| encoded[slot].1.as_slice())
                    .collect();
                // One sequence is one example.
                let read =
                    batch::read_rows(&rows, options.batch, rows.len() / forward.len(), |pass| {
                        runtime.forward_train_rows(pass)
                    })?;
                for (position, &slot) in forward.iter().enumerate() {
                    let (index, ids, boundary) = &encoded[slot];
                    let value = loss::completion_loss(&read[position], ids, *boundary, runtime)
                        .with_context(|| format!("example {index} produced no usable loss"))?;
                    losses.push(value.to_scalar::<f32>()? as f64);
                    let scaled = (value / scale)?;
                    summed = Some(match summed {
                        Some(total) => (total + scaled)?,
                        None => scaled,
                    });
                }
            }
            let Some(summed) = summed else {
                // `plan` never yields a step with no forwards and never a
                // forward with no rows; refusing beats stepping on nothing.
                bail!("an accumulation group contained no examples");
            };
            optimizer
                .backward_step(&summed)
                .context("failed to backpropagate the accumulated loss")?;

            let group_loss: f64 = losses.iter().sum();
            let step_loss = (group_loss / plan.units as f64) as f32;
            running.epoch_losses.push(group_loss);
            running.epoch_units.push(plan.units);
            running.first_loss.get_or_insert(step_loss);
            running.final_loss = Some(step_loss);
            let step = optimizer.rates.len();
            let seen: usize = running.epoch_units.iter().sum();
            workflow::progress(format!(
                "epochs done {}/{} step {step}/{total_steps} example {seen}/{} loss {step_loss:.4}",
                running.epoch_means.len(),
                options.epochs,
                encoded.len()
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

    let (Some(first_loss), Some(final_loss), Some(&mean_final_epoch_loss)) = (
        running.first_loss,
        running.final_loss,
        running.epoch_means.last(),
    ) else {
        bail!("the run took no step, so it has no loss to report");
    };
    Ok(SftReport {
        examples: examples.examples.len(),
        trained_examples: encoded.len(),
        skipped_long: examples.examples.len() - encoded.len(),
        epochs: options.epochs,
        steps: optimizer.rates.len(),
        trainable_tensors,
        trainable_parameters,
        first_loss,
        final_loss,
        mean_final_epoch_loss,
        rank: spec.rank,
        alpha: spec.alpha,
        targets: spec
            .targets
            .iter()
            .map(|target| target.name().to_owned())
            .collect(),
        layers: spec.layers.clone(),
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        batch: options.batch,
        resumed_from: resumed.map(|position| position.step),
        checkpoints,
    })
}
