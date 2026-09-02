//! sft.rs — supervised fine-tuning: masked next-token cross-entropy.
//!
//! The plainest objective Ster has, and the only one that learns from a
//! completion someone already wrote. Everything about it that is a decision
//! rather than a default is one of two things:
//!
//! * **The prompt is never a target.** The loss window starts at the
//!   completion boundary. An operator writing `{"prompt": …, "completion": …}`
//!   is not asking the model to learn to reproduce the prompt.
//! * **An over-long example is skipped, not truncated.** A cut completion is a
//!   different completion, and one that ends mid-sentence teaches the model to
//!   stop early.

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap, loss};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use super::{ExampleSet, Preflight, Trainable, batch, schedule};
use crate::{lora, runtime::Runtime, workflow};

#[derive(Debug, Clone)]
pub struct SftOptions {
    pub spec: lora::Spec,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
    /// Examples folded into one forward pass. One is the unbatched pass every
    /// run recorded before batching existed.
    pub batch: usize,
    pub warmup_steps: usize,
    pub max_sequence: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SftReport {
    pub examples: usize,
    pub trained_examples: usize,
    pub skipped_long: usize,
    pub epochs: usize,
    pub steps: usize,
    pub trainable_tensors: usize,
    pub trainable_parameters: usize,
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
}

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
    let Trainable { spec, vars, tensors: trainable_tensors, parameters: trainable_parameters, limit } =
        Preflight {
            subject: "supervised fine-tuning",
            unit: "example",
            pass: "epoch",
            noun: "adapter tensors",
            epochs: options.epochs,
            accumulation: options.accumulation,
            batch: options.batch,
            learning_rate: options.learning_rate,
            max_sequence: options.max_sequence,
        }
        .open(runtime, varmap, &options.spec)?;

    // Tokenized once, up front. Encoding is deterministic, so repeating it per
    // epoch would buy nothing, and doing it here means the skip report is
    // complete before the first gradient is taken rather than trickling out
    // over the whole run.
    let mut encoded: Vec<(usize, Vec<u32>, usize)> = Vec::with_capacity(examples.examples.len());
    let mut skipped_long = 0usize;
    for (index, example) in examples.examples.iter().enumerate() {
        let (ids, boundary) = runtime
            .encode_example(&example.prompt, &example.completion)
            .with_context(|| format!("example {index} could not be encoded"))?;
        if ids.len() > limit {
            skipped_long += 1;
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

    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW { lr: options.learning_rate, ..Default::default() },
    )
    .context("failed to initialize the AdamW optimizer")?;

    let lengths: Vec<usize> = encoded.iter().map(|(_, ids, _)| ids.len()).collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let steps_per_epoch =
        batch::steps_per_epoch(encoded.len(), options.batch, options.accumulation);
    let total_steps = steps_per_epoch * options.epochs;
    let mut order: Vec<usize> = (0..encoded.len()).collect();
    let mut step = 0usize;
    let mut first_loss: Option<f32> = None;
    let mut final_loss = 0f32;
    let mut mean_final_epoch_loss = 0f32;

    for epoch in 0..options.epochs {
        // Reseeded per epoch rather than carried across epochs so that a run
        // is reproducible from `seed` alone, and so that resuming an epoch
        // would see the same order it saw the first time.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);

        let mut epoch_loss = 0f64;
        let mut epoch_examples = 0usize;

        for plan in batch::plan(&order, &lengths, options.batch, options.accumulation) {
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                step,
                total_steps,
                options.warmup_steps,
            ));

            // `batch` examples per forward, `accumulation` forwards per
            // optimizer step. The rows of one forward are right-padded to a
            // common width and the decoder is handed a mask that hides both
            // the future and the filler, so no example is trained on another
            // example's padding; each row's own positions are read back out
            // with `batch::row`. The per-example loss is divided by the
            // constant `accumulation * batch` rather than by what this step
            // held, which is what makes a short tail step proportionally
            // smaller instead of as loudly as a full one.
            let mut summed: Option<Tensor> = None;
            let mut group_loss = 0f64;
            for forward in &plan.forwards {
                let rows: Vec<&[u32]> =
                    forward.iter().map(|&slot| encoded[slot].1.as_slice()).collect();
                let logits = runtime.forward_train_rows(&rows)?;
                for (position, &slot) in forward.iter().enumerate() {
                    let (index, ids, boundary) = &encoded[slot];
                    let logits = batch::row(&logits, position, ids.len())?;
                    let value = completion_loss(&logits, ids, *boundary, runtime)
                        .with_context(|| format!("example {index} produced no usable loss"))?;
                    group_loss += value.to_scalar::<f32>()? as f64;
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

            let step_loss = (group_loss / plan.units as f64) as f32;
            epoch_loss += group_loss;
            epoch_examples += plan.units;
            step += 1;
            if first_loss.is_none() {
                first_loss = Some(step_loss);
            }
            final_loss = step_loss;
            workflow::progress(format!(
                "epoch {}/{} step {step}/{total_steps} example {epoch_examples}/{} loss {step_loss:.4}",
                epoch + 1,
                options.epochs,
                encoded.len()
            ));
        }

        let epoch_mean = (epoch_loss / epoch_examples.max(1) as f64) as f32;
        if epoch + 1 == options.epochs {
            mean_final_epoch_loss = epoch_mean;
        }
        workflow::progress(format!(
            "epoch {}/{} mean loss {epoch_mean:.4}",
            epoch + 1,
            options.epochs
        ));
    }

    Ok(SftReport {
        examples: examples.examples.len(),
        trained_examples: encoded.len(),
        skipped_long,
        epochs: options.epochs,
        steps: step,
        trainable_tensors,
        trainable_parameters,
        // At least one group ran: `encoded` is non-empty and `epochs` is at
        // least one. The fallback keeps the report serializable rather than
        // emitting a JSON null for a float field.
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
    })
}

/// Next-token cross-entropy over the completion only.
///
/// `logits` is `[1, n, vocab]`, and the distribution that predicts token `t`
/// sits at logit index `t - 1`. Scoring the completion therefore means the
/// window `[boundary - 1, n - 1)` — `n - boundary` positions — against the
/// targets `ids[boundary..]`. A single-token completion is the tight end of
/// that arithmetic: the window is one position wide and starts at `boundary -
/// 1`, which is why `boundary` must be at least one, and it always is because
/// `encode_example` prepends the prompt's special tokens.
fn completion_loss(
    logits: &Tensor,
    ids: &[u32],
    boundary: usize,
    runtime: &Runtime,
) -> Result<Tensor> {
    if boundary == 0 {
        bail!("a training example must begin with at least one prompt token");
    }
    let predicted = ids.len().saturating_sub(boundary);
    if predicted == 0 {
        bail!("a training example must end with at least one completion token");
    }
    let (_, positions, vocab) = logits.dims3()?;
    if positions != ids.len() {
        bail!(
            "the forward pass returned {positions} positions for {} tokens",
            ids.len()
        );
    }
    let window = logits
        .narrow(1, boundary - 1, predicted)?
        .reshape((predicted, vocab))?;
    let targets = Tensor::new(&ids[boundary..], runtime.device())?;
    Ok(loss::cross_entropy(&window, &targets)?)
}
