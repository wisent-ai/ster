//! tune.rs — supervised fine-tuning of LoRA adapters.
//!
//! Everything else in Ster is forward-only: it reads hidden states, fits a
//! direction over them, and adds that direction back during decode. Nothing
//! there needs a gradient with respect to a weight. This module is the one
//! place that does, and it is deliberately narrow — one objective (masked
//! next-token cross-entropy over the completion), one optimizer (AdamW over
//! the adapter variables and nothing else), and one artifact.
//!
//! Two properties are structural rather than conventional and are worth
//! stating once:
//!
//! * **Only the adapters train.** `Runtime::load_trainable` maps the base
//!   weights read-only and registers nothing but the low-rank pairs in the
//!   `VarMap`. The optimizer is constructed from `varmap.all_vars()`, so there
//!   is no base weight it could reach even if the loss asked for one.
//! * **The prompt is never a target.** The loss window starts at the
//!   completion boundary. An operator writing `{"prompt": …, "completion": …}`
//!   is not asking the model to learn to reproduce the prompt.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap, loss};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{lora, runtime::Runtime, workflow};

/// The cosine schedule decays to this fraction of the peak learning rate
/// rather than to zero. LoRA runs are short; a schedule that reaches zero
/// spends a meaningful share of its last steps not learning at all.
const DECAY_FLOOR: f64 = 0.1;

// MARK: - Example sets

#[derive(Debug, Clone, Deserialize)]
pub struct Example {
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExampleSet {
    #[serde(default)]
    pub name: String,
    pub examples: Vec<Example>,
}

impl ExampleSet {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read example set {}", path.display()))?;
        let value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid example set JSON in {}", path.display()))?;
        value.validate(&path.display().to_string())?;
        Ok(value)
    }

    /// Checks the two content invariants every consumer of an example set
    /// relies on.
    ///
    /// `label` is the identity quoted in the refusal sentence, exactly as
    /// `PairSet::validate` uses it: the loader passes the path, and a caller
    /// validating a set assembled from an API request passes whatever names it
    /// to the operator. Both sentences are published in the runbook, so they
    /// must stay byte-identical regardless of which caller triggers them.
    pub fn validate(&self, label: &str) -> Result<()> {
        if self.examples.is_empty() {
            bail!("example set {label} contains no examples");
        }
        if self
            .examples
            .iter()
            .any(|example| example.prompt.trim().is_empty() || example.completion.trim().is_empty())
        {
            bail!("example set {label} contains an empty prompt or completion");
        }
        Ok(())
    }

    /// How the set names itself in a refusal when it never came from a file.
    fn label(&self) -> String {
        match self.name.trim() {
            "" => "(unnamed)".to_owned(),
            name => name.to_owned(),
        }
    }
}

// MARK: - Training

#[derive(Debug, Clone)]
pub struct SftOptions {
    pub spec: lora::Spec,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
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
    if options.epochs == 0 {
        bail!("supervised fine-tuning requires at least one epoch");
    }
    if options.accumulation == 0 {
        bail!("supervised fine-tuning requires an accumulation of at least one example");
    }
    if !options.learning_rate.is_finite() || options.learning_rate <= 0.0 {
        bail!("supervised fine-tuning requires a finite learning rate above zero");
    }
    if options.max_sequence < 2 {
        bail!("max_sequence must be at least two tokens, so that one token can predict another");
    }
    examples.validate(&examples.label())?;

    let layer_count = runtime.layer_count();
    options.spec.validate(layer_count)?;
    let spec = options.spec.resolved(layer_count);

    // Every variable in the map is an adapter matrix: the base was mapped
    // read-only and never registered. Handing the whole map to AdamW is
    // therefore the same statement as "train only the adapters".
    let vars = varmap.all_vars();
    if vars.is_empty() {
        bail!("this run has no trainable adapters; load the model with load_trainable");
    }
    let trainable_tensors = vars.len();
    let trainable_parameters: usize = vars.iter().map(|var| var.elem_count()).sum();
    workflow::progress(format!(
        "training {trainable_tensors} adapter tensors ({trainable_parameters} parameters) at rank {} across {} layers",
        spec.rank,
        spec.layers.len()
    ));

    // Tokenized once, up front. Encoding is deterministic, so repeating it per
    // epoch would buy nothing, and doing it here means the skip report is
    // complete before the first gradient is taken rather than trickling out
    // over the whole run.
    let limit = options.max_sequence.min(runtime.context_length());
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

    let steps_per_epoch = encoded.len().div_ceil(options.accumulation);
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

        for group in order.chunks(options.accumulation) {
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                step,
                total_steps,
                options.warmup_steps,
            ));

            // One example per forward, `accumulation` forwards per optimizer
            // step. This is deliberate, not a simplification: Ster's decoder
            // builds its causal mask from the sequence length alone and has no
            // padding token and no attention mask, so stacking two examples of
            // different lengths into one batch would train the adapter on
            // whatever filler the shorter row was padded with — silently, with
            // a loss that still looks reasonable. Summing scaled per-example
            // losses gives the same gradient a real batch would, at the cost
            // of holding one graph at a time.
            let mut summed: Option<Tensor> = None;
            let mut group_loss = 0f64;
            for &slot in group {
                let (index, ids, boundary) = &encoded[slot];
                let logits = runtime.forward_train(ids)?;
                let value = completion_loss(&logits, ids, *boundary, runtime)
                    .with_context(|| format!("example {index} produced no usable loss"))?;
                group_loss += value.to_scalar::<f32>()? as f64;
                let scaled = (value / options.accumulation as f64)?;
                summed = Some(match summed {
                    Some(total) => (total + scaled)?,
                    None => scaled,
                });
            }
            let Some(summed) = summed else {
                // `chunks` never yields an empty slice, so this is unreachable
                // in practice; refusing beats stepping on nothing.
                bail!("an accumulation group contained no examples");
            };
            optimizer
                .backward_step(&summed)
                .context("failed to backpropagate the accumulated loss")?;

            let step_loss = (group_loss / group.len() as f64) as f32;
            epoch_loss += group_loss;
            epoch_examples += group.len();
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

/// Linear warmup for `warmup` steps, then cosine decay from `base` down to
/// `DECAY_FLOOR * base` over whatever steps remain.
///
/// Warmup counts from one so that the very first step is not taken at a zero
/// learning rate, which would waste the one step whose gradient is largest.
fn schedule(base: f64, step: usize, total: usize, warmup: usize) -> f64 {
    if step < warmup {
        return base * (step + 1) as f64 / warmup as f64;
    }
    let floor = base * DECAY_FLOOR;
    let decaying = total.saturating_sub(warmup);
    if decaying <= 1 {
        return base;
    }
    let progress = ((step - warmup) as f64 / (decaying - 1) as f64).clamp(0.0, 1.0);
    floor + (base - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

// MARK: - Inspection

/// The adapter equivalent of `workflow::artifact_summary`: everything the
/// artifact knows about itself, including the shape of every tensor it
/// carries, without loading a model. Auditing which layers and projections an
/// adapter touches should not cost a multi-gigabyte mmap.
pub fn inspect(artifact: &lora::Artifact) -> serde_json::Value {
    // `Spec::scale` is the same ratio, but an artifact is not a spec and a
    // rank of zero would never have been written; guarding here keeps the
    // inspector total over files it did not produce.
    let scale = if artifact.rank == 0 {
        0.0
    } else {
        artifact.alpha / artifact.rank as f64
    };
    serde_json::json!({
        "adapter": {
            "schema_version": artifact.schema_version,
            "product": artifact.product,
            "model": artifact.model,
            "model_revision": artifact.model_revision,
            "rank": artifact.rank,
            "alpha": artifact.alpha,
            "scale": scale,
            "targets": artifact.targets.iter().map(|target| target.name()).collect::<Vec<_>>(),
            "layers": artifact.layers,
            "hidden_size": artifact.hidden_size,
            "tensors": artifact.tensors.iter().map(|(name, tensor)| serde_json::json!({
                "name": name,
                "shape": tensor.dims(),
            })).collect::<Vec<_>>(),
            "train": artifact.train,
        }
    })
}
