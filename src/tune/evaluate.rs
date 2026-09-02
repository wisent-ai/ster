//! evaluate.rs — held-out scoring of a checkpoint, with or without an adapter.
//!
//! Every other file in this module changes weights. This one only measures
//! them, and the absence of an optimizer is the point: the number it reports is
//! meaningful precisely because nothing about the run could have moved to
//! produce it. It answers the question a training report structurally cannot —
//! a training loss is measured on the data that produced the gradient, so it
//! falls whether or not the model learned anything transferable.
//!
//! Three things worth stating:
//!
//! * **It takes the fused kernels.** No gradient is wanted, so paying for the
//!   composed rope, softmax and norm would buy an autograd tape that is
//!   discarded. That is what `Mode::score` exists for.
//! * **Two aggregates, because they answer different questions.** The
//!   token-weighted loss is total negative log-likelihood over total tokens,
//!   and its exponential is corpus perplexity — the number comparable with
//!   every other perplexity anyone reports. The macro average weighs each
//!   example equally regardless of length, which is what an operator comparing
//!   two adapters on a curated set usually means. Reporting one and calling it
//!   "the" loss would silently pick a side.
//! * **The report is F64.** Perplexity is the exponential of a loss and
//!   overflows F32 at a loss of about 89, which a broken adapter can reach. A
//!   measurement that reports infinity as a JSON null is worse than useless, so
//!   the arithmetic and the fields are both double.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use super::{ExampleSet, batch, sequence_logprob};
use crate::{artifact::Document, lora, model::Route, runtime::Runtime, workflow};

/// Warns when an artifact this run is consuming was produced in a different
/// encoding, or at a different precision, than this run is using.
///
/// Every command that consumes an artifact already documents that the two must
/// agree — `tune evaluate --chat-template` says a mismatch "measures a format
/// the adapter never saw", and `--precision` says the same about the space a
/// direction was fitted in. The artifact records both, so the product can say
/// it rather than leaving the operator to remember: an adapter trained on chat
/// markers and scored on raw text produces a plausible number that answers a
/// different question, and nothing about the output otherwise looks wrong.
///
/// One function covers both kinds of artifact because the question is one
/// question. Where the record lives differs — an adapter keeps it in the
/// `train` object of the sidecar written beside its safetensors, a steering
/// artifact is itself a JSON document and carries it at the top level — and
/// `provenance` resolves that difference once, so a field added to either
/// shape is warned on without a second copy of this logic growing beside it.
///
/// A warning and not a refusal, deliberately. Both combinations are things an
/// operator may want on purpose — measuring how far an adapter transfers out
/// of its training format is a real question, and so is reading a direction in
/// a space it was not fitted in — and a refusal would make the experiment
/// impossible rather than merely deliberate. Only an unnoticed mismatch is a
/// defect.
///
/// Silent when nothing can be read or the record does not carry the field: an
/// artifact written before a field existed is not a mismatch, and this is a
/// courtesy on top of the run rather than a gate in front of it.
pub fn warn_on_provenance(artifact: &Path, subject: &str, runtime: &Runtime) {
    let Some(record) = provenance(artifact) else { return };
    if let Some(trained) = record["chat_template"].as_str() {
        let now = runtime.chat_status().label();
        if trained != now {
            workflow::progress(format!(
                "warning: this {subject} was trained with chat template {trained} and this run encodes {now}, so the number below describes a format it was not trained in"
            ));
        }
    }
    if let Some(trained) = record["precision"].as_str() {
        let now = runtime.precision().name();
        if trained != now {
            workflow::progress(format!(
                "warning: this {subject} was trained at precision {trained} and this run maps the base weights at {now}, so it is being read in a different space than it was fitted in"
            ));
        }
    }
}

/// The object recording how `artifact` was produced, whichever kind it is.
///
/// An adapter's is the `train` object of the sidecar written beside its
/// safetensors; a steering artifact is itself JSON and records at the top
/// level. Which one this is cannot be decided by looking for a sidecar:
/// `sidecar_path` swaps the extension for `json`, so for a steering artifact —
/// which is already `.json` — it names the artifact itself, that read succeeds,
/// and the adapter branch then looks for a `train` object that a direction
/// never had. So the bytes are read once and `Document::recognise` says what
/// they are, which is honest where both the file name and the mere existence
/// of a neighbouring file are not.
///
/// `Unrecognised` takes the adapter reading. A document that is neither shape
/// has no provenance to disagree with, and the field lookups above miss
/// silently on the `null` that produces.
fn provenance(artifact: &Path) -> Option<serde_json::Value> {
    let sidecar = lora::Artifact::sidecar_path(artifact);
    let bytes = std::fs::read(&sidecar).or_else(|_| std::fs::read(artifact)).ok()?;
    let document = serde_json::from_slice::<serde_json::Value>(&bytes).ok()?;
    match Document::recognise(&bytes) {
        Document::Steering => Some(document),
        Document::AdapterSidecar | Document::Unrecognised => Some(document["train"].clone()),
    }
}

#[derive(Debug, Clone)]
pub struct EvaluateOptions {
    /// Examples longer than this are skipped rather than truncated, exactly as
    /// supervised fine-tuning skips them: a cut completion is a different
    /// completion, and scoring one would report a loss for text the operator
    /// never wrote.
    pub max_sequence: usize,
    /// Examples folded into one forward pass. One is the unbatched pass every
    /// score recorded before batching existed.
    pub batch: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluatedExample {
    pub index: usize,
    pub prompt: String,
    pub completion: String,
    pub completion_tokens: usize,
    pub loss: f64,
    pub perplexity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluateReport {
    pub model: String,
    pub model_revision: Option<String>,
    /// The adapter that was attached, or `null` for the bare checkpoint. This
    /// is the field that makes two reports comparable, so it is recorded even
    /// when it is absent.
    pub adapter: Option<String>,
    pub name: String,
    pub examples: usize,
    pub evaluated: usize,
    pub skipped_long: usize,
    pub completion_tokens: usize,
    /// Total negative log-likelihood over total completion tokens. Long
    /// examples count for more, which is what makes its exponential comparable
    /// with corpus perplexity anywhere else.
    pub loss: f64,
    pub perplexity: f64,
    /// The mean of the per-example losses. Every example counts once whatever
    /// its length, which is usually what an operator comparing two adapters on
    /// a curated set means by "the loss".
    pub mean_example_loss: f64,
    pub mean_example_perplexity: f64,
    pub entries: Vec<EvaluatedExample>,
}

/// Scores `examples` under `runtime`. Nothing is trained and nothing is written.
///
/// `runtime` is whatever the caller loaded — a bare checkpoint, or one with a
/// frozen adapter attached. `adapter` is only what the report should say it
/// was; the runtime already carries it.
pub fn evaluate(
    runtime: &Runtime,
    examples: &ExampleSet,
    adapter: Option<&Path>,
    options: &EvaluateOptions,
) -> Result<EvaluateReport> {
    if options.max_sequence < 2 {
        bail!("max_sequence must be at least two tokens, so that one token can predict another");
    }
    examples.validate(&examples.label())?;
    if let Some(adapter) = adapter {
        warn_on_provenance(adapter, "adapter", runtime);
    }

    let limit = options.max_sequence.min(runtime.context_length());
    let device = runtime.device();
    let mut entries = Vec::with_capacity(examples.examples.len());
    let mut skipped_long = 0usize;
    let mut total_tokens = 0usize;
    let mut total_negative_log_likelihood = 0f64;

    // Encoded up front, so every skip is reported before the first score and
    // so the batch planner can see how long each example is.
    let mut encoded: Vec<(usize, Vec<u32>, usize)> = Vec::with_capacity(examples.examples.len());
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

    let lengths: Vec<usize> = encoded.iter().map(|(_, ids, _)| ids.len()).collect();
    let order: Vec<usize> = (0..encoded.len()).collect();
    // Each example's negative log-likelihood, kept beside its entry so the
    // corpus total can be summed in the operator's order rather than in
    // whatever order the batches were scored in.
    let mut likelihoods: Vec<(usize, f64)> = Vec::with_capacity(encoded.len());
    // One step per forward here: there is no optimizer, so accumulation is one
    // and a step is a batch. Scoring is order-free — every entry carries its
    // own index and the report is sorted back into it below — so length
    // grouping costs nothing an operator can see.
    for plan in batch::plan(&order, &lengths, options.batch, 1) {
        for forward in &plan.forwards {
            let rows: Vec<&[u32]> =
                forward.iter().map(|&slot| encoded[slot].1.as_slice()).collect();
            // Adapted, because the adapters this runtime carries — if it
            // carries any — are the thing being evaluated. A bare checkpoint
            // has none and this is the base model's own score.
            let read = batch::read_rows(&rows, options.batch, 1, |pass| {
                runtime.forward_scored_rows(pass, Route::Adapted)
            })?;
            for (position, &slot) in forward.iter().enumerate() {
                let (index, ids, boundary) = &encoded[slot];
                let log_likelihood = sequence_logprob(&read[position], ids, *boundary, device)
                    .with_context(|| format!("example {index} produced no usable score"))?
                    .to_scalar::<f32>()? as f64;
                let tokens = ids.len() - boundary;
                let loss = -log_likelihood / tokens as f64;
                workflow::progress(format!(
                    "example {}/{} loss {loss:.4} perplexity {:.3}",
                    index + 1,
                    examples.examples.len(),
                    loss.exp()
                ));
                likelihoods.push((*index, -log_likelihood));
                entries.push(EvaluatedExample {
                    index: *index,
                    prompt: examples.examples[*index].prompt.clone(),
                    completion: examples.examples[*index].completion.clone(),
                    completion_tokens: tokens,
                    loss,
                    perplexity: loss.exp(),
                });
            }
        }
    }

    // Back into the operator's order, and the totals summed in that order, so
    // the corpus numbers depend on the example set and not on the grouping.
    entries.sort_by_key(|entry| entry.index);
    likelihoods.sort_by_key(|&(index, _)| index);
    for entry in &entries {
        total_tokens += entry.completion_tokens;
    }
    for (_, negative_log_likelihood) in &likelihoods {
        total_negative_log_likelihood += negative_log_likelihood;
    }

    if entries.is_empty() {
        bail!("every example is longer than the sequence limit, so there is nothing to evaluate");
    }

    let loss = total_negative_log_likelihood / total_tokens as f64;
    let mean_example_loss =
        entries.iter().map(|entry| entry.loss).sum::<f64>() / entries.len() as f64;
    workflow::progress(format!(
        "{} examples, {total_tokens} completion tokens, loss {loss:.4}, perplexity {:.3}",
        entries.len(),
        loss.exp()
    ));

    Ok(EvaluateReport {
        model: runtime.model_id.clone(),
        model_revision: runtime.revision.clone(),
        adapter: adapter.map(|path| path.display().to_string()),
        name: examples.label(),
        examples: examples.examples.len(),
        evaluated: entries.len(),
        skipped_long,
        completion_tokens: total_tokens,
        loss,
        perplexity: loss.exp(),
        mean_example_loss,
        mean_example_perplexity: mean_example_loss.exp(),
        entries,
    })
}
