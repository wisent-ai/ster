//! The training rows of calibrated decision training, and the loss each one
//! produces.

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::ops::log_softmax;

use crate::{
    decide::{ExampleSet, Request, render_rows},
    runtime::Runtime,
    workflow,
};

/// One training row: the tokens, which letter position holds the correct
/// option, and the token ids each letter position may be spelled with.
pub(super) struct TrainingRow {
    pub(super) ids: Vec<u32>,
    truth: usize,
    labels: Vec<Vec<u32>>,
}

/// Every row the labelled questions of `examples` render to, with what was
/// counted on the way: the labelled questions, every rendering, and the ones
/// skipped for exceeding `limit` tokens.
pub(super) struct Rows {
    pub(super) rows: Vec<TrainingRow>,
    pub(super) questions: usize,
    pub(super) rendered: usize,
    pub(super) skipped_long: usize,
}

/// Renders every labelled question of `examples` in `permutations` option
/// orders (every cyclic shift when none is named), skipping a rendering
/// longer than `limit` tokens rather than truncating it.
pub(super) fn training_rows(
    runtime: &Runtime,
    examples: &ExampleSet,
    permutations: usize,
    limit: usize,
) -> Result<Rows> {
    let mut rows = Vec::new();
    let mut questions = Vec::new();
    let mut rendered = Vec::new();
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
        questions.extend(request.questions.iter().map(|(id, _)| id.clone()));
        let truths = request
            .questions
            .iter()
            .map(|(id, question)| {
                question.truth_index(&example.answers[id]).with_context(|| {
                    format!("example {index} labels question {id} with an answer that is none of its options")
                })
            })
            .collect::<Result<Vec<usize>>>()?;
        let (renderings, labels) = render_rows(runtime, &request, permutations)
            .with_context(|| format!("example {index} could not be rendered"))?;
        for row in renderings {
            rendered.push(row.ids.len());
            if row.ids.len() > limit {
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
                .with_context(|| {
                    format!("a rendering of example {index} does not show its correct option")
                })?;
            let spellings = row
                .order
                .iter()
                .enumerate()
                .map(|(position, _)| {
                    labels.get(&position).cloned().with_context(|| {
                        format!("answer letter {position} has no spelling in this vocabulary")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            rows.push(TrainingRow {
                ids: row.ids,
                truth,
                labels: spellings,
            });
        }
    }
    let skipped_long = rendered.len() - rows.len();
    Ok(Rows {
        rows,
        questions: questions.len(),
        rendered: rendered.len(),
        skipped_long,
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
/// the adapters through the same arithmetic the reader uses. Every reduction
/// is over the last axis, the only one the vectors it works on have.
pub(super) fn decision_loss(
    logits: &Tensor,
    row: &TrainingRow,
    runtime: &Runtime,
) -> Result<Tensor> {
    let (_, positions, _) = logits.dims3()?;
    if positions != row.ids.len() {
        bail!(
            "the forward pass returned {positions} positions for {} tokens",
            row.ids.len()
        );
    }
    let Some((answer, _)) = row.ids.iter().enumerate().last() else {
        bail!("a training row holds no token, so no distribution answers it");
    };
    // `[1, n, vocab]` read as `[n, vocab]`, then the answering position.
    let last = logits
        .flatten_to(D::Minus2)?
        .i(answer)?
        .to_dtype(DType::F32)?;
    let mut letters = Vec::with_capacity(row.labels.len());
    for spellings in &row.labels {
        let picked = last.index_select(
            &Tensor::new(spellings.as_slice(), runtime.device())?,
            D::Minus1,
        )?;
        let max = picked.max_keepdim(D::Minus1)?;
        let summed = picked
            .broadcast_sub(&max)?
            .exp()?
            .sum_keepdim(D::Minus1)?
            .log()?;
        letters.push((summed + max)?);
    }
    let letters = Tensor::cat(&letters, D::Minus1)?;
    let log_probabilities = log_softmax(&letters, D::Minus1)?;
    Ok(log_probabilities.i(row.truth)?.neg()?)
}
