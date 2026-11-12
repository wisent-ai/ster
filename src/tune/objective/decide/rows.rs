//! The training rows of calibrated decision training, and the loss each one
//! produces.

use anyhow::Result;
use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::ops::log_softmax;

use crate::{
    decide::{ExampleSet, Question, Request, render_rows},
    failures::tune::run::nothing_fits,
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
        // The labelled questions, each with the canonical index of the
        // option its label names; `ExampleSet::validate` refused a label
        // naming none, so none is dropped here.
        let labelled: Vec<(&String, &Question, usize)> = example
            .request
            .questions
            .iter()
            .filter_map(|(id, question)| {
                let truth = question.truth_index(example.answers.get(id)?)?;
                Some((id, question, truth))
            })
            .collect();
        if labelled.is_empty() {
            continue;
        }
        let request = Request {
            state: example.request.state.clone(),
            model: None,
            questions: labelled
                .iter()
                .map(|(id, question, _)| ((*id).clone(), (*question).clone()))
                .collect(),
        };
        questions.extend(labelled.iter().map(|(id, _, _)| (*id).clone()));
        // In the order `request.questions` iterates: both follow the ids.
        let truths: Vec<usize> = labelled.iter().map(|(_, _, truth)| *truth).collect();
        let (renderings, labels) = render_rows(runtime, &request, permutations).map_err(|error| {
            crate::failures::amend(error, format!("it was rendering example {index}"))
        })?;
        for row in renderings {
            rendered.push(row.ids.len());
            if row.ids.len() > limit {
                workflow::progress(format!(
                    "skipping a rendering of example {index}: {} tokens exceed the {limit} token limit",
                    row.ids.len()
                ));
                continue;
            }
            // Each row's order is a cyclic shift of its question's options,
            // so the labelled option has a position in it, and `labels`
            // holds the spellings of every position below its length.
            let Some(truth) = truths
                .get(row.question)
                .and_then(|truth| row.order.iter().position(|option| option == truth))
            else {
                continue;
            };
            let spellings = labels
                .range(..row.order.len())
                .map(|(_, spelling)| spelling.clone())
                .collect();
            rows.push(TrainingRow {
                ids: row.ids,
                truth,
                labels: spellings,
            });
        }
    }
    if rows.is_empty() {
        return Err(nothing_fits("rendering", "train on", rendered, limit));
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
    assert_eq!(
        positions,
        row.ids.len(),
        "the forward pass answers one position per token"
    );
    // A rendering always holds its question's text, so it has a last token.
    let answer = row.ids.len() - 1;
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
