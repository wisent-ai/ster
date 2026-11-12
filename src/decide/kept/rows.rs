//! Turning a decision request into the token sequences the model reads, and
//! reading the answer-letter logits back from one pass over them.

use std::collections::BTreeMap;

use anyhow::Result;

use super::{
    Options,
    answer::{self, QuestionLogits, Usage},
    prompt,
    request::Request,
};
use crate::{
    Runtime,
    failures::{
        decide::{LETTER_OUTSIDE, QUESTION_TOO_LONG},
        refuse,
    },
    workflow,
};

/// One sequence the model reads: which question, which option order, and the
/// tokens.
pub struct Row {
    pub question: usize,
    pub order: Vec<usize>,
    pub ids: Vec<u32>,
}

/// Every rendering of every question in `request`, tokenized, plus the token
/// ids each answer letter position may be spelled with. This is the one
/// place a decision prompt is turned into tokens, for reading and for
/// training alike, so the two cannot drift apart.
///
/// The request is checked first, so a question with more options than there
/// are letters is refused as `options-too-many` here too, and every question
/// renders to at least one row.
pub fn render_rows(
    runtime: &Runtime,
    request: &Request,
    permutations: usize,
) -> Result<(Vec<Row>, BTreeMap<usize, Vec<u32>>)> {
    request.validate()?;
    let mut rows = Vec::new();
    let mut labels: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (index, (id, question)) in request.questions.iter().enumerate() {
        let texts = question.options();
        for (position, label) in prompt::LABELS.iter().enumerate().take(texts.len()) {
            if let std::collections::btree_map::Entry::Vacant(slot) = labels.entry(position) {
                slot.insert(runtime.label_tokens(label)?);
            }
        }
        for order in prompt::orders(texts.len(), permutations) {
            let text = prompt::render(request, question, &texts, &order);
            let ids = runtime.encode_prompt(&text)?;
            let context_length = runtime.context_length()?;
            if ids.len() > context_length {
                return Err(refuse(
                    &QUESTION_TOO_LONG,
                    &[
                        ("id", id),
                        ("tokens", &ids.len().to_string()),
                        ("context", &context_length.to_string()),
                    ],
                ));
            }
            rows.push(Row {
                question: index,
                order,
                ids,
            });
        }
    }
    Ok((rows, labels))
}

/// The answer-letter logits of every question in `request`, in canonical
/// option order, plus what it cost.
///
/// The rows all begin with the same text — the header and the state — so the
/// longest run of tokens they share goes through the model once, and each row
/// then costs only what comes after it: its question, its options, and the
/// answer prompt. The shared run is found on the tokens rather than the text,
/// because a tokenizer may merge across the point where the texts diverge
/// and a prefix cut by characters could then end mid-token.
pub(super) fn question_logits(
    runtime: &Runtime,
    request: &Request,
    options: Options,
) -> Result<(Vec<QuestionLogits>, Usage)> {
    let (rows, labels) = render_rows(runtime, request, options.permutations)?;
    // `render_rows` checked the request: at least one question, each shown
    // in at least one order, so there is a first row.
    let first: &[u32] = &rows[0].ids;
    let shared = shared_prefix(first, &rows);
    let suffixes: Vec<&[u32]> = rows.iter().map(|row| &row.ids[shared..]).collect();
    let mut usage = Usage::for_passes(usize::from(shared > 0) + rows.len());
    usage.input_tokens = shared + suffixes.iter().map(|suffix| suffix.len()).sum::<usize>();
    workflow::progress(format!(
        "reading the state once ({shared} tokens), then {} question renderings ({} tokens between them)",
        rows.len(),
        usage.input_tokens - shared
    ));
    let distributions = runtime.next_token_logits_after(&first[..shared], &suffixes)?;
    let mut logits: Vec<QuestionLogits> = request
        .questions
        .iter()
        .map(|_| QuestionLogits::default())
        .collect();
    for (row, vocabulary) in rows.iter().zip(distributions) {
        // Letter `position` shows option `order[position]`; its logit is kept
        // at that option's place, so a row reads in canonical option order.
        // `labels` holds the spellings of every position below the largest
        // option count, so the range yields one entry per letter shown.
        let mut canonical = vec![0f32; row.order.len()];
        let shown = row
            .order
            .iter()
            .zip(prompt::LABELS)
            .zip(labels.range(..row.order.len()));
        for ((&option, label), (_, spellings)) in shown {
            let mut scores = Vec::with_capacity(spellings.len());
            for &token in spellings {
                let Some(&score) = vocabulary.get(token as usize) else {
                    return Err(refuse(
                        &LETTER_OUTSIDE,
                        &[
                            ("letter", label),
                            ("token", &token.to_string()),
                            ("vocabulary", &vocabulary.len().to_string()),
                        ],
                    ));
                };
                scores.push(score);
            }
            if let Some(slot) = canonical.get_mut(option) {
                *slot = answer::log_sum_exp(scores.into_iter());
            }
        }
        if let Some(question) = logits.get_mut(row.question) {
            question.rows.push(canonical);
            question.orders.push(row.order.clone());
        }
    }
    Ok((logits, usage))
}

/// How many leading tokens every row has in common with `first`, short of
/// any row's whole length: each row must keep at least one token of its own,
/// because the position read is the last one and it has to be the row's, not
/// the prefix's.
fn shared_prefix(first: &[u32], rows: &[Row]) -> usize {
    let mut shared = first.len().saturating_sub(1);
    for row in rows {
        let common = row
            .ids
            .iter()
            .zip(first)
            .take_while(|(a, b)| a == b)
            .count();
        shared = shared.min(common).min(row.ids.len().saturating_sub(1));
    }
    shared
}
