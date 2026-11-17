//! A multiple-choice row of a fetched dataset as one labelled choice: its
//! question is the state, its choices are the options and its correct choice
//! is the label, each read where the JSON pointers say (`ChoiceFields`, the
//! same reading `ster pairs import --benchmark choices` does). A row that
//! cannot become one is refused under its cause and listed in the report;
//! nothing about it is guessed.

use std::collections::BTreeMap;

use anyhow::Error;
use serde_json::Value;

use super::{FetchOptions, Kept, RefusedRow, example};
use crate::decide::{Example, MAX_OPTIONS};
use crate::failures::decide::{fetch as causes, request};
use crate::failures::{Rejection, Rejections, excerpt, pairs::rows, refuse};
use crate::pairs::benchmark::{Choice, ChoiceFields, choice_of};

/// Row `index` of the split, kept as an example or listed as refused.
pub(super) fn read(
    options: &FetchOptions,
    fields: &ChoiceFields,
    index: usize,
    row: &Value,
    kept: &mut Kept,
) {
    match labelled(options, fields, row) {
        Ok((made, correct)) => {
            *kept.label_counts.entry(correct.to_string()).or_default() += 1;
            kept.examples.push(made);
        }
        Err(rejection) => {
            let rejection = rejection
                .with("row", format!("row {index}"))
                .with("source", options.dataset.as_str());
            kept.refused.push(RefusedRow {
                row: index,
                refused: kept.rejections.record(&rejection),
            });
        }
    }
}

/// The refusal of a fetch that kept no multiple-choice row: the first refused
/// row's own, or, when the split held no row from the offset, that.
pub(super) fn none_kept(options: &FetchOptions, rejections: Rejections, read: usize) -> Error {
    let kept_none = "every row was refused, so no labelled decision could be written";
    match rejections.refusal(kept_none, read) {
        Some(refusal) => refusal,
        None => refuse(
            &causes::ROWS_NONE,
            &[
                ("dataset", &options.dataset),
                ("split", &options.split),
                ("first", &options.offset.to_string()),
            ],
        ),
    }
}

/// `row` as an example, with the position of its correct choice among the
/// row's, or why it cannot be one. Choices of the same text are one option:
/// the label names the option by that text either way.
fn labelled(
    options: &FetchOptions,
    fields: &ChoiceFields,
    row: &Value,
) -> Result<(Example, usize), Rejection> {
    let Choice {
        question,
        choices,
        correct,
    } = choice_of(row, fields)?;
    let criteria: BTreeMap<String, Value> = choices
        .iter()
        .map(|choice| ((*choice).to_owned(), Value::Null))
        .collect();
    let count = criteria.len().to_string();
    if criteria.len() > MAX_OPTIONS {
        return Err(Rejection::new(
            &request::OPTIONS_TOO_MANY,
            vec![
                ("kind", "choice".to_owned()),
                ("id", options.question_id.clone()),
                ("count", count),
                ("unit", "options".to_owned()),
                ("max", MAX_OPTIONS.to_string()),
            ],
        ));
    }
    if criteria.len() <= 1 {
        return Err(Rejection::new(
            &rows::CHOICES_FEW,
            vec![
                ("count", count),
                ("pointer", fields.choices.clone()),
                ("choices", excerpt(&format!("{choices:?}"))),
            ],
        ));
    }
    let answer = choices[correct].to_owned();
    let made = example(options, question.to_owned(), criteria, answer);
    Ok((made, correct))
}
