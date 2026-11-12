//! Multiple-choice rows of any dataset export: a question, its choices and
//! the answer, each found where the caller says it sits (a JSON pointer,
//! RFC 6901), because the dataset's schema is what states them. This replaces
//! the per-benchmark Python extractors of wisent-extractors, which guessed:
//! an answer letter mapped by arithmetic, the next choice taken as the wrong
//! one, a missing answer read as index zero and a missing score read as safe.
//!
//! Here an answer is read in the one form the caller names (an index, one of
//! the row's labels, or the text of a choice); a row whose answer is missing
//! or does not resolve is skipped under its cause (`ster.pairs.import.rows`),
//! never repaired by a guess. The incorrect side is one of the other choices
//! drawn with the import's seed, as TruthfulQA's is.

use std::path::Path;

use anyhow::Result;
use rand::{rngs::StdRng, seq::IndexedRandom};
use serde_json::{Value, error::Category};
use wisent_errors::Cause;

use super::fields::{AnswerForm, ChoiceFields};
use super::pair;
use crate::ContrastivePair;
use crate::failures::{
    Rejection, excerpt, file::json, pairs::export, pairs::rows as causes, refuse, shape,
};

/// Rows and records are numbered from one in every message, as the GNU
/// Coding Standards number lines in diagnostics.
pub(super) const FIRST_ROW: usize = 1; // https://www.gnu.org/prep/standards/html_node/Errors.html "Line numbers should start from 1 at the beginning of the file"

/// What a multiple-choice export that opens with `[` must be, for the
/// `ster.file` refusal of one whose JSON does not read.
const LIST_FORM: &str = "one JSON list of row objects, each holding the question, choices and answer at the pointers given";

/// The rows of a JSON list, or of JSON Lines, each with the number a reader
/// counts it by: its position in the list, or its line.
pub(super) fn rows(text: &str, path: &str) -> Result<Vec<(String, Value)>> {
    if text.trim_start().starts_with('[') {
        let rows: Vec<Value> = serde_json::from_str(text).map_err(|error| {
            json::refusal(
                "the multiple-choice rows",
                LIST_FORM,
                Path::new(path),
                &error,
            )
        })?;
        return Ok(rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| ((index + FIRST_ROW).to_string(), row))
            .collect());
    }
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            let number = (index + FIRST_ROW).to_string();
            serde_json::from_str(line)
                .map_err(|error| {
                    // A value that runs out at the line's end continues on
                    // the next: a document spread over lines, not JSON Lines.
                    let cause = match error.classify() {
                        Category::Eof => &export::ROW_UNCLOSED,
                        _ => &export::ROW_NOT_JSON,
                    };
                    refuse(
                        cause,
                        &[
                            ("path", path),
                            ("line", &number),
                            ("column", &error.column().to_string()),
                            ("excerpt", &excerpt(line)),
                            ("error", &error.to_string()),
                        ],
                    )
                })
                .map(|row| (number, row))
        })
        .collect()
}

fn refused<T>(cause: &'static Cause, values: Vec<(&'static str, String)>) -> Result<T, Rejection> {
    Err(Rejection::new(cause, values))
}

/// The list at `pointer` of `row`, every element non-empty text, or the
/// refusal naming the first that is not: `not_list` when there is no list,
/// `not_text` with the element's index and the whole list under `list`
/// (`choices`, `labels`).
fn texts<'a>(
    row: &'a Value,
    pointer: &str,
    list: &'static str,
    not_list: &'static Cause,
    not_text: &'static Cause,
) -> Result<Vec<&'a str>, Rejection> {
    let found = row.pointer(pointer);
    let (Some(stated), Some(items)) = (found, found.and_then(Value::as_array)) else {
        return refused(
            not_list,
            vec![("pointer", pointer.to_owned()), ("found", shape(found))],
        );
    };
    let mut texts = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item.as_str().map(str::trim).filter(|text| !text.is_empty()) {
            Some(text) => texts.push(text),
            None => {
                return refused(
                    not_text,
                    vec![
                        ("index", index.to_string()),
                        ("found", shape(Some(item))),
                        ("pointer", pointer.to_owned()),
                        (list, excerpt(&stated.to_string())),
                    ],
                );
            }
        }
    }
    Ok(texts)
}

/// One row's pair, or why the row cannot become one.
pub(super) fn pair_of(
    row: &Value,
    fields: &ChoiceFields,
    rng: &mut StdRng,
) -> Result<ContrastivePair, Rejection> {
    let stated = row.pointer(&fields.question);
    let Some(question) = stated
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|question| !question.is_empty())
    else {
        return refused(
            &causes::CHOICE_QUESTION_MISSING,
            vec![
                ("pointer", fields.question.clone()),
                ("found", shape(stated)),
            ],
        );
    };
    let choices = texts(
        row,
        &fields.choices,
        "choices",
        &causes::CHOICES_NOT_LIST,
        &causes::CHOICE_NOT_TEXT,
    )?;
    let shown_choices = excerpt(&format!("{choices:?}"));
    if choices.len() < 2 {
        return refused(
            &causes::CHOICES_FEW,
            vec![
                ("count", choices.len().to_string()),
                ("pointer", fields.choices.clone()),
                ("choices", shown_choices),
            ],
        );
    }
    let Some(answer) = row
        .pointer(&fields.answer)
        .filter(|answer| !answer.is_null())
    else {
        return refused(
            &causes::ANSWER_MISSING,
            vec![
                ("pointer", fields.answer.clone()),
                ("found", shape(row.pointer(&fields.answer))),
            ],
        );
    };
    let correct = match fields.answer_form {
        AnswerForm::Index => index_of(answer, &fields.answer, &choices, shown_choices.clone())?,
        AnswerForm::Label => {
            // `ImportOptions::check` refused --answer-form label without
            // --labels before the export was read.
            let Some(pointer) = fields.labels.as_deref() else {
                return refused(
                    &causes::LABELS_NOT_LIST,
                    vec![
                        ("pointer", "--labels".to_owned()),
                        ("found", "no --labels given".to_owned()),
                    ],
                );
            };
            let labels = texts(
                row,
                pointer,
                "labels",
                &causes::LABELS_NOT_LIST,
                &causes::LABEL_NOT_TEXT,
            )?;
            let shown_labels = excerpt(&format!("{labels:?}"));
            if labels.len() != choices.len() {
                return refused(
                    &causes::LABELS_MISALIGNED,
                    vec![
                        ("labels", labels.len().to_string()),
                        ("choices", choices.len().to_string()),
                        ("shown_labels", shown_labels),
                        ("shown_choices", shown_choices),
                    ],
                );
            }
            let Some(named) = answer.as_str().map(str::trim) else {
                return refused(
                    &causes::ANSWER_LABEL_NOT_TEXT,
                    vec![
                        ("pointer", fields.answer.clone()),
                        ("found", shape(Some(answer))),
                        ("labels", shown_labels),
                    ],
                );
            };
            match labels.iter().position(|label| *label == named) {
                Some(position) => position,
                None => {
                    return refused(
                        &causes::ANSWER_LABEL_UNKNOWN,
                        vec![("answer", format!("{named:?}")), ("labels", shown_labels)],
                    );
                }
            }
        }
        AnswerForm::Text => {
            let Some(named) = answer.as_str().map(str::trim) else {
                return refused(
                    &causes::ANSWER_TEXT_NOT_TEXT,
                    vec![
                        ("pointer", fields.answer.clone()),
                        ("found", shape(Some(answer))),
                        ("choices", shown_choices),
                    ],
                );
            };
            match choices.iter().position(|choice| *choice == named) {
                Some(position) => position,
                None => {
                    return refused(
                        &causes::ANSWER_TEXT_UNKNOWN,
                        vec![
                            ("answer", excerpt(&format!("{named:?}"))),
                            ("choices", shown_choices),
                        ],
                    );
                }
            }
        }
    };
    let positive = choices[correct];
    let incorrect: Vec<&str> = choices
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != correct)
        .map(|(_, choice)| *choice)
        .collect();
    // Two or more choices were required above, so another one is there.
    let Some(negative) = incorrect.choose(rng) else {
        return refused(
            &causes::CHOICES_FEW,
            vec![
                ("count", choices.len().to_string()),
                ("pointer", fields.choices.clone()),
                ("choices", shown_choices),
            ],
        );
    };
    Ok(pair(question, positive, negative))
}

/// The position `answer` names under --answer-form index: a whole number
/// from 0, or text holding one, inside the row's choices.
fn index_of(
    answer: &Value,
    pointer: &str,
    choices: &[&str],
    shown: String,
) -> Result<usize, Rejection> {
    let stated = match (answer.as_u64(), answer.as_str()) {
        (Some(index), _) => index,
        (None, Some(text)) => match text.trim().parse::<u64>() {
            Ok(index) => index,
            Err(error) => {
                return refused(
                    &causes::ANSWER_NOT_INDEX,
                    vec![
                        ("pointer", pointer.to_owned()),
                        ("found", shape(Some(answer))),
                        ("detail", format!("the text is not a whole number: {error}")),
                    ],
                );
            }
        },
        (None, None) => {
            return refused(
                &causes::ANSWER_NOT_INDEX,
                vec![
                    ("pointer", pointer.to_owned()),
                    ("found", shape(Some(answer))),
                    ("detail", "it is neither a whole number nor text".to_owned()),
                ],
            );
        }
    };
    // An index too large for this machine is past the choices too.
    match usize::try_from(stated) {
        Ok(index) if index < choices.len() => Ok(index),
        _ => refused(
            &causes::ANSWER_INDEX_OUTSIDE,
            vec![
                ("index", stated.to_string()),
                ("count", choices.len().to_string()),
                ("last", (choices.len() - 1).to_string()),
                ("pointer", pointer.to_owned()),
                ("choices", shown),
            ],
        ),
    }
}
