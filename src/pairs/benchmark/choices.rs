//! Multiple-choice rows of any dataset export: a question, its choices and
//! the answer, each found where the caller says it sits (a JSON pointer,
//! RFC 6901), because the dataset's schema is what states them. This replaces
//! the per-benchmark Python extractors of wisent-extractors, which guessed:
//! an answer letter mapped by arithmetic, the next choice taken as the wrong
//! one, a missing answer read as index zero and a missing score read as safe.
//!
//! Here an answer is read in the one form the caller names (an index, one of
//! the row's labels, or the text of a choice); a row whose answer is missing
//! or does not resolve is skipped with the reason, never repaired by a guess.
//! The incorrect side is one of the other choices drawn with the import's
//! seed, as TruthfulQA's is.

use std::num::NonZeroUsize;

use anyhow::{Context, Result, anyhow, bail};
use rand::{rngs::StdRng, seq::IndexedRandom};
use serde::Serialize;
use serde_json::Value;

use super::pair;
use crate::ContrastivePair;

/// Readers count rows and lines from one, the way an editor numbers lines.
const FIRST_ROW: usize = NonZeroUsize::MIN.get();

/// How a row states its correct choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnswerForm {
    /// The position of the correct choice, counted from zero, as a number or
    /// as text holding one (HellaSwag's "label": "2").
    Index,
    /// One of the row's labels, aligned with its choices (ARC's "answerKey":
    /// "B" against "choices": {"label": ["A", "B", …]}).
    Label,
    /// The correct choice's own text.
    Text,
}

impl AnswerForm {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "index" => Ok(Self::Index),
            "label" => Ok(Self::Label),
            "text" => Ok(Self::Text),
            other => bail!("unknown answer form {other}: choose index, label or text"),
        }
    }
}

/// Where each part of a multiple-choice row sits, as JSON pointers.
#[derive(Debug, Clone)]
pub struct ChoiceFields {
    pub question: String,
    pub choices: String,
    pub answer: String,
    pub answer_form: AnswerForm,
    /// The labels the answer names; only for [`AnswerForm::Label`].
    pub labels: Option<String>,
}

impl ChoiceFields {
    /// The fields a caller gave. None given is `None`; any given needs the
    /// question, choices, answer and answer form, and labels exactly when the
    /// answer is a label.
    pub fn from_parts(
        question: Option<String>,
        choices: Option<String>,
        answer: Option<String>,
        answer_form: Option<String>,
        labels: Option<String>,
    ) -> Result<Option<Self>> {
        if question.is_none()
            && choices.is_none()
            && answer.is_none()
            && answer_form.is_none()
            && labels.is_none()
        {
            return Ok(None);
        }
        let missing = |flag: &str| {
            anyhow!(
                "multiple-choice rows need {flag}: --question, --choices, --answer and --answer-form name where each part of a row sits"
            )
        };
        let fields = Self {
            question: question.ok_or_else(|| missing("--question"))?,
            choices: choices.ok_or_else(|| missing("--choices"))?,
            answer: answer.ok_or_else(|| missing("--answer"))?,
            answer_form: AnswerForm::parse(&answer_form.ok_or_else(|| missing("--answer-form"))?)?,
            labels,
        };
        let pointers = [
            ("--question", Some(&fields.question)),
            ("--choices", Some(&fields.choices)),
            ("--answer", Some(&fields.answer)),
            ("--labels", fields.labels.as_ref()),
        ];
        for (flag, pointer) in pointers {
            if let Some(pointer) = pointer.filter(|pointer| !pointer.starts_with('/')) {
                bail!(
                    "{flag} {pointer:?} is not a JSON pointer: it starts with / (for example /question or /choices/text)"
                );
            }
        }
        match (fields.answer_form, fields.labels.is_some()) {
            (AnswerForm::Label, false) => {
                bail!("--answer-form label needs --labels, the list of labels the answer names")
            }
            (AnswerForm::Index | AnswerForm::Text, true) => {
                bail!("--labels applies only to --answer-form label")
            }
            _ => Ok(Some(fields)),
        }
    }
}

/// The rows of a JSON list, or of JSON Lines, each with the number a reader
/// counts it by: its position in the list, or its line.
pub(super) fn rows(text: &str, label: &str) -> Result<Vec<(String, Value)>> {
    if text.trim_start().starts_with('[') {
        let rows: Vec<Value> = serde_json::from_str(text)
            .with_context(|| format!("{label} is not a JSON list of rows"))?;
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
                .with_context(|| format!("{label} line {number} is not JSON"))
                .map(|row| (number, row))
        })
        .collect()
}

/// A list whose every element is non-empty text.
fn texts(value: Option<&Value>) -> Option<Vec<&str>> {
    value?
        .as_array()?
        .iter()
        .map(|element| element.as_str().map(str::trim).filter(|text| !text.is_empty()))
        .collect()
}

/// One row's pair, or why the row cannot become one.
pub(super) fn pair_of(
    row: &Value,
    fields: &ChoiceFields,
    rng: &mut StdRng,
) -> Result<ContrastivePair, &'static str> {
    let question = row
        .pointer(&fields.question)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|question| !question.is_empty())
        .ok_or("no question")?;
    let choices =
        texts(row.pointer(&fields.choices)).ok_or("choices are not a list of non-empty text")?;
    let answer = row
        .pointer(&fields.answer)
        .filter(|answer| !answer.is_null())
        .ok_or("no answer")?;
    let correct = match fields.answer_form {
        AnswerForm::Index => answer
            .as_u64()
            .or_else(|| answer.as_str().and_then(|text| text.trim().parse().ok()))
            .and_then(|index| usize::try_from(index).ok())
            .ok_or("answer is not a whole-number index")?,
        AnswerForm::Label => {
            let labels = texts(fields.labels.as_deref().and_then(|pointer| row.pointer(pointer)))
                .ok_or("labels are not a list of non-empty text")?;
            if labels.len() != choices.len() {
                return Err("labels and choices differ in length");
            }
            let named = answer.as_str().map(str::trim).ok_or("answer is not a label")?;
            labels
                .iter()
                .position(|label| *label == named)
                .ok_or("answer is not one of the labels")?
        }
        AnswerForm::Text => {
            let named = answer.as_str().map(str::trim).ok_or("answer is not text")?;
            choices
                .iter()
                .position(|choice| *choice == named)
                .ok_or("answer is not one of the choices")?
        }
    };
    let positive = choices
        .get(correct)
        .ok_or("answer index is outside the choices")?;
    let incorrect: Vec<&str> = choices
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != correct)
        .map(|(_, choice)| *choice)
        .collect();
    let negative = incorrect.choose(rng).ok_or("no incorrect choice")?;
    Ok(pair(question, positive, negative))
}
