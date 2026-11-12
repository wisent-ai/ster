//! Where a multiple-choice row holds its parts: the row flags of
//! `--benchmark choices`, each a JSON pointer (RFC 6901), and the form its
//! answer takes. All of it is checked before any export is read.

use anyhow::Result;
use serde::Serialize;

use super::Benchmark;
use crate::failures::{pairs::import, refuse};

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
            other => Err(refuse(
                &import::ANSWER_FORM_UNKNOWN,
                &[("form", other), ("benchmark", Benchmark::Choices.name())],
            )),
        }
    }

    /// The name `--answer-form` takes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Label => "label",
            Self::Text => "text",
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

/// The row flags given, as a refusal shows them: `--question /q, --labels /l`.
fn shown_flags(flags: &[(&str, Option<&str>)]) -> String {
    flags
        .iter()
        .filter_map(|(flag, value)| value.map(|value| format!("{flag} {value}")))
        .collect::<Vec<_>>()
        .join(", ")
}

impl ChoiceFields {
    /// The fields a caller gave for `benchmark`, checked before any export
    /// is read. Only `choices` takes them, and needs the question, choices,
    /// answer and answer form, and labels exactly when the answer is a label;
    /// every other benchmark is `None` and refuses any of them.
    pub fn from_parts(
        benchmark: Benchmark,
        question: Option<String>,
        choices: Option<String>,
        answer: Option<String>,
        answer_form: Option<String>,
        labels: Option<String>,
    ) -> Result<Option<Self>> {
        let shown = shown_flags(&[
            ("--question", question.as_deref()),
            ("--choices", choices.as_deref()),
            ("--answer", answer.as_deref()),
            ("--answer-form", answer_form.as_deref()),
            ("--labels", labels.as_deref()),
        ]);
        let name = benchmark.name();
        match (benchmark, shown.is_empty()) {
            (Benchmark::Choices, true) => {
                return Err(refuse(
                    &import::CHOICE_FIELDS_MISSING,
                    &[("benchmark", name)],
                ));
            }
            (Benchmark::Choices, false) => {}
            (_, true) => return Ok(None),
            (_, false) => {
                return Err(refuse(
                    &import::CHOICE_FIELDS_NOT_APPLICABLE,
                    &[("benchmark", name), ("given", &shown)],
                ));
            }
        }
        let missing = |flag: &str| {
            refuse(
                &import::CHOICE_FIELD_MISSING,
                &[("flag", flag), ("given", &shown), ("benchmark", name)],
            )
        };
        let fields = Self {
            question: question.ok_or_else(|| missing("--question"))?,
            choices: choices.ok_or_else(|| missing("--choices"))?,
            answer: answer.ok_or_else(|| missing("--answer"))?,
            answer_form: AnswerForm::parse(&answer_form.ok_or_else(|| missing("--answer-form"))?)?,
            labels,
        };
        fields.check(name)?;
        Ok(Some(fields))
    }

    /// Every pointer starts with `/`, and labels are given exactly when the
    /// answer is one.
    fn check(&self, benchmark: &str) -> Result<()> {
        let pointers = [
            ("--question", Some(&self.question)),
            ("--choices", Some(&self.choices)),
            ("--answer", Some(&self.answer)),
            ("--labels", self.labels.as_ref()),
        ];
        for (flag, pointer) in pointers {
            if let Some(pointer) = pointer.filter(|pointer| !pointer.starts_with('/')) {
                return Err(refuse(
                    &import::POINTER_INVALID,
                    &[
                        ("flag", flag),
                        ("pointer", &format!("{pointer:?}")),
                        ("benchmark", benchmark),
                    ],
                ));
            }
        }
        let form = self.answer_form.name();
        match (self.answer_form, &self.labels) {
            (AnswerForm::Label, None) => Err(refuse(
                &import::LABELS_MISSING,
                &[("benchmark", benchmark), ("answer", &self.answer), ("choices", &self.choices)],
            )),
            (AnswerForm::Index | AnswerForm::Text, Some(labels)) => Err(refuse(
                &import::LABELS_NOT_APPLICABLE,
                &[("form", form), ("labels", labels), ("benchmark", benchmark)],
            )),
            _ => Ok(()),
        }
    }

    /// The flags these fields were given as, for a refusal to show.
    pub(super) fn given(&self) -> String {
        shown_flags(&[
            ("--question", Some(self.question.as_str())),
            ("--choices", Some(self.choices.as_str())),
            ("--answer", Some(self.answer.as_str())),
            ("--answer-form", Some(self.answer_form.name())),
            ("--labels", self.labels.as_deref()),
        ])
    }
}
