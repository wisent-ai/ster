//! Labelled decisions: the requests a calibration or a decision adapter is
//! fitted on, each with the answer a person gave to some of its questions.
//! The set is checked whole before a weight is mapped, and every refusal
//! names the example it is about.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::request::{Question, Request};
use crate::failures::{
    decide::calibration::{
        EXAMPLES_FORM, EXAMPLES_MISSING, EXAMPLES_NONE, LABEL_NOT_OPTION, LABEL_UNASKED,
        LABELS_NONE,
    },
    file, refuse,
};

/// What a labelled example set is called in a refusal of its file.
const ROLE: &str = "the labelled example set";

/// Labelled decisions a calibration is fitted on: each example is a request
/// plus the answer a person gave to some or all of its questions.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExampleSet {
    pub examples: Vec<Example>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Example {
    #[serde(flatten)]
    pub request: Request,
    /// Question id to the correct answer: an option name for a choice, a
    /// level index for a score, `true` or `false` for a noul.
    pub answers: BTreeMap<String, Value>,
}

impl ExampleSet {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .map_err(|error| file::read::refusal_or(&EXAMPLES_MISSING, ROLE, path, &error))?;
        let set: Self = serde_json::from_slice(&bytes)
            .map_err(|error| file::json::refusal(ROLE, EXAMPLES_FORM, path, &error))?;
        set.validate_from(&format!("{} ({})", path.display(), file::state(path)))?;
        Ok(set)
    }

    /// Refuses a set no temperature can be fitted on; one built in this run.
    pub fn validate(&self) -> Result<()> {
        self.validate_from("the set this run built")
    }

    /// [`ExampleSet::validate`], `source` naming where the set came from.
    pub fn validate_from(&self, source: &str) -> Result<()> {
        if self.examples.is_empty() {
            return Err(refuse(&EXAMPLES_NONE, &[("source", source)]));
        }
        let mut labelled = 0usize;
        for (index, example) in self.examples.iter().enumerate() {
            // The request's own declared refusal, told which example it is in.
            example.request.validate().map_err(|error| {
                crate::failures::amend(error, format!("it is labelled example {index} of {}", self.examples.len()))
            })?;
            for (id, answer) in &example.answers {
                let Some(question) = example.request.questions.get(id) else {
                    let asked: Vec<&str> =
                        example.request.questions.keys().map(String::as_str).collect();
                    return Err(refuse(
                        &LABEL_UNASKED,
                        &[
                            ("index", &index.to_string()),
                            ("id", id),
                            ("asked", &asked.join(", ")),
                        ],
                    ));
                };
                if question.truth_index(answer).is_none() {
                    return Err(refuse(
                        &LABEL_NOT_OPTION,
                        &[
                            ("index", &index.to_string()),
                            ("id", id),
                            ("answer", &answer.to_string()),
                            ("options", &question.answer_forms()),
                        ],
                    ));
                }
                labelled += 1;
            }
        }
        if labelled == 0 {
            return Err(refuse(
                &LABELS_NONE,
                &[("count", &self.examples.len().to_string())],
            ));
        }
        Ok(())
    }
}

impl Question {
    /// The canonical option index a labelled answer names, or `None` when the
    /// label is not one of this question's options.
    pub fn truth_index(&self, answer: &Value) -> Option<usize> {
        match self {
            Self::Choice { criteria, .. } => {
                let name = answer.as_str()?;
                criteria.keys().position(|option| option == name)
            }
            Self::Score { criteria, .. } => {
                let level = answer.as_u64()? as usize;
                (level < criteria.len()).then_some(level)
            }
            Self::Noul { .. } => answer.as_bool().map(|yes| if yes { 0 } else { 1 }),
        }
    }

    /// The answers a label may give this question, as a refusal lists them.
    pub fn answer_forms(&self) -> String {
        match self {
            Self::Choice { criteria, .. } => criteria
                .keys()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", "),
            Self::Score { criteria, .. } => {
                format!("level indexes from 0, below {}", criteria.len())
            }
            Self::Noul { .. } => "true, false".to_owned(),
        }
    }
}
