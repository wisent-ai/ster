//! What a decision asks: one state, and the typed questions put to it. The
//! document is checked whole before a single weight is mapped, and every
//! refusal names the question it is about.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::prompt::LABELS;
use crate::failures::{
    decide::request::{
        FORM, ID_EMPTY, INSTRUCTIONS_EMPTY, LEVEL_EMPTY, MISSING, OPTION_NAME_EMPTY,
        OPTIONS_TOO_FEW, OPTIONS_TOO_MANY, QUESTIONS_NONE, STATE_EMPTY,
    },
    file, refuse,
};

/// What a decision request is called in a refusal of its file.
pub const ROLE: &str = "the decision request";

/// The most options one question may carry. Every option is answered by one
/// letter of [`LABELS`], read from one token position, and a second letter
/// would be a second position; the cap is read from the letters themselves
/// so the two cannot diverge.
pub const MAX_OPTIONS: usize = LABELS.len();

/// A decision request: TypeSafe's System One request shape, so a document
/// written for that API runs here unchanged. `model` is accepted and ignored,
/// because the model is the checkpoint Ster was started with.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Request {
    /// The content every question is judged against: a string, or structured
    /// data that is rendered as JSON.
    pub state: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Question id to question. The id is the caller's; the model never sees
    /// it, and the answer comes back under it.
    pub questions: BTreeMap<String, Question>,
}

/// The three question types, tagged exactly as the request carries them.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// One option out of a named set. `criteria` maps option name to a
    /// description, or `null` when the name says enough.
    Choice {
        instructions: Value,
        criteria: BTreeMap<String, Value>,
    },
    /// A position on an ordered scale. `criteria` is the levels in order;
    /// level `i` is the `i`th entry.
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
    /// A yes/no judgement, optionally with what each side means.
    Noul {
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<Value>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub no: Option<Value>,
}

impl Request {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            fs::read(path).map_err(|error| file::read::refusal_or(&MISSING, ROLE, path, &error))?;
        Self::parse(&bytes, path)
    }

    /// The request `bytes` hold, read from `source` (a path, or /dev/stdin
    /// for standard input), checked whole.
    pub fn parse(bytes: &[u8], source: &Path) -> Result<Self> {
        let request: Self = serde_json::from_slice(bytes)
            .map_err(|error| file::json::refusal(ROLE, FORM, source, &error))?;
        request.validate()?;
        Ok(request)
    }

    /// Every refusal a request can earn, checked before any model work.
    pub fn validate(&self) -> Result<()> {
        if text_of(&self.state).trim().is_empty() {
            return Err(refuse(&STATE_EMPTY, &[("state", &self.state.to_string())]));
        }
        if self.questions.is_empty() {
            // The empty object it holds is a value of the observed line.
            return Err(refuse(&QUESTIONS_NONE, &[("questions", "{}")]));
        }
        for (id, question) in &self.questions {
            if id.trim().is_empty() {
                return Err(refuse(&ID_EMPTY, &[("id", id)]));
            }
            let instructions = question.instructions();
            if text_of(instructions).trim().is_empty() {
                return Err(refuse(
                    &INSTRUCTIONS_EMPTY,
                    &[("id", id), ("instructions", &instructions.to_string())],
                ));
            }
            let (kind, unit) = match question {
                Question::Choice { .. } => ("choice", "options"),
                Question::Score { .. } => ("score", "levels"),
                Question::Noul { .. } => continue,
            };
            let count = question.option_count();
            let (count_text, max_text) = (count.to_string(), MAX_OPTIONS.to_string());
            let values: [(&str, &str); 5] = [
                ("kind", kind),
                ("id", id.as_str()),
                ("unit", unit),
                ("count", count_text.as_str()),
                ("max", max_text.as_str()),
            ];
            if count < 2 {
                return Err(refuse(&OPTIONS_TOO_FEW, &values));
            }
            if count > MAX_OPTIONS {
                return Err(refuse(&OPTIONS_TOO_MANY, &values));
            }
            match question {
                Question::Choice { criteria, .. } => {
                    if let Some(name) = criteria.keys().find(|name| name.trim().is_empty()) {
                        return Err(refuse(&OPTION_NAME_EMPTY, &[("id", id), ("name", name)]));
                    }
                }
                Question::Score { criteria, .. } => {
                    if let Some((index, level)) = criteria
                        .iter()
                        .enumerate()
                        .find(|(_, level)| text_of(level).trim().is_empty())
                    {
                        return Err(refuse(
                            &LEVEL_EMPTY,
                            &[
                                ("id", id),
                                ("index", &index.to_string()),
                                ("level", &level.to_string()),
                            ],
                        ));
                    }
                }
                Question::Noul { .. } => {}
            }
        }
        Ok(())
    }
}

impl Question {
    pub fn instructions(&self) -> &Value {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }

    /// The option texts in canonical order: choice options by name, score
    /// levels by index, and yes then no for a noul. Answers are reported in
    /// this order whatever order the model was shown them in.
    pub fn options(&self) -> Vec<String> {
        match self {
            Self::Choice { criteria, .. } => criteria
                .iter()
                .map(|(name, description)| describe(name, Some(description)))
                .collect(),
            Self::Score { criteria, .. } => criteria.iter().map(text_of).collect(),
            Self::Noul { criteria, .. } => {
                // A noul without `criteria`, or without one side of it, shows
                // that side by its name alone.
                let (yes, no) = match criteria {
                    Some(NoulCriteria { yes, no }) => (yes.as_ref(), no.as_ref()),
                    None => (None, None),
                };
                vec![describe("yes", yes), describe("no", no)]
            }
        }
    }

    pub fn option_count(&self) -> usize {
        match self {
            Self::Choice { criteria, .. } => criteria.len(),
            Self::Score { criteria, .. } => criteria.len(),
            Self::Noul { .. } => 2,
        }
    }
}

/// A name and its description as one option line: `name: description`, or
/// the name alone when there is no description.
fn describe(name: &str, description: Option<&Value>) -> String {
    let description = match description {
        Some(description) => text_of(description),
        None => String::new(),
    };
    if description.trim().is_empty() {
        name.to_owned()
    } else {
        format!("{name}: {description}")
    }
}

/// A request field as the text the model reads: a string as itself, `null` as
/// nothing, and any structured value as its JSON.
pub fn text_of(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => format!("{other:#}"),
    }
}
