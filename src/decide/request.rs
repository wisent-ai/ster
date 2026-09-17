//! What a decision asks: one state, and the typed questions put to it. The
//! document is checked whole before a single weight is mapped, and every
//! refusal names the question it is about.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The most options one question may carry. Every option is answered by one
/// letter of the Latin alphabet, read from one token position, and a second
/// letter would be a second position.
pub const MAX_OPTIONS: usize = 26;

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
    Choice { instructions: Value, criteria: BTreeMap<String, Value> },
    /// A position on an ordered scale. `criteria` is the levels in order;
    /// level `i` is the `i`th entry.
    Score { instructions: Value, criteria: Vec<Value> },
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
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let request: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid decide request {}", path.display()))?;
        request.validate()?;
        Ok(request)
    }

    /// Every refusal a request can earn, checked before any model work.
    pub fn validate(&self) -> Result<()> {
        if text_of(&self.state).trim().is_empty() {
            bail!("a decide request has an empty state");
        }
        if self.questions.is_empty() {
            bail!("a decide request needs at least one question");
        }
        for (id, question) in &self.questions {
            if id.trim().is_empty() {
                bail!("a decide request has a question with an empty id");
            }
            if text_of(question.instructions()).trim().is_empty() {
                bail!("question '{id}' has empty instructions");
            }
            match question {
                Question::Choice { criteria, .. } => {
                    if criteria.len() < 2 {
                        bail!("choice question '{id}' needs at least two options");
                    }
                    if criteria.len() > MAX_OPTIONS {
                        bail!(
                            "choice question '{id}' has {} options; Ster labels at most {MAX_OPTIONS}",
                            criteria.len()
                        );
                    }
                    if let Some(name) = criteria.keys().find(|name| name.trim().is_empty()) {
                        bail!("choice question '{id}' has an option with an empty name {name:?}");
                    }
                }
                Question::Score { criteria, .. } => {
                    if criteria.len() < 2 {
                        bail!("score question '{id}' needs at least two levels");
                    }
                    if criteria.len() > MAX_OPTIONS {
                        bail!(
                            "score question '{id}' has {} levels; Ster labels at most {MAX_OPTIONS}",
                            criteria.len()
                        );
                    }
                    if let Some(index) =
                        criteria.iter().position(|level| text_of(level).trim().is_empty())
                    {
                        bail!("score question '{id}' has an empty description at level {index}");
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
                .map(|(name, description)| describe(name, description))
                .collect(),
            Self::Score { criteria, .. } => criteria.iter().map(text_of).collect(),
            Self::Noul { criteria, .. } => {
                let criteria = criteria.clone().unwrap_or_default();
                vec![
                    describe("yes", &criteria.yes.unwrap_or(Value::Null)),
                    describe("no", &criteria.no.unwrap_or(Value::Null)),
                ]
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
fn describe(name: &str, description: &Value) -> String {
    let description = text_of(description);
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
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

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
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let set: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid calibration examples {}", path.display()))?;
        set.validate()?;
        Ok(set)
    }

    pub fn validate(&self) -> Result<()> {
        if self.examples.is_empty() {
            bail!("calibration needs at least one labelled example");
        }
        let mut labelled = 0usize;
        for (index, example) in self.examples.iter().enumerate() {
            example
                .request
                .validate()
                .with_context(|| format!("calibration example {index} is not a valid request"))?;
            for (id, answer) in &example.answers {
                let Some(question) = example.request.questions.get(id) else {
                    bail!("example {index} labels question '{id}', which it does not ask");
                };
                question.truth_index(answer).with_context(|| {
                    format!(
                        "example {index} answers question '{id}' with {answer}, which is not one of its options"
                    )
                })?;
                labelled += 1;
            }
        }
        if labelled == 0 {
            bail!("calibration needs at least one labelled question");
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
}
