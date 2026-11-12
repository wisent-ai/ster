//! Whether a synthesised answer declines the role it was asked to play.
//!
//! Synthesis asks a model to answer as a personality; a safety-tuned model
//! often answers that request with a refusal instead, and a refusal is a
//! useless side of a pair because it differs along the refusal axis rather
//! than along the trait. Which answer is a refusal is judged by the same
//! generator that wrote it, asked one question per answer and held to one
//! exact answer form, not by a list of phrasings: a refusal worded any other
//! way, or in another language, is still a refusal.

use anyhow::Result;
use serde::Deserialize;
use serde_json::error::Category;

use crate::failures::{pairs::synthesize::{VERDICT_MALFORMED, VERDICT_SHAPE}, refuse};

/// The judgement the generator is asked for: one JSON object, nothing else.
pub(crate) fn judgement_prompt(question: &str, personality: &str, answer: &str) -> String {
    format!(
        "A model was asked to answer a question as if it had a personality.\n\nQuestion: {question}\nPersonality: {personality}\nAnswer: {answer}\n\nDid the answer decline or refuse to answer as that personality, instead of answering in character? Reply with exactly one JSON object and nothing else: {{\"refuses\": true}} or {{\"refuses\": false}}."
    )
}

/// The one answer form the judgement accepts.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Verdict {
    refuses: bool,
}

/// The generator's verdict, read from `text`; any answer that is not exactly
/// `{"refuses": <bool>}` is refused, naming the generator and its text: not
/// JSON, or JSON of another form.
pub(crate) fn verdict(generator: &str, text: &str) -> Result<bool> {
    let trimmed = text.trim();
    let verdict: Verdict = serde_json::from_str(trimmed).map_err(|error| {
        let cause = match error.classify() {
            Category::Data => &VERDICT_SHAPE,
            Category::Syntax | Category::Eof | Category::Io => &VERDICT_MALFORMED,
        };
        refuse(
            cause,
            &[
                ("generator", generator),
                ("text", &format!("{text:?}")),
                ("length", &trimmed.chars().count().to_string()),
                ("column", &error.column().to_string()),
                ("error", &error.to_string()),
            ],
        )
    })?;
    Ok(verdict.refuses)
}
