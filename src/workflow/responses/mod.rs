//! A model-free audit of what a model answered: the `{"prompt",
//! "model_output"}` list `ster generate --prompts` writes, read for refusals
//! and for how varied the answers are. With a steered and an unsteered run of
//! the same prompts it is the refusal rate wisent's evaluate-refusal measured,
//! and the textual half of its evaluate-responses, each judged from the text
//! at a threshold the caller states.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::pairs::{RefusalFlag, quality::diversity, refusal_flag};

/// One answer, as `ster generate --prompts` writes it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub prompt: String,
    pub model_output: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseEntry {
    pub index: usize,
    pub prompt: String,
    pub chars: usize,
    pub words: usize,
    pub refusal: Option<RefusalFlag>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesReport {
    pub path: String,
    pub response_count: usize,
    pub refusal_count: usize,
    /// Refused answers over all answers.
    pub refusal_rate: f64,
    /// The score at or above which an answer counted as a refusal, as the
    /// caller stated it.
    pub refusal_threshold: f32,
    /// How varied the answers are, read across every answer.
    pub diversity: diversity::Scores,
    pub entries: Vec<ResponseEntry>,
}

/// Read the answers at `path` and audit them at `refusal_threshold`.
pub fn inspect_responses(path: &Path, refusal_threshold: f32) -> Result<ResponsesReport> {
    if !refusal_threshold.is_finite() {
        bail!("--refusal-threshold {refusal_threshold} is not a number a score can reach");
    }
    let bytes = fs::read(path).with_context(|| format!("failed to read responses {}", path.display()))?;
    let responses: Vec<Response> = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "{} is not a list of {{\"prompt\", \"model_output\"}} answers as ster generate --prompts writes",
            path.display()
        )
    })?;
    if responses.is_empty() {
        bail!("{} holds no answers to inspect", path.display());
    }
    let entries: Vec<ResponseEntry> = responses
        .iter()
        .enumerate()
        .map(|(index, response)| ResponseEntry {
            index,
            prompt: response.prompt.clone(),
            chars: response.model_output.chars().count(),
            words: response.model_output.split_whitespace().count(),
            refusal: refusal_flag(&response.model_output, refusal_threshold),
        })
        .collect();
    let refusal_count = entries.iter().filter(|entry| entry.refusal.is_some()).count();
    let outputs: Vec<String> = responses.iter().map(|response| response.model_output.clone()).collect();
    Ok(ResponsesReport {
        path: path.display().to_string(),
        response_count: entries.len(),
        refusal_count,
        refusal_rate: refusal_count as f64 / entries.len() as f64,
        refusal_threshold,
        diversity: diversity::compute(&outputs),
        entries,
    })
}
