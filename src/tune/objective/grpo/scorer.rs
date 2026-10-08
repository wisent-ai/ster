//! A reward read from an outside scorer over HTTP, for a quality Ster has no
//! model of (an AI-text detector, a moderation score, a house judge, a game
//! environment that pays a move).
//!
//! `--reward https://host/score#/ai_probability` posts each completion as
//! `{"prompt": "<prompt>", "text": "<completion>"}` to `https://host/score`
//! and reads the reward at the JSON pointer the fragment names
//! (`/ai_probability`); the fragment is never sent. The prompt is the one the
//! group was sampled from: a scorer that judges an answer needs the question.
//! `STER_REWARD_BEARER`, when set, is sent as the bearer. A scorer that
//! refuses, answers something other than JSON, or has no number at the
//! pointer stops the run with what it answered: a reward that is missing is
//! not zero.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

pub struct Scorer {
    url: String,
    pointer: String,
    bearer: Option<String>,
    agent: ureq::Agent,
}

impl Scorer {
    /// The scorer `source` names, when it is an `http(s)://` address.
    pub fn parse(source: &str) -> Result<Option<Self>> {
        let lowered = source.to_ascii_lowercase();
        if !lowered.starts_with("http://") && !lowered.starts_with("https://") {
            return Ok(None);
        }
        let Some((url, pointer)) = source.split_once('#') else {
            bail!(
                "reward scorer {source:?} names no field: end it with #/<json pointer> to the reward, for example #/score"
            );
        };
        if !pointer.starts_with('/') {
            bail!("reward scorer field {pointer:?} is not a JSON pointer; it starts with /");
        }
        Ok(Some(Self {
            url: url.to_owned(),
            pointer: pointer.to_owned(),
            bearer: std::env::var("STER_REWARD_BEARER")
                .ok()
                .filter(|bearer| !bearer.trim().is_empty()),
            agent: ureq::AgentBuilder::new().build(),
        }))
    }

    /// The scorer's reward for `text`, the answer to `prompt`.
    pub fn score(&self, prompt: &str, text: &str) -> Result<f64> {
        let mut request = self.agent.post(&self.url);
        if let Some(bearer) = &self.bearer {
            request = request.set("Authorization", &format!("Bearer {bearer}"));
        }
        let body = match request
            .set("content-type", "application/json")
            .send_string(&json!({"prompt": prompt, "text": text}).to_string())
        {
            Ok(response) => response
                .into_string()
                .with_context(|| format!("reading the reward scorer's answer from {}", self.url))?,
            Err(ureq::Error::Status(status, response)) => bail!(
                "the reward scorer {} answered HTTP {status}: {}",
                self.url,
                response.into_string().unwrap_or_default()
            ),
            Err(ureq::Error::Transport(transport)) => {
                bail!(
                    "the reward scorer {} could not be reached: {transport}",
                    self.url
                )
            }
        };
        let answer: Value = serde_json::from_str(&body).with_context(|| {
            format!(
                "the reward scorer {} answered with something other than JSON: {body}",
                self.url
            )
        })?;
        answer
            .pointer(&self.pointer)
            .and_then(Value::as_f64)
            .with_context(|| {
                format!(
                    "the reward scorer {} answered without a number at {}: {body}",
                    self.url, self.pointer
                )
            })
    }
}
