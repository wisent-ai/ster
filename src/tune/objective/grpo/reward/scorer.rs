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

use anyhow::Result;
use serde_json::{Value, json};

use crate::failures::{
    refuse,
    tune::policy::{
        BEARER_EMPTY, BEARER_NOT_UNICODE, SCORER_ANSWER_UNREADABLE, SCORER_FAILING,
        SCORER_FIELD_ABSENT, SCORER_LIMITED, SCORER_NO_NUMBER, SCORER_NOT_JSON, SCORER_NOT_TEXT,
        SCORER_POINTER_INVALID, SCORER_REFUSED, SCORER_UNAUTHORIZED, SCORER_UNREACHABLE,
    },
};

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
        let shown = format!("{source:?}");
        let Some((url, pointer)) = source.split_once('#') else {
            return Err(refuse(&SCORER_FIELD_ABSENT, &[("value", &shown)]));
        };
        if !pointer.starts_with('/') {
            return Err(refuse(
                &SCORER_POINTER_INVALID,
                &[("value", &shown), ("pointer", &format!("{pointer:?}"))],
            ));
        }
        let bearer = match std::env::var("STER_REWARD_BEARER") {
            Ok(value) if value.trim().is_empty() => {
                return Err(refuse(
                    &BEARER_EMPTY,
                    &[("bytes", &value.len().to_string()), ("url", url)],
                ));
            }
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(raw)) => {
                return Err(refuse(
                    &BEARER_NOT_UNICODE,
                    &[("bytes", &raw.len().to_string()), ("url", url)],
                ));
            }
        };
        Ok(Some(Self {
            url: url.to_owned(),
            pointer: pointer.to_owned(),
            bearer,
            agent: ureq::AgentBuilder::new().build(),
        }))
    }

    /// The scorer's reward for `text`, the answer to `prompt`.
    pub fn score(&self, prompt: &str, text: &str) -> Result<f64> {
        let mut request = self.agent.post(&self.url);
        if let Some(bearer) = &self.bearer {
            request = request.set("Authorization", &format!("Bearer {bearer}"));
        }
        let sent = stado_wait::blocking(
            stado_wait::Kind::Network,
            "the reward scorer's score of one completion",
            &self.url,
            || {
                request
                    .set("content-type", "application/json")
                    .send_string(&json!({"prompt": prompt, "text": text}).to_string())
            },
        );
        let url = self.url.as_str();
        let unreadable = |error: std::io::Error| {
            let cause = if error.kind() == std::io::ErrorKind::InvalidData {
                &SCORER_NOT_TEXT
            } else {
                &SCORER_ANSWER_UNREADABLE
            };
            refuse(
                cause,
                &[
                    ("url", url),
                    ("kind", &crate::failures::file::kind(&error)),
                    ("error", &error.to_string()),
                ],
            )
        };
        let body = match sent {
            Ok(response) => response.into_string().map_err(unreadable)?,
            Err(ureq::Error::Status(status, response)) => {
                let body = response.into_string().map_err(unreadable)?;
                let cause = match status {
                    401 | 403 => &SCORER_UNAUTHORIZED,
                    429 => &SCORER_LIMITED,
                    500..=599 => &SCORER_FAILING,
                    _ => &SCORER_REFUSED,
                };
                let bearer = if self.bearer.is_some() { "set" } else { "unset" };
                return Err(refuse(
                    cause,
                    &[
                        ("url", url),
                        ("status", &status.to_string()),
                        ("bearer", bearer),
                        ("body", &wisent_errors::trim_detail(&body, 240)),
                    ],
                ));
            }
            Err(ureq::Error::Transport(transport)) => {
                return Err(refuse(
                    &SCORER_UNREACHABLE,
                    &[
                        ("url", url),
                        ("kind", &format!("{:?}", transport.kind())),
                        ("error", &transport.to_string()),
                    ],
                ));
            }
        };
        let answer: Value = serde_json::from_str(&body).map_err(|error| {
            refuse(
                &SCORER_NOT_JSON,
                &[
                    ("url", url),
                    ("error", &error.to_string()),
                    ("size", &body.len().to_string()),
                    ("line", &error.line().to_string()),
                    ("column", &error.column().to_string()),
                    ("body", &wisent_errors::trim_detail(&body, 240)),
                ],
            )
        })?;
        answer
            .pointer(&self.pointer)
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                refuse(
                    &SCORER_NO_NUMBER,
                    &[
                        ("url", url),
                        ("pointer", &self.pointer),
                        ("body", &wisent_errors::trim_detail(&body, 240)),
                    ],
                )
            })
    }
}
