//! Ster's completions, asked of Stado's inference alternatives.
//!
//! Ster steers locally because steering needs hidden states, but writing
//! contrastive pairs or decision examples is plain text generation: no
//! activations, no local weights. That writer may therefore be any
//! alternative Stado lists, and Ster names none of them: no gateway, no key,
//! no provider. Each request is handed to `stado capabilities call inference
//! --consumer ster --json`, which sends it to the first alternative in
//! Stado's selected order that answers it (Brama, or a statement recorded for
//! that exact request) and prints which one answered beside its answer. An
//! alternative Stado adds, or another order, changes what writes Ster's text
//! without Ster changing. How this layer fails is `ster.inference`
//! (src/failures/inference.rs); where `stado` is found is [`locate`].

mod locate;

use std::process::Command;
use std::sync::Mutex;

use anyhow::Result;
use serde_json::{Value, json};

use crate::failures::{inference as causes, refuse};

/// The consumer Ster asks Stado's capabilities as, as Stado's service
/// directory names it.
const CONSUMER: &str = "ster";

/// The call every completion is, after the program, as a refusal names it.
const CALL: &str = "capabilities call inference --consumer ster --json";

/// One route asked through Stado's inference alternatives.
pub struct Inference {
    model: String,
    /// The alternative that answered the last completion, `stado` before any.
    provider: Mutex<String>,
}

impl Inference {
    /// The route Ster asks for: a Brama alias, a canonical provider/model
    /// route or a selector, given to whichever alternative answers.
    pub fn new(route: &str) -> Result<Self> {
        let model = route.trim();
        if model.is_empty() {
            return Err(refuse(&causes::ROUTE_BLANK, &[("route", route)]));
        }
        Ok(Self {
            model: model.to_owned(),
            provider: Mutex::new("stado".to_owned()),
        })
    }

    /// The route this writer asks for, for reports.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The alternative that answered the last completion (`stado` before
    /// any), for reports.
    pub fn provider(&self) -> String {
        self.provider
            .lock()
            .map(|provider| provider.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
    }

    /// One buffered completion: the assistant text, trimmed.
    ///
    /// The request carries exactly `model`, `messages`, `max_tokens` and
    /// `temperature`, an OpenAI chat-completions request every inference
    /// alternative takes; the bounds on `max_tokens` and `temperature` are the
    /// answering alternative's, and its refusal reaches the reader in Stado's
    /// words. `top_p` and the seed do not travel: no alternative takes them.
    pub fn complete(&self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String> {
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        .to_string();
        let printed = self.call(&body)?;
        if let Some(provider) = printed["resolution"]["provider"].as_str() {
            if let Ok(mut answered) = self.provider.lock() {
                *answered = provider.to_owned();
            }
        }
        let content = printed["answer"]["choices"][0]["message"]["content"].as_str();
        match content {
            Some(content) => Ok(content.trim().to_owned()),
            None => Err(refuse(
                &causes::COMPLETION_MALFORMED,
                &[
                    ("route", &self.model),
                    ("provider", &self.provider()),
                    ("answer", &quoted(&printed["answer"].to_string())),
                ],
            )),
        }
    }

    /// Hand `body` to Stado and read what it printed: the resolution and the
    /// answer, or the refusal it names every alternative in.
    fn call(&self, body: &str) -> Result<Value> {
        let found = locate::program()?;
        let stado = &found.path;
        let shown = format!("{} {CALL}", stado.display());
        let mut command = Command::new(stado);
        command.args([
            "capabilities",
            "call",
            "inference",
            "--consumer",
            CONSUMER,
            "--json",
        ]);
        let output = match stado_wait::output_with_input(&mut command, body.as_bytes()) {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                return Err(locate::not_executable(&found, &shown, &error));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(locate::unlaunchable(&found, &shown, &error));
            }
            Err(error) => {
                return Err(refuse(
                    &causes::CALL_UNFINISHED,
                    &[
                        ("call", &shown),
                        ("error", &error.to_string()),
                        ("kind", &format!("{:?}", error.kind())),
                    ],
                ));
            }
        };
        let printed = serde_json::from_slice::<Value>(&output.stdout);
        if output.status.success() {
            let quoted_out = quoted(&String::from_utf8_lossy(&output.stdout));
            return match printed {
                Ok(printed) if printed.get("answer").is_some() => Ok(printed),
                Ok(_) => Err(refuse(
                    &causes::ANSWER_FIELDLESS,
                    &[("call", &shown), ("printed", &quoted_out)],
                )),
                Err(error) => Err(refuse(
                    &causes::ANSWER_UNREADABLE,
                    &[
                        ("call", &shown),
                        ("printed", &quoted_out),
                        ("error", &error.to_string()),
                    ],
                )),
            };
        }
        // A refusal Stado printed as its `--json` envelope names its cause,
        // its fix and whether asking again can help; without one, Ster has
        // only what Stado wrote on standard error, quoted as it is.
        let envelope = match printed {
            Ok(Value::Object(fields)) => Some(Value::Object(fields)),
            _ => None,
        };
        let text = |key: &str| {
            envelope
                .as_ref()
                .and_then(|envelope| envelope[key].as_str())
                .map(str::trim)
        };
        let named = match (text("functionality"), text("cause")) {
            (Some(functionality), Some(cause)) => format!("{functionality}.{cause}"),
            (None, Some(cause)) => cause.to_owned(),
            _ => "none: Stado printed no --json envelope".to_owned(),
        };
        // Stado's own sentence is kept whole: it names every alternative and
        // the command that answers a request it kept.
        let said = match text("message") {
            Some(message) => message.to_owned(),
            None => quoted(&String::from_utf8_lossy(&output.stderr)),
        };
        let fix = match text("remedy") {
            Some(remedy) => remedy.to_owned(),
            None => "none given".to_owned(),
        };
        let directory = match std::env::current_dir() {
            Ok(directory) => directory.display().to_string(),
            Err(error) => format!("unknown ({error})"),
        };
        let status = output.status.to_string();
        let values = [
            ("call", shown.as_str()),
            ("route", self.model.as_str()),
            ("directory", directory.as_str()),
            ("status", status.as_str()),
            ("named", named.as_str()),
            ("said", said.as_str()),
            ("fix", fix.as_str()),
        ];
        // Stado's envelope says whether asking again can help: no
        // alternative reachable now is; a request it refuses for its form,
        // or a configuration it refuses, is not. Without an envelope, its
        // exit status says the same.
        let retryable = match envelope.as_ref().and_then(|envelope| envelope["retryable"].as_bool()) {
            Some(retryable) => retryable,
            None => output.status.code() == Some(wisent_errors::Code::RETRY_EXIT),
        };
        if retryable {
            Err(refuse(&causes::UNANSWERED, &values))
        } else {
            Err(refuse(&causes::STADO_REFUSED, &values))
        }
    }
}

/// `text` as a refusal quotes it: whole, its ends trimmed.
fn quoted(text: &str) -> String {
    text.trim().to_owned()
}
