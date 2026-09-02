//! Client for Brama, the gateway every Wisent model call goes through.
//!
//! Ster steers locally because steering needs hidden states, but writing
//! contrastive pairs is plain text generation: no activations, no local
//! weights. That writer may therefore be a hosted model, and Wisent's rule is
//! that a hosted model is reached through Brama and never through a provider
//! SDK. This module is the whole of that reach — one synchronous POST, no
//! async runtime, no provider credentials of its own.

use std::{env, time::Duration};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

/// The gateway's own documented client variables.
///
/// Ster reads the base URL and the bearer from its own environment and never
/// talks to Skarbiec itself. Nothing in this repository, and nothing on the
/// fleet, puts the bearer there: no launcher script, no wrapper, no service
/// unit, and not Ster Desktop, which spawns `ster serve` with the environment
/// it inherited and sets no variable of its own. Whoever runs Ster exports
/// `BRAMA_BEARER` themselves. Say so plainly here rather than naming a
/// launcher, because a comment that points at a mechanism nobody wrote stops
/// the reader looking for the one that is missing.
///
/// The credential to export is Ster's own, not another product's. Ster is a
/// declared consumer of `brama` in Stado's service directory, and Skarbiec
/// holds a client identity minted for the consumer `ster` whose capability is
/// `call:brama#<route>` — the grant Brama resolves by introspection when it
/// meets a bearer its own start did not preload.
pub const URL_VAR: &str = "BRAMA_URL";
pub const BEARER_VAR: &str = "BRAMA_BEARER";
pub const DEFAULT_URL: &str = "https://brama.wisent.com";

/// Matches Brama's documented `requestDeadlineSeconds`, so Ster gives up at the
/// same moment the gateway does instead of holding a socket the other side has
/// already abandoned.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Brama refuses `max_tokens` outside `1..=32768`; see [`Gateway::complete`].
const MAX_TOKENS_LIMIT: usize = 32_768;

/// How much of an unrecognized error body is worth quoting back. Long enough to
/// identify a proxy's HTML error page, short enough not to flood a terminal.
const BODY_EXCERPT: usize = 240;

pub struct Gateway {
    base: String,
    bearer: String,
    model: String,
    /// Built once per gateway: a `ureq` agent owns the connection pool and the
    /// TLS setup, so rebuilding it per call would pay a fresh handshake for
    /// every pair in a synthesis run.
    agent: ureq::Agent,
}

impl Gateway {
    /// Reads `BRAMA_URL` (defaulted) and `BRAMA_BEARER` (required).
    pub fn from_env(model: &str) -> Result<Self> {
        let model = model.trim();
        if model.is_empty() {
            bail!(
                "the generator model must be a Brama alias, a canonical provider/model route, or a selector"
            );
        }
        let base = base_url()?;
        require_secure_transport(&base)?;
        Ok(Self {
            base,
            bearer: bearer()?,
            model: model.to_owned(),
            agent: ureq::AgentBuilder::new().timeout(REQUEST_TIMEOUT).build(),
        })
    }

    /// The route this gateway generates with, for reports.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// One buffered completion. Returns the assistant text, trimmed.
    ///
    /// The two bound checks below duplicate limits Brama already enforces. That
    /// is deliberate: a run that asks for 40000 tokens is wrong on the first
    /// pair and every pair after it, so refusing here saves a round trip per
    /// pair. The sentences are Brama's own, verbatim, so the operator reads the
    /// same words whichever side of the wire refuses.
    ///
    /// The request body carries exactly `model`, `messages`, `max_tokens` and
    /// `temperature`. Brama's deserializer refuses unknown fields by name, so
    /// nothing else may appear — that is why the sampling `top_p` that
    /// `ster generate` exposes is not forwarded here: the gateway has no such
    /// field and would answer `400 invalid JSON: unknown field `top_p``.
    pub fn complete(&self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String> {
        if prompt.trim().is_empty() {
            bail!("the generator prompt is empty");
        }
        if max_tokens == 0 || max_tokens > MAX_TOKENS_LIMIT {
            bail!("max_tokens must be between one and 32768");
        }
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            bail!("temperature must be finite and between zero and 2");
        }
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        .to_string();
        let response = self
            .agent
            .post(&format!("{}/v1/chat/completions", self.base))
            .set("authorization", &format!("Bearer {}", self.bearer))
            .set("content-type", "application/json")
            .send_string(&body);
        let response = match response {
            Ok(response) => response,
            // `ureq` hands back every non-2xx as this arm, response body intact.
            Err(ureq::Error::Status(status, response)) => {
                let body = response.into_string().unwrap_or_default();
                return Err(refusal(status, &body));
            }
            // The transport error quotes the URL it failed on; the bearer lives
            // in a header and never appears in it.
            Err(ureq::Error::Transport(transport)) => {
                let detail = match transport.message() {
                    Some(message) => format!("{} ({message})", transport.kind()),
                    None => transport.kind().to_string(),
                };
                bail!("failed to reach Brama at {}: {detail}", self.base);
            }
        };
        let text = response
            .into_string()
            .context("failed to read Brama's completion body")?;
        let value: Value =
            serde_json::from_str(&text).context("Brama returned a completion body that is not JSON")?;
        let content = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str);
        match content {
            Some(content) => Ok(content.trim().to_owned()),
            None => bail!("Brama's answer carried no assistant message"),
        }
    }
}

/// `BRAMA_URL` with trailing slashes trimmed, or the documented default.
///
/// A non-unicode value is refused by name and never echoed: the same launcher
/// exports the bearer, and a message that quotes environment values is one
/// copy-paste away from leaking it.
fn base_url() -> Result<String> {
    let raw = match env::var(URL_VAR) {
        Ok(raw) => raw,
        Err(env::VarError::NotPresent) => return Ok(DEFAULT_URL.to_owned()),
        Err(env::VarError::NotUnicode(_)) => bail!("{URL_VAR} is not valid unicode"),
    };
    let base = raw.trim().trim_end_matches('/');
    if base.is_empty() {
        return Ok(DEFAULT_URL.to_owned());
    }
    Ok(base.to_owned())
}

/// Refuses a base Brama itself would answer `426 secure_transport_required`.
///
/// Catching it here keeps the operator debugging their own configuration
/// instead of reading a gateway status code they never asked for. Loopback is
/// allowed unencrypted because that is how a developer runs Brama locally.
fn require_secure_transport(base: &str) -> Result<()> {
    if base.starts_with("https://") || is_loopback(base) {
        return Ok(());
    }
    bail!(
        "{URL_VAR} must be an https:// base or an explicit http:// loopback address, because Brama answers plain http elsewhere with 426 secure_transport_required"
    );
}

fn is_loopback(base: &str) -> bool {
    let Some(authority) = base.strip_prefix("http://") else {
        return false;
    };
    let host = authority
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    // Strip the port, but not the colons inside a bracketed IPv6 literal.
    let host = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or_default(),
        None => host.split(':').next().unwrap_or_default(),
    };
    host == "127.0.0.1" || host == "localhost" || host == "::1"
}

/// `BRAMA_BEARER`, trimmed and required.
///
/// Every failure collapses to one sentence that names the variable and nothing
/// else. `env::VarError` is swallowed rather than propagated on purpose: its
/// `NotUnicode` variant formats the offending value, which here is the bearer.
///
/// The sentence tells the operator to export it, because that is the only way
/// it ever arrives: no launcher in this repository or on the fleet sets it.
fn bearer() -> Result<String> {
    let bearer = env::var(BEARER_VAR).unwrap_or_default();
    let bearer = bearer.trim();
    if bearer.is_empty() {
        bail!(
            "{BEARER_VAR} is unset or empty; export Ster's own Brama bearer before running this command"
        );
    }
    Ok(bearer.to_owned())
}

/// Turns a non-2xx into Brama's own words.
///
/// Every Brama route fails with `{"error":{"code","message","type","retryable",
/// "attempts"}}`, so the operator should read the gateway's sentence, not a
/// paraphrase of it. Anything else on the wire is a proxy or a misconfigured
/// host, and saying so plus a bounded excerpt is more useful than pretending
/// the envelope was there.
fn refusal(status: u16, body: &str) -> anyhow::Error {
    let message = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    match message {
        Some(message) => anyhow::anyhow!("brama refused the completion: {status} {message}"),
        None => anyhow::anyhow!(
            "brama refused the completion with {status} and a body that is not its error envelope: {}",
            excerpt(body)
        ),
    }
}

fn excerpt(body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        return "<empty body>".to_owned();
    }
    match body.char_indices().nth(BODY_EXCERPT) {
        Some((cut, _)) => format!("{}…", &body[..cut]),
        None => body.to_owned(),
    }
}
