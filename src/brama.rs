//! Client for Brama, the gateway every Wisent model call goes through.
//!
//! Ster steers locally because steering needs hidden states, but writing
//! contrastive pairs is plain text generation: no activations, no local
//! weights. That writer may therefore be a hosted model, and Wisent's rule is
//! that a hosted model is reached through Brama and never through a provider
//! SDK. This module is the whole of that reach — one synchronous POST, no
//! async runtime, no provider credentials of its own.

use std::env;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

/// The gateway's own documented client variables.
///
/// Ster reads the base URL and the bearer from its own environment and never
/// talks to Skarbiec itself. Nothing in this repository, and nothing on the
/// fleet, puts the bearer there: no launcher script, no wrapper, no service
/// unit, and not Ster Desktop, which runs `ster request` with the environment
/// it inherited and sets no variable of its own. Whoever runs Ster exports
/// `BRAMA_BEARER` themselves. Say so plainly here rather than naming a
/// launcher, because a comment that points at a mechanism nobody wrote stops
/// the reader looking for the one that is missing.
///
/// The credential to export is Ster's own, not another product's. Ster is a
/// declared consumer of `brama` in Stado's service directory, and the bearer
/// is a Skarbiec grant for the consumer `ster` whose capability is
/// `call:brama#<route>` — the grant Brama resolves by introspection when it
/// meets a bearer its own start did not preload. It is minted with
/// `stado host vault-token-mint <gateway-host> ster --capabilities
/// 'call:brama#<route>' --audience ster --raw-token`.
///
/// The host argument is load-bearing. Brama introspects the vault on its own
/// host, so a grant minted in a workstation's vault authenticates against that
/// workstation and is still answered `401 unauthenticated` by the gateway,
/// which never asks it. Mint it where the gateway looks.
pub const URL_VAR: &str = "BRAMA_URL";
pub const BEARER_VAR: &str = "BRAMA_BEARER";

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
    /// Reads `BRAMA_URL` and `BRAMA_BEARER`, both required.
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
            agent: ureq::AgentBuilder::new().build(),
        })
    }

    /// The route this gateway generates with, for reports.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// One buffered completion. Returns the assistant text, trimmed.
    ///
    /// The bounds on `max_tokens` and `temperature` are Brama's: it refuses a
    /// value outside them on the first pair, and that refusal reaches the
    /// operator in Brama's own words through [`refusal`], so no second copy of
    /// its limits is kept here to drift from them.
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
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        .to_string();
        let url = format!("{}/v1/chat/completions", self.base);
        let response = stado_wait::blocking(
            stado_wait::Kind::Network,
            format!("Brama completion from {}", self.model),
            &url,
            || {
                self.agent
                    .post(&url)
                    .set("authorization", &format!("Bearer {}", self.bearer))
                    .set("content-type", "application/json")
                    .send_string(&body)
            },
        );
        let response = match response {
            Ok(response) => response,
            // `ureq` hands back every non-2xx as this arm, response body intact.
            Err(ureq::Error::Status(status, response)) => {
                let body = response.into_string().with_context(|| {
                    format!(
                        "failed to read Brama refusal from {} with status {status}",
                        self.base
                    )
                })?;
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
        let value: Value = serde_json::from_str(&text).with_context(|| {
            format!("Brama returned a completion body that is not JSON: {text}")
        })?;
        let content = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str);
        match content {
            Some(content) => Ok(content.trim().to_owned()),
            None => bail!("Brama's answer carried no assistant message: {text}"),
        }
    }
}

/// `BRAMA_URL` with trailing slashes trimmed. There is no built-in gateway:
/// an unset or empty value is refused by name (cli.md rule 14).
///
/// A non-unicode value is refused by name and never echoed: the same launcher
/// exports the bearer, and a message that quotes environment values is one
/// copy-paste away from leaking it.
fn base_url() -> Result<String> {
    let raw = match env::var(URL_VAR) {
        Ok(raw) => raw,
        Err(env::VarError::NotPresent) => String::new(),
        Err(env::VarError::NotUnicode(_)) => bail!("{URL_VAR} is not valid unicode"),
    };
    let base = raw.trim().trim_end_matches('/');
    if base.is_empty() {
        bail!("{URL_VAR} is unset or empty; export the Brama gateway base Ster should call");
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
    let Some(rest) = base.strip_prefix("http://") else {
        return false;
    };
    let authority = match rest.split_once(['/', '?', '#']) {
        Some((authority, _)) => authority,
        None => rest,
    };
    // Strip the port, but not the colons inside a bracketed IPv6 literal.
    let host = match authority.strip_prefix('[') {
        Some(bracketed) => match bracketed.split_once(']') {
            Some((host, _)) => host,
            None => return false,
        },
        None => match authority.split_once(':') {
            Some((host, _)) => host,
            None => authority,
        },
    };
    host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

/// `BRAMA_BEARER`, trimmed and required.
///
/// A value that is not Unicode is refused without being formatted, because
/// the value is the bearer. The sentence tells the operator to export it,
/// because that is the only way it ever arrives: no launcher in this
/// repository or on the fleet sets it.
fn bearer() -> Result<String> {
    match env::var(BEARER_VAR) {
        Ok(bearer) if !bearer.trim().is_empty() => Ok(bearer.trim().to_owned()),
        Ok(_) | Err(env::VarError::NotPresent) => bail!(
            "{BEARER_VAR} is unset or empty; export Ster's own Brama bearer before running this command"
        ),
        Err(env::VarError::NotUnicode(_)) => bail!(
            "{BEARER_VAR} is not Unicode; export Ster's own Brama bearer as text before running this command"
        ),
    }
}

/// Turns a non-2xx into Brama's own words.
///
/// Every Brama route fails with `{"error":{"code","message","type","retryable",
/// "attempts"}}`, so the operator should read the gateway's sentence, not a
/// paraphrase of it. Anything else on the wire is a proxy or a misconfigured
/// host; preserve that answer whole instead of pretending the envelope was there.
fn refusal(status: u16, body: &str) -> anyhow::Error {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        let message = value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str);
        if let Some(message) = message {
            return anyhow::anyhow!("brama refused the completion: {status} {message}");
        }
    }
    anyhow::anyhow!(
        "brama refused the completion with {status} and a body that is not its error envelope: {body}"
    )
}
