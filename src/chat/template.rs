//! A checkpoint's chat template: reading it out of the files the checkpoint
//! publishes, compiling it once, and rendering turns through it.

use std::{
    collections::BTreeMap,
    fs,
    path::Path,
};

use anyhow::{bail, Context, Result};
use minijinja::Environment;

use super::{jinja, Message};

/// The name the template is compiled under; it appears in engine errors, so it
/// reads as the thing the operator would name.
const TEMPLATE_NAME: &str = "chat_template";

/// A checkpoint's chat template, compiled once.
///
/// The compiled program is kept rather than the source because training
/// renders it twice per example, and re-parsing a two-kilobyte template for
/// every example in a set is work with no result to show for it.
pub struct Template {
    environment: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl Template {
    /// Reads the template a checkpoint publishes, or `None` if it publishes
    /// none.
    ///
    /// Two locations, because Hugging Face moved: the field
    /// `tokenizer_config.json:chat_template` is where every model shipped
    /// before mid-2025 carries it, and `chat_template.jinja` beside it is
    /// where newer repositories do. The standalone file wins when both exist,
    /// which is the precedence `transformers` itself uses — a repository that
    /// carries both left the JSON copy behind for older readers.
    pub fn load(tokenizer_config: Option<&Path>, chat_template: Option<&Path>) -> Result<Option<Self>> {
        let config = match tokenizer_config {
            Some(path) => {
                let bytes = fs::read(path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                let value: serde_json::Value = serde_json::from_slice(&bytes)
                    .with_context(|| format!("invalid tokenizer config {}", path.display()))?;
                Some(value)
            }
            None => None,
        };
        let source = match chat_template {
            Some(path) => Some(
                fs::read_to_string(path)
                    .with_context(|| format!("failed to read {}", path.display()))?,
            ),
            None => match config.as_ref() {
                Some(config) => embedded_template(config)?,
                None => None,
            },
        };
        let Some(source) = source else {
            return Ok(None);
        };
        let bos_token = config.as_ref().and_then(|config| token_text(config, "bos_token"));
        let eos_token = config.as_ref().and_then(|config| token_text(config, "eos_token"));
        // A template that writes `bos_token` into its output and is handed no
        // value for it renders a sequence with no begin-of-sequence marker at
        // all — every token shifted one position off what the model was
        // trained on, under a loss that still looks reasonable. That is the
        // silent failure this module exists to prevent, so it is a refusal.
        for (name, value) in [("bos_token", &bos_token), ("eos_token", &eos_token)] {
            if value.is_none() && source.contains(name) {
                bail!("this model's chat template uses {name}, but its tokenizer config declares none");
            }
        }
        let mut environment = Environment::new();
        environment.set_keep_trailing_newline(true);
        environment.add_function("raise_exception", jinja::raise_exception);
        environment.add_function("strftime_now", jinja::strftime_now);
        environment.set_unknown_method_callback(jinja::python_method);
        environment
            .add_template_owned(TEMPLATE_NAME, jinja::plain_generation_blocks(source))
            .map_err(|error| anyhow::anyhow!("this model's chat template does not parse: {error}"))?;
        Ok(Some(Self { environment, bos_token, eos_token }))
    }

    /// Renders `messages`, optionally followed by the marker that opens the
    /// assistant's turn.
    pub fn render(&self, messages: &[Message<'_>], add_generation_prompt: bool) -> Result<String> {
        let messages: Vec<BTreeMap<&str, &str>> = messages
            .iter()
            .map(|message| {
                BTreeMap::from([("role", message.role), ("content", message.content)])
            })
            .collect();
        let context = minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
        };
        self.environment
            .get_template(TEMPLATE_NAME)
            .and_then(|template| template.render(context))
            .map_err(|error| anyhow::anyhow!("this model's chat template failed to render: {error}"))
    }

    /// One prompt as a user turn, ending exactly where the assistant's own
    /// tokens begin.
    pub fn prompt(&self, prompt: &str) -> Result<String> {
        if prompt.trim().is_empty() {
            bail!("prompt must not be empty");
        }
        self.render(&[Message { role: "user", content: prompt }], true)
    }

    /// A training example split at the boundary the loss starts from: the
    /// whole rendered conversation, cut where the assistant's own words begin.
    ///
    /// Both pieces are slices of one render rather than two renders glued
    /// together, which matters more than it sounds. The obvious construction —
    /// render the prompt with a generation prompt, render the conversation,
    /// and subtract the first from the second — assumes the second begins with
    /// the first, and real templates do not oblige. TinyLlama's is typical:
    /// it emits the generation prompt *inside* its message loop, guarded by
    /// `loop.last and add_generation_prompt`, so the run of newlines between
    /// the user turn and the assistant header differs between the two renders
    /// and the prefix does not match. Nothing is wrong with that template; the
    /// subtraction was wrong.
    ///
    /// So the split is located instead. The two renders agree up to the end of
    /// the prompt's own turn, and the search for the completion starts there,
    /// which is what keeps a completion that also appears inside the prompt
    /// from matching the wrong occurrence. Everything before the completion is
    /// the prompt side — markers, headers and all — and everything from it to
    /// the end is the completion plus whatever the template closes the turn
    /// with, normally the end-of-turn marker. That marker belongs in the loss:
    /// the model has to learn to stop.
    pub fn example(&self, prompt: &str, completion: &str) -> Result<(String, String)> {
        let head = self.prompt(prompt)?;
        let full = self.render(
            &[
                Message { role: "user", content: prompt },
                Message { role: "assistant", content: completion },
            ],
            false,
        )?;
        let agreed = common_prefix(&head, &full);
        // A template may put the content through `trim`, so the exact text is
        // tried first and the trimmed text second. Anything else — a template
        // that rewrites, escapes or drops the content — cannot be split, and
        // guessing where the assistant started would be the silent
        // wrong-boundary failure this whole path exists to avoid.
        let start = full[agreed..]
            .find(completion)
            .or_else(|| full[agreed..].find(completion.trim()))
            .map(|offset| agreed + offset);
        let Some(start) = start else {
            bail!(
                "this model's chat template does not render the assistant turn after the prompt, so the completion boundary cannot be located"
            );
        };
        let tail = &full[start..];
        if tail.is_empty() {
            bail!("this model's chat template rendered an empty assistant turn");
        }
        Ok((full[..start].to_owned(), tail.to_owned()))
    }

    /// One whole utterance as an assistant turn.
    ///
    /// A contrastive pair carries no prompt — both sides are complete
    /// responses — so the only faithful rendering is the turn the model would
    /// have produced. A template that refuses a conversation opening on the
    /// assistant says so through the engine rather than being worked around
    /// here: inventing a user turn to satisfy it would put words in the
    /// operator's data that the operator did not write.
    pub fn response(&self, text: &str) -> Result<String> {
        if text.trim().is_empty() {
            bail!("prompt must not be empty");
        }
        self.render(&[Message { role: "assistant", content: text }], false)
    }
}

/// The byte length of the longest common prefix of two renders, always on a
/// character boundary so the result can slice either of them.
fn common_prefix(left: &str, right: &str) -> usize {
    left.char_indices()
        .zip(right.chars())
        .take_while(|((_, a), b)| a == b)
        .map(|((index, a), _)| index + a.len_utf8())
        .last()
        .unwrap_or(0)
}

/// The `chat_template` field, in either shape `transformers` ever wrote it.
///
/// The list form pairs a template with a name, and only the one called
/// `default` is the conversation format; the others are tool or RAG variants
/// that expect inputs Ster does not have.
fn embedded_template(config: &serde_json::Value) -> Result<Option<String>> {
    match config.get("chat_template") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(source)) => Ok(Some(source.clone())),
        Some(serde_json::Value::Array(entries)) => {
            let named = entries.iter().find(|entry| {
                entry.get("name").and_then(serde_json::Value::as_str) == Some("default")
            });
            match named.or_else(|| entries.first().filter(|_| entries.len() == 1)) {
                Some(entry) => match entry.get("template").and_then(serde_json::Value::as_str) {
                    Some(source) => Ok(Some(source.to_owned())),
                    None => bail!("this model's chat template list has an entry with no template"),
                },
                None => bail!(
                    "this model publishes several named chat templates and none of them is the default"
                ),
            }
        }
        Some(_) => bail!("this model's chat_template is neither a template nor a list of them"),
    }
}

/// A special token as its text, from either the bare string or the
/// `AddedToken` object `transformers` also writes.
fn token_text(config: &serde_json::Value, key: &str) -> Option<String> {
    match config.get(key)? {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Object(object) => {
            object.get("content")?.as_str().map(str::to_owned)
        }
        _ => None,
    }
}
