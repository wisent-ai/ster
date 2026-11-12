//! A checkpoint's chat template: reading it out of the files the checkpoint
//! publishes, compiling it once, and rendering turns through it.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use minijinja::Environment;

use super::{Message, jinja};
use crate::failures::{
    chat as causes, model::tokenizer::read_refusal, refuse, runtime::encode::TEXT_BLANK,
};
use crate::runtime::vocabulary::tokenizer_config as read_tokenizer_config;

mod config;

use config::{embedded_template, token_text};

/// Refuses `text`, the `what` a turn is made of, when it holds no text.
fn filled(text: &str, what: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(refuse(
            &TEXT_BLANK,
            &[("what", what), ("text", &format!("{text:?}"))],
        ));
    }
    Ok(())
}

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
    pub fn load(
        tokenizer_config: Option<&Path>,
        chat_template: Option<&Path>,
    ) -> Result<Option<Self>> {
        let config = match tokenizer_config {
            Some(path) => Some(read_tokenizer_config(path)?),
            None => None,
        };
        let config_shown =
            tokenizer_config.map_or("absent".to_owned(), |path| path.display().to_string());
        let (source, from) = match chat_template {
            Some(path) => (
                Some(
                    fs::read_to_string(path)
                        .map_err(|error| read_refusal("chat template", path, &error))?,
                ),
                path.display().to_string(),
            ),
            None => match config.as_ref() {
                Some(config) => (
                    embedded_template(config, &config_shown)?,
                    format!("the chat_template of {config_shown}"),
                ),
                None => (None, String::new()),
            },
        };
        let Some(source) = source else {
            return Ok(None);
        };
        let bos_token = config
            .as_ref()
            .and_then(|config| token_text(config, "bos_token"));
        let eos_token = config
            .as_ref()
            .and_then(|config| token_text(config, "eos_token"));
        // A template that writes `bos_token` into its output and is handed no
        // value for it renders a sequence with no begin-of-sequence marker at
        // all — every token shifted one position off what the model was
        // trained on, under a loss that still looks reasonable. That is the
        // silent failure this module exists to prevent, so it is a refusal.
        for (name, value) in [("bos_token", &bos_token), ("eos_token", &eos_token)] {
            if value.is_none() && source.contains(name) {
                return Err(refuse(
                    &causes::TOKEN_UNDECLARED,
                    &[("name", name), ("config", &config_shown), ("source", &from)],
                ));
            }
        }
        let mut environment = Environment::new();
        environment.set_keep_trailing_newline(true);
        environment.add_function("raise_exception", jinja::raise_exception);
        environment.add_function("strftime_now", jinja::strftime_now);
        environment.set_unknown_method_callback(jinja::python_method);
        environment
            .add_template_owned(TEMPLATE_NAME, jinja::plain_generation_blocks(source))
            .map_err(|error| {
                refuse(
                    &causes::UNPARSABLE,
                    &[("line", &line(&error)), ("error", &error.to_string())],
                )
            })?;
        Ok(Some(Self {
            environment,
            bos_token,
            eos_token,
        }))
    }

    /// Renders `messages`, optionally followed by the marker that opens the
    /// assistant's turn.
    pub fn render(&self, messages: &[Message<'_>], add_generation_prompt: bool) -> Result<String> {
        let roles: Vec<&str> = messages.iter().map(|message| message.role).collect();
        let count = messages.len();
        let messages: Vec<BTreeMap<&str, &str>> = messages
            .iter()
            .map(|message| BTreeMap::from([("role", message.role), ("content", message.content)]))
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
            .map_err(|error| {
                let cause = if error.kind() == minijinja::ErrorKind::InvalidOperation {
                    &causes::RENDER_REFUSED
                } else {
                    &causes::RENDER_FAILED
                };
                refuse(
                    cause,
                    &[
                        ("count", &count.to_string()),
                        ("roles", &roles.join(", ")),
                        ("kind", &format!("{:?}", error.kind())),
                        ("line", &line(&error)),
                        ("error", &error.to_string()),
                    ],
                )
            })
    }

    /// One prompt as a user turn, ending exactly where the assistant's own
    /// tokens begin.
    pub fn prompt(&self, prompt: &str) -> Result<String> {
        filled(prompt, "prompt")?;
        self.render(
            &[Message {
                role: "user",
                content: prompt,
            }],
            true,
        )
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
    ///
    /// `system`, when given, opens the conversation as the system turn the
    /// model is served with; it is part of the prompt side and never a target.
    pub fn example(
        &self,
        system: Option<&str>,
        prompt: &str,
        completion: &str,
    ) -> Result<(String, String)> {
        filled(prompt, "prompt")?;
        filled(completion, "completion")?;
        let mut turns: Vec<Message<'_>> = system
            .map(|content| Message {
                role: "system",
                content,
            })
            .into_iter()
            .collect();
        turns.push(Message {
            role: "user",
            content: prompt,
        });
        let head = self.render(&turns, true)?;
        turns.push(Message {
            role: "assistant",
            content: completion,
        });
        let full = self.render(&turns, false)?;
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
            let after: String = full[agreed..].chars().take(60).collect();
            let wanted: String = completion.chars().take(40).collect();
            return Err(refuse(
                &causes::BOUNDARY_ABSENT,
                &[
                    ("head", &head.len().to_string()),
                    ("full", &full.len().to_string()),
                    ("agreed", &agreed.to_string()),
                    ("after", &after),
                    ("completion", &wanted),
                ],
            ));
        };
        // The tail starts with the completion's own non-blank text.
        let tail = &full[start..];
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
        filled(text, "response")?;
        self.render(
            &[Message {
                role: "assistant",
                content: text,
            }],
            false,
        )
    }
}

/// The byte length of the longest common prefix of two renders, always on a
/// character boundary so the result can slice either of them.
fn common_prefix(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a.len_utf8())
        .sum()
}

/// The template line a minijinja error points at, or `unknown`.
fn line(error: &minijinja::Error) -> String {
    error
        .line()
        .map_or("unknown".to_owned(), |line| line.to_string())
}
