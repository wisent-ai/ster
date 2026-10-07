//! chat.rs — the conversation format a checkpoint was actually trained in.
//!
//! Every other tokenizer path in Ster treats text as a bare completion prompt,
//! which is exactly right for a base model and wrong for every instruct
//! checkpoint published in the last two years. Those models were post-trained
//! with a *chat template*: roles wrapped in special markers, recorded as a
//! Jinja string in `tokenizer_config.json` under `chat_template` (or, on newer
//! repositories, in a standalone `chat_template.jinja`). Fine-tuning such a
//! model on untemplated text teaches it a format it will never see at
//! inference, and generating from it without the template produces the
//! rambling continuations that make people think a checkpoint is broken.
//!
//! **The Jinja decision.** These templates are real Jinja2 — loops, tests,
//! filters, `raise_exception`, namespaces — and a hand-rolled renderer that
//! understands most of that subset is the worst possible outcome: it fails
//! silently into plausible-looking, wrongly-marked training data rather than
//! loudly into an error. So this module takes exactly one dependency,
//! [`minijinja`], a pure-Rust Jinja2 engine with no C code and one transitive
//! crate, and adds only what Hugging Face's renderer adds on top of stock
//! Jinja: the `raise_exception` and `strftime_now` globals, and the handful of
//! Python string and mapping methods templates call as methods rather than
//! filters. Anything outside that is an error from the engine, with the
//! template's own line number, which is the failure mode this module wants.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

mod jinja;
mod template;

pub use template::Template;

/// Whether a run applies the model's own chat template.
///
/// Two settings rather than three: there is no `on`, because a model with no
/// template cannot be forced into one and an operator who asked for `on` would
/// only ever get a refusal. `auto` applies the template when the checkpoint
/// publishes one and says so when it does not; `off` is the raw-text encoding
/// every Ster release before this one used, which is what a base model wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Auto,
    Off,
}

impl Choice {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            _ => bail!("unknown chat template mode {value:?}; expected auto or off"),
        }
    }
}

/// What a run decided to do about the chat template, once the checkpoint has
/// been resolved and the question can actually be answered.
///
/// This is reported rather than inferred by the caller because the answer
/// depends on a file the caller never reads. It travels into the run's report
/// and onto one progress line, so an operator reading either can tell whether
/// the adapter was trained on the shape the model will be asked to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The checkpoint publishes a template and every prompt and completion
    /// goes through it.
    Applied,
    /// `auto` found no template, so text is encoded raw.
    Absent,
    /// The operator asked for raw text.
    Off,
}

impl Status {
    /// The word the report carries. Machine-readable, so it is a bare token
    /// rather than the sentence.
    pub fn label(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::Absent => "absent",
            Self::Off => "off",
        }
    }

    /// The one sentence an operator reads on the progress stream.
    ///
    /// The absent case is the refusal that matters: a model that publishes no
    /// template has no conversation format to guess at, so Ster says so and
    /// encodes raw text rather than inventing markers the model never saw.
    pub fn sentence(self) -> &'static str {
        match self {
            Self::Applied => {
                "applying the model's own chat template to every prompt and completion"
            }
            Self::Absent => {
                "this model publishes no chat template, so prompts and completions are encoded as raw text"
            }
            Self::Off => "chat template off, so prompts and completions are encoded as raw text",
        }
    }

    /// Records this decision in a run's own report.
    ///
    /// The report is the document that gets folded into the adapter artifact,
    /// so an adapter carries the shape it was trained in rather than leaving
    /// an operator to guess months later which encoding produced it.
    pub fn annotate(self, report: &mut serde_json::Value) -> Result<()> {
        report
            .as_object_mut()
            .context("a run report must be a JSON object to record its chat template")?
            .insert(
                "chat_template".to_owned(),
                serde_json::Value::from(self.label()),
            );
        Ok(())
    }
}

/// One turn of a conversation, as the template sees it.
#[derive(Debug, Clone, Copy)]
pub struct Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// The two files a checkpoint may carry its template in, resolved beside the
/// three files Ster already resolves.
///
/// Both are optional and their absence is not a failure: a base checkpoint
/// publishes neither, which is exactly the case `auto` reports rather than
/// refuses.
pub fn local_files(root: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let config = root.join("tokenizer_config.json");
    let template = root.join("chat_template.jinja");
    (
        config.is_file().then_some(config),
        template.is_file().then_some(template),
    )
}
