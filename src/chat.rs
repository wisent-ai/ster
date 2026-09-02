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

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use minijinja::{
    Environment, Error, ErrorKind, State, Value,
    value::{ValueKind, from_args},
};

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
            Self::Applied => "applying the model's own chat template to every prompt and completion",
            Self::Absent => "this model publishes no chat template, so prompts and completions are encoded as raw text",
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
            .insert("chat_template".to_owned(), serde_json::Value::from(self.label()));
        Ok(())
    }
}

/// One turn of a conversation, as the template sees it.
#[derive(Debug, Clone, Copy)]
pub struct Message<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

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

/// The name the template is compiled under; it appears in engine errors, so it
/// reads as the thing the operator would name.
const TEMPLATE_NAME: &str = "chat_template";

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
        environment.add_function("raise_exception", raise_exception);
        environment.add_function("strftime_now", strftime_now);
        environment.set_unknown_method_callback(python_method);
        environment
            .add_template_owned(TEMPLATE_NAME, source)
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

    /// A training example split at the boundary the loss starts from.
    ///
    /// Both halves come out of the *same* template rather than being glued
    /// together by hand: the second render is the whole conversation, the
    /// first is that conversation cut off before the assistant speaks, and the
    /// tail is whatever the template added — the completion plus the turn's
    /// own end marker, which is precisely the span the model must learn to
    /// produce. Deriving the tail by subtraction is what keeps the boundary
    /// exact; a boundary measured on the untemplated prompt would be short by
    /// the length of every marker the template emitted.
    pub fn example(&self, prompt: &str, completion: &str) -> Result<(String, String)> {
        let head = self.prompt(prompt)?;
        let full = self.render(
            &[
                Message { role: "user", content: prompt },
                Message { role: "assistant", content: completion },
            ],
            false,
        )?;
        let Some(tail) = full.strip_prefix(&head) else {
            bail!(
                "this model's chat template does not render the assistant turn after the prompt, so the completion boundary cannot be located"
            );
        };
        if tail.is_empty() {
            bail!("this model's chat template rendered an empty assistant turn");
        }
        Ok((head, tail.to_owned()))
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

/// Hugging Face's own template globals, which stock Jinja does not have.
///
/// Templates call `raise_exception` to reject a conversation they cannot
/// represent — alternating-role checks, mostly — and the message is the
/// template author's, so it travels out unchanged.
fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now(format)`, as the Llama 3.2 family's template calls it to
/// stamp today's date into the system prompt.
///
/// UTC, and only the directives a chat template plausibly uses. An unknown
/// directive is an error rather than a passthrough: a template asking for a
/// field this cannot produce would otherwise render a literal `%V` into the
/// model's context.
fn strftime_now(format: String) -> Result<Value, Error> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::new(ErrorKind::InvalidOperation, "the system clock is before 1970"))?
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let time = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (time / 3600, (time % 3600) / 60, time % 60);
    let weekday = (days + 4).rem_euclid(7) as usize;
    let year_day = days - days_from_civil(year, 1, 1) + 1;
    const MONTHS: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December",
    ];
    const DAYS: [&str; 7] = [
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    ];
    let mut out = String::with_capacity(format.len() + 8);
    let mut chars = format.chars();
    while let Some(character) = chars.next() {
        if character != '%' {
            out.push(character);
            continue;
        }
        // `%-d` is glibc's "no leading zero"; templates that format a date the
        // way a human writes it use it, so it is understood rather than
        // rejected.
        let (pad, directive) = match chars.next() {
            Some('-') => (false, chars.next()),
            other => (true, other),
        };
        let Some(directive) = directive else {
            return Err(Error::new(ErrorKind::InvalidOperation, "strftime format ends in a bare %"));
        };
        match directive {
            '%' => out.push('%'),
            'Y' => out.push_str(&year.to_string()),
            'y' => out.push_str(&two(year.rem_euclid(100) as i64, pad)),
            'm' => out.push_str(&two(month as i64, pad)),
            'd' | 'e' => out.push_str(&two(day as i64, pad)),
            'H' => out.push_str(&two(hour, pad)),
            'M' => out.push_str(&two(minute, pad)),
            'S' => out.push_str(&two(second, pad)),
            'b' | 'h' => out.push_str(&MONTHS[month as usize - 1][..3]),
            'B' => out.push_str(MONTHS[month as usize - 1]),
            'a' => out.push_str(&DAYS[weekday][..3]),
            'A' => out.push_str(DAYS[weekday]),
            'j' => out.push_str(&format!("{year_day:03}")),
            other => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("strftime directive %{other} is not supported"),
                ));
            }
        }
    }
    Ok(Value::from(out))
}

fn two(value: i64, pad: bool) -> String {
    if pad { format!("{value:02}") } else { value.to_string() }
}

/// Days since 1970-01-01 to a civil date, and back. Howard Hinnant's
/// algorithm, exact for every year this will ever be handed.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 { shifted } else { shifted - 146_096 } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 } as i64;
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i64 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The Python methods chat templates call on strings and mappings.
///
/// Jinja2 runs on Python, so a template author writes `content.strip()` and
/// `message.items()` as naturally as a filter. MiniJinja has the filters but
/// not the methods, and this is the documented hook for closing that gap. Only
/// methods whose Python semantics can be reproduced exactly are here; anything
/// else falls through to the engine's own "unknown method" error, naming the
/// method and the line, because a method that quietly returns the wrong string
/// is a mis-rendered template.
fn python_method(
    state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    let unknown = || Error::from(ErrorKind::UnknownMethod);
    if let Some(text) = value.as_str() {
        let argument = |index: usize| -> Result<&str, Error> {
            args.get(index)
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, format!("{method} expects a string argument")))
        };
        return match method {
            "strip" | "lstrip" | "rstrip" => {
                let trimmed = match args.first().and_then(Value::as_str) {
                    Some(cut) => {
                        let cut: Vec<char> = cut.chars().collect();
                        match method {
                            "strip" => text.trim_matches(cut.as_slice()),
                            "lstrip" => text.trim_start_matches(cut.as_slice()),
                            _ => text.trim_end_matches(cut.as_slice()),
                        }
                    }
                    None => match method {
                        "strip" => text.trim(),
                        "lstrip" => text.trim_start(),
                        _ => text.trim_end(),
                    },
                };
                Ok(Value::from(trimmed))
            }
            "lower" => Ok(Value::from(text.to_lowercase())),
            "upper" => Ok(Value::from(text.to_uppercase())),
            "title" => Ok(Value::from(title_case(text))),
            "capitalize" => Ok(Value::from(capitalize(text))),
            "startswith" => Ok(Value::from(text.starts_with(argument(0)?))),
            "endswith" => Ok(Value::from(text.ends_with(argument(0)?))),
            "replace" => Ok(Value::from(text.replace(argument(0)?, argument(1)?))),
            "split" => Ok(Value::from_iter(match args.first().and_then(Value::as_str) {
                Some(separator) => text.split(separator).map(Value::from).collect::<Vec<_>>(),
                None => text.split_whitespace().map(Value::from).collect(),
            })),
            "splitlines" => Ok(Value::from_iter(text.lines().map(Value::from))),
            _ => Err(unknown()),
        };
    }
    if value.kind() == ValueKind::Map {
        return match method {
            "items" | "keys" | "values" => {
                let () = from_args(args)?;
                state.apply_filter(method, &[value.clone()])
            }
            "get" => {
                let (key, default): (Value, Option<Value>) = from_args(args)?;
                Ok(value
                    .get_item(&key)
                    .ok()
                    .filter(|found| !found.is_undefined())
                    .unwrap_or_else(|| default.unwrap_or(Value::from(()))))
            }
            _ => Err(unknown()),
        };
    }
    Err(unknown())
}

fn title_case(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_word = false;
    for character in text.chars() {
        if character.is_alphanumeric() {
            if in_word {
                out.extend(character.to_lowercase());
            } else {
                out.extend(character.to_uppercase());
            }
            in_word = true;
        } else {
            out.push(character);
            in_word = false;
        }
    }
    out
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars.flat_map(char::to_lowercase)).collect(),
        None => String::new(),
    }
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
    (config.is_file().then_some(config), template.is_file().then_some(template))
}
