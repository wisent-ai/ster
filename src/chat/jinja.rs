//! What Hugging Face's renderer adds on top of stock Jinja2: the two globals
//! chat templates call, and the Python string and mapping methods they invoke
//! as methods rather than filters. Anything outside this is an error from the
//! engine, naming the template's own line.

use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::{
    Error, ErrorKind, State, Value,
    value::{ValueKind, from_args},
};

mod calendar;
use calendar::{civil_from_days, days_from_civil};

/// The source with Transformers' `{% generation %}` … `{% endgeneration %}`
/// blocks, which mark the assistant's tokens for `return_assistant_tokens_mask`
/// and otherwise render their body unchanged (Transformers'
/// `AssistantTracker` extension), written as an always-true `if` so stock
/// Jinja renders them the same, whitespace control included.
pub(super) fn plain_generation_blocks(source: String) -> String {
    let opening =
        regex::Regex::new(r"\{%(-?)\s*generation\s*(-?)%\}").expect("a fixed pattern compiles");
    let closing =
        regex::Regex::new(r"\{%(-?)\s*endgeneration\s*(-?)%\}").expect("a fixed pattern compiles");
    let source = opening.replace_all(&source, "{%${1} if true ${2}%}");
    closing
        .replace_all(&source, "{%${1} endif ${2}%}")
        .into_owned()
}

/// Hugging Face's own template globals, which stock Jinja does not have.
///
/// Templates call `raise_exception` to reject a conversation they cannot
/// represent — alternating-role checks, mostly — and the message is the
/// template author's, so it travels out unchanged.
pub(super) fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now(format)`, as the Llama 3.2 family's template calls it to
/// stamp today's date into the system prompt.
///
/// UTC, and only the directives a chat template plausibly uses. An unknown
/// directive is an error rather than a passthrough: a template asking for a
/// field this cannot produce would otherwise render a literal `%V` into the
/// model's context.
pub(super) fn strftime_now(format: String) -> Result<Value, Error> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| {
            Error::new(
                ErrorKind::InvalidOperation,
                "the system clock is before 1970",
            )
        })?
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400); /* https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap04.html#tag_04_19 */
    let time = seconds.rem_euclid(86_400); /* https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap04.html#tag_04_19 */
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (time / 3600, (time % 3600) / 60, time % 60); /* https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap04.html#tag_04_19 */
    let weekday = (days + 4).rem_euclid(7) as usize; /* https://howardhinnant.github.io/date_algorithms.html */
    let year_day = days - days_from_civil(year, 1, 1) + 1; /* https://pubs.opengroup.org/onlinepubs/9799919799/functions/strftime.html */
    const MONTHS: &[&str] = &[
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: &[&str] = &[
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    let mut out = String::with_capacity(format.len());
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
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "strftime format ends in a bare %",
            ));
        };
        match directive {
            '%' => out.push('%'),
            'Y' => out.push_str(&year.to_string()),
            'y' => out.push_str(&two(year.rem_euclid(100) as i64, pad)), /* https://pubs.opengroup.org/onlinepubs/9799919799/functions/strftime.html */
            'm' => out.push_str(&two(month as i64, pad)),
            'd' | 'e' => out.push_str(&two(day as i64, pad)),
            'H' => out.push_str(&two(hour, pad)),
            'M' => out.push_str(&two(minute, pad)),
            'S' => out.push_str(&two(second, pad)),
            'b' | 'h' => out.push_str(&MONTHS[month as usize - 1][..3]), /* https://pubs.opengroup.org/onlinepubs/9799919799/functions/strftime.html */
            'B' => out.push_str(MONTHS[month as usize - 1]),
            'a' => out.push_str(&DAYS[weekday][..3]), /* https://pubs.opengroup.org/onlinepubs/9799919799/functions/strftime.html */
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
    if pad {
        format!("{value:02}")
    } else {
        value.to_string()
    }
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
pub(super) fn python_method(
    state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    let unknown = || Error::from(ErrorKind::UnknownMethod);
    if let Some(text) = value.as_str() {
        let argument = |index: usize| -> Result<&str, Error> {
            args.get(index).and_then(Value::as_str).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    format!("{method} expects a string argument"),
                )
            })
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
            "split" => Ok(Value::from_iter(
                match args.first().and_then(Value::as_str) {
                    Some(separator) => text.split(separator).map(Value::from).collect::<Vec<_>>(),
                    None => text.split_whitespace().map(Value::from).collect(),
                },
            )),
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
                // Python's `dict.get(key, default=None)`: https://docs.python.org/3/library/stdtypes.html#dict.get
                let (key, default): (Value, Option<Value>) = from_args(args)?;
                let found = value.get_item(&key)?;
                Ok(match (found.is_undefined(), default) {
                    (false, _) => found,
                    (true, Some(default)) => default,
                    (true, None) => Value::from(()),
                })
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
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}
