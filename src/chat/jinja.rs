//! What Hugging Face's renderer adds on top of stock Jinja2: the two globals
//! chat templates call, and the Python string and mapping methods they invoke
//! as methods rather than filters. Anything outside this is an error from the
//! engine, naming the template's own line.

use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::{
    value::{from_args, ValueKind},
    Error, ErrorKind, State, Value,
};

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

// The calendar arithmetic below counts years in 400-year eras of 146 097 days, starting each year in
// March so the leap day falls last; 719 468 days separate 0000-03-01 from the Unix epoch.
const DAYS_PER_ERA: i64 = 146_097;
const YEARS_PER_ERA: i64 = 400;
const EPOCH_OFFSET_DAYS: i64 = 719_468;
const MARCH: i64 = 3;
const MONTHS_PER_YEAR: i64 = 12;
const MONTHS_FROM_MARCH_TO_DECEMBER: i64 = 10;

/// Days since 1970-01-01 to a civil date, and back. Howard Hinnant's
/// algorithm, exact for every year this will ever be handed.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + EPOCH_OFFSET_DAYS;
    let era = if shifted >= 0 { shifted } else { shifted - (DAYS_PER_ERA - 1) } / DAYS_PER_ERA;
    let day_of_era = shifted - era * DAYS_PER_ERA;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / (DAYS_PER_ERA - 1)) / 365;
    let year = year_of_era + era * YEARS_PER_ERA;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < MONTHS_FROM_MARCH_TO_DECEMBER { shifted_month + MARCH } else { shifted_month - (MONTHS_PER_YEAR - MARCH) } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - (YEARS_PER_ERA - 1) } / YEARS_PER_ERA;
    let year_of_era = year - era * YEARS_PER_ERA;
    let shifted_month = if month > 2 { month as i64 - MARCH } else { month as i64 + (MONTHS_PER_YEAR - MARCH) };
    let day_of_year = (153 * shifted_month + 2) / 5 + day as i64 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * DAYS_PER_ERA + day_of_era - EPOCH_OFFSET_DAYS
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
