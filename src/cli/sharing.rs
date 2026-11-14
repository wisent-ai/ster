//! What this program's `sources discover` (and the import it stands for) saw, reported
//! with the person's consent so the fleet learns which sources people keep (the import
//! plan's priorities): one `<PRODUCT>.sources.discovered` event each, sent to Wisent
//! Integrations' onboarding domain at the route Stado's `onboarding-events` alternative
//! names (`stado capabilities resolve onboarding-events --consumer <PRODUCT> --json`), with
//! the token `TOKEN` names; Integrations refuses any field but these. An event holds the
//! adapter, whether its source was found, its counts rounded into buckets and the outcome,
//! under a random subject kept with the answer; never a name, a path or a value. Nothing
//! is sent before the person answers once (`<PRODUCT> sources share --yes | --no`), kept in
//! `$XDG_CONFIG_HOME/<PRODUCT>/sources-share.json` (else `~/.config/<PRODUCT>/`); the
//! listing shows the question with exactly what an event holds until then. Every step
//! works the same either way, and each answer's `reported` says whether it was sent and
//! why not.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{json, Map, Value};

const PRODUCT: &str = "ster";
const TOKEN: &str = "STER_STADO_INTEGRATION_TOKEN";
/// The adapter the question's example event names.
const EXAMPLE: &str = "IMP-STER-06";

fn kept() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join(PRODUCT).join("sources-share.json"))
}

fn random_hex(count: usize) -> String {
    let mut bytes = vec![0u8; count];
    if let Ok(mut random) = std::fs::File::open("/dev/urandom") {
        let _ = random.read_exact(&mut bytes);
    }
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The person's answer and the subject it was given under: none until they answer.
fn consent() -> Option<(bool, String)> {
    let answer: Value = serde_json::from_str(&std::fs::read_to_string(kept()?).ok()?).ok()?;
    Some((answer["share"].as_bool()?, answer["subject"].as_str()?.to_string()))
}

/// `PRODUCT sources share --yes | --no`: keep the person's answer; with neither, the question.
pub fn share(answer: Option<bool>) -> Result<Value, String> {
    let Some(share) = answer else { return Ok(question()) };
    let path = kept().ok_or_else(|| "neither XDG_CONFIG_HOME nor HOME is set, so the answer has nowhere to be kept; set one and answer again".to_string())?;
    let subject = consent().map_or_else(|| random_hex(32), |(_, subject)| subject);
    let kept = json!({"share": share, "subject": subject, "answered_at": occurred_at()});
    path.parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&path, format!("{kept:#}\n")))
        .map_err(|error| format!("whether import counts are shared is kept in {}, and it could not be written: {error}; set XDG_CONFIG_HOME to a directory this user can write, then answer again", path.display()))?;
    Ok(json!({"share": share, "kept": path}))
}

fn bucket(count: usize) -> &'static str {
    match count {
        0 => "0",
        1..=9 => "1-9",
        10..=99 => "10-99",
        100..=999 => "100-999",
        1000..=9999 => "1000-9999",
        _ => "10000+",
    }
}

/// Seconds since the epoch as an RFC 3339 UTC time (Howard Hinnant's `civil_from_days`).
fn occurred_at() -> String {
    let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |since| since.as_secs());
    let (days, rest) = ((seconds / 86_400) as i64, seconds % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rest / 3600, rest % 3600 / 60, rest % 60)
}

/// A version-4 UUID (RFC 9562 §5.4).
fn event_id() -> String {
    let hex = random_hex(16);
    let variant = ["8", "9", "a", "b"][usize::from(u8::from_str_radix(&hex[16..17], 16).unwrap_or(0) & 3)];
    format!("{}-{}-4{}-{variant}{}-{}", &hex[0..8], &hex[8..12], &hex[13..16], &hex[17..20], &hex[20..32])
}

fn event(subject: &str, adapter: &str, found: bool, counts: &[(&str, usize)], outcome: &str) -> Value {
    let counts: Map<String, Value> = counts.iter().map(|(kind, count)| ((*kind).to_string(), json!(bucket(*count)))).collect();
    json!({
        "event_id": event_id(),
        "product_id": PRODUCT,
        "subject_hash": subject,
        "adapter": adapter,
        "found": found,
        "counts": counts,
        "outcome": outcome,
        "scope_kind": "user",
        "occurred_at": occurred_at(),
    })
}

/// The question the listing asks until it is answered, with exactly what an event holds.
pub fn question() -> Value {
    match consent() {
        Some((share, _)) => json!({"share": share, "change_with": format!("{PRODUCT} sources share --yes | --no")}),
        None => json!({
            "asked": format!("May {PRODUCT} send, for each source it reads, which source it was, whether it was there and how many records it held, rounded, so the sources people keep are read first? No name, path or value is sent; everything works the same either way."),
            "sends": event(&"0".repeat(64), EXAMPLE, true, &[("records", 12)], "discovered"),
            "answer_with": format!("{PRODUCT} sources share --yes | --no"),
        }),
    }
}

/// `answer` with `reported`: whether the event of `adapter` was sent, and why not.
pub fn reported(mut answer: Value, adapter: &str, found: bool, counts: &[(&str, usize)], outcome: &str) -> Value {
    let sent = send(adapter, found, counts, outcome);
    if let Some(fields) = answer.as_object_mut() {
        fields.insert("reported".to_string(), sent);
    }
    answer
}

fn send(adapter: &str, found: bool, counts: &[(&str, usize)], outcome: &str) -> Value {
    let subject = match consent() {
        None => return json!({"sent": false, "why": format!("not asked yet: {PRODUCT} sources share --yes | --no")}),
        Some((false, _)) => return json!({"sent": false, "why": "you chose not to share"}),
        Some((true, subject)) => subject,
    };
    let Some(token) = std::env::var(TOKEN).ok().filter(|token| !token.trim().is_empty()) else {
        return json!({"sent": false, "why": format!("{TOKEN} is unset, so there is no credential to report with")});
    };
    let stado = std::env::var("STADO_BIN").unwrap_or_else(|_| "stado".to_string());
    let resolved = stado_wait::output(Command::new(&stado).args(["capabilities", "resolve", "onboarding-events", "--consumer", PRODUCT, "--json"]).stdin(Stdio::null()));
    let resolved: Value = match resolved {
        Ok(output) if output.status.success() => match serde_json::from_slice(&output.stdout) {
            Ok(resolved) => resolved,
            Err(error) => return json!({"sent": false, "why": format!("Stado's resolution is not JSON: {error}")}),
        },
        Ok(output) => return json!({"sent": false, "why": format!("Stado resolved no onboarding route: {}", String::from_utf8_lossy(&output.stderr).trim())}),
        Err(error) => return json!({"sent": false, "why": format!("{stado} cannot be run: {error}")}),
    };
    let (Some(route), Some(provider)) = (resolved["route"].as_str(), resolved["provider"].as_str()) else {
        return json!({"sent": false, "why": format!("Stado's resolution names no route: {resolved}")});
    };
    let address = format!("{}/api/integration/onboarding/{PRODUCT}.sources.discovered", route.trim_end_matches('/'));
    let body = event(&subject, adapter, found, counts, outcome).to_string();
    // The token is handed to curl on its standard input, never on its command line.
    let config = format!("header = \"Authorization: Bearer {}\"\n", token.trim().replace('\\', "\\\\").replace('"', "\\\""));
    let sent = stado_wait::output_with_input(
        Command::new("curl").args(["--silent", "--show-error", "--fail-with-body", "--config", "-", "--header", "Content-Type: application/json", "--data", &body, &address]),
        config.as_bytes(),
    );
    match sent {
        Ok(output) if output.status.success() => json!({"sent": true, "to": provider}),
        Ok(output) => json!({"sent": false, "why": format!("{provider} answered: {}", String::from_utf8_lossy(&output.stdout).trim())}),
        Err(error) => json!({"sent": false, "why": format!("curl did not run to its end: {error}")}),
    }
}
