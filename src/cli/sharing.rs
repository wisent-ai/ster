//! What this program's `sources discover` (and the import it stands for) saw, reported
//! with the person's consent so the fleet learns which sources people keep (the import
//! plan's priorities): one `<PRODUCT>.sources.discovered` event each, sent to Wisent
//! Integrations' onboarding domain at the route Stado's `onboarding-events` alternative
//! names (`stado capabilities resolve onboarding-events --consumer <PRODUCT> --json`), with
//! the token `TOKEN` names; Integrations refuses any field but these. An event holds the
//! adapter, whether its source was found, its counts rounded into buckets and the outcome,
//! under a random subject kept with the answer; never a name, a path or a value. Nothing
//! is sent before the person answers once (`<PRODUCT> sources share yes | no`), kept in
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
const EXAMPLE: &str = "dpo";

fn kept() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join(PRODUCT).join("sources-share.json"))
}

/// `count` random bytes as hex, or why /dev/urandom gave none: a buffer it
/// did not fill was all zeros, the same subject and event id for everyone.
fn random_hex(count: usize) -> Result<String, String> {
    let mut bytes = vec![u8::MIN; count];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut random| random.read_exact(&mut bytes))
        .map_err(|error| format!("{count} random bytes could not be read from /dev/urandom: {error}; nothing was kept or sent"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// The person's answer and the subject it was given under: none until they answer.
fn consent() -> Option<(bool, String)> {
    let answer: Value = serde_json::from_str(&std::fs::read_to_string(kept()?).ok()?).ok()?;
    Some((answer["share"].as_bool()?, answer["subject"].as_str()?.to_string()))
}

/// `PRODUCT sources share yes | no`: keep the person's answer; alone, the question.
pub fn share(answer: Option<bool>) -> Result<Value, String> {
    let Some(share) = answer else { return Ok(question()) };
    let path = kept().ok_or_else(|| "neither XDG_CONFIG_HOME nor HOME is set, so the answer has nowhere to be kept; set one and answer again".to_string())?;
    let subject = match consent() {
        Some((_, subject)) => subject,
        None => random_hex(/* a SHA-256 digest's bytes: https://csrc.nist.gov/pubs/fips/180-4/upd1/final */ 32)?,
    };
    let kept = json!({"share": share, "subject": subject, "answered_at": occurred_at()});
    path.parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&path, format!("{kept:#}\n")))
        .map_err(|error| format!("whether import counts are shared is kept in {}, and it could not be written: {error}; set XDG_CONFIG_HOME to a directory this user can write, then answer again", path.display()))?;
    Ok(json!({"share": share, "kept": path}))
}

fn bucket(count: usize) -> &'static str {
    // The count buckets Integrations accepts, as echo-web's onboarding store declares
    // them (COUNT_BUCKETS in src/lib/onboarding/store/writes/sources.ts): none, then
    // one bucket per count of decimal digits, the last one open.
    const COUNT_BUCKETS: &[&str] = &["0", "1-9", "10-99", "100-999", "1000-9999", "10000+"];
    let &[none, .., open] = COUNT_BUCKETS else {
        unreachable!("COUNT_BUCKETS names the buckets Integrations accepts");
    };
    if std::num::NonZeroUsize::new(count).is_none() {
        return none;
    }
    let digits = count.to_string().len();
    let closed = format!("{:0<digits$}-{}", "1", "9".repeat(digits));
    COUNT_BUCKETS.iter().copied().find(|bucket| *bucket == closed).unwrap_or(open)
}

/// Seconds since the epoch as an RFC 3339 UTC time (Howard Hinnant's `civil_from_days`).
fn occurred_at() -> String {
    // A POSIX day, hour and minute are exactly these many seconds: https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap04.html#tag_04_19
    const SECONDS_PER_DAY: u64 = 86_400;
    // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap04.html#tag_04_19
    const SECONDS_PER_HOUR: u64 = 3_600;
    // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap04.html#tag_04_19
    const SECONDS_PER_MINUTE: u64 = 60;
    // A clock before 1970 is said as the epoch itself.
    let seconds = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(u64::MIN, |since| since.as_secs());
    let (days, rest) = ((seconds / SECONDS_PER_DAY) as i64, seconds % SECONDS_PER_DAY);
    // Hinnant's civil_from_days, each constant his: https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let z = days + 719_468;
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let era = z.div_euclid(146_097);
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let doe = z - era * 146_097;
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let mp = (5 * doy + 2) / 153;
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let day = doy - (153 * mp + 2) / 5 + 1;
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rest / SECONDS_PER_HOUR, rest % SECONDS_PER_HOUR / SECONDS_PER_MINUTE, rest % SECONDS_PER_MINUTE)
}

/// A version-4 UUID (RFC 9562 §5.4), or why no random bytes were read.
fn event_id() -> Result<String, String> {
    // A UUID is 128 bits, sixteen random bytes: https://www.rfc-editor.org/rfc/rfc9562#section-4
    let hex = random_hex(16)?;
    // The variant's nibble, read as hexadecimal: https://www.rfc-editor.org/rfc/rfc9562#section-4.1
    let nibble = u8::from_str_radix(&hex[16..17], 16).map_err(|error| format!("the random hex {hex} is not hex: {error}"))?;
    // Its two high bits are 10: https://www.rfc-editor.org/rfc/rfc9562#section-4.1
    let variant = ["8", "9", "a", "b"][usize::from(nibble & 3)];
    // The 8-4-4-4-12 layout, version 4: https://www.rfc-editor.org/rfc/rfc9562#section-5.4
    Ok(format!("{}-{}-4{}-{variant}{}-{}", &hex[0..8], &hex[8..12], &hex[13..16], &hex[17..20], &hex[20..32]))
}

fn event(id: &str, subject: &str, adapter: &str, found: bool, counts: &[(&str, usize)], outcome: &str) -> Value {
    let counts: Map<String, Value> = counts.iter().map(|(kind, count)| ((*kind).to_string(), json!(bucket(*count)))).collect();
    json!({
        "event_id": id,
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
        Some((share, _)) => json!({"share": share, "change_with": format!("{PRODUCT} sources share yes | no")}),
        None => json!({
            "asked": format!("May {PRODUCT} send, for each source it reads, which source it was, whether it was there and how many records it held, rounded, so the sources people keep are read first? No name, path or value is sent; everything works the same either way."),
            "sends": event("a fresh random version-4 UUID for each event", "the SHA-256 digest of a random subject, kept with your answer", EXAMPLE, true, &[("records", usize::MIN)], "discovered"),
            "answer_with": format!("{PRODUCT} sources share yes | no"),
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
        None => return json!({"sent": false, "why": format!("not asked yet: {PRODUCT} sources share yes | no")}),
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
    let id = match event_id() {
        Ok(id) => id,
        Err(why) => return json!({"sent": false, "why": why}),
    };
    let body = event(&id, &subject, adapter, found, counts, outcome).to_string();
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
