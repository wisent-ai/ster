//! `ster errors check`: the catalogue held to the declaration rules, to the
//! tests this checkout holds, and to the commands the binary answers.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{Value, json};
use wisent_errors::{Operation, catalogue_problems};

use super::{catalogue, operations, refuse};

/// The commands no functionality names as an entry: `ster <command>` is no
/// entry's first words.
fn uncatalogued<'a>(operations: &[&Operation], commands: &'a [String]) -> Vec<&'a str> {
    commands
        .iter()
        .map(String::as_str)
        .filter(|name| {
            let entry = format!("ster {name}");
            !operations.iter().any(|operation| {
                operation
                    .entries
                    .iter()
                    .any(|named| *named == entry || named.starts_with(&format!("{entry} ")))
            })
        })
        .collect()
}

/// Every declared test `repository` does not hold, by its declared path:
/// the file the cause names is absent, unreadable, or never names the test.
fn missing_tests(operations: &[&Operation], repository: &Path) -> Vec<(String, String)> {
    let mut missing = Vec::new();
    for operation in operations {
        for cause in operation.causes {
            let Some((file, name)) = cause.test.split_once("::") else {
                // Operation::problems already refuses a test without `::`.
                continue;
            };
            let path = repository.join(file);
            let finding = match std::fs::read_to_string(&path) {
                Ok(text) if text.contains(name) => continue,
                Ok(_) => format!(
                    "{} {}: {} does not name the declared test '{name}'; write the test there, correct the declaration, or list {} in the untested baseline",
                    operation.id,
                    cause.id,
                    path.display(),
                    cause.test
                ),
                Err(error) => format!(
                    "{} {}: the declared test file {} could not be read ({error}); write the test there, correct the declaration, or list {} in the untested baseline",
                    operation.id,
                    cause.id,
                    path.display(),
                    cause.test
                ),
            };
            missing.push((cause.test.to_owned(), finding));
        }
    }
    missing
}

/// One baseline file: a JSON list of strings.
fn baseline(path: Option<&Path>, what: &str) -> anyhow::Result<BTreeSet<String>> {
    let Some(path) = path else {
        return Ok(BTreeSet::new());
    };
    let shown = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|error| {
        refuse(
            &catalogue::BASELINE_UNREADABLE,
            &[
                ("what", what),
                ("path", &shown),
                ("error", &error.to_string()),
            ],
        )
    })?;
    serde_json::from_str(&text).map_err(|error| {
        refuse(
            &catalogue::BASELINE_MALFORMED,
            &[
                ("what", what),
                ("path", &shown),
                ("error", &error.to_string()),
            ],
        )
    })
}

/// Every declaration problem, every problem only the whole catalogue shows,
/// with `repository` every declared test that checkout does not hold unless
/// the `untested` baseline lists it, and every command of `commands` no
/// functionality names unless the `uncatalogued` baseline lists it. Both
/// baselines may only shrink: an entry no longer needed is a finding too.
/// Refuses with the findings in the answer when there is one.
pub fn check(
    repository: Option<&Path>,
    uncatalogued_baseline: Option<&Path>,
    untested_baseline: Option<&Path>,
    commands: &[String],
) -> anyhow::Result<Value> {
    let allowed_commands = baseline(uncatalogued_baseline, "uncatalogued-command")?;
    let allowed_tests = baseline(untested_baseline, "untested-cause")?;
    let operations = operations();
    let mut findings: Vec<String> = operations
        .iter()
        .flat_map(|operation| operation.problems())
        .collect();
    findings.extend(catalogue_problems(&operations));
    let mut untested = Vec::new();
    if let Some(repository) = repository {
        let missing = missing_tests(&operations, repository);
        for (test, finding) in &missing {
            if allowed_tests.contains(test) {
                untested.push(test.clone());
            } else {
                findings.push(finding.clone());
            }
        }
        for test in &allowed_tests {
            if !missing.iter().any(|(missing, _)| missing == test) {
                findings.push(format!(
                    "the untested baseline still lists {test}, but no declared cause is missing that test; remove it from the baseline"
                ));
            }
        }
    }
    let uncatalogued = uncatalogued(&operations, commands);
    for command in &uncatalogued {
        if !allowed_commands.contains(*command) {
            findings.push(format!(
                "ster {command}: no functionality names it as an entry, so its failures are declared nowhere; declare them in src/failures"
            ));
        }
    }
    for command in &allowed_commands {
        if !uncatalogued.contains(&command.as_str()) {
            findings.push(format!(
                "the uncatalogued baseline still allows ster {command}, but a functionality names it now; remove it from the baseline"
            ));
        }
    }
    let answer = json!({
        "ok": findings.is_empty(),
        "functionalities": operations.len(),
        "causes": operations.iter().map(|operation| operation.causes.len()).sum::<usize>(),
        "uncatalogued_commands": uncatalogued,
        "untested_causes": untested,
        "findings": findings,
    });
    Ok(answer)
}
