//! The operation Ster Desktop's Decide screen runs, driven on the real binary.

use std::{
    io::Write,
    process::{Command, Stdio},
};

use serde_json::Value;

use crate::{probabilities_sum, MODEL};

/// The desktop never builds a command line for a decision: it runs
/// `ster request decide`, writes the request document inline on stdin and
/// reads the NDJSON events. This drives that operation on the real binary
/// with the body Ster Desktop's Decide screen composes, and a body the
/// operation refuses before loading a model.
#[test]
fn request_answers_the_desktops_decide_body_and_refuses_a_bad_one() {
    let (refused_exit, refused) = request(
        "decide",
        &serde_json::json!({
            "model": MODEL, "device": "cpu",
            "request": {"state": "x", "questions": {"only": {"type": "choice", "instructions": "?", "criteria": {"a": null}}}}
        }),
    );
    assert_eq!(
        refused.len(),
        1,
        "a refusal is one result event and nothing else: {refused:?}"
    );
    let refusal = &refused[0];
    assert_eq!(refusal["type"], "result", "{refusal}");
    assert_eq!(
        refusal["json"]["error"],
        "choice question 'only' needs at least two options"
    );
    assert_eq!(
        refused_exit,
        refusal["status"].as_i64().map(|status| status as i32)
    );
    assert_ne!(refused_exit, Some(0), "{refusal}");

    let (exit, events) = request(
        "decide",
        &serde_json::json!({
            "model": MODEL, "device": "cpu", "precision": "f32", "chatTemplate": "auto",
            "permutations": 0, "explain": false,
            "request": {
                "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
                "questions": {
                    "department": {"type": "choice", "instructions": "Which team should handle this?",
                                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "billing": null}},
                    "is_urgent": {"type": "noul", "instructions": "Does this message convey urgency?"}
                }
            }
        }),
    );
    assert_eq!(exit, Some(0), "{events:?}");
    let result = events.last().expect("a result event");
    assert_eq!(result["type"], "result", "{result}");
    assert_eq!(result["status"], 0, "{result}");
    let logs: Vec<&str> = events
        .iter()
        .filter(|event| event["type"] == "log")
        .map(|event| event["chunk"].as_str().unwrap())
        .collect();
    assert!(
        logs.iter()
            .any(|chunk| chunk.contains("reading the state once (")),
        "{logs:?}"
    );
    let answers = &result["json"]["answers"];
    assert_eq!(answers["department"]["choice"], "returns", "{answers}");
    assert!((probabilities_sum(&answers["department"]["probabilities"]) - 1.0).abs() < 1e-6);
    assert!(answers["is_urgent"]["noul"].is_f64(), "{answers}");
    assert_eq!(result["json"]["usage"]["output_tokens"], 0);
}

/// Run `ster request <operation>` with `body` on stdin, the way Ster Desktop
/// does, and return the exit status and every event it printed. The process
/// has ended by the time this returns: nothing of Ster outlives the request.
fn request(operation: &str, body: &Value) -> (Option<i32>, Vec<Value>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ster"))
        .args(["request", operation])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("ster request starts");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(body.to_string().as_bytes())
        .expect("request body written");
    let output = child.wait_with_output().expect("ster request finishes");
    let events = String::from_utf8(output.stdout)
        .expect("UTF-8 events")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("NDJSON event"))
        .collect();
    (output.status.code(), events)
}
