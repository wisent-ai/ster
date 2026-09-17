//! The route Ster Desktop's Decide screen calls, driven on the real backend.

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    process::{Command, Stdio},
};

use serde_json::Value;

use crate::{probabilities_sum, MODEL};

/// The desktop never builds a command line: it posts the request document
/// inline to `/v1/decide` on `ster serve` and reads the NDJSON stream. This
/// drives that route on the real backend with the body Ster Desktop's
/// Decide screen composes, and a body the backend refuses before loading.
#[test]
fn serve_answers_the_desktops_decide_request_and_refuses_a_bad_one() {
    let mut server = Command::new(env!("CARGO_BIN_EXE_ster"))
        .args(["serve", "--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("ster serve starts");
    let mut ready = String::new();
    BufReader::new(server.stdout.take().expect("stdout")).read_line(&mut ready).expect("ready line");
    let ready: Value = serde_json::from_str(&ready).expect("ready JSON");
    assert_eq!(ready["ready"], true, "{ready}");
    let port = ready["port"].as_u64().expect("port");

    let refused = post(port, "/v1/decide", &serde_json::json!({
        "model": MODEL, "device": "cpu",
        "request": {"state": "x", "questions": {"only": {"type": "choice", "instructions": "?", "criteria": {"a": null}}}}
    }));
    assert!(refused.0.starts_with("HTTP/1.1 400"), "{}", refused.0);
    let body: Value = serde_json::from_str(refused.1.trim()).expect("error envelope");
    assert_eq!(body["error"], "choice question 'only' needs at least two options");

    let answered = post(port, "/v1/decide", &serde_json::json!({
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
    }));
    server.kill().expect("serve stops");
    assert!(answered.0.starts_with("HTTP/1.1 200"), "{}", answered.0);
    let events: Vec<Value> = answered
        .1
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("NDJSON event"))
        .collect();
    let result = events.last().expect("a result event");
    assert_eq!(result["type"], "result", "{result}");
    assert_eq!(result["status"], 0, "{result}");
    let logs: Vec<&str> =
        events.iter().filter(|event| event["type"] == "log").map(|event| event["chunk"].as_str().unwrap()).collect();
    assert!(logs.iter().any(|chunk| chunk.contains("reading the state once (")), "{logs:?}");
    let answers = &result["json"]["answers"];
    assert_eq!(answers["department"]["choice"], "returns", "{answers}");
    assert!((probabilities_sum(&answers["department"]["probabilities"]) - 1.0).abs() < 1e-6);
    assert!(answers["is_urgent"]["noul"].is_f64(), "{answers}");
    assert_eq!(result["json"]["usage"]["output_tokens"], 0);
}

/// One HTTP POST to the serve backend, returning the status line and the
/// whole body. The backend closes the connection after each response, so
/// reading to the end is reading the body.
fn post(port: u64, path: &str, body: &Value) -> (String, String) {
    let payload = body.to_string();
    let mut stream = TcpStream::connect(("127.0.0.1", port as u16)).expect("connect to serve");
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    )
    .expect("request written");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("response read");
    let (head, body) = response.split_once("\r\n\r\n").expect("a response head");
    let status = head.lines().next().unwrap_or_default().to_owned();
    (status, body.to_owned())
}
