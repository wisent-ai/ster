//! `ster decide` and `ster calibrate`, driven as a person drives them: the
//! real binary, the real HuggingFaceTB/SmolLM2-1.7B-Instruct checkpoint from
//! the Hugging Face cache, on the CPU, with every document read back from
//! disk. Nothing is mocked. Each story is one command with its refusals.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::Value;

mod refusals;
mod request;

const MODEL: &str = "HuggingFaceTB/SmolLM2-1.7B-Instruct";
const OTHER_MODEL: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/decide/fixtures").join(name)
}

/// A fresh directory under the package's own build tree for one story.
fn scratch(story: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("decide-{story}"));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("scratch directory");
    root
}

fn ster(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ster")).args(args).output().expect("ster runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("a JSON document")
}

fn probabilities_sum(map: &Value) -> f64 {
    map.as_object().expect("a map").values().map(|value| value.as_f64().expect("a number")).sum()
}

#[test]
fn decide_answers_every_question_from_the_state_and_writes_the_response() {
    let root = scratch("answers");
    let output_path = root.join("response.json");
    let request = fixture("request.json");
    let output = ster(&[
        "decide",
        "--model",
        MODEL,
        "--request",
        request.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "decide failed: {}", stderr(&output));
    let printed = json(&output.stdout);
    let written = json(&fs::read(&output_path).expect("response written"));
    assert_eq!(printed, written, "the printed response and the written one differ");

    assert_eq!(written["model"], MODEL);
    assert_eq!(written["chat_template"], "applied");
    assert_eq!(written["temperature"], 1.0);
    assert!(written["calibration"].is_null());
    assert!(written.get("explain").is_none(), "explain is present without --explain");
    assert_eq!(written["usage"]["output_tokens"], 0);
    // One prefix pass for the state plus one rendering per option order:
    // three orders each for the choice and the score, two for the noul.
    assert_eq!(written["usage"]["forward_passes"], 9);
    assert!(stderr(&output).contains("reading the state once ("), "progress line missing: {}", stderr(&output));

    let department = &written["answers"]["department"];
    assert_eq!(department["type"], "choice");
    assert_eq!(department["choice"], "returns", "a wrong-size exchange belongs to returns: {department}");
    assert!((probabilities_sum(&department["probabilities"]) - 1.0).abs() < 1e-6);
    assert!(department["probabilities"]["returns"].as_f64().unwrap() > 0.5, "{department}");
    let confidence = department["confidence"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&confidence));

    let frustration = &written["answers"]["frustration"];
    assert_eq!(frustration["type"], "score");
    assert_eq!(frustration["legend"]["0"], "Calm, just stating facts");
    assert_eq!(frustration["legend"].as_object().unwrap().len(), 3);
    assert!((probabilities_sum(&frustration["probabilities"]) - 1.0).abs() < 1e-6);
    let score = frustration["score"].as_f64().unwrap();
    assert!((0.0..=2.0).contains(&score), "{frustration}");

    let urgent = &written["answers"]["is_urgent"];
    assert_eq!(urgent["type"], "noul");
    let noul = urgent["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&noul), "{urgent}");
    assert!(urgent.get("confidence").is_none(), "a noul carries no confidence");
}

#[test]
fn decide_explains_every_order_it_showed() {
    let request = fixture("request.json");
    let output = ster(&["decide", "--model", MODEL, "--request", request.to_str().unwrap(), "--explain"]);
    assert!(output.status.success(), "decide failed: {}", stderr(&output));
    let response = json(&output.stdout);
    let department = &response["explain"]["department"];
    let orders = department["orders"].as_array().expect("orders");
    assert_eq!(orders.len(), 3, "three options, three cyclic orders");
    for order in orders {
        let letters: Vec<&str> = order["letters"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let mut sorted = letters.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, ["A", "B", "C"], "every option wears one letter per order: {order}");
        let total: f64 = order["probabilities"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).sum();
        assert!((total - 1.0).abs() < 1e-6, "{order}");
    }
    assert_eq!(response["explain"]["is_urgent"]["options"], serde_json::json!(["yes", "no"]));
}

#[test]
fn calibrate_fits_a_temperature_that_the_next_decide_reads() {
    let root = scratch("calibrate");
    let artifact = root.join("calibration.json");
    let examples = fixture("examples.json");
    let output = ster(&[
        "calibrate",
        "--model",
        MODEL,
        "--examples",
        examples.to_str().unwrap(),
        "--output",
        artifact.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "calibrate failed: {}", stderr(&output));
    let written = json(&fs::read(&artifact).expect("calibration written"));
    assert_eq!(json(&output.stdout), written);
    assert_eq!(written["schema"], "ster-calibration/1");
    assert_eq!(written["model"], MODEL);
    assert_eq!(written["examples"], 4);
    assert_eq!(written["questions"], 11, "three labels on three examples and two on the fourth");
    let temperature = written["temperature"].as_f64().unwrap();
    assert!(temperature > 0.0 && temperature.is_finite());
    assert_eq!(written["after"]["temperature"], written["temperature"]);
    assert_eq!(written["before"]["temperature"], 1.0);
    // Scaling never changes the winner, so accuracy is identical before and
    // after; the fit can only lower the negative log-likelihood.
    assert_eq!(written["before"]["accuracy"], written["after"]["accuracy"]);
    assert!(written["after"]["nll"].as_f64().unwrap() <= written["before"]["nll"].as_f64().unwrap() + 1e-9);
    // The model reads the state: its labels score better on the real pairing
    // than on the shuffled one.
    let real = written["after"]["accuracy"].as_f64().unwrap();
    let control = written["control"]["accuracy"].as_f64().unwrap();
    assert!(real > control, "accuracy {real} does not beat the shuffled-state control {control}");
    assert!(stderr(&output).contains("on shuffled states"), "{}", stderr(&output));

    let request = fixture("request.json");
    let output = ster(&[
        "decide",
        "--model",
        MODEL,
        "--request",
        request.to_str().unwrap(),
        "--calibration",
        artifact.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "calibrated decide failed: {}", stderr(&output));
    let response = json(&output.stdout);
    assert_eq!(response["temperature"], written["temperature"]);
    assert_eq!(response["calibration"], artifact.to_str().unwrap());

    let output = ster(&[
        "decide",
        "--model",
        OTHER_MODEL,
        "--request",
        request.to_str().unwrap(),
        "--calibration",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains(&format!(
            "Error: calibration {} was fitted for model '{MODEL}', not '{OTHER_MODEL}'",
            artifact.display()
        )),
        "{}",
        stderr(&output)
    );
}

