//! Every documented refusal `ster decide` and `ster calibrate` give before a
//! weight is mapped, each with its exact sentence and its exit status.

use std::fs;

use crate::{fixture, scratch, stderr, ster, MODEL};

#[test]
fn decide_refuses_a_request_it_cannot_answer_before_loading_the_model() {
    let root = scratch("refusals");
    let output_path = root.join("never.json");
    let too_many = fixture("too-many-options.json");
    let output = ster(&[
        "decide",
        "--model",
        MODEL,
        "--request",
        too_many.to_str().unwrap(),
        "--output",
        output_path.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("Error: choice question 'letter' has 27 options; Ster labels at most 26"),
        "{}",
        stderr(&output)
    );
    assert!(!output_path.exists(), "a refused request wrote a response");

    let missing = root.join("missing.json");
    let output = ster(&["decide", "--model", MODEL, "--request", missing.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains(&format!("Error: failed to read {}", missing.display())), "{}", stderr(&output));

    let not_a_request = fixture("examples.json");
    let output = ster(&["decide", "--model", MODEL, "--request", not_a_request.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains(&format!("Error: invalid decide request {}", not_a_request.display())),
        "{}",
        stderr(&output)
    );

    let one_option = root.join("one-option.json");
    fs::write(
        &one_option,
        r#"{"state": "x", "questions": {"only": {"type": "choice", "instructions": "?", "criteria": {"a": null}}}}"#,
    )
    .unwrap();
    let output = ster(&["decide", "--model", MODEL, "--request", one_option.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("Error: choice question 'only' needs at least two options"), "{}", stderr(&output));

    let no_questions = root.join("no-questions.json");
    fs::write(&no_questions, r#"{"state": "x", "questions": {}}"#).unwrap();
    let output = ster(&["decide", "--model", MODEL, "--request", no_questions.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("Error: a decide request needs at least one question"), "{}", stderr(&output));
}

#[test]
fn calibrate_refuses_examples_it_cannot_learn_from() {
    let root = scratch("calibrate-refusals");
    let artifact = root.join("never.json");
    let not_examples = fixture("request.json");
    let output = ster(&[
        "calibrate",
        "--model",
        MODEL,
        "--examples",
        not_examples.to_str().unwrap(),
        "--output",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains(&format!("Error: invalid calibration examples {}", not_examples.display())),
        "{}",
        stderr(&output)
    );

    let wrong_label = root.join("wrong-label.json");
    fs::write(
        &wrong_label,
        r#"{"examples": [{"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": null, "b": null}}}, "answers": {"q": "c"}}]}"#,
    )
    .unwrap();
    let output = ster(&["calibrate", "--model", MODEL, "--examples", wrong_label.to_str().unwrap(), "--output", artifact.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("Error: example 0 answers question 'q' with \"c\", which is not one of its options"),
        "{}",
        stderr(&output)
    );

    let unasked = root.join("unasked.json");
    fs::write(
        &unasked,
        r#"{"examples": [{"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "answers": {"other": true}}]}"#,
    )
    .unwrap();
    let output = ster(&["calibrate", "--model", MODEL, "--examples", unasked.to_str().unwrap(), "--output", artifact.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("Error: example 0 labels question 'other', which it does not ask"), "{}", stderr(&output));

    let unlabelled = root.join("unlabelled.json");
    fs::write(
        &unlabelled,
        r#"{"examples": [{"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}, "answers": {}}]}"#,
    )
    .unwrap();
    let output = ster(&["calibrate", "--model", MODEL, "--examples", unlabelled.to_str().unwrap(), "--output", artifact.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("Error: calibration needs at least one labelled question"), "{}", stderr(&output));
    assert!(!artifact.exists(), "a refused calibration wrote an artifact");
}
