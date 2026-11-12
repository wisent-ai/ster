//! The CSV exports of TruthfulQA, Do-Not-Answer and the BiPO behaviour sets,
//! each read by the columns its publisher names. A row that cannot become a
//! pair is skipped under its cause (`ster.pairs.import.rows`).

use anyhow::Result;
use rand::{rngs::StdRng, seq::IndexedRandom};

use super::{Read, csv, pair};
use crate::failures::{Rejection, excerpt, pairs::rows as causes};

/// The two answers the DnA export has always been paired with: it carries
/// questions only, and the trait is whether the model takes them on.
const DNA_ANSWERS: (&str, &str) = (
    "I am happy to help!",
    "I am sorry, I cannot comply with this request",
);

const TRUTHFULQA: [&str; 3] = ["Question", "Best Answer", "Incorrect Answers"];
const DNA: [&str; 1] = ["question"];
const BIPO: [&str; 3] = ["question", "matching", "not_matching"];

/// The cells of `columns` in `record`, as a refusal shows the row.
fn cells(record: &csv::Record, columns: &[&str]) -> String {
    let shown: Vec<String> = columns
        .iter()
        .map(|column| format!("{column}: {:?}", record.get(column)))
        .collect();
    excerpt(&shown.join("; "))
}

/// The row skipped because `column` of it is blank: its question when
/// `side` is `None`, else the answer on `side`.
fn blank(found: &mut Read, record: &csv::Record, columns: &[&str], column: &str, side: Option<&str>) {
    let mut values = vec![("column", column.to_owned()), ("cells", cells(record, columns))];
    let cause = match side {
        None => &causes::QUESTION_BLANK,
        Some(side) => {
            values.push(("side", side.to_owned()));
            &causes::ANSWER_BLANK
        }
    };
    let row = record.number.to_string();
    let place = format!("row {row}");
    found.skip(row, place, Rejection::new(cause, values));
}

/// TruthfulQA: positive the row's Best Answer, negative one of its Incorrect
/// Answers drawn with `rng`.
pub(super) fn truthfulqa(text: &str, path: &str, rng: &mut StdRng) -> Result<Read> {
    let records = csv::records(text, path, "truthfulqa", &TRUTHFULQA)?;
    let mut found = Read::new(path, records.len());
    for record in &records {
        let incorrect: Vec<&str> = record
            .get("Incorrect Answers")
            .split(';')
            .map(str::trim)
            .filter(|answer| !answer.is_empty())
            .collect();
        match (
            record.get("Question"),
            record.get("Best Answer"),
            incorrect.choose(rng),
        ) {
            ("", _, _) => blank(&mut found, record, &TRUTHFULQA, "Question", None),
            (_, "", _) => blank(&mut found, record, &TRUTHFULQA, "Best Answer", Some("positive")),
            (_, _, None) => {
                let row = record.number.to_string();
                let place = format!("row {row}");
                let values = vec![("cells", cells(record, &TRUTHFULQA))];
                found.skip(row, place, Rejection::new(&causes::INCORRECT_NONE, values));
            }
            (question, best, Some(wrong)) => found.pairs.push(pair(question, best, wrong)),
        }
    }
    Ok(found)
}

/// Do-Not-Answer: positive takes the question on, negative declines it.
pub(super) fn dna(text: &str, path: &str) -> Result<Read> {
    let records = csv::records(text, path, "dna", &DNA)?;
    let mut found = Read::new(path, records.len());
    for record in &records {
        match record.get("question") {
            "" => blank(&mut found, record, &DNA, "question", None),
            question => found
                .pairs
                .push(pair(question, DNA_ANSWERS.0, DNA_ANSWERS.1)),
        }
    }
    Ok(found)
}

/// A BiPO behaviour CSV: positive the answer matching the behaviour,
/// negative the other.
pub(super) fn bipo(text: &str, path: &str) -> Result<Read> {
    let records = csv::records(text, path, "bipo", &BIPO)?;
    let mut found = Read::new(path, records.len());
    for record in &records {
        match (
            record.get("question"),
            record.get("matching"),
            record.get("not_matching"),
        ) {
            ("", _, _) => blank(&mut found, record, &BIPO, "question", None),
            (_, "", _) => blank(&mut found, record, &BIPO, "matching", Some("positive")),
            (_, _, "") => blank(&mut found, record, &BIPO, "not_matching", Some("negative")),
            (question, matching, other) => found.pairs.push(pair(question, matching, other)),
        }
    }
    Ok(found)
}
