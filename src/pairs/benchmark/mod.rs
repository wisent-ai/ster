//! Contrastive pair sets read from published benchmark exports: TruthfulQA,
//! Do-Not-Answer (DnA), LiveCodeBench, the BiPO behaviour CSVs, and the
//! multiple-choice rows of any dataset whose schema the caller names. This is
//! the pair generation the retired Python `wisent-benchmark` and
//! `wisent-extractors` packages did, in the pair-set format every Ster command
//! already reads.
//!
//! Each benchmark row becomes one pair written the way `pairs synthesize`
//! writes one — `Question: …\nAnswer: …` on both sides — so an imported set and
//! a synthesized set steer the same way. Unlike the Python loaders, a file that
//! cannot be read fails the import instead of yielding an empty list, and a row
//! that cannot become a pair is reported by number with the reason.

mod choices;
mod csv;

pub use choices::{AnswerForm, ChoiceFields};

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rand::{SeedableRng, rngs::StdRng, seq::IndexedRandom};
use serde::Serialize;
use serde_json::Value;

use crate::{ContrastivePair, PairSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Benchmark {
    /// Positive: the row's Best Answer. Negative: one of its Incorrect Answers.
    Truthfulqa,
    /// Do-Not-Answer: positive answers the question, negative declines it.
    Dna,
    /// Positive: the problem's good example code. Negative: its bad example.
    Livecodebench,
    /// The BiPO paper's behaviour CSVs (question, matching, not_matching):
    /// positive the answer matching the behaviour, negative the other.
    Bipo,
    /// Multiple-choice rows (JSON Lines or a JSON list) of any dataset:
    /// positive the correct choice, negative another choice drawn with the
    /// seed, each part read where [`ChoiceFields`] says it sits.
    Choices,
}

impl Benchmark {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "truthfulqa" => Ok(Self::Truthfulqa),
            "dna" => Ok(Self::Dna),
            "livecodebench" => Ok(Self::Livecodebench),
            "bipo" => Ok(Self::Bipo),
            "choices" => Ok(Self::Choices),
            other => {
                bail!(
                    "unknown benchmark {other}: choose truthfulqa, dna, livecodebench, bipo or choices"
                )
            }
        }
    }
}

/// The two answers the DnA export has always been paired with: it carries
/// questions only, and the trait is whether the model takes them on.
const DNA_ANSWERS: (&str, &str) = (
    "I am happy to help!",
    "I am sorry, I cannot comply with this request",
);

pub struct ImportOptions {
    pub benchmark: Benchmark,
    /// The benchmark export: TruthfulQA_en.csv, DnA_en.csv, problems.json, a
    /// BiPO CSV, or multiple-choice rows.
    pub source: PathBuf,
    /// LiveCodeBench's question_examples.json; defaults to the file of that
    /// name beside `source`.
    pub examples: Option<PathBuf>,
    /// Where a multiple-choice row holds its parts; required by
    /// [`Benchmark::Choices`] and refused by every other benchmark.
    pub fields: Option<ChoiceFields>,
    /// Keep this many pairs, drawn with `seed`; every pair when absent.
    pub count: Option<usize>,
    pub seed: u64,
    pub trait_name: String,
}

#[derive(Debug, Serialize)]
pub struct Skipped {
    pub row: String,
    pub reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ImportReport {
    pub benchmark: Benchmark,
    pub source: String,
    pub rows: usize,
    pub skipped: Vec<Skipped>,
    pub pairs: usize,
}

fn pair(question: &str, positive: &str, negative: &str) -> ContrastivePair {
    ContrastivePair {
        positive: format!("Question: {question}\nAnswer: {positive}"),
        negative: format!("Question: {question}\nAnswer: {negative}"),
    }
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

pub fn import(options: &ImportOptions) -> Result<(PairSet, ImportReport)> {
    let mut rng = StdRng::seed_from_u64(options.seed);
    let label = options.source.display().to_string();
    let mut skipped = Vec::new();
    let mut pairs = Vec::new();
    let rows;
    match (options.benchmark, &options.fields) {
        (Benchmark::Choices, None) => bail!(
            "--benchmark choices needs --question, --choices, --answer and --answer-form: where each part of a row sits"
        ),
        (Benchmark::Choices, Some(_)) | (_, None) => {}
        (_, Some(_)) => bail!(
            "--question, --choices, --answer, --answer-form and --labels apply only to --benchmark choices"
        ),
    }
    match options.benchmark {
        Benchmark::Truthfulqa | Benchmark::Dna | Benchmark::Bipo => {
            let records = csv::records(&read(&options.source)?, &label)?;
            rows = records.len();
            for record in &records {
                let row = record.number.to_string();
                if options.benchmark == Benchmark::Dna {
                    match record.get("question") {
                        "" => skipped.push(Skipped {
                            row,
                            reason: "no question",
                        }),
                        question => pairs.push(pair(question, DNA_ANSWERS.0, DNA_ANSWERS.1)),
                    }
                    continue;
                }
                if options.benchmark == Benchmark::Bipo {
                    match (
                        record.get("question"),
                        record.get("matching"),
                        record.get("not_matching"),
                    ) {
                        ("", _, _) => skipped.push(Skipped {
                            row,
                            reason: "no question",
                        }),
                        (_, "", _) => skipped.push(Skipped {
                            row,
                            reason: "no matching answer",
                        }),
                        (_, _, "") => skipped.push(Skipped {
                            row,
                            reason: "no not_matching answer",
                        }),
                        (question, matching, other) => pairs.push(pair(question, matching, other)),
                    }
                    continue;
                }
                let (question, best) = (record.get("Question"), record.get("Best Answer"));
                let incorrect: Vec<&str> = record
                    .get("Incorrect Answers")
                    .split(';')
                    .map(str::trim)
                    .filter(|answer| !answer.is_empty())
                    .collect();
                match (question, best, incorrect.choose(&mut rng)) {
                    ("", _, _) => skipped.push(Skipped {
                        row,
                        reason: "no question",
                    }),
                    (_, "", _) => skipped.push(Skipped {
                        row,
                        reason: "no best answer",
                    }),
                    (_, _, None) => skipped.push(Skipped {
                        row,
                        reason: "no incorrect answer",
                    }),
                    (question, best, Some(wrong)) => pairs.push(pair(question, best, wrong)),
                }
            }
        }
        Benchmark::Livecodebench => {
            let problems: Vec<Value> = serde_json::from_str(&read(&options.source)?)
                .with_context(|| format!("{label} is not a JSON list of problems"))?;
            let examples_path = options
                .examples
                .clone()
                .unwrap_or_else(|| options.source.with_file_name("question_examples.json"));
            let examples: Value = serde_json::from_str(&read(&examples_path)?)
                .with_context(|| format!("{} is not JSON", examples_path.display()))?;
            let examples = examples
                .get("questions")
                .and_then(Value::as_array)
                .with_context(|| format!("{} has no questions list", examples_path.display()))?;
            let code = |example: &Value, side: &str| {
                example
                    .pointer(&format!("/{side}/code"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            };
            rows = problems.len();
            for problem in &problems {
                let id = problem
                    .get("question_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let row = if id.is_empty() {
                    "(no question_id)".to_string()
                } else {
                    id.to_string()
                };
                let content = problem
                    .get("question_content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim();
                let example = examples.iter().find(|example| {
                    !id.is_empty() && example.get("question_id").and_then(Value::as_str) == Some(id)
                });
                let (good, bad) = example
                    .map(|example| (code(example, "good_example"), code(example, "bad_example")))
                    .unwrap_or_default();
                if content.is_empty() {
                    skipped.push(Skipped {
                        row,
                        reason: "no question_content",
                    });
                } else if good.is_empty() || bad.is_empty() {
                    skipped.push(Skipped {
                        row,
                        reason: "no good and bad example code",
                    });
                } else {
                    pairs.push(pair(content, &good, &bad));
                }
            }
        }
        Benchmark::Choices => {
            let fields = options
                .fields
                .as_ref()
                .context("--benchmark choices needs its row fields")?;
            let records = choices::rows(&read(&options.source)?, &label)?;
            rows = records.len();
            for (row, record) in records {
                match choices::pair_of(&record, fields, &mut rng) {
                    Ok(found) => pairs.push(found),
                    Err(reason) => skipped.push(Skipped { row, reason }),
                }
            }
        }
    }
    if let Some(count) = options.count {
        if count > pairs.len() {
            bail!(
                "{label} yields {} pairs, fewer than --count {count}",
                pairs.len()
            );
        }
        pairs = pairs.choose_multiple(&mut rng, count).cloned().collect();
    }
    let set = PairSet {
        trait_name: options.trait_name.clone(),
        pairs,
    };
    set.validate(&label)?;
    let report = ImportReport {
        benchmark: options.benchmark,
        source: label,
        rows,
        skipped,
        pairs: set.pairs.len(),
    };
    Ok((set, report))
}
