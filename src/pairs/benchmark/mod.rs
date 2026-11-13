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
//! that cannot become a pair is reported by number with the reason. Every
//! flag is checked before the export is read.

mod choices;
mod preference;
mod csv;
mod fields;
mod livecodebench;
mod published;

pub use fields::{AnswerForm, ChoiceFields};
pub use preference::pair_of as preference_pair;

use std::path::{Path, PathBuf};

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng, seq::IndexedRandom};
use serde::Serialize;

use crate::failures::{
    Refused, Rejection, Rejections, file,
    pairs::{export, import},
    refuse,
};
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
    /// Preference rows (DPO: prompt, chosen, rejected; JSON Lines or a JSON
    /// list): positive the chosen answer, negative the rejected one.
    Preference,
}

impl Benchmark {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "truthfulqa" => Ok(Self::Truthfulqa),
            "dna" => Ok(Self::Dna),
            "livecodebench" => Ok(Self::Livecodebench),
            "bipo" => Ok(Self::Bipo),
            "choices" => Ok(Self::Choices),
            "preference" => Ok(Self::Preference),
            other => Err(refuse(
                &import::BENCHMARK_UNKNOWN,
                &[("benchmark", other), ("benchmarks", "truthfulqa, dna, livecodebench, bipo, choices, preference")],
            )),
        }
    }

    /// The name `--benchmark` takes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Truthfulqa => "truthfulqa",
            Self::Dna => "dna",
            Self::Livecodebench => "livecodebench",
            Self::Bipo => "bipo",
            Self::Choices => "choices",
            Self::Preference => "preference",
        }
    }
}

pub struct ImportOptions {
    pub benchmark: Benchmark,
    /// The benchmark export: TruthfulQA_en.csv, DnA_en.csv, problems.json, a
    /// BiPO CSV, or multiple-choice rows.
    pub source: PathBuf,
    /// LiveCodeBench's question_examples.json; `--benchmark livecodebench`
    /// refuses without it and every other benchmark refuses it.
    pub examples: Option<PathBuf>,
    /// Where a multiple-choice row holds its parts; required by
    /// [`Benchmark::Choices`] and refused by every other benchmark.
    pub fields: Option<ChoiceFields>,
    /// Keep this many pairs, drawn with `seed`; every pair when absent.
    pub count: Option<usize>,
    pub seed: u64,
    pub trait_name: String,
}

impl ImportOptions {
    /// Every flag against the benchmark, before the export is read.
    fn check(&self) -> Result<()> {
        let source = self.source.display().to_string();
        let name = self.benchmark.name();
        let count = self.count.map_or_else(|| "every pair".to_owned(), |count| count.to_string());
        let values = [("benchmark", name), ("source", source.as_str()), ("count", count.as_str())];
        match (self.benchmark, &self.fields) {
            (Benchmark::Choices, None) => {
                return Err(refuse(&import::CHOICE_FIELDS_MISSING, &values));
            }
            (Benchmark::Choices, Some(_)) | (_, None) => {}
            (_, Some(fields)) => {
                return Err(refuse(
                    &import::CHOICE_FIELDS_NOT_APPLICABLE,
                    &[("benchmark", name), ("given", &fields.given())],
                ));
            }
        }
        match (self.benchmark, &self.examples) {
            (Benchmark::Livecodebench, None) => {
                return Err(refuse(&import::EXAMPLES_MISSING, &values));
            }
            (Benchmark::Livecodebench, Some(_)) | (_, None) => {}
            (_, Some(examples)) => {
                return Err(refuse(
                    &import::EXAMPLES_NOT_APPLICABLE,
                    &[
                        ("benchmark", name),
                        ("source", &source),
                        ("examples", &examples.display().to_string()),
                    ],
                ));
            }
        }
        if self.count == Some(0) {
            return Err(refuse(&import::COUNT_ZERO, &values));
        }
        Ok(())
    }
}

/// One row that could not become a pair: its number (or its problem's
/// question_id), the refusal as a reader meets it, and its declared cause
/// (`ster.pairs.import.rows`).
#[derive(Debug, Serialize)]
pub struct Skipped {
    pub row: String,
    #[serde(flatten)]
    pub refused: Refused,
}

#[derive(Debug, Serialize)]
pub struct ImportReport {
    pub benchmark: Benchmark,
    pub source: String,
    pub rows: usize,
    pub skipped: Vec<Skipped>,
    pub pairs: usize,
}

/// What reading one export gave: how many rows it holds, the pairs they
/// became, and every row that could not become one under its cause.
#[derive(Default)]
struct Read {
    rows: usize,
    pairs: Vec<ContrastivePair>,
    skipped: Vec<Skipped>,
    rejections: Rejections,
    /// The export, as every skipped row's refusal names it.
    source: String,
}

impl Read {
    fn new(source: &str, rows: usize) -> Self {
        Self {
            rows,
            source: source.to_owned(),
            ..Self::default()
        }
    }

    /// Skip the row the report counts as `row` and its refusal names as
    /// `place` (`row 2`, `problem 3`), for `rejection`.
    fn skip(&mut self, row: impl Into<String>, place: String, rejection: Rejection) {
        let rejection = rejection.with("row", place).with("source", self.source.as_str());
        let refused = self.rejections.record(&rejection);
        self.skipped.push(Skipped {
            row: row.into(),
            refused,
        });
    }
}

fn pair(question: &str, positive: &str, negative: &str) -> ContrastivePair {
    ContrastivePair {
        positive: format!("Question: {question}\nAnswer: {positive}"),
        negative: format!("Question: {question}\nAnswer: {negative}"),
    }
}

/// The export named by `flag` at `path`, as UTF-8 text.
fn read_export(path: &Path, flag: &str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|error| export::read_refusal(path, flag, &error))
}

pub fn import(options: &ImportOptions) -> Result<(PairSet, ImportReport)> {
    options.check()?;
    let mut rng = StdRng::seed_from_u64(options.seed);
    let label = options.source.display().to_string();
    let name = options.benchmark.name();
    let text = read_export(&options.source, "--source")?;
    // `check` already refused a missing --examples or missing row flags; the
    // let-else arms below give the same refusal where the types need one.
    let values = [("benchmark", name), ("source", label.as_str())];
    let mut found = match options.benchmark {
        Benchmark::Livecodebench => {
            let Some(examples) = &options.examples else {
                return Err(refuse(&import::EXAMPLES_MISSING, &values));
            };
            let examples_text = read_export(examples, "--examples")?;
            livecodebench::read(&text, &label, &examples_text, examples)?
        }
        Benchmark::Choices => {
            let Some(fields) = &options.fields else {
                return Err(refuse(&import::CHOICE_FIELDS_MISSING, &values));
            };
            let records = choices::rows(&text, &label)?;
            let mut found = Read::new(&label, records.len());
            for (row, record) in records {
                match choices::pair_of(&record, fields, &mut rng) {
                    Ok(made) => found.pairs.push(made),
                    Err(rejection) => {
                        let place = format!("row {row}");
                        found.skip(row, place, rejection);
                    }
                }
            }
            found
        }
        Benchmark::Preference => {
            let records = choices::rows(&text, &label)?;
            let mut found = Read::new(&label, records.len());
            for (row, record) in records {
                match preference::pair_of(&record) {
                    Ok(made) => found.pairs.push(made),
                    Err(rejection) => {
                        let place = format!("row {row}");
                        found.skip(row, place, rejection);
                    }
                }
            }
            found
        }
        Benchmark::Truthfulqa => published::truthfulqa(&text, &label, &mut rng)?,
        Benchmark::Dna => published::dna(&text, &label)?,
        Benchmark::Bipo => published::bipo(&text, &label)?,
    };
    let skipped = found.skipped.len().to_string();
    if found.pairs.is_empty() {
        // The first skipped row's own cause, so its fix is the row's, with
        // how every row was skipped after it.
        let kept_none = format!("no row of {label} became a pair, and a set without pairs is refused");
        let rows = found.rows;
        return Err(match found.rejections.refusal(&kept_none, rows) {
            Some(refusal) => refusal,
            None => refuse(
                &import::NO_PAIRS,
                &[
                    ("benchmark", name),
                    ("source", &label),
                    ("state", &file::state(&options.source)),
                ],
            ),
        });
    }
    if let Some(count) = options.count {
        if count > found.pairs.len() {
            return Err(refuse(
                &import::COUNT_EXCEEDS,
                &[
                    ("count", &count.to_string()),
                    ("source", &label),
                    ("pairs", &found.pairs.len().to_string()),
                    ("rows", &found.rows.to_string()),
                    ("skipped", &skipped),
                ],
            ));
        }
        found.pairs = found
            .pairs
            .choose_multiple(&mut rng, count)
            .cloned()
            .collect();
    }
    let set = PairSet {
        trait_name: options.trait_name.clone(),
        pairs: found.pairs,
    };
    set.validate(&label)?;
    let report = ImportReport {
        benchmark: options.benchmark,
        source: label,
        rows: found.rows,
        skipped: found.skipped,
        pairs: set.pairs.len(),
    };
    Ok((set, report))
}
