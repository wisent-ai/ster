//! The benchmarks `pairs import` reads, by the name `--benchmark` takes.

use anyhow::Result;
use serde::Serialize;

use crate::failures::{pairs::import, refuse};

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
