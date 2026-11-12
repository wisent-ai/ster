//! What a synthesis run is asked for, checked before any model loads, and
//! the report it answers with.

use std::num::NonZeroUsize;

use anyhow::Result;
use serde::Serialize;

use crate::{
    failures::{pairs::synthesize, refuse},
    pairs::quality::{dedupe::DedupeOptions, diversity},
    runtime::GenerationOptions,
};

#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    pub trait_description: String,
    /// The set's label; unnamed or blank, the trait description labels it.
    pub trait_name: Option<String>,
    pub opposite: Option<String>,
    pub count: NonZeroUsize,
    pub retry_multiplier: NonZeroUsize,
    pub dedupe: DedupeOptions,
    pub generation: GenerationOptions,
}

impl SynthesisOptions {
    /// The settings a run cannot start with, checked before any model loads
    /// or any route is asked. `count` and `retry_multiplier` are typed above
    /// zero where they are read.
    pub fn validate(&self) -> Result<()> {
        if self.trait_description.trim().is_empty() {
            return Err(refuse(
                &synthesize::TRAIT_BLANK,
                &[("trait", &self.trait_description)],
            ));
        }
        // Every prompt of the loop is a constant, repeated for the whole run:
        // the same question instruction, the same two persona prompts. A
        // greedy decoder (Ster's own argmax path at or below zero
        // temperature, and any hosted sampler asked for the same) would return
        // the same question and answers forever and the run would dedupe to
        // one pair. The reason is the loop's, so the refusal holds for every
        // route; a temperature that is not a number is not above zero either.
        // Sampling temperature is strictly positive, as Transformers'
        // `TemperatureLogitsWarper` states:
        // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
        let temperature = self.generation.temperature;
        if temperature.is_nan() || temperature <= 0.0 {
            return Err(refuse(
                &synthesize::TEMPERATURE_NOT_POSITIVE,
                &[("temperature", &temperature.to_string())],
            ));
        }
        self.dedupe.validate()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesisReport {
    /// What wrote these pairs: `local:<model id>` or `brama:<route>`. First in
    /// the report because every other number below is only interpretable once
    /// the reader knows which model produced the text.
    pub generator: String,
    pub trait_name: String,
    pub trait_description: String,
    pub opposite: String,
    pub requested: usize,
    pub attempts: usize,
    pub kept: usize,
    pub rejected_empty: usize,
    pub rejected_refusals: usize,
    pub rejected_duplicates: usize,
    pub refusal_retries: usize,
    pub diversity: diversity::Scores,
}
