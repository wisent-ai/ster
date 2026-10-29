//! A model-free audit of a pair set: duplicates and the length confound, each
//! judged from the text alone so the desktop app can show the answer the
//! moment a file is opened. Whether a side is a refusal needs a model's
//! judgement, which synthesis asks for as it writes each pair.

use anyhow::Result;
use serde::Serialize;

use crate::{
    artifact::PairSet,
    pairs::quality::dedupe::{self, DedupeOptions, Duplicate},
    pairs::quality::diversity,
};

// MARK: - Inspection

/// Every judgement threshold is the caller's; Ster assumes none.
#[derive(Debug, Clone)]
pub struct InspectOptions {
    pub dedupe: DedupeOptions,
    /// A pair whose longer side has more than this many times the other's
    /// characters is counted as unbalanced. Length is the confound the
    /// product documents at https://ster.wisent.com/docs/concept-contrastive-pair:
    /// a direction trained on consistently longer positives encodes length,
    /// not the trait.
    pub unbalanced_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntryReport {
    pub index: usize,
    pub positive: String,
    pub negative: String,
    pub positive_chars: usize,
    pub negative_chars: usize,
    pub positive_words: usize,
    pub negative_words: usize,
    pub duplicate: Option<Duplicate>,
    pub length_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetReport {
    pub trait_name: String,
    pub pair_count: usize,
    pub duplicate_count: usize,
    pub unbalanced_count: usize,
    /// The ratio `unbalanced_count` was counted against, as the caller stated it.
    pub unbalanced_ratio: f64,
    pub diversity: diversity::Scores,
    pub entries: Vec<EntryReport>,
}

/// Audit a pair set without loading a model. Every judgement here is textual,
/// which is why the desktop app can show it the moment a file is opened.
pub fn inspect(pairs: &PairSet, options: &InspectOptions) -> Result<SetReport> {
    // The set inspected is the set `PairSet::load` admits: at least one pair,
    // and no side blank after trimming, so every side has a character and
    // `length_ratio` never divides by zero.
    pairs.validate(&format!("{:?}", pairs.trait_name))?;
    let duplicates = dedupe::classify(&pairs.pairs, options.dedupe)?;
    let entries: Vec<EntryReport> = pairs
        .pairs
        .iter()
        .zip(duplicates)
        .enumerate()
        .map(|(index, (pair, duplicate))| {
            let positive_chars = pair.positive.chars().count();
            let negative_chars = pair.negative.chars().count();
            EntryReport {
                index,
                positive: pair.positive.clone(),
                negative: pair.negative.clone(),
                positive_chars,
                negative_chars,
                positive_words: pair.positive.split_whitespace().count(),
                negative_words: pair.negative.split_whitespace().count(),
                duplicate,
                length_ratio: length_ratio(positive_chars, negative_chars),
            }
        })
        .collect();

    // Diversity reads the positive side only: the two sides of a pair are
    // near-copies of each other by construction, so scoring both would report
    // the contrast as repetition.
    let positives: Vec<String> = pairs
        .pairs
        .iter()
        .map(|pair| pair.positive.clone())
        .collect();
    let diversity = diversity::compute(&positives);

    Ok(SetReport {
        trait_name: pairs.trait_name.clone(),
        pair_count: pairs.pairs.len(),
        duplicate_count: entries
            .iter()
            .filter(|entry| entry.duplicate.is_some())
            .count(),
        unbalanced_count: entries
            .iter()
            .filter(|entry| entry.length_ratio > options.unbalanced_ratio)
            .count(),
        unbalanced_ratio: options.unbalanced_ratio,
        diversity,
        entries,
    })
}

/// Longer side over shorter side, in characters. `inspect` refused a blank
/// side before calling it, so the shorter side has at least one character.
fn length_ratio(positive_chars: usize, negative_chars: usize) -> f64 {
    positive_chars.max(negative_chars) as f64 / positive_chars.min(negative_chars) as f64
}
