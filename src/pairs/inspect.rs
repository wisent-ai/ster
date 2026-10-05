//! A model-free audit of a pair set: duplicates, refusals, and the length
//! confound, each judged from the text alone so the desktop app can show the
//! answer the moment a file is opened.

use anyhow::Result;
use serde::Serialize;

use crate::{
    artifact::PairSet,
    pairs::quality::dedupe::{self, DedupeOptions, Duplicate},
    pairs::quality::{diversity, refusal},
};

// MARK: - Inspection

/// Every judgement threshold is the caller's; Ster assumes none.
#[derive(Debug, Clone)]
pub struct InspectOptions {
    pub dedupe: DedupeOptions,
    pub refusal_threshold: f32,
    /// A pair whose longer side has more than this many times the other's
    /// characters is counted as unbalanced. Length is the confound the
    /// product documents at https://ster.wisent.com/docs/concept-contrastive-pair:
    /// a direction trained on consistently longer positives encodes length,
    /// not the trait.
    pub unbalanced_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RefusalFlag {
    pub score: f32,
    pub family: String,
    pub snippet: String,
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
    pub positive_refusal: Option<RefusalFlag>,
    pub negative_refusal: Option<RefusalFlag>,
    pub length_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetReport {
    pub trait_name: String,
    pub pair_count: usize,
    pub duplicate_count: usize,
    pub refusal_count: usize,
    pub unbalanced_count: usize,
    /// The ratio `unbalanced_count` was counted against, as the caller stated it.
    pub unbalanced_ratio: f64,
    pub diversity: diversity::Scores,
    pub entries: Vec<EntryReport>,
}

/// Audit a pair set without loading a model. Every judgement here is textual,
/// which is why the desktop app can show it the moment a file is opened.
pub fn inspect(pairs: &PairSet, options: &InspectOptions) -> Result<SetReport> {
    let duplicates = dedupe::classify(&pairs.pairs, options.dedupe)?;
    let mut entries = Vec::with_capacity(pairs.pairs.len());
    let mut duplicate_count = 0usize;
    let mut refusal_count = 0usize;
    let mut unbalanced_count = 0usize;

    for (index, pair) in pairs.pairs.iter().enumerate() {
        let duplicate = duplicates[index];
        if duplicate.is_some() {
            duplicate_count += 1;
        }
        let positive_refusal = flag(&pair.positive, options.refusal_threshold);
        let negative_refusal = flag(&pair.negative, options.refusal_threshold);
        if positive_refusal.is_some() || negative_refusal.is_some() {
            refusal_count += 1;
        }
        let positive_chars = pair.positive.chars().count();
        let negative_chars = pair.negative.chars().count();
        let length_ratio = length_ratio(positive_chars, negative_chars);
        if length_ratio > options.unbalanced_ratio {
            unbalanced_count += 1;
        }
        entries.push(EntryReport {
            index,
            positive: pair.positive.clone(),
            negative: pair.negative.clone(),
            positive_chars,
            negative_chars,
            positive_words: pair.positive.split_whitespace().count(),
            negative_words: pair.negative.split_whitespace().count(),
            duplicate,
            positive_refusal,
            negative_refusal,
            length_ratio,
        });
    }

    // Diversity reads the positive side only: the two sides of a pair are
    // near-copies of each other by construction, so scoring both would report
    // the contrast as repetition.
    let positives: Vec<String> = pairs.pairs.iter().map(|pair| pair.positive.clone()).collect();
    let diversity = diversity::compute(&positives);

    Ok(SetReport {
        trait_name: pairs.trait_name.clone(),
        pair_count: pairs.pairs.len(),
        duplicate_count,
        refusal_count,
        unbalanced_count,
        unbalanced_ratio: options.unbalanced_ratio,
        diversity,
        entries,
    })
}

fn flag(text: &str, threshold: f32) -> Option<RefusalFlag> {
    let scored = refusal::score(text);
    if scored.score < threshold {
        return None;
    }
    Some(RefusalFlag {
        score: scored.score,
        family: scored.family.map(|family| family.name().to_owned()).unwrap_or_default(),
        snippet: scored.snippet,
    })
}

/// Longer side over shorter side. Two empty sides are perfectly balanced and
/// 0/0 has no value, so they report 1.0; a single empty side would divide by
/// zero, so the non-empty length stands in for the ratio and the pair reads
/// as maximally unbalanced.
fn length_ratio(positive_chars: usize, negative_chars: usize) -> f64 {
    let longer = positive_chars.max(negative_chars);
    let shorter = positive_chars.min(negative_chars);
    if longer == 0 {
        return 1.0;
    }
    longer as f64 / shorter.max(1) as f64
}
