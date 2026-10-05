//! Near-duplicate detection for contrastive pairs.
//!
//! A port of Wisent's `SimHashDeduper`. A synthesised pair set collapses fast:
//! sample the same model a hundred times for "be concise" and a third of the
//! answers are paraphrases of each other. Paraphrases are worse than useless for
//! training a direction — they weight one phrasing heavily and make the training
//! margin look better than it is. SimHash plus banded LSH catches paraphrases in
//! roughly linear time, which matters because this runs inside the generation
//! loop, once per candidate, while a model is loaded.

use std::collections::HashMap;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::artifact::ContrastivePair;

mod fingerprint;

pub use fingerprint::{hamming, normalize, simhash64};
use fingerprint::SIMHASH_BIT_WIDTH;

/// Knobs for fingerprinting and bucketing.
#[derive(Debug, Clone, Copy)]
pub struct DedupeOptions {
    /// Hamming distance at or below which two fingerprints are near-duplicates.
    pub threshold_bits: u32,
    /// Word-shingle size used for non-CJK text.
    pub word_ngram: usize,
    /// Character-shingle size used for CJK/Kana/Hangul text.
    pub char_ngram: usize,
    /// Number of LSH bands the 64-bit fingerprint is split into.
    pub num_bands: u32,
}

impl DedupeOptions {
    /// The caller's threshold and band count, with the shingle sizes of
    /// Wisent's `SimHashDeduper`, which this module ports: single words, and
    /// four characters for CJK/Kana/Hangul text.
    pub fn new(threshold_bits: u32, num_bands: u32) -> Self {
        Self { threshold_bits, word_ngram: 1, char_ngram: 4, num_bands }
    }
}

impl DedupeOptions {
    pub fn validate(&self) -> Result<()> {
        if self.num_bands == 0 || SIMHASH_BIT_WIDTH % self.num_bands != 0 {
            bail!(
                "dedupe band count {} must divide the 64-bit simhash evenly, such as 4, 8, 16, or 32",
                self.num_bands
            );
        }
        if self.word_ngram < 1 || self.char_ngram < 1 {
            bail!("dedupe n-gram sizes must be at least 1");
        }
        if self.threshold_bits > SIMHASH_BIT_WIDTH {
            bail!(
                "dedupe threshold {} exceeds the 64-bit simhash width",
                self.threshold_bits
            );
        }
        Ok(())
    }
}

/// Why a pair was rejected, and which earlier pair it collided with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Duplicate {
    /// Byte-identical after normalisation.
    Exact { of: usize },
    /// Within `threshold_bits` Hamming distance of an earlier fingerprint.
    Near { of: usize, distance: u32 },
}

/// Running first-occurrence-wins index.
///
/// `insert` returns `None` when the pair is new (and keeps it), `Some(Duplicate)`
/// when an earlier kept pair matches. Indices in the returned `Duplicate` are
/// positions in the sequence of *kept* pairs, which for a caller that only ever
/// inserts is the position that pair would occupy in the deduplicated output.
pub struct Index {
    options: DedupeOptions,
    /// `64 / num_bands`.
    band_size: u32,
    band_mask: u64,
    /// Normalised `(positive, negative)` -> kept index, the exact-match pass.
    exact: HashMap<(String, String), usize>,
    fingerprints: Vec<u64>,
    /// One bucket map per band: band value -> kept indices sharing it.
    buckets: Vec<HashMap<u64, Vec<usize>>>,
}

impl Index {
    pub fn new(options: DedupeOptions) -> Result<Self> {
        options.validate()?;
        let band_size = SIMHASH_BIT_WIDTH / options.num_bands;
        // A single band covers the whole word, and `1u64 << 64` would overflow.
        let band_mask =
            if band_size >= SIMHASH_BIT_WIDTH { u64::MAX } else { (1u64 << band_size) - 1 };
        Ok(Self {
            options,
            band_size,
            band_mask,
            exact: HashMap::new(),
            fingerprints: Vec::new(),
            buckets: vec![HashMap::new(); options.num_bands as usize],
        })
    }

    pub fn insert(&mut self, pair: &ContrastivePair) -> Option<Duplicate> {
        let key = (normalize(&pair.positive), normalize(&pair.negative));
        if let Some(of) = self.exact.get(&key) {
            return Some(Duplicate::Exact { of: *of });
        }

        let fingerprint =
            simhash64(&[pair.positive.as_str(), pair.negative.as_str()], self.options);

        let mut candidates: Vec<usize> = Vec::new();
        for band in 0..self.options.num_bands as usize {
            let value = self.band_value(fingerprint, band);
            if let Some(bucket) = self.buckets[band].get(&value) {
                candidates.extend_from_slice(bucket);
            }
        }
        // The Python falls back to a full scan when no band agrees but the index
        // is non-empty, which makes the LSH a speed-up rather than a filter: the
        // result is exact-Hamming recall, not approximate recall. Ported as-is.
        if candidates.is_empty() && !self.fingerprints.is_empty() {
            candidates.extend(0..self.fingerprints.len());
        }
        // Deviation, deliberate. The Python only asks `any(...)` over an
        // unordered set and never reports which pair collided. The contract here
        // reports `of`, so candidates are scanned in ascending kept order and the
        // earliest colliding pair is named — deterministic, and consistent with
        // the first-occurrence-wins rule the rest of the algorithm follows.
        candidates.sort_unstable();
        candidates.dedup();
        for candidate in candidates {
            let distance = hamming(fingerprint, self.fingerprints[candidate]);
            if distance <= self.options.threshold_bits {
                return Some(Duplicate::Near { of: candidate, distance });
            }
        }

        let index = self.fingerprints.len();
        self.fingerprints.push(fingerprint);
        self.exact.insert(key, index);
        for band in 0..self.options.num_bands as usize {
            let value = self.band_value(fingerprint, band);
            self.buckets[band].entry(value).or_default().push(index);
        }
        None
    }

    /// Number of pairs kept so far.
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    fn band_value(&self, fingerprint: u64, band: usize) -> u64 {
        (fingerprint >> (band as u32 * self.band_size)) & self.band_mask
    }
}

/// Classifies every pair against the ones before it, one slot per input pair, in
/// input order.
///
/// Indices carried in the returned `Duplicate`s are positions in `pairs`, not
/// positions in the kept subsequence, so a report can point straight at the
/// offending entry the operator is looking at.
pub fn classify(pairs: &[ContrastivePair], options: DedupeOptions) -> Result<Vec<Option<Duplicate>>> {
    let mut index = Index::new(options)?;
    let mut kept_to_input: Vec<usize> = Vec::with_capacity(pairs.len());
    let mut verdicts: Vec<Option<Duplicate>> = Vec::with_capacity(pairs.len());
    for (position, pair) in pairs.iter().enumerate() {
        match index.insert(pair) {
            None => {
                kept_to_input.push(position);
                verdicts.push(None);
            }
            Some(Duplicate::Exact { of }) => {
                verdicts.push(Some(Duplicate::Exact { of: kept_to_input[of] }));
            }
            Some(Duplicate::Near { of, distance }) => {
                verdicts.push(Some(Duplicate::Near { of: kept_to_input[of], distance }));
            }
        }
    }
    Ok(verdicts)
}

