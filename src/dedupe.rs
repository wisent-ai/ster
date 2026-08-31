//! Near-duplicate detection for contrastive pairs.
//!
//! A port of Wisent's `SimHashDeduper`. A synthesised pair set collapses fast:
//! sample the same model a hundred times for "be concise" and a third of the
//! answers are paraphrases of each other. Paraphrases are worse than useless for
//! training a direction — they weight one phrasing heavily and make the training
//! margin look better than it is. SimHash plus banded LSH catches paraphrases in
//! roughly linear time, which matters because this runs inside the generation
//! loop, once per candidate, while a model is loaded.

use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use anyhow::{Result, bail};
use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use regex::Regex;
use serde::Serialize;
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

use crate::artifact::ContrastivePair;

/// Width of the SimHash fingerprint; the Python `SIMHASH_BIT_WIDTH`.
const SIMHASH_BIT_WIDTH: u32 = 64;

/// The Python `BLAKE2B_DIGEST_SIZE`: an 8-byte, i.e. 64-bit, digest.
const BLAKE2B_DIGEST_SIZE: usize = 8;

/// Ported `_default_stopwords()`.
///
/// Deviation, forced. `SimHashDeduper.__init__` calls `self._default_stopwords()`
/// whenever the caller leaves `stopwords` unset, but that method is defined
/// neither on the class nor on the abstract `Deduper` base, and no definition
/// exists anywhere in the Wisent tree — so the shipped default constructor
/// raises `AttributeError` before it can return a set. There is no word list to
/// port. The empty set is the only faithful reading, and it is also the
/// behaviour every caller that *does* pass `stopwords=set()` already gets.
/// Filtering stays in the token pipeline so a real list can be dropped in later.
const STOPWORDS: [&str; 0] = [];

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

impl Default for DedupeOptions {
    fn default() -> Self {
        Self { threshold_bits: 3, word_ngram: 1, char_ngram: 4, num_bands: 8 }
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

/// Computes the 64-bit SimHash fingerprint of a pair's fields.
///
/// Deviation, deliberate. The Python looks each field's weight up with
/// `float(self.field_weights[field])` against a dict that defaults to empty, so
/// the shipped call path raises `KeyError` on the first field; the `if w != 1.0`
/// branch immediately below that lookup only makes sense if 1.0 is the intended
/// default, so every field is weighted 1.0 here. The defect is recorded rather
/// than reproduced because reproducing it would mean refusing every input.
///
/// Second deviation, forced by the data model: Ster's `ContrastivePair` has no
/// `prompt` field, so the hashed fields are `positive` and `negative` — the
/// Python's `fields_to_hash` default of `("prompt",)` has no counterpart here.
pub fn simhash64(fields: &[&str], options: DedupeOptions) -> u64 {
    // Ordered so the ± accumulation below is bit-for-bit reproducible across runs.
    let mut features: BTreeMap<String, f64> = BTreeMap::new();
    for field in fields.iter().copied() {
        if field.is_empty() {
            continue;
        }
        for (feature, count) in extract_features(field, options) {
            *features.entry(feature).or_insert(0.0) += count;
        }
    }

    let mut accumulators = [0.0f64; SIMHASH_BIT_WIDTH as usize];
    for (feature, weight) in &features {
        let hash = hash64(feature);
        for (bit, accumulator) in accumulators.iter_mut().enumerate() {
            if hash & (1u64 << bit) != 0 {
                *accumulator += weight;
            } else {
                *accumulator -= weight;
            }
        }
    }

    // Note the consequence of `>=`: a text with no features yields all-zero
    // accumulators and therefore an all-ones fingerprint, not zero. That is what
    // the Python does, and it keeps featureless texts colliding with each other.
    let mut fingerprint = 0u64;
    for (bit, accumulator) in accumulators.iter().enumerate() {
        if *accumulator >= 0.0 {
            fingerprint |= 1u64 << bit;
        }
    }
    fingerprint
}

/// Number of differing bits between two fingerprints.
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://\S+").expect("url pattern is a fixed constant"));
static EMAIL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\S+@\S+\b").expect("email pattern is a fixed constant"));
static WHITESPACE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("whitespace pattern is a fixed constant"));
static WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\w+").expect("word pattern is a fixed constant"));

/// Canonicalises text so that cosmetic differences stop producing distinct
/// fingerprints.
///
/// URLs and e-mail addresses become placeholders because two otherwise identical
/// answers that quote different links are the same answer for our purposes.
///
/// Deviations, both forced by the standard library:
/// - Python casefolds; Rust has no `casefold`, so `str::to_lowercase` is used.
///   It is the closest stable equivalent and differs only for a handful of
///   scripts (German `ß` folds to `ss` but lowercases to `ß`, Cherokee, and a
///   few Greek finals), none of which change the outcome for the Latin-script
///   text this pipeline sees.
/// - Python drops characters with a non-zero canonical combining class
///   (`unicodedata.combining`); this uses `General_Category=Mark`, which is the
///   property `unicode-normalization` exposes. The two sets agree on every mark
///   NFKD actually produces from precomposed Latin, Greek, and Cyrillic.
pub fn normalize(text: &str) -> String {
    let text = URL_RE.replace_all(text, " <URL> ");
    let text = EMAIL_RE.replace_all(&text, " <EMAIL> ");
    let text: String = text.nfkc().collect::<String>().to_lowercase();
    let text: String = text.nfkd().filter(|c| !is_combining_mark(*c)).collect();
    WHITESPACE_RE.replace_all(&text, " ").trim().to_string()
}

/// Weighted shingle counts for one field.
fn extract_features(text: &str, options: DedupeOptions) -> BTreeMap<String, f64> {
    let normalized = normalize(text);
    if uses_char_mode(&normalized) {
        char_features(&normalized, options.char_ngram)
    } else {
        word_features(&normalized, options.word_ngram)
    }
}

/// Auto tokenizer selection: scripts that do not delimit words with spaces have
/// to be shingled by character, or every sentence collapses to one token.
fn uses_char_mode(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{3400}'..='\u{9FFF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{3040}'..='\u{30FF}'
            | '\u{AC00}'..='\u{D7AF}')
    })
}

fn word_features(text: &str, word_ngram: usize) -> BTreeMap<String, f64> {
    let tokens: Vec<&str> = WORD_RE
        .find_iter(text)
        .map(|token| token.as_str())
        .filter(|token| !STOPWORDS.contains(token))
        .collect();
    let mut counts: BTreeMap<String, f64> = BTreeMap::new();
    if word_ngram == 1 {
        for token in tokens {
            *counts.entry(token.to_string()).or_insert(0.0) += 1.0;
        }
        return counts;
    }
    for window in tokens.windows(word_ngram) {
        *counts.entry(window.join(" ")).or_insert(0.0) += 1.0;
    }
    counts
}

fn char_features(text: &str, char_ngram: usize) -> BTreeMap<String, f64> {
    let mut counts: BTreeMap<String, f64> = BTreeMap::new();
    if char_ngram == 1 {
        for c in text.chars().filter(|c| *c != ' ') {
            *counts.entry(c.to_string()).or_insert(0.0) += 1.0;
        }
        return counts;
    }
    // Spaces become a visible marker so that a shingle straddling a word break
    // stays distinguishable from the same letters run together. `normalize` has
    // already collapsed whitespace runs to a single space.
    let marked: Vec<char> = text.chars().map(|c| if c == ' ' { '␠' } else { c }).collect();
    for window in marked.windows(char_ngram) {
        *counts.entry(window.iter().collect::<String>()).or_insert(0.0) += 1.0;
    }
    counts
}

/// Stable 64-bit feature hash: BLAKE2b with an 8-byte digest, read big-endian.
///
/// The digest length is a BLAKE2b parameter-block field rather than a
/// truncation, so `Blake2bVar::new(8)` reproduces `hashlib.blake2b(digest_size=8)`
/// byte for byte.
fn hash64(value: &str) -> u64 {
    let mut hasher =
        Blake2bVar::new(BLAKE2B_DIGEST_SIZE).expect("8 is a valid blake2b digest size");
    hasher.update(value.as_bytes());
    let mut digest = [0u8; BLAKE2B_DIGEST_SIZE];
    hasher
        .finalize_variable(&mut digest)
        .expect("digest buffer is exactly the configured size");
    u64::from_be_bytes(digest)
}
