//! Lexical and structural diversity metrics for a set of texts.
//!
//! A port of Wisent's `FastDiversity`. A pair set that passes deduplication can
//! still be monotonous — a hundred distinct sentences that all open "Certainly!
//! Here is" teach a direction more about that opening than about the trait.
//! These five numbers are the cheap summary an operator needs to decide whether
//! a synthesis run produced a usable set or a template with the nouns swapped.
//! Everything here is O(n^2) in the sample, which is why the sample is capped.

use std::collections::HashSet;

use rand::{SeedableRng, rngs::StdRng, seq::IndexedRandom};
use serde::Serialize;

/// Width of the SimHash fingerprint; the Python `SIMHASH_BIT_WIDTH`.
const SIMHASH_BIT_WIDTH: u32 = 64;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x100_0000_01B3;

/// Default cap on the number of texts drawn for the pairwise passes.
pub const DEFAULT_MAX_SAMPLE: usize = 256;

/// The five diversity numbers.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Scores {
    /// Distinct-1: unique unigrams over total unigrams, across every text.
    pub unique_unigrams: f64,
    /// Distinct-2: unique bigrams over total bigrams, across every text.
    pub unique_bigrams: f64,
    /// Mean pairwise Jaccard similarity of token sets. Lower is more diverse.
    pub avg_jaccard: f64,
    /// Mean pairwise SimHash Hamming distance. Higher is more diverse.
    pub mean_simhash_hamming: f64,
    /// Smallest pairwise SimHash Hamming distance; the closest near-collision.
    pub min_simhash_hamming: u32,
}

/// Computes the diversity summary for `texts`.
///
/// The distinct-N ratios cover every text; the pairwise passes are quadratic, so
/// they run over a sample of at most `max_sample` texts drawn without
/// replacement from a `seed`-derived generator, which keeps a report
/// reproducible for a given seed.
///
/// Deviation, forced. The Python samples with `numpy.random.default_rng(seed)`
/// (PCG64) and `Generator.choice`; Rust draws with `StdRng` and
/// `choose_multiple`. Both are seeded and both are uniform without replacement,
/// but the two generators are different algorithms, so the *particular* subset
/// differs from Python's for the same seed. Nothing downstream depends on which
/// subset is drawn, only that repeated runs agree with each other.
pub fn compute(texts: &[String], seed: u64, max_sample: usize) -> Scores {
    let unique_unigrams = distinct_n(texts, 1);
    let unique_bigrams = distinct_n(texts, 2);

    let sample: Vec<&String> = if texts.len() <= max_sample {
        texts.iter().collect()
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        texts.choose_multiple(&mut rng, max_sample).collect()
    };

    if sample.len() < 2 {
        return Scores {
            unique_unigrams,
            unique_bigrams,
            avg_jaccard: 0.0,
            mean_simhash_hamming: 0.0,
            min_simhash_hamming: SIMHASH_BIT_WIDTH,
        };
    }

    let token_sets: Vec<Vec<String>> = sample.iter().map(|text| tokenize(text.as_str())).collect();
    let fingerprints: Vec<u64> = token_sets.iter().map(|tokens| simhash64(tokens)).collect();

    let mut jaccard_total = 0.0f64;
    let mut distance_total = 0u64;
    let mut min_distance = SIMHASH_BIT_WIDTH;
    let mut comparisons = 0u64;
    for left in 0..sample.len() {
        for right in (left + 1)..sample.len() {
            jaccard_total += jaccard(&token_sets[left], &token_sets[right]);
            let distance = hamming(fingerprints[left], fingerprints[right]);
            distance_total += u64::from(distance);
            min_distance = min_distance.min(distance);
            comparisons += 1;
        }
    }

    // `sample.len() >= 2` guarantees at least one comparison.
    let comparisons = comparisons as f64;
    Scores {
        unique_unigrams,
        unique_bigrams,
        avg_jaccard: jaccard_total / comparisons,
        mean_simhash_hamming: distance_total as f64 / comparisons,
        min_simhash_hamming: min_distance,
    }
}

/// Unique n-grams over total n-grams, pooled across every text.
fn distinct_n(texts: &[String], n: usize) -> f64 {
    let mut total = 0usize;
    let mut unique: HashSet<Vec<String>> = HashSet::new();
    for text in texts {
        // `windows` yields nothing when the text is shorter than `n`, which is
        // the `max(0, len - n + 1)` guard the Python spells out.
        let tokens = tokenize(text);
        for window in tokens.windows(n) {
            unique.insert(window.to_vec());
            total += 1;
        }
    }
    if total == 0 { 0.0 } else { unique.len() as f64 / total as f64 }
}

/// Jaccard similarity of two token *sets*, built from token lists.
///
/// Two empty texts count as identical, matching the Python; an empty text
/// against a non-empty one counts as maximally different.
fn jaccard(left: &[String], right: &[String]) -> f64 {
    let left: HashSet<&String> = left.iter().collect();
    let right: HashSet<&String> = right.iter().collect();
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(&right).count();
    let union = left.union(&right).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

/// The Python `_TOKEN_RE`, `[A-Za-z0-9']+|[^\w\s]`, applied to lowercased text.
///
/// Deviation, deliberate: hand-rolled rather than compiled as a regex. The
/// pattern is two trivial character classes, this runs once per text per report,
/// and a scan avoids both a second regex compilation and an allocation per
/// match. Punctuation is emitted as its own single-character token, exactly as
/// the `[^\w\s]` alternative does.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() || c == '\'' {
            current.push(c);
            continue;
        }
        if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
        // `[^\w\s]`: anything that is neither a word character nor whitespace.
        // Non-ASCII letters and digits are word characters under Unicode `\w`,
        // so they fall through here and are dropped, just as in the Python —
        // the `[A-Za-z0-9']+` alternative never claimed them either.
        if !c.is_alphanumeric() && c != '_' && !c.is_whitespace() {
            tokens.push(c.to_string());
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// 64-bit FNV-1a hash.
///
/// Deviation, deliberate: this module hashes with FNV-1a while [`crate::dedupe`]
/// hashes with BLAKE2b. That is not an oversight on our side — the two Python
/// modules genuinely use two different hashes, and both are ported as written so
/// that neither module's numbers shift relative to the original.
fn hash64(value: &str) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// SimHash over raw tokens, unweighted.
///
/// Returns 0 for a text with no tokens. Note that this differs from
/// [`crate::dedupe::simhash64`], where an empty feature set yields all ones;
/// the two Python modules disagree here too and both are ported faithfully.
fn simhash64(tokens: &[String]) -> u64 {
    if tokens.is_empty() {
        return 0;
    }
    let mut accumulators = [0i64; SIMHASH_BIT_WIDTH as usize];
    for token in tokens {
        let hash = hash64(token);
        for (bit, accumulator) in accumulators.iter_mut().enumerate() {
            if (hash >> bit) & 1 == 1 {
                *accumulator += 1;
            } else {
                *accumulator -= 1;
            }
        }
    }
    let mut fingerprint = 0u64;
    for (bit, accumulator) in accumulators.iter().enumerate() {
        if *accumulator >= 0 {
            fingerprint |= 1u64 << bit;
        }
    }
    fingerprint
}

fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}
