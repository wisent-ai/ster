//! Pair sets drawn out of one by a shuffle the caller's seed fixes: a set
//! parted into a training set and a held-out set, and a sample of a set. The
//! shuffle is the one every seeded order in Ster is drawn with
//! (`StdRng::seed_from_u64`, as `ster decisions split` and `ster pairs
//! import --count` draw), so one set and one seed give the same pairs on every
//! run, and a protocol of several seeds is repeatable pair for pair. Each
//! drawn set keeps the order the source set lists its pairs in.

use std::path::Path;

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use crate::artifact::PairSet;
use crate::failures::{file, pairs::author, refuse};
use crate::workflow::split_holdout;

/// One drawn set as a report names it.
#[derive(Debug, Clone, Serialize)]
pub struct Drawn {
    pub path: String,
    pub pairs: usize,
}

/// What `ster pairs split` wrote.
#[derive(Debug, Clone, Serialize)]
pub struct SplitReport {
    pub trait_name: String,
    /// Pairs in the set split.
    pub pair_count: usize,
    pub holdout: f64,
    pub seed: u64,
    pub train: Drawn,
    pub test: Drawn,
}

/// What `ster pairs sample` wrote.
#[derive(Debug, Clone, Serialize)]
pub struct SampleReport {
    pub trait_name: String,
    /// Pairs in the set sampled.
    pub pair_count: usize,
    pub seed: u64,
    pub sample: Drawn,
}

/// Refuses an output where a file already is, named by `flag`, before any
/// set is read: a drawn set never replaces one. Only a path the system says
/// holds nothing is free; one it cannot look at (a directory this account
/// may not search, a path running through a file) is refused for that.
pub fn check_output(flag: &str, path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            let role = format!("the --{flag} pair set");
            return Err(file::write::refusal("checking", &role, path, "nothing was written", &error));
        }
        Ok(_) => {}
    }
    Err(refuse(
        &author::OUTPUT_EXISTS,
        &[
            ("flag", flag),
            ("path", &path.display().to_string()),
            ("state", &file::state(path)),
        ],
    ))
}

/// Refuses a split whose two sides would be written to one file.
pub fn check_sides(train: &Path, test: &Path) -> Result<()> {
    if train != test {
        return Ok(());
    }
    Err(refuse(
        &author::OUTPUTS_SAME,
        &[("path", &train.display().to_string())],
    ))
}

/// `set` parted into a training set and a held-out set: a shuffle seeded by
/// `seed`, whose first `holdout` fraction, rounded to whole pairs, is held
/// out. A fraction outside zero and one, or one that leaves a side without a
/// pair, is refused as `ster vector optimize` refuses it.
pub fn split(set: &PairSet, holdout: f64, seed: u64) -> Result<(PairSet, PairSet)> {
    let parted = split_holdout(set, holdout)?;
    let order = shuffled(set, seed);
    let (held, kept) = order.split_at(parted.holdout_pairs);
    Ok((picked(set, kept), picked(set, held)))
}

/// `count` pairs of `set`, the first `count` of a shuffle seeded by `seed`;
/// `path` names the set in the refusal of a count it cannot give.
pub fn sample(set: &PairSet, count: usize, seed: u64, path: &Path) -> Result<PairSet> {
    let total = set.pairs.len();
    if count == 0 || count > total {
        return Err(refuse(
            &author::COUNT_OUTSIDE,
            &[
                ("count", &count.to_string()),
                ("total", &total.to_string()),
                ("path", &path.display().to_string()),
            ],
        ));
    }
    let order = shuffled(set, seed);
    let (drawn, _) = order.split_at(count);
    Ok(picked(set, drawn))
}

/// The positions of `set`'s pairs in the order a shuffle seeded by `seed`
/// puts them.
fn shuffled(set: &PairSet, seed: u64) -> Vec<usize> {
    let pairs = set.pairs.iter().enumerate();
    let mut order: Vec<usize> = pairs.map(|(index, _)| index).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));
    order
}

/// The pairs of `set` at `positions`, in the order `set` lists them, under
/// its trait name.
fn picked(set: &PairSet, positions: &[usize]) -> PairSet {
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    PairSet {
        trait_name: set.trait_name.clone(),
        pairs: sorted
            .into_iter()
            .map(|index| set.pairs[index].clone())
            .collect(),
    }
}
