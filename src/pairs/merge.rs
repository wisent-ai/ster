//! One pair set out of several: every pair of every source, in the order
//! the sources were named, under one trait name the caller gives. A
//! direction fitted on the result is one direction for all of them — the
//! unified direction wisent's train-unified-goodness and geometry-search
//! fitted across benchmarks.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use serde::Serialize;

use crate::artifact::{ContrastivePair, PairSet};

/// How many pairs each source gave.
#[derive(Debug, Clone, Serialize)]
pub struct MergedSource {
    pub path: String,
    pub trait_name: String,
    pub pairs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MergeReport {
    pub trait_name: String,
    pub pair_count: usize,
    pub sources: Vec<MergedSource>,
}

/// Every pair of every set at `sources`, in order, under `trait_name`. Two
/// sets at least: one is already itself. Each source is loaded with the
/// same validation every pair set passes.
pub fn merge(sources: &[PathBuf], trait_name: &str) -> Result<(PairSet, MergeReport)> {
    let [_, _, ..] = sources else {
        bail!(
            "pairs merge needs at least two pair sets and got {}",
            sources.len()
        );
    };
    if trait_name.trim().is_empty() {
        bail!("pairs merge needs the trait name the merged set is fitted for");
    }
    let mut pairs: Vec<ContrastivePair> = Vec::new();
    let mut merged = Vec::with_capacity(sources.len());
    for path in sources {
        let set = PairSet::load(Path::new(path))?;
        merged.push(MergedSource {
            path: path.display().to_string(),
            trait_name: set.trait_name.clone(),
            pairs: set.pairs.len(),
        });
        pairs.extend(set.pairs);
    }
    let set = PairSet {
        trait_name: trait_name.to_owned(),
        pairs,
    };
    let report = MergeReport {
        trait_name: set.trait_name.clone(),
        pair_count: set.pairs.len(),
        sources: merged,
    };
    Ok((set, report))
}
