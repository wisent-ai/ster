//! `ster pairs edit`: the counterpart of `add` for changing one pair in
//! place, by the zero-based index `remove` takes. Only the sides given change;
//! an edit that changes nothing is refused, and so is an index past the end.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::json;
use ster::PairSet;

pub(super) fn run(file: &Path, index: usize, positive: Option<String>, negative: Option<String>, trait_name: Option<String>) -> Result<()> {
    if positive.is_none() && negative.is_none() && trait_name.is_none() {
        bail!("pairs edit changes nothing without --positive, --negative or --trait");
    }
    let mut pair_set = PairSet::load(file)?;
    let count = pair_set.pairs.len();
    let Some(pair) = pair_set.pairs.get_mut(index) else {
        bail!("pair index {index} is outside the set: {} holds {count} pair(s), indexed from 0", file.display());
    };
    if let Some(positive) = positive {
        pair.positive = positive;
    }
    if let Some(negative) = negative {
        pair.negative = negative;
    }
    let edited = json!({"index": index, "positive": pair.positive, "negative": pair.negative});
    if let Some(name) = trait_name {
        pair_set.trait_name = name;
    }
    pair_set.save(file)?;
    super::super::answer(&json!({
        "path": file.display().to_string(),
        "pair_count": pair_set.pairs.len(),
        "edited": edited,
    }))
}
