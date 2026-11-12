//! `ster pairs edit`: the counterpart of `add` for changing one pair in
//! place, by the zero-based index `remove` takes. Only the sides given change;
//! an edit that changes nothing, or gives a side no text, is refused before
//! the set is read, and an index past the end once it is.

use std::path::Path;

use anyhow::Result;
use serde_json::json;
use ster::{
    PairSet,
    failures::{pairs::author, refuse},
};

use super::{outside, side};

pub(super) fn run(
    file: &Path,
    index: usize,
    positive: Option<String>,
    negative: Option<String>,
    trait_name: Option<String>,
) -> Result<()> {
    if positive.is_none() && negative.is_none() && trait_name.is_none() {
        return Err(refuse(
            &author::NOTHING_TO_CHANGE,
            &[
                ("index", &index.to_string()),
                ("path", &file.display().to_string()),
                ("state", &ster::failures::file::state(file)),
            ],
        ));
    }
    if let Some(text) = &positive {
        side("positive", text, file)?;
    }
    if let Some(text) = &negative {
        side("negative", text, file)?;
    }
    let mut pair_set = PairSet::load(file)?;
    outside(index, pair_set.pairs.len(), file)?;
    let pair = &mut pair_set.pairs[index];
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
