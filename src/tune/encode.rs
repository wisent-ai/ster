//! Tokenizing what a run trains or scores on, up front: examples and the
//! two sides of each preference pair, skipping what does not fit the
//! sequence limit and refusing a pair side the tokenizer leaves nothing to
//! predict in. An example the tokenizer leaves no prompt or completion
//! token of is refused by `Runtime::encode_example` itself.

use anyhow::Result;

use crate::failures::{
    amend, refuse,
    tune::run::{SIDE_UNPREDICTABLE, nothing_fits},
};
use crate::{artifact::PairSet, runtime::Runtime, workflow};

use super::examples::ExampleSet;

/// One example, tokenized: its index in the set, its ids, and where its
/// completion starts.
pub(crate) type EncodedExample = (usize, Vec<u32>, usize);

/// Tokenizes every example, dropping the ones longer than `limit` tokens.
/// `purpose` is what the run does with them, as a refusal of a set nothing
/// of which fits says it.
pub(crate) fn encode_examples(
    runtime: &Runtime,
    examples: &ExampleSet,
    limit: usize,
    purpose: &str,
) -> Result<Vec<EncodedExample>> {
    let mut encoded = Vec::with_capacity(examples.examples.len());
    let mut lengths = Vec::with_capacity(examples.examples.len());
    for (index, example) in examples.examples.iter().enumerate() {
        let (ids, boundary) = runtime
            .encode_example(
                example.system.as_deref(),
                &example.prompt,
                &example.completion,
            )
            .map_err(|error| {
                amend(error, format!("it was example {index} of {}", examples.examples.len()))
            })?;
        lengths.push(ids.len());
        if ids.len() > limit {
            workflow::progress(format!(
                "skipping example {index}: {} tokens exceed the {limit} token limit",
                ids.len()
            ));
            continue;
        }
        encoded.push((index, ids, boundary));
    }
    if encoded.is_empty() {
        return Err(nothing_fits("example", purpose, lengths, limit));
    }
    Ok(encoded)
}

/// One preference pair, tokenized.
pub(crate) struct EncodedPair {
    /// The pair's index in the original set, so a refusal names the entry the
    /// operator wrote rather than a position in a filtered list.
    pub index: usize,
    pub chosen: Vec<u32>,
    pub rejected: Vec<u32>,
}

/// Tokenizes both sides of every pair, dropping the ones that do not fit.
///
/// Over-long pairs are skipped rather than truncated for the reason supervised
/// fine-tuning skips over-long examples: a cut sequence is a different
/// sequence, and preferring a truncation of the chosen side over a truncation
/// of the rejected side is not the preference the operator stated. Both sides
/// go or neither does — half a pair states no preference at all.
///
/// Each side goes through `Runtime::encode_response` rather than the raw
/// encoder, so a run that asked for the model's chat template compares two
/// assistant turns wearing the markers inference will put around them, and a
/// run that did not gets byte-identical ids to every release before this one.
///
/// Tokenizing up front rather than per epoch is what makes the skip report
/// complete before the first gradient is taken.
pub(crate) fn encode_pairs(
    runtime: &Runtime,
    pairs: &PairSet,
    limit: usize,
    purpose: &str,
) -> Result<Vec<EncodedPair>> {
    let mut encoded = Vec::with_capacity(pairs.pairs.len());
    let mut lengths = Vec::with_capacity(pairs.pairs.len());
    for (index, pair) in pairs.pairs.iter().enumerate() {
        let total = pairs.pairs.len();
        let chosen = runtime.encode_response(&pair.positive).map_err(|error| {
            amend(error, format!("it was the chosen side of pair {index} of {total}"))
        })?;
        let rejected = runtime.encode_response(&pair.negative).map_err(|error| {
            amend(error, format!("it was the rejected side of pair {index} of {total}"))
        })?;
        for (side, text, ids) in [
            ("chosen", &pair.positive, &chosen),
            ("rejected", &pair.negative, &rejected),
        ] {
            if ids.len() < 2 {
                return Err(refuse(
                    &SIDE_UNPREDICTABLE,
                    &[
                        ("index", &index.to_string()),
                        ("side", side),
                        ("text", &format!("{text:?}")),
                        ("tokens", &format!("{ids:?}")),
                    ],
                ));
            }
        }
        let longest = chosen.len().max(rejected.len());
        lengths.push(longest);
        if longest > limit {
            workflow::progress(format!(
                "skipping pair {index}: {longest} tokens exceed the {limit} token limit"
            ));
            continue;
        }
        encoded.push(EncodedPair {
            index,
            chosen,
            rejected,
        });
    }
    if encoded.is_empty() {
        return Err(nothing_fits("pair", purpose, lengths, limit));
    }
    Ok(encoded)
}

/// How a pair set names itself in a refusal when it carries no trait name.
pub(crate) fn pair_set_label(pairs: &PairSet) -> String {
    match pairs.trait_name.trim() {
        "" => "(unnamed)".to_owned(),
        name => name.to_owned(),
    }
}
