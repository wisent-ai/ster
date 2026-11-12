//! The decision pipeline: where labelled decisions come from, how a set is
//! split, and how a model is measured on one.
//!
//! Training a decision model is four steps, each a command and each a
//! function here: get labelled decisions (`fetch` from a Hugging Face
//! classification dataset, `import` from a JSONL file, or `synthesize`
//! through Stado's inference alternatives), `split` them into a training and
//! a held-out set, train (`ster tune decide`), and `benchmark` the base model
//! and the adapter on
//! the held-out set. Every step reads and writes the one labelled-decision
//! document `ster calibrate` already reads, so nothing is converted twice.

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use super::examples::ExampleSet;
use crate::failures::{decide::pipeline as causes, refuse};

mod benchmark;
mod fetch;
mod import;
mod synthesize;

pub use benchmark::{Benchmark, BenchmarkOptions, Latency, TypeMetrics, benchmark};
pub use fetch::{FetchOptions, FetchReport, fetch};
pub use import::{ImportReport, import_jsonl};
pub use synthesize::{Schema, SynthesizeOptions, SynthesizeReport, synthesize};

/// Refuses the one question a `source` ("fetched", "imported") set asks
/// when its id or its instructions are blank; checked before any row is read.
pub fn check_question(source: &str, id: &str, instructions: &str) -> Result<()> {
    if id.trim().is_empty() {
        return Err(refuse(
            &causes::QUESTION_ID_BLANK,
            &[("source", source), ("value", &format!("{id:?}"))],
        ));
    }
    if instructions.trim().is_empty() {
        return Err(refuse(
            &causes::INSTRUCTIONS_BLANK,
            &[("source", source), ("value", &format!("{instructions:?}"))],
        ));
    }
    Ok(())
}

/// Refuses a held-out fraction that parts no example from the rest; checked
/// before the set is read.
pub fn check_holdout(holdout: f64) -> Result<()> {
    if holdout > 0.0 && holdout < 1.0 {
        return Ok(());
    }
    Err(refuse(&causes::HOLDOUT_OUTSIDE, &[("holdout", &holdout.to_string())]))
}

/// Splits `set` into a training and a held-out set, `holdout` being the
/// fraction held out, after a shuffle seeded by `seed`.
///
/// Examples are the unit, never questions: every question of one example
/// lands on the same side, so a held-out state is one the model never saw
/// under any question. Both sides are non-empty when the set has two or more
/// examples; a fraction that would empty one side is clamped to one example.
pub fn split(set: &ExampleSet, holdout: f64, seed: u64) -> (ExampleSet, ExampleSet) {
    let mut order: Vec<usize> = (0..set.examples.len()).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));
    let total = order.len();
    let held = ((total as f64 * holdout).round() as usize)
        .clamp(usize::from(total > 1), total.saturating_sub(1));
    let (held_out, training) = order.split_at(held);
    let pick = |indices: &[usize]| ExampleSet {
        examples: {
            let mut sorted = indices.to_vec();
            sorted.sort_unstable();
            sorted
                .into_iter()
                .map(|index| set.examples[index].clone())
                .collect()
        },
    };
    (pick(training), pick(held_out))
}
