//! The decision pipeline: where labelled decisions come from, how a set is
//! split, and how a model is measured on one.
//!
//! Training a decision model is four steps, each a command and each a
//! function here: get labelled decisions (`fetch` from a Hugging Face
//! classification dataset, `import` from a JSONL file, or `synthesize`
//! through Brama), `split` them into a training and a held-out set, train
//! (`ster tune decide`), and `benchmark` the base model and the adapter on
//! the held-out set. Every step reads and writes the one labelled-decision
//! document `ster calibrate` already reads, so nothing is converted twice.

use rand::{seq::SliceRandom, rngs::StdRng, SeedableRng};

use super::request::ExampleSet;

mod benchmark;
mod fetch;
mod import;
mod synthesize;

pub use benchmark::{benchmark, Benchmark, BenchmarkOptions, Latency, TypeMetrics};
pub use fetch::{fetch, FetchOptions, FetchReport};
pub use import::{import_jsonl, ImportReport};
pub use synthesize::{synthesize, Schema, SynthesizeOptions, SynthesizeReport};

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
    let held = ((total as f64 * holdout).round() as usize).clamp(usize::from(total > 1), total.saturating_sub(1));
    let (held_out, training) = order.split_at(held);
    let pick = |indices: &[usize]| ExampleSet {
        examples: {
            let mut sorted = indices.to_vec();
            sorted.sort_unstable();
            sorted.into_iter().map(|index| set.examples[index].clone()).collect()
        },
    };
    (pick(training), pick(held_out))
}
