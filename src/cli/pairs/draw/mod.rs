//! `ster pairs split` and `ster pairs sample`: pair sets drawn out of one by
//! a shuffle the caller's seed fixes, for a protocol run over several seeds.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use ster::{PairSet, pairs};

#[derive(Debug, Args)]
pub(in crate::cli) struct SplitArgs {
    /// Pair-set JSON to split.
    #[arg(long)]
    pairs: PathBuf,
    /// Fraction of the pairs held out for --test, above zero and below one;
    /// Ster assumes none.
    #[arg(long)]
    holdout: f64,
    /// The seed of the shuffle the held-out pairs are drawn from; Ster
    /// assumes none.
    #[arg(long)]
    seed: u64,
    /// Pair-set JSON the training pairs are written to; refused when a file
    /// is there.
    #[arg(long)]
    train: PathBuf,
    /// Pair-set JSON the held-out pairs are written to; refused when a file
    /// is there.
    #[arg(long)]
    test: PathBuf,
}

#[derive(Debug, Args)]
pub(in crate::cli) struct SampleArgs {
    /// Pair-set JSON to draw from.
    #[arg(long)]
    pairs: PathBuf,
    /// Pairs to draw, from 1 to the pairs the set holds; Ster assumes none.
    #[arg(long)]
    count: usize,
    /// The seed of the shuffle the pairs are drawn from; Ster assumes none.
    #[arg(long)]
    seed: u64,
    /// Pair-set JSON the drawn pairs are written to; refused when a file is
    /// there.
    #[arg(long)]
    output: PathBuf,
}

pub(in crate::cli) fn split(args: SplitArgs) -> Result<()> {
    // Where the sides go is checked before the set is read, so nothing is
    // read for a split that could not be written.
    pairs::check_sides(&args.train, &args.test)?;
    pairs::check_output("train", &args.train)?;
    pairs::check_output("test", &args.test)?;
    let set = PairSet::load(&args.pairs)?;
    let (train, test) = pairs::split(&set, args.holdout, args.seed)?;
    train.save(&args.train)?;
    test.save(&args.test)?;
    super::super::answer(&pairs::SplitReport {
        trait_name: set.trait_name.clone(),
        pair_count: set.pairs.len(),
        holdout: args.holdout,
        seed: args.seed,
        train: pairs::Drawn {
            path: args.train.display().to_string(),
            pairs: train.pairs.len(),
        },
        test: pairs::Drawn {
            path: args.test.display().to_string(),
            pairs: test.pairs.len(),
        },
    })
}

pub(in crate::cli) fn sample(args: SampleArgs) -> Result<()> {
    pairs::check_output("output", &args.output)?;
    let set = PairSet::load(&args.pairs)?;
    let drawn = pairs::sample(&set, args.count, args.seed, &args.pairs)?;
    drawn.save(&args.output)?;
    super::super::answer(&pairs::SampleReport {
        trait_name: set.trait_name.clone(),
        pair_count: set.pairs.len(),
        seed: args.seed,
        sample: pairs::Drawn {
            path: args.output.display().to_string(),
            pairs: drawn.pairs.len(),
        },
    })
}
