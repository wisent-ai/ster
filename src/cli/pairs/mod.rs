//! `ster pairs`: authoring and auditing contrastive pair sets from the
//! terminal. Each arm does the work and prints one pretty JSON document.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Subcommand;
use serde_json::json;
use ster::{
    ContrastivePair, PairSet,
    failures::{pairs::author, refuse},
    pairs::{self, InspectOptions, quality::dedupe::DedupeOptions},
};

use super::resolve_pairs;

mod edit;
mod import;
mod merge;
mod synthesize;

#[derive(Debug, Subcommand)]
pub(super) enum PairsCommand {
    /// Report duplicates, length balance, and diversity for a set.
    Inspect {
        /// Pair-set JSON. Omit it to inspect the active imported set.
        #[arg(long)]
        pairs: Option<PathBuf>,
        /// SimHash Hamming distance below which two pairs count as near-duplicates; Ster assumes none.
        #[arg(long)]
        dedupe_bits: u32,
        /// Banded-LSH band count, a divisor of 64; more bands catch more near-duplicates.
        #[arg(long)]
        dedupe_bands: u32,
        /// Longer-over-shorter character ratio above which a pair is
        /// counted as unbalanced; Ster assumes none.
        #[arg(long)]
        unbalanced_ratio: f64,
    },
    /// Append one pair, creating the set when the file does not exist.
    Add {
        #[arg(long)]
        pairs: PathBuf,
        #[arg(long)]
        positive: String,
        #[arg(long)]
        negative: String,
        /// Set or replace the trait name on the file.
        #[arg(long = "trait")]
        trait_name: Option<String>,
    },
    /// Remove one pair by its zero-based index.
    Remove {
        #[arg(long)]
        pairs: PathBuf,
        #[arg(long)]
        index: usize,
    },
    /// Change one pair's sides, or the set's trait name, by the index remove takes.
    Edit {
        #[arg(long)]
        pairs: PathBuf,
        #[arg(long)]
        index: usize,
        #[arg(long)]
        positive: Option<String>,
        #[arg(long)]
        negative: Option<String>,
        #[arg(long = "trait")]
        trait_name: Option<String>,
    },
    /// Generate a contrastive pair set locally or with a hosted model.
    Synthesize(synthesize::SynthesizeArgs),
    /// Write a pair set from a TruthfulQA, Do-Not-Answer or LiveCodeBench export.
    Import(import::ImportArgs),
    /// Write one pair set out of several, every pair of each in order, for
    /// a direction fitted across all of them.
    Merge(merge::MergeArgs),
}

/// The `ster pairs` arms. Each one does the work and prints one pretty JSON
/// document, exactly like the arms above; they live here rather than inline
/// only to keep the top-level match readable.
pub(super) fn run(command: PairsCommand) -> Result<()> {
    match command {
        PairsCommand::Inspect {
            pairs,
            dedupe_bits,
            dedupe_bands,
            unbalanced_ratio,
        } => {
            // The settings are the caller's and are checked before the
            // workspace or the set is read.
            let options = InspectOptions {
                dedupe: DedupeOptions::new(dedupe_bits, dedupe_bands),
                unbalanced_ratio,
            };
            options.validate()?;
            let file = resolve_pairs(pairs)?;
            let pair_set = PairSet::load(&file)?;
            let report = pairs::inspect(&pair_set, &options)?;
            super::answer(&report)?;
        }
        PairsCommand::Import(args) => import::run(args)?,
        PairsCommand::Merge(args) => merge::run(args)?,
        PairsCommand::Add {
            pairs: file,
            positive,
            negative,
            trait_name,
        } => {
            side("positive", &positive, &file)?;
            side("negative", &negative, &file)?;
            // `PairSet::load` refuses a set with no pairs, so the first `add`
            // to a path that does not exist yet builds the set in memory
            // rather than loading one.
            let mut pair_set = if file.exists() {
                PairSet::load(&file)?
            } else {
                PairSet {
                    trait_name: String::new(),
                    pairs: Vec::new(),
                }
            };
            if let Some(name) = trait_name {
                pair_set.trait_name = name;
            }
            let index = pair_set.pairs.len();
            pair_set.pairs.push(ContrastivePair { positive, negative });
            pair_set.save(&file)?;
            super::answer(&json!({
                "path": file.display().to_string(),
                "pair_count": pair_set.pairs.len(),
                "added": {"index": index},
            }))?;
        }
        PairsCommand::Remove { pairs: file, index } => {
            let mut pair_set = PairSet::load(&file)?;
            outside(index, pair_set.pairs.len(), &file)?;
            let removed = pair_set.pairs.remove(index);
            // Removing the last pair leaves a set no loader would accept, so
            // `save` refuses and the file on disk is left as it was.
            pair_set.save(&file)?;
            super::answer(&json!({
                "path": file.display().to_string(),
                "pair_count": pair_set.pairs.len(),
                "removed": {
                    "index": index,
                    "positive": removed.positive,
                    "negative": removed.negative,
                },
            }))?;
        }
        PairsCommand::Edit {
            pairs: file,
            index,
            positive,
            negative,
            trait_name,
        } => edit::run(&file, index, positive, negative, trait_name)?,
        PairsCommand::Synthesize(args) => synthesize::run(args)?,
    }
    Ok(())
}

/// The refusal a side given as empty text meets, before any set is read:
/// every pair needs text on both sides.
fn side(name: &str, value: &str, file: &Path) -> Result<()> {
    if value.trim().is_empty() {
        return Err(refuse(
            &author::SIDE_EMPTY,
            &[
                ("side", name),
                ("value", value),
                ("path", &file.display().to_string()),
            ],
        ));
    }
    Ok(())
}

/// The refusal an index past the end of a set of `count` pairs meets.
/// `count` is at least one: `PairSet::load` refuses a set without pairs.
fn outside(index: usize, count: usize, file: &Path) -> Result<()> {
    if index < count {
        return Ok(());
    }
    Err(refuse(
        &author::INDEX_OUTSIDE,
        &[
            ("index", &index.to_string()),
            ("count", &count.to_string()),
            ("last", &count.saturating_sub(1).to_string()),
            ("path", &file.display().to_string()),
        ],
    ))
}
