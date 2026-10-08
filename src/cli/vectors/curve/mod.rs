//! `ster vector curve`: how many pairs a direction needs, read off one
//! method fitted on growing first parts of a pair set and scored on the same
//! held-out pairs.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
use ster::{
    ChatChoice, PairSet, Precision, TrainingMethod,
    workflow::{self, parse_layers},
};

use super::super::{ModelArgs, resolve_pairs};

/// `ster vector curve`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct CurveArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Pair-set JSON. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    /// Comma-separated layers, half-open ranges such as 8..16, or all.
    #[arg(long, default_value = "all")]
    layers: String,
    /// Direction training method: caa, pca, or logistic.
    #[arg(long, value_parser = TrainingMethod::parse)]
    method: TrainingMethod,
    /// Fraction of the pairs held out to score every size on, above zero and
    /// below one; Ster assumes none.
    #[arg(long)]
    holdout: f64,
    /// Comma-separated numbers of pairs to fit on, each at most the pairs
    /// left after the holdout; Ster assumes none.
    #[arg(long, value_delimiter = ',', required = true)]
    sizes: Vec<NonZeroUsize>,
    /// auto reads every pair through the model's own chat template when it
    /// publishes one, off reads it as raw text.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

pub(in crate::cli) fn curve(args: CurveArgs) -> Result<()> {
    let CurveArgs {
        model,
        pairs,
        layers,
        method,
        holdout,
        sizes,
        chat_template,
        precision,
    } = args;
    let pairs = resolve_pairs(pairs)?;
    let pair_set = PairSet::load(&pairs)?;
    let mut runtime = model.load_at(precision)?;
    let chat = runtime.set_chat_template(chat_template)?;
    let layers = parse_layers(&layers, runtime.layer_count())?;
    let report = workflow::curve(&runtime, &pair_set, &layers, method, holdout, &sizes)?;
    let mut report = serde_json::to_value(report)?;
    chat.annotate(&mut report)?;
    super::super::answer(&report)
}
