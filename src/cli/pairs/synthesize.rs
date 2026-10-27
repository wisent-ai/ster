//! `ster pairs synthesize`: a contrastive pair set written by a local model
//! or a hosted route. Which side declined its role is judged by the same
//! generator as the pairs are written, so no refusal score is asked for.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Args, ValueEnum};
use serde_json::json;
use ster::{
    ChatChoice, DeviceChoice, GenerationOptions, Precision, Runtime, brama,
    pairs::{self, SynthesisOptions, quality::dedupe::DedupeOptions},
};

/// Where the pair text comes from.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub(super) enum GeneratorKind {
    /// A model this machine loads.
    Local,
    /// A route Brama serves.
    Brama,
}

#[derive(Debug, Args)]
pub(super) struct SynthesizeArgs {
    /// Where the pair text comes from: local or brama. Steering always
    /// needs a local model; writing pairs does not, so this route may be
    /// hosted. Neither is assumed.
    #[arg(long, value_enum)]
    generator: GeneratorKind,
    /// Route the Brama generator writes with: a Brama alias, a canonical
    /// provider/model route, or a selector. Wisent's own served model is
    /// wisent-backend/chat/primary. Required with --generator brama.
    #[arg(long)]
    generator_model: Option<String>,
    /// Hugging Face model id or local model directory.
    #[arg(long)]
    model: Option<String>,
    /// Immutable Hugging Face revision; defaults to main.
    #[arg(long)]
    revision: Option<String>,
    /// Runtime device: cpu, metal, or cuda; an unknown one is a usage error.
    #[arg(long, default_value = "cpu", value_parser = DeviceChoice::parse)]
    device: DeviceChoice,
    /// Dtype the local generator's base weights are mapped at: f32, f16,
    /// or bf16. Ignored by --generator brama, which loads no weights.
    /// bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    /// auto asks the local generator through the model's own chat
    /// template when it publishes one, off asks it as raw text. An
    /// instruct checkpoint spoken to without its markers answers with
    /// meta-instructional debris — "Step 3: Make sure your emojis are
    /// visually appealing" — rather than the pair text that was asked
    /// for. Ignored by --generator brama, which is already a chat API.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// One-sentence description of the trait the positive side shows.
    #[arg(long = "trait")]
    trait_description: String,
    /// Number of pairs to keep after refusal and duplicate rejection; above zero.
    #[arg(long)]
    count: NonZeroUsize,
    /// Pair-set JSON the generated pairs are written to.
    #[arg(long)]
    output: PathBuf,
    /// Artifact label; unnamed, the trait description labels the set.
    #[arg(long)]
    trait_name: Option<String>,
    /// Skip the opposite-trait generation step and use this text.
    #[arg(long)]
    opposite: Option<String>,
    /// Attempt budget is count times this multiplier; Ster assumes none.
    /// Above zero; clap refuses a zero by name.
    #[arg(long)]
    retry_multiplier: NonZeroUsize,
    /// SimHash Hamming distance below which two pairs count as near-duplicates.
    #[arg(long)]
    dedupe_bits: u32,
    /// Banded-LSH band count, a divisor of 64; more bands catch more near-duplicates.
    #[arg(long)]
    dedupe_bands: u32,
    /// Token budget of one generated side.
    #[arg(long)]
    max_new_tokens: usize,
    /// Must exceed zero; argmax generation would repeat one pair.
    #[arg(long)]
    temperature: f64,
    #[arg(long)]
    top_p: f64,
    #[arg(long)]
    seed: u64,
}

pub(super) fn run(args: SynthesizeArgs) -> Result<()> {
    let options = SynthesisOptions {
        trait_description: args.trait_description,
        trait_name: args.trait_name,
        opposite: args.opposite,
        count: args.count,
        retry_multiplier: args.retry_multiplier,
        dedupe: DedupeOptions::new(args.dedupe_bits, args.dedupe_bands),
        generation: GenerationOptions {
            strength: None,
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_p: Some(args.top_p),
            seed: args.seed,
        },
    };
    // The runtime or the gateway is built inside the arm that uses it:
    // `--generator brama` must not load weights or touch a device, and
    // `--generator local` must not read the gateway's environment.
    let (pair_set, report) = match args.generator {
        GeneratorKind::Local => {
            let Some(model) = args.model else {
                bail!("pairs synthesize with --generator local requires --model");
            };
            let mut runtime = Runtime::load_at(
                &model,
                args.revision.as_deref(),
                args.device,
                args.precision,
            )?;
            // Synthesis is the first step of the funnel and everything
            // downstream inherits what it writes. Addressed without its
            // markers, an instruct checkpoint answers a pair request with
            // instructions about answering pair requests.
            runtime.set_chat_template(args.chat_template)?;
            pairs::synthesize(pairs::Generator::Local(&runtime), &options)?
        }
        GeneratorKind::Brama => {
            let Some(route) = args.generator_model else {
                bail!("pairs synthesize with --generator brama requires --generator-model");
            };
            let gateway = brama::Gateway::from_env(&route)?;
            pairs::synthesize(pairs::Generator::Gateway(&gateway), &options)?
        }
    };
    pair_set.save(&args.output)?;
    super::super::answer(&json!({
        "path": args.output.display().to_string(),
        "report": report,
    }))
}
