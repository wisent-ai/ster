//! `ster pairs`: authoring and auditing contrastive pair sets from the
//! terminal. Each arm does the work and prints one pretty JSON document.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Subcommand;
use serde_json::json;
use ster::{
    brama,
    pairs::{self, quality::dedupe::DedupeOptions, quality::diversity::DEFAULT_MAX_SAMPLE,
        InspectOptions, SynthesisOptions},
    ChatChoice, ContrastivePair, DeviceChoice, GenerationOptions, PairSet, Precision, Runtime,
};

use super::resolve_pairs;

#[derive(Debug, Subcommand)]
pub(super) enum PairsCommand {
    /// Report duplicates, refusals, length balance, and diversity for a set.
    Inspect {
        /// Pair-set JSON. Omit it to inspect the active imported set.
        #[arg(long)]
        pairs: Option<PathBuf>,
        /// SimHash Hamming distance below which two pairs count as near-duplicates.
        #[arg(long, default_value_t = 3)]
        dedupe_bits: u32,
        /// Banded-LSH band count; more bands catch more near-duplicates.
        #[arg(long, default_value_t = 8)]
        dedupe_bands: u32,
        /// Refusal score at or above which a side is flagged.
        #[arg(long, default_value_t = 0.5)]
        refusal_threshold: f32,
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
    /// Generate a contrastive pair set locally or with a hosted model.
    Synthesize {
        /// Where the pair text comes from: local or brama. Steering always
        /// needs a local model; writing pairs does not, so this route may be
        /// hosted.
        #[arg(long, default_value = "local")]
        generator: String,
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
        /// Runtime device: cpu, metal, or cuda.
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Dtype the local generator's base weights are mapped at: f32, f16,
        /// or bf16. Ignored by --generator brama, which loads no weights.
        /// bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
        /// auto asks the local generator through the model's own chat
        /// template when it publishes one, off asks it as raw text. An
        /// instruct checkpoint spoken to without its markers answers with
        /// meta-instructional debris — "Step 3: Make sure your emojis are
        /// visually appealing" — rather than the pair text that was asked
        /// for. Ignored by --generator brama, which is already a chat API.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// One-sentence description of the trait the positive side shows.
        #[arg(long = "trait")]
        trait_description: String,
        /// Number of pairs to keep after refusal and duplicate rejection.
        #[arg(long)]
        count: usize,
        /// Pair-set JSON the generated pairs are written to.
        #[arg(long)]
        output: PathBuf,
        /// Artifact label; defaults to the trait description, truncated.
        #[arg(long)]
        trait_name: Option<String>,
        /// Skip the opposite-trait generation step and use this text.
        #[arg(long)]
        opposite: Option<String>,
        /// Attempt budget is count times this multiplier.
        #[arg(long, default_value_t = 3)]
        retry_multiplier: usize,
        /// SimHash Hamming distance below which two pairs count as near-duplicates.
        #[arg(long, default_value_t = 3)]
        dedupe_bits: u32,
        /// Banded-LSH band count; more bands catch more near-duplicates.
        #[arg(long, default_value_t = 8)]
        dedupe_bands: u32,
        /// Refusal score at or above which a generated side is rejected.
        #[arg(long, default_value_t = 0.5)]
        refusal_threshold: f32,
        #[arg(long, default_value_t = 96)]
        max_new_tokens: usize,
        /// Must exceed zero; argmax generation would repeat one pair.
        #[arg(long, default_value_t = 0.9)]
        temperature: f64,
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}


/// The `ster pairs` arms. Each one does the work and prints one pretty JSON
/// document, exactly like the arms above; they live here rather than inline
/// only to keep the top-level match readable.
pub(super) fn run(command: PairsCommand) -> Result<()> {
    match command {
        PairsCommand::Inspect { pairs, dedupe_bits, dedupe_bands, refusal_threshold } => {
            let file = resolve_pairs(pairs)?;
            let pair_set = PairSet::load(&file)?;
            let options = InspectOptions {
                dedupe: DedupeOptions {
                    threshold_bits: dedupe_bits,
                    num_bands: dedupe_bands,
                    ..DedupeOptions::default()
                },
                refusal_threshold,
                ..InspectOptions::default()
            };
            let report = pairs::inspect(&pair_set, &options)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        PairsCommand::Add { pairs: file, positive, negative, trait_name } => {
            // `PairSet::load` refuses a set with no pairs, so the first `add`
            // to a path that does not exist yet builds the set in memory
            // rather than loading one.
            let mut pair_set = if file.exists() {
                PairSet::load(&file)?
            } else {
                PairSet { trait_name: String::new(), pairs: Vec::new() }
            };
            if let Some(name) = trait_name {
                pair_set.trait_name = name;
            }
            pair_set.pairs.push(ContrastivePair { positive, negative });
            let index = pair_set.pairs.len() - 1;
            pair_set.save(&file)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path": file.display().to_string(),
                    "pair_count": pair_set.pairs.len(),
                    "added": {"index": index},
                }))?
            );
        }
        PairsCommand::Remove { pairs: file, index } => {
            let mut pair_set = PairSet::load(&file)?;
            if index >= pair_set.pairs.len() {
                // Inclusive upper bound, matching the layer-range refusals.
                bail!(
                    "pair index {index} is outside the set's 0..{} range",
                    pair_set.pairs.len() - 1
                );
            }
            let removed = pair_set.pairs.remove(index);
            // Removing the last pair leaves a set no loader would accept, so
            // `save` refuses and the file on disk is left as it was.
            pair_set.save(&file)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path": file.display().to_string(),
                    "pair_count": pair_set.pairs.len(),
                    "removed": {
                        "index": index,
                        "positive": removed.positive,
                        "negative": removed.negative,
                    },
                }))?
            );
        }
        PairsCommand::Synthesize {
            generator,
            generator_model,
            model,
            revision,
            device,
            precision,
            chat_template,
            trait_description,
            count,
            output,
            trait_name,
            opposite,
            retry_multiplier,
            dedupe_bits,
            dedupe_bands,
            refusal_threshold,
            max_new_tokens,
            temperature,
            top_p,
            seed,
        } => {
            let options = SynthesisOptions {
                trait_description,
                trait_name: trait_name.unwrap_or_default(),
                opposite,
                count,
                retry_multiplier,
                dedupe: DedupeOptions {
                    threshold_bits: dedupe_bits,
                    num_bands: dedupe_bands,
                    ..DedupeOptions::default()
                },
                refusal_threshold,
                generation: GenerationOptions {
                    strength: 1.0,
                    max_new_tokens,
                    temperature,
                    top_p: Some(top_p),
                    seed,
                },
                diversity_seed: seed,
                diversity_max_sample: DEFAULT_MAX_SAMPLE,
            };
            // The runtime or the gateway is built inside the arm that uses it:
            // `--generator brama` must not load weights or touch a device, and
            // `--generator local` must not read the gateway's environment.
            let (pair_set, report) = match generator.as_str() {
                "local" => {
                    let Some(model) = model else {
                        bail!("pairs synthesize with --generator local requires --model");
                    };
                    let mut runtime = Runtime::load_at(
                        &model,
                        revision.as_deref(),
                        DeviceChoice::parse(&device)?,
                        Precision::parse(&precision)?,
                    )?;
                    // Synthesis is the first step of the funnel and everything
                    // downstream inherits what it writes. Addressed without
                    // its markers, an instruct checkpoint answers a pair
                    // request with instructions about answering pair requests.
                    runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
                    pairs::synthesize(pairs::Generator::Local(&runtime), &options)?
                }
                "brama" => {
                    let Some(route) = generator_model else {
                        bail!("pairs synthesize with --generator brama requires --generator-model");
                    };
                    let gateway = brama::Gateway::from_env(&route)?;
                    pairs::synthesize(pairs::Generator::Gateway(&gateway), &options)?
                }
                value => bail!("unknown generator {value:?}; expected local or brama"),
            };
            pair_set.save(&output)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path": output.display().to_string(),
                    "report": report,
                }))?
            );
        }
    }
    Ok(())
}
