use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use ster::{
    ContrastivePair, DeviceChoice, GenerationOptions, PairSet, Runtime, SteeringArtifact,
    SynthesisOptions, TrainingMethod,
    brama,
    dedupe::DedupeOptions,
    diversity::DEFAULT_MAX_SAMPLE,
    pairs::{self, InspectOptions},
    workflow::{self, parse_layers},
};

#[derive(Debug, Parser)]
#[command(
    name = "ster",
    version,
    about = "Understand, measure, and control model representations",
    long_about = "Ster reads hidden representations from open-weight Llama-family models, trains steering directions from contrastive pairs, evaluates those directions, and applies them during generation."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train steering vectors from positive and negative prompts.
    Train {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON file with trait_name and contrastive pairs.
        #[arg(long)]
        pairs: PathBuf,
        /// Output Ster steering artifact.
        #[arg(long)]
        output: PathBuf,
        /// Comma-separated layers, half-open ranges such as 8..16, or all.
        #[arg(long, default_value = "all")]
        layers: String,
        /// Direction training method: caa, pca, or logistic.
        #[arg(long, default_value = "caa")]
        method: String,
    },
    /// Select the best method and layer on an 80/20 holdout.
    Optimize {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        pairs: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "all")]
        layers: String,
    },
    /// Measure pair ordering for a steering artifact.
    Evaluate {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        pairs: PathBuf,
        #[arg(long)]
        vector: PathBuf,
    },
    /// Generate text with an optional steering artifact.
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        vector: Option<PathBuf>,
        #[arg(long, default_value_t = 1.0)]
        strength: f64,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Zero selects deterministic argmax generation.
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long)]
        top_p: Option<f64>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Export hidden representations for arbitrary prompts.
    Extract {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON file shaped as {"prompts": ["..."]}.
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "all")]
        layers: String,
    },
    /// Print and validate a Ster steering artifact.
    Inspect {
        #[arg(value_name = "ARTIFACT")]
        artifact: PathBuf,
    },
    /// Author, inspect, and synthesize contrastive pair sets.
    Pairs {
        #[command(subcommand)]
        command: PairsCommand,
    },
    /// Loopback HTTP/JSON backend for desktop apps.
    Serve {
        /// Port to bind; 0 selects an ephemeral port.
        #[arg(long, default_value_t = 0)]
        port: u16,
    },
}

#[derive(Debug, Subcommand)]
enum PairsCommand {
    /// Report duplicates, refusals, length balance, and diversity for a set.
    Inspect {
        #[arg(value_name = "FILE")]
        file: PathBuf,
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

#[derive(Debug, Args)]
struct ModelArgs {
    /// Hugging Face model id or local model directory.
    #[arg(long)]
    model: String,
    /// Immutable Hugging Face revision; defaults to main.
    #[arg(long)]
    revision: Option<String>,
    /// Runtime device: cpu, metal, or cuda.
    #[arg(long, default_value = "cpu")]
    device: String,
}

impl ModelArgs {
    fn load(&self) -> Result<Runtime> {
        let device = DeviceChoice::parse(&self.device)?;
        Runtime::load(&self.model, self.revision.as_deref(), device)
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train { model, pairs, output, layers, method } => {
            let runtime = model.load()?;
            let pair_set = PairSet::load(&pairs)?;
            let layers = parse_layers(&layers, runtime.layer_count())?;
            let method = TrainingMethod::parse(&method)?;
            let artifact = workflow::train(&runtime, &pair_set, &layers, method)?;
            artifact.save(&output)?;
            println!("{}", serde_json::to_string_pretty(&workflow::artifact_summary(&artifact))?);
        }
        Command::Optimize { model, pairs, output, layers } => {
            let runtime = model.load()?;
            let pair_set = PairSet::load(&pairs)?;
            let layers = parse_layers(&layers, runtime.layer_count())?;
            let artifact = workflow::optimize(&runtime, &pair_set, &layers)?;
            artifact.save(&output)?;
            println!("{}", serde_json::to_string_pretty(&workflow::artifact_summary(&artifact))?);
        }
        Command::Evaluate { model, pairs, vector } => {
            let runtime = model.load()?;
            let pair_set = PairSet::load(&pairs)?;
            let artifact = SteeringArtifact::load(&vector)?;
            let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Generate {
            model,
            prompt,
            vector,
            strength,
            max_new_tokens,
            temperature,
            top_p,
            seed,
        } => {
            let runtime = model.load()?;
            let artifact = vector.as_deref().map(SteeringArtifact::load).transpose()?;
            let generated = runtime.generate(
                &prompt,
                artifact.as_ref(),
                GenerationOptions { strength, max_new_tokens, temperature, top_p, seed },
            )?;
            println!("{generated}");
        }
        Command::Extract { model, input, output, layers } => {
            let runtime = model.load()?;
            let layers = parse_layers(&layers, runtime.layer_count())?;
            workflow::extract(&runtime, &input, &output, &layers)?;
            println!("{}", output.display());
        }
        Command::Inspect { artifact } => {
            let artifact = SteeringArtifact::load(&artifact)
                .with_context(|| format!("failed to inspect {}", artifact.display()))?;
            println!("{}", serde_json::to_string_pretty(&artifact)?);
        }
        Command::Pairs { command } => run_pairs(command)?,
        Command::Serve { port } => {
            ster::serve::run(port)?;
        }
    }
    Ok(())
}

/// The `ster pairs` arms. Each one does the work and prints one pretty JSON
/// document, exactly like the arms above; they live here rather than inline
/// only to keep the top-level match readable.
fn run_pairs(command: PairsCommand) -> Result<()> {
    match command {
        PairsCommand::Inspect { file, dedupe_bits, dedupe_bands, refusal_threshold } => {
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
                    let runtime =
                        Runtime::load(&model, revision.as_deref(), DeviceChoice::parse(&device)?)?;
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
