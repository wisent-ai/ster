use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use ster::{
    DeviceChoice, GenerationOptions, PairSet, Runtime, SteeringArtifact, TrainingMethod,
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
    /// Loopback HTTP/JSON backend for desktop apps.
    Serve {
        /// Port to bind; 0 selects an ephemeral port.
        #[arg(long, default_value_t = 0)]
        port: u16,
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
        Command::Serve { port } => {
            ster::serve::run(port)?;
        }
    }
    Ok(())
}
