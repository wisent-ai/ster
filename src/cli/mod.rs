//! Everything the `ster` binary itself owns: the command surface a person
//! types, and the arms behind it. The library does the work; this decides
//! what was asked for and prints the answer.

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use clap::{Args, Parser};
use serde::Serialize;
use serde_json::Value;
use ster::{DeviceChoice, Precision, Runtime};

mod command;
mod converse;
mod decide;
mod decisions;
mod onboarding;
mod pairs;
mod toy;
mod tune;
mod vectors;
mod workspace;

use command::Command;

#[derive(Debug, Parser)]
#[command(
    name = "ster",
    version,
    about = "Understand, measure, and control model representations",
    long_about = "Ster reads hidden representations from open-weight decoder models — Llama, Mistral, Mixtral, Qwen2, Qwen2-MoE, Qwen3, Qwen3-MoE, Phi-2, Phi-3, PhiMoE, Granite, GraniteMoE, Granite 4.0, StableLM, Starcoder2, Command R, Nemotron, Nemotron-H, OLMo, OLMo 2, OLMo 3, OLMoE, EXAONE 3, EXAONE 4, InternLM2, InternLM3, Seed-OSS, Arcee, Jais 2, Tele-FLM, ERNIE 4.5, ERNIE 4.5 MoE, MiniCPM, MiniCPM3, Orion, GLM-4, GLM-4.5, GPT-NeoX, Pythia, GPT-J, GPT-2, StarCoder, OPT, BLOOM, Falcon, MPT, DBRX, DeepSeek-V2, DeepSeek-V3, GPT-OSS, Ling, Mamba, Falcon-Mamba, Mamba-2, Jamba, Bamba, LFM2, LFM2-MoE, HunYuan, HunYuan-MoE, TeleChat2, SmolLM3, Gemma, Gemma 2 and Gemma 3 — trains steering directions from contrastive pairs, evaluates those directions, and applies them during generation."
)]
struct Cli {
    /// Print every answer as `path: value` lines for a person instead of
    /// the JSON document machines read; both come from the same data.
    #[arg(long, global = true)]
    text: bool,
    #[command(subcommand)]
    command: Command,
}

/// Whether this invocation asked for text; set once before any command runs.
static TEXT: OnceLock<bool> = OnceLock::new();

/// Print one answer: the pretty JSON document, or with `--text` one
/// `path: value` line per leaf field, nested keys joined with dots.
pub(crate) fn answer<T: Serialize>(value: &T) -> Result<()> {
    let value = serde_json::to_value(value)?;
    if !TEXT.get().copied().unwrap_or(false) {
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    let mut lines = String::new();
    render(&value, "", &mut lines);
    print!("{lines}");
    Ok(())
}

fn render(value: &Value, path: &str, lines: &mut String) {
    match value {
        Value::Object(fields) => {
            for (key, field) in fields {
                let child = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                render(field, &child, lines);
            }
        }
        Value::Array(items) => {
            if items.is_empty() {
                lines.push_str(&format!("{path}: []\n"));
            }
            for (index, item) in items.iter().enumerate() {
                render(item, &format!("{path}[{index}]"), lines);
            }
        }
        Value::String(text) => lines.push_str(&format!("{path}: {text}\n")),
        Value::Null => lines.push_str(&format!("{path}: -\n")),
        other => lines.push_str(&format!("{path}: {other}\n")),
    }
}

#[derive(Debug, Args)]
pub(crate) struct ModelArgs {
    /// Hugging Face model id or local model directory.
    #[arg(long)]
    model: String,
    /// Immutable Hugging Face revision; defaults to main.
    #[arg(long)]
    revision: Option<String>,
    /// Runtime device: cpu, metal, or cuda. Parsed by clap, so an unknown
    /// device is a usage error (exit 2) before anything loads.
    #[arg(long, default_value = "cpu", value_parser = DeviceChoice::parse)]
    device: DeviceChoice,
}

impl ModelArgs {
    /// The shared load. Every command that maps a checkpoint goes through
    /// here, so `--precision` means the same thing on all of them and a new
    /// command cannot quietly forget it.
    fn load_at(&self, precision: Precision) -> Result<Runtime> {
        Runtime::load_at(
            &self.model,
            self.revision.as_deref(),
            self.device,
            precision,
        )
    }
}

pub(crate) fn run() -> Result<()> {
    let cli = Cli::parse();
    let _ = TEXT.set(cli.text);
    match cli.command {
        Command::Train(args) => vectors::train(args),
        Command::Optimize(args) => vectors::optimize(args),
        Command::Evaluate(args) => vectors::evaluate(args),
        Command::Generate(args) => vectors::generate(args),
        Command::Extract(args) => vectors::extract(args),
        Command::Inspect(args) => vectors::inspect(args),
        Command::Onboarding(args) => vectors::onboarding(args),
        Command::Decide(args) => decide::decide(args),
        Command::Calibrate(args) => decide::calibrate(args),
        Command::Decisions { command } => decisions::run(command),
        Command::Workspace { command } => workspace::run(command),
        Command::Pairs { command } => pairs::run(command),
        Command::Tune { command } => tune::run(command),
        Command::ToyModel { out } => toy::run(&out),
        Command::Converse(args) => converse::run(args),
        Command::Request { operation } => {
            let status = ster::request::run(&operation)?;
            if status != 0 {
                std::process::exit(status);
            }
            Ok(())
        }
    }
}

fn resolve_pairs(selected: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = selected {
        return Ok(path);
    }
    ster::workspace::active_pair_set()?.context(
        "no active Ster pair set; pass --pairs or import one with `ster workspace import-pairs`",
    )
}
