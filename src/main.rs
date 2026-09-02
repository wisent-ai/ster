use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::Device;
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use ster::{
    ChatChoice, ContrastivePair, DeviceChoice, DpoLoss, DpoOptions, EvaluateOptions, ExampleSet, GenerationOptions, GrpoOptions,
    PairSet, Precision, PromptSet, Reward, RewardHead, RewardOptions, Runtime, SftOptions, SteeringArtifact,
    SynthesisOptions, TrainingMethod,
    brama,
    dedupe::DedupeOptions,
    diversity::DEFAULT_MAX_SAMPLE,
    lora,
    pairs::{self, InspectOptions},
    tune,
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
        /// auto reads every pair through the model's own chat template when
        /// it publishes one, off reads it as raw text. A direction is fitted
        /// in whatever space the pairs were read in and added in whatever
        /// space generation runs in, so a direction fitted off and applied
        /// auto is measured in one space and steers another.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. A
        /// direction is fitted in whatever space the prompts were read in, so
        /// two artifacts trained at different precisions are not
        /// interchangeable. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
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
        /// auto reads every pair through the model's own chat template when
        /// it publishes one, off reads it as raw text.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. bf16 needs
        /// --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
    },
    /// Measure pair ordering for a steering artifact.
    Evaluate {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        pairs: PathBuf,
        #[arg(long)]
        vector: PathBuf,
        /// auto reads every pair through the model's own chat template when
        /// it publishes one, off reads it as raw text. It should match the
        /// run that trained the artifact for the same reason --precision
        /// should.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. It should
        /// match the run that trained the artifact, or the score measures the
        /// direction in a space it was not fitted in.
        #[arg(long, default_value = "f32")]
        precision: String,
    },
    /// Generate text with an optional steering artifact.
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        vector: Option<PathBuf>,
        /// Frozen LoRA adapter artifact to load the model with. It must have
        /// been trained for this exact model: Ster refuses a mismatch rather
        /// than steering the wrong residual stream.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// auto renders the prompt through the model's own chat template when
        /// it publishes one, off sends the prompt as raw text. An instruct
        /// checkpoint asked a bare question continues the text instead of
        /// answering it, which is what auto exists to prevent.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. Half
        /// precision holds a checkpoint in half the memory; a steering vector
        /// is cast to it on the way in. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
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
        /// auto reads every prompt through the model's own chat template when
        /// it publishes one, off reads it as raw text. The exported
        /// activations are the states the model reached; this is what it was
        /// reading when it reached them.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. The
        /// exported activations are F32 either way; this is the width they
        /// were computed in. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
    },
    /// Summarize and validate a Ster steering artifact.
    Inspect {
        #[arg(value_name = "ARTIFACT")]
        artifact: PathBuf,
    },
    /// Author, inspect, and synthesize contrastive pair sets.
    Pairs {
        #[command(subcommand)]
        command: PairsCommand,
    },
    /// Train, merge, score, and inspect LoRA adapters.
    Tune {
        #[command(subcommand)]
        command: TuneCommand,
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
        #[arg(long)]
        pairs: PathBuf,
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

#[derive(Debug, Subcommand)]
enum TuneCommand {
    /// Train LoRA adapters on prompt and completion examples.
    Sft {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON file shaped as {"examples": [{"prompt": "...", "completion": "..."}]}.
        #[arg(long)]
        examples: PathBuf,
        /// Output LoRA adapter safetensors; the identity sidecar is written beside it.
        #[arg(long)]
        output: PathBuf,
        /// Low-rank dimension shared by every adapter.
        #[arg(long, default_value_t = 8)]
        rank: usize,
        /// LoRA scaling numerator; each update is scaled by alpha over rank.
        #[arg(long, default_value_t = 16.0)]
        alpha: f64,
        /// Comma-separated projections to adapt: query, key, value, output,
        /// gate, up, or down. The default query,value is the pair the LoRA
        /// papers adapt first, and it is the cheapest useful choice.
        #[arg(long, default_value = "query,value")]
        targets: String,
        /// Comma-separated layers, half-open ranges such as 8..16, or all.
        #[arg(long, default_value = "all")]
        layers: String,
        /// Passes over the example set.
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,
        /// Examples folded into one optimizer step.
        #[arg(long, default_value_t = 8)]
        accumulation: usize,
        /// Steps over which the learning rate ramps up from zero.
        #[arg(long, default_value_t = 0)]
        warmup_steps: usize,
        /// Examples longer than this many tokens are skipped rather than
        /// truncated; a cut completion would teach the model to stop early.
        #[arg(long, default_value_t = 512)]
        max_sequence: usize,
        /// auto encodes every prompt and completion through the model's own
        /// chat template when it publishes one, off encodes raw text. An
        /// instruct checkpoint trained on raw text learns a format it will
        /// never be prompted in.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Examples folded into one forward pass. One is the unbatched pass
        /// every run recorded so far took; --accumulation still counts
        /// forwards, so a step sees up to batch-size times accumulation
        /// examples and nothing changes at the default.
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
        /// Adapters, any head, and every optimizer moment stay in f32
        /// whatever this says, because a low-rank update below the weight's
        /// own ulp rounds to nothing in half. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Train LoRA adapters to prefer one side of each contrastive pair.
    Dpo {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON file with trait_name and contrastive pairs. The positive side
        /// is the chosen response and the negative side the rejected one, so a
        /// steering pair set trains a preference without being rewritten.
        #[arg(long)]
        pairs: PathBuf,
        /// Output LoRA adapter safetensors; the identity sidecar is written beside it.
        #[arg(long)]
        output: PathBuf,
        /// Low-rank dimension shared by every adapter.
        #[arg(long, default_value_t = 8)]
        rank: usize,
        /// LoRA scaling numerator; each update is scaled by alpha over rank.
        #[arg(long, default_value_t = 16.0)]
        alpha: f64,
        /// Comma-separated projections to adapt: query, key, value, output,
        /// gate, up, or down.
        #[arg(long, default_value = "query,value")]
        targets: String,
        /// Comma-separated layers, half-open ranges such as 8..16, or all.
        #[arg(long, default_value = "all")]
        layers: String,
        /// How hard the frozen reference pulls the policy back.
        #[arg(long, default_value_t = 0.1)]
        beta: f64,
        /// Preference objective: dpo for the sigmoid loss, ipo for the squared
        /// error against 1/(2*beta) over length-normalized log-probabilities.
        #[arg(long, default_value = "dpo")]
        loss: String,
        /// Passes over the pair set.
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,
        /// Pairs folded into one optimizer step.
        #[arg(long, default_value_t = 8)]
        accumulation: usize,
        /// Steps over which the learning rate ramps up from zero.
        #[arg(long, default_value_t = 0)]
        warmup_steps: usize,
        /// Pairs with a side longer than this many tokens are skipped rather
        /// than truncated; a cut response is not the response that was preferred.
        #[arg(long, default_value_t = 512)]
        max_sequence: usize,
        /// auto encodes both sides of every pair as the assistant turn the
        /// model's own chat template renders, when it publishes one; off
        /// encodes raw text.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Pairs folded into one forward pass; a pair is two rows. One is the
        /// unbatched pass every run recorded so far took.
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
        /// Adapters and every optimizer moment stay in f32. bf16 needs
        /// --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Train a scalar reward head that ranks the two sides of each pair.
    Reward {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON file with trait_name and contrastive pairs. The positive side
        /// is the response the head learns to score higher.
        #[arg(long)]
        pairs: PathBuf,
        /// Output reward artifact safetensors, carrying the adapters and the
        /// head together; the identity sidecar is written beside it.
        #[arg(long)]
        output: PathBuf,
        /// Low-rank dimension shared by every adapter.
        #[arg(long, default_value_t = 8)]
        rank: usize,
        /// LoRA scaling numerator; each update is scaled by alpha over rank.
        #[arg(long, default_value_t = 16.0)]
        alpha: f64,
        /// Comma-separated projections to adapt: query, key, value, output,
        /// gate, up, or down.
        #[arg(long, default_value = "query,value")]
        targets: String,
        /// Comma-separated layers, half-open ranges such as 8..16, or all.
        #[arg(long, default_value = "all")]
        layers: String,
        /// Passes over the pair set.
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,
        /// Pairs folded into one optimizer step.
        #[arg(long, default_value_t = 8)]
        accumulation: usize,
        /// Steps over which the learning rate ramps up from zero.
        #[arg(long, default_value_t = 0)]
        warmup_steps: usize,
        /// Pairs with a side longer than this many tokens are skipped rather
        /// than truncated; a cut response is not the response that was ranked.
        #[arg(long, default_value_t = 512)]
        max_sequence: usize,
        /// auto encodes both sides of every pair as the assistant turn the
        /// model's own chat template renders, when it publishes one; off
        /// encodes raw text.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Pairs folded into one forward pass; a pair is two rows. One is the
        /// unbatched pass every run recorded so far took.
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
        /// The adapters and the scalar head stay in f32. bf16 needs
        /// --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Optimize the policy against a reward, using a sampled group as baseline.
    Grpo {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON file shaped as {"prompts": ["..."]}, the shape ster extract takes.
        #[arg(long)]
        prompts: PathBuf,
        /// Output LoRA adapter safetensors; the identity sidecar is written beside it.
        #[arg(long)]
        output: PathBuf,
        /// Where a completion's reward comes from: the keyword length, which
        /// counts sampled tokens and needs no judge, or the path to a reward
        /// artifact written by ster tune reward.
        #[arg(long, default_value = "length")]
        reward: String,
        /// Completions sampled per prompt. Their mean is the baseline, which is
        /// why two is the smallest group that means anything.
        #[arg(long, default_value_t = 4)]
        group: usize,
        /// Passes over the prompt set; each one re-samples from the current policy.
        #[arg(long, default_value_t = 1)]
        iterations: usize,
        /// Weight on the KL penalty pulling the policy back to the frozen base.
        #[arg(long, default_value_t = 0.04)]
        beta: f64,
        /// Low-rank dimension shared by every adapter.
        #[arg(long, default_value_t = 8)]
        rank: usize,
        /// LoRA scaling numerator; each update is scaled by alpha over rank.
        #[arg(long, default_value_t = 16.0)]
        alpha: f64,
        /// Comma-separated projections to adapt: query, key, value, output,
        /// gate, up, or down.
        #[arg(long, default_value = "query,value")]
        targets: String,
        /// Comma-separated layers, half-open ranges such as 8..16, or all.
        #[arg(long, default_value = "all")]
        layers: String,
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,
        /// Prompt groups folded into one optimizer step. One group is already
        /// --group sequences, so a step per group is the natural unit.
        #[arg(long, default_value_t = 1)]
        accumulation: usize,
        /// Steps over which the learning rate ramps up from zero.
        #[arg(long, default_value_t = 0)]
        warmup_steps: usize,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        /// Must exceed zero; argmax sampling would draw one identical
        /// completion per group and leave the baseline with nothing to compare.
        #[arg(long, default_value_t = 0.9)]
        temperature: f64,
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
        /// Prompts whose prompt plus its longest completion exceed this many
        /// tokens are skipped rather than truncated.
        #[arg(long, default_value_t = 512)]
        max_sequence: usize,
        /// auto samples every completion from the prompt as the model's own
        /// chat template renders it, when it publishes one; off samples from
        /// raw text.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
        /// Adapters and every optimizer moment stay in f32. bf16 needs
        /// --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Fold a LoRA adapter into the base weights as a standalone checkpoint.
    Merge {
        #[command(flatten)]
        model: ModelArgs,
        /// The adapter to fold in. It must have been trained for this exact
        /// model, and must be a generation adapter rather than a reward model.
        #[arg(long)]
        adapter: PathBuf,
        /// Directory to write. It receives model.safetensors beside the
        /// source's own config.json and tokenizer.json, plus whichever of
        /// tokenizer_config.json and chat_template.jinja the source
        /// published, which together are what --model accepts. Those last
        /// two are where a chat template lives, so a source that published
        /// one merges to a checkpoint that still reports applied rather
        /// than absent.
        #[arg(long)]
        output: PathBuf,
    },
    /// Score a checkpoint on held-out examples: loss and perplexity, no training.
    Evaluate {
        #[command(flatten)]
        model: ModelArgs,
        /// JSON file shaped as {"examples": [{"prompt": "...", "completion": "..."}]}.
        #[arg(long)]
        examples: PathBuf,
        /// Frozen LoRA adapter to attach before scoring. Omit it to score the
        /// bare checkpoint, which is the run the adapter is compared against.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Examples longer than this many tokens are skipped rather than
        /// truncated; a cut completion is not the completion being scored.
        #[arg(long, default_value_t = 512)]
        max_sequence: usize,
        /// auto scores every example in the shape the model's own chat
        /// template renders, when it publishes one; off scores raw text. It
        /// must match the run that trained the adapter, or the score measures
        /// a format the adapter never saw.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Examples folded into one forward pass. One is the unbatched pass
        /// every run recorded so far took.
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
        /// A score is only comparable with another score taken at the same
        /// precision. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
    },
    /// Print and validate a Ster LoRA adapter artifact.
    Inspect {
        #[arg(value_name = "ARTIFACT")]
        artifact: PathBuf,
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
    /// The shared load. Every command that maps a checkpoint goes through
    /// here, so `--precision` means the same thing on all of them and a new
    /// command cannot quietly forget it.
    fn load_at(&self, precision: Precision) -> Result<Runtime> {
        let device = DeviceChoice::parse(&self.device)?;
        Runtime::load_at(&self.model, self.revision.as_deref(), device, precision)
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train { model, pairs, output, layers, method, chat_template, precision } => {
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let pair_set = PairSet::load(&pairs)?;
            let layers = parse_layers(&layers, runtime.layer_count())?;
            let method = TrainingMethod::parse(&method)?;
            let artifact = workflow::train(&runtime, &pair_set, &layers, method)?;
            artifact.save(&output)?;
            let mut summary = workflow::artifact_summary(&artifact);
            chat.annotate(&mut summary)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Optimize { model, pairs, output, layers, chat_template, precision } => {
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let pair_set = PairSet::load(&pairs)?;
            let layers = parse_layers(&layers, runtime.layer_count())?;
            let selection = workflow::optimize(&runtime, &pair_set, &layers)?;
            selection.artifact.save(&output)?;
            let mut summary = selection.summary();
            chat.annotate(&mut summary)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Evaluate { model, pairs, vector, chat_template, precision } => {
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let pair_set = PairSet::load(&pairs)?;
            let artifact = SteeringArtifact::load(&vector)?;
            // The artifact now records the precision and the format it was
            // fitted in, so the advice `--precision` has always given can
            // finally be checked. Same helper the tune half uses.
            tune::warn_on_provenance(&vector, "direction", &runtime);
            let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
            let mut report = serde_json::to_value(report)?;
            chat.annotate(&mut report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Generate {
            model,
            prompt,
            vector,
            adapter,
            chat_template,
            precision,
            strength,
            max_new_tokens,
            temperature,
            top_p,
            seed,
        } => {
            let precision = Precision::parse(&precision)?;
            // Both documents are read before a single weight is mapped, so
            // the two halves of the wrong-document refusal cost the same. An
            // adapter was already refused this early because it is attached
            // during the load; a steering vector was not, and handing one the
            // wrong file paid for a full checkpoint load before being told.
            // `Reward::parse` resolves its source ahead of the policy load for
            // this reason and says so.
            let artifact = vector.as_deref().map(SteeringArtifact::load).transpose()?;
            // An adapter rewrites the projections themselves, so it is
            // attached while the weights are mapped rather than applied per
            // token the way a steering vector is.
            let mut runtime = match adapter.as_deref() {
                Some(adapter) => Runtime::load_with_adapter_at(
                    &model.model,
                    model.revision.as_deref(),
                    DeviceChoice::parse(&model.device)?,
                    adapter,
                    precision,
                )?,
                None => model.load_at(precision)?,
            };
            runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            if let Some(vector) = vector.as_deref() {
                tune::warn_on_provenance(vector, "direction", &runtime);
            }
            let generated = runtime.generate(
                &prompt,
                artifact.as_ref(),
                GenerationOptions { strength, max_new_tokens, temperature, top_p, seed },
            )?;
            println!("{generated}");
        }
        Command::Extract { model, input, output, layers, chat_template, precision } => {
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let layers = parse_layers(&layers, runtime.layer_count())?;
            workflow::extract(&runtime, &input, &output, &layers)?;
            println!("{}", output.display());
        }
        Command::Inspect { artifact } => {
            let artifact = SteeringArtifact::load(&artifact)
                .with_context(|| format!("failed to inspect {}", artifact.display()))?;
            println!("{}", serde_json::to_string_pretty(&workflow::artifact_summary(&artifact))?);
        }
        Command::Pairs { command } => run_pairs(command)?,
        Command::Tune { command } => run_tune(command)?,
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
        PairsCommand::Inspect { pairs: file, dedupe_bits, dedupe_bands, refusal_threshold } => {
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

/// The `ster tune` arms. Like `run_pairs`, each one goes through the same
/// `tune` and `lora` functions the serve endpoints call and prints one pretty
/// JSON document.
fn run_tune(command: TuneCommand) -> Result<()> {
    match command {
        TuneCommand::Sft {
            model,
            examples,
            output,
            rank,
            alpha,
            targets,
            layers,
            epochs,
            learning_rate,
            accumulation,
            warmup_steps,
            max_sequence,
            chat_template,
            batch_size,
            precision,
            seed,
        } => {
            let device = DeviceChoice::parse(&model.device)?;
            let spec = lora::Spec {
                rank,
                alpha,
                targets: parse_targets(&targets)?,
                layers: parse_adapter_layers(&layers)?,
                seed,
            };
            // The adapters have to exist before the first forward pass, so
            // the runtime is built from the spec rather than patched after
            // loading; the returned VarMap owns every trainable tensor.
            let precision = Precision::parse(&precision)?;
            let (mut runtime, varmap) = Runtime::load_trainable_at(
                &model.model,
                model.revision.as_deref(),
                device,
                &spec,
                precision,
            )?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let example_set = ExampleSet::load(&examples)?;
            let options = SftOptions {
                spec: spec.clone(),
                epochs,
                learning_rate,
                accumulation,
                batch: batch_size,
                warmup_steps,
                max_sequence,
                seed,
            };
            let report = tune::sft(&runtime, &varmap, &example_set, &options)?;
            // The report is folded into the artifact so a trained adapter
            // always carries the run that produced it, and the encoding it was
            // produced in travels with it.
            let mut report = serde_json::to_value(&report)?;
            chat.annotate(&mut report)?;
            note_precision(&mut report, precision)?;
            let artifact = runtime.adapter_artifact(&spec, report.clone())?;
            artifact.save(&output)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path": output.display().to_string(),
                    "report": report,
                }))?
            );
        }
        TuneCommand::Dpo {
            model,
            pairs,
            output,
            rank,
            alpha,
            targets,
            layers,
            beta,
            loss,
            epochs,
            learning_rate,
            accumulation,
            warmup_steps,
            max_sequence,
            chat_template,
            batch_size,
            precision,
            seed,
        } => {
            let device = DeviceChoice::parse(&model.device)?;
            let spec = lora::Spec {
                rank,
                alpha,
                targets: parse_targets(&targets)?,
                layers: parse_adapter_layers(&layers)?,
                seed,
            };
            // The reference the objective measures against is this same
            // runtime with the adapters skipped, so exactly one model is
            // loaded however many times each sequence is scored.
            let precision = Precision::parse(&precision)?;
            let (mut runtime, varmap) = Runtime::load_trainable_at(
                &model.model,
                model.revision.as_deref(),
                device,
                &spec,
                precision,
            )?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let pair_set = PairSet::load(&pairs)?;
            let options = DpoOptions {
                spec: spec.clone(),
                loss: DpoLoss::parse(&loss)?,
                beta,
                epochs,
                learning_rate,
                accumulation,
                batch: batch_size,
                warmup_steps,
                max_sequence,
                seed,
            };
            let report = tune::dpo(&runtime, &varmap, &pair_set, &options)?;
            let mut report = serde_json::to_value(&report)?;
            chat.annotate(&mut report)?;
            note_precision(&mut report, precision)?;
            let artifact = runtime.adapter_artifact(&spec, report.clone())?;
            artifact.save(&output)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path": output.display().to_string(),
                    "report": report,
                }))?
            );
        }
        TuneCommand::Reward {
            model,
            pairs,
            output,
            rank,
            alpha,
            targets,
            layers,
            epochs,
            learning_rate,
            accumulation,
            warmup_steps,
            max_sequence,
            chat_template,
            batch_size,
            precision,
            seed,
        } => {
            let device = DeviceChoice::parse(&model.device)?;
            let spec = lora::Spec {
                rank,
                alpha,
                targets: parse_targets(&targets)?,
                layers: parse_adapter_layers(&layers)?,
                seed,
            };
            let precision = Precision::parse(&precision)?;
            let (mut runtime, varmap) = Runtime::load_trainable_at(
                &model.model,
                model.revision.as_deref(),
                device,
                &spec,
                precision,
            )?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            // The head joins the same VarMap the adapters live in, so one
            // optimizer steps the pair and the artifact holds both. It is
            // registered at the parameter dtype, never the base dtype: a
            // scalar head is exactly the small trained weight that rounds away
            // in half precision.
            let head = RewardHead::fresh(
                &varmap,
                runtime.hidden_size(),
                runtime.device(),
                runtime.param_dtype(),
            )?;
            let pair_set = PairSet::load(&pairs)?;
            let options = RewardOptions {
                spec: spec.clone(),
                epochs,
                learning_rate,
                accumulation,
                batch: batch_size,
                warmup_steps,
                max_sequence,
                seed,
            };
            let report = tune::reward(&runtime, &varmap, &head, &pair_set, &options)?;
            let mut report = serde_json::to_value(&report)?;
            chat.annotate(&mut report)?;
            note_precision(&mut report, precision)?;
            let artifact = runtime.reward_artifact(&spec, head.weight(), report.clone())?;
            artifact.save(&output)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path": output.display().to_string(),
                    "report": report,
                }))?
            );
        }
        TuneCommand::Grpo {
            model,
            prompts,
            output,
            reward,
            group,
            iterations,
            beta,
            rank,
            alpha,
            targets,
            layers,
            learning_rate,
            accumulation,
            warmup_steps,
            max_new_tokens,
            temperature,
            top_p,
            max_sequence,
            chat_template,
            precision,
            seed,
        } => {
            let device = DeviceChoice::parse(&model.device)?;
            let spec = lora::Spec {
                rank,
                alpha,
                targets: parse_targets(&targets)?,
                layers: parse_adapter_layers(&layers)?,
                seed,
            };
            // The reward source is resolved before the policy is loaded: a
            // reward artifact for the wrong checkpoint should be refused
            // before an operator waits out a policy load to hear it.
            let source = Reward::parse(&reward, &model.model, model.revision.as_deref(), device)?;
            let precision = Precision::parse(&precision)?;
            let (mut runtime, varmap) = Runtime::load_trainable_at(
                &model.model,
                model.revision.as_deref(),
                device,
                &spec,
                precision,
            )?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let prompt_set = PromptSet::load(&prompts)?;
            let options = GrpoOptions {
                spec: spec.clone(),
                group,
                iterations,
                beta,
                learning_rate,
                accumulation,
                warmup_steps,
                max_sequence,
                generation: GenerationOptions {
                    strength: 1.0,
                    max_new_tokens,
                    temperature,
                    top_p: Some(top_p),
                    seed,
                },
            };
            let report =
                tune::grpo(&runtime, &varmap, &prompt_set, &source, &reward, &options)?;
            let mut report = serde_json::to_value(&report)?;
            chat.annotate(&mut report)?;
            note_precision(&mut report, precision)?;
            let artifact = runtime.adapter_artifact(&spec, report.clone())?;
            artifact.save(&output)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path": output.display().to_string(),
                    "report": report,
                }))?
            );
        }
        TuneCommand::Merge { model, adapter, output } => {
            // No device and no runtime: merging rewrites tensors and never runs
            // the model, so it resolves the checkpoint's files without mapping
            // them.
            let report =
                tune::merge(&model.model, model.revision.as_deref(), &adapter, &output)?;
            println!("{}", serde_json::to_string_pretty(&json!({ "report": report }))?);
        }
        TuneCommand::Evaluate {
            model,
            examples,
            adapter,
            max_sequence,
            chat_template,
            batch_size,
            precision,
        } => {
            // The adapter is attached while the weights are mapped, exactly as
            // `generate --adapter` attaches one, so the score is the score of
            // the model an operator would actually run.
            let precision = Precision::parse(&precision)?;
            let mut runtime = match adapter.as_deref() {
                Some(adapter) => Runtime::load_with_adapter_at(
                    &model.model,
                    model.revision.as_deref(),
                    DeviceChoice::parse(&model.device)?,
                    adapter,
                    precision,
                )?,
                None => Runtime::load_at(
                    &model.model,
                    model.revision.as_deref(),
                    DeviceChoice::parse(&model.device)?,
                    precision,
                )?,
            };
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let example_set = ExampleSet::load(&examples)?;
            let report = tune::evaluate(
                &runtime,
                &example_set,
                adapter.as_deref(),
                &EvaluateOptions { max_sequence, batch: batch_size },
            )?;
            let mut report = serde_json::to_value(&report)?;
            chat.annotate(&mut report)?;
            note_precision(&mut report, precision)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        TuneCommand::Inspect { artifact } => {
            // Inspection reads the adapter document alone: no model is
            // loaded, so the tensors land on the CPU whatever trained them.
            let loaded = lora::Artifact::load(&artifact, &Device::Cpu)
                .with_context(|| format!("failed to inspect {}", artifact.display()))?;
            println!("{}", serde_json::to_string_pretty(&tune::inspect(&loaded))?);
        }
    }
    Ok(())
}

/// Records the dtype the base weights were mapped at in a run's own report,
/// beside the chat-template decision and for the same reason.
///
/// Two runs of the same command at different precisions produce different
/// losses, and an adapter that does not say which one made it leaves an
/// operator comparing two numbers that were never comparable.
fn note_precision(report: &mut serde_json::Value, precision: Precision) -> Result<()> {
    report
        .as_object_mut()
        .context("a run report must be a JSON object to record its precision")?
        .insert("precision".to_owned(), json!(precision.name()));
    Ok(())
}

/// `--targets` is a comma-separated projection list. Repeats collapse, so
/// `query,query` builds one adapter, and the order follows the flag.
fn parse_targets(value: &str) -> Result<Vec<lora::Target>> {
    let mut targets = Vec::new();
    for segment in value.split(',').map(str::trim).filter(|segment| !segment.is_empty()) {
        let target = lora::Target::parse(segment)?;
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    if targets.is_empty() {
        bail!("no targets selected");
    }
    Ok(targets)
}

/// `--layers` means here what it means everywhere else in Ster, with one
/// difference: `all` cannot be expanded yet. `parse_layers` needs the model's
/// layer count, and the count is only known once the weights are mapped —
/// which happens inside `Runtime::load_trainable`, after the spec exists. An
/// empty layer list is the spec's way of saying every layer, and the loader
/// resolves it against the real count before it builds any adapter.
fn parse_adapter_layers(value: &str) -> Result<Vec<usize>> {
    if value.trim() == "all" {
        return Ok(Vec::new());
    }
    parse_layers(value, usize::MAX)
}
