//! The decision arms of `ster`: answering typed questions about a state, and
//! fitting the temperature that makes those answers' probabilities honest.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use ster::{
    decide::{self, Calibration, ExampleSet, Request, RAW_TEMPERATURE},
    ChatChoice, DecideOptions, Precision,
};

use super::ModelArgs;

/// The flags every decision run shares with a calibration run.
#[derive(Debug, clap::Args)]
pub(super) struct ReadArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// auto renders every prompt through the model's own chat template when
    /// it publishes one, off sends raw text. A decision is read at the
    /// position where the assistant's answer begins, so an instruct
    /// checkpoint wants auto.
    #[arg(long, default_value = "auto")]
    chat_template: String,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32")]
    precision: String,
    /// Option orders each question is shown in. 0 shows every option under
    /// every letter once, which cancels the model's letter preference; 1 is a
    /// single pass with no correction.
    #[arg(long, default_value_t = 0)]
    permutations: usize,
}

/// `ster decide`
#[derive(Debug, clap::Args)]
pub(super) struct DecideArgs {
    #[command(flatten)]
    read: ReadArgs,
    /// The request document: a state and its typed questions. `-` reads it
    /// from standard input.
    #[arg(long)]
    request: PathBuf,
    /// A calibration written by `ster calibrate` for this exact model.
    #[arg(long)]
    calibration: Option<PathBuf>,
    /// Also write the response document here.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Add per-order detail to the response: what the model said in each
    /// option order on its own, so a flat answer can be read for whether the
    /// content or the letter decided it.
    #[arg(long)]
    explain: bool,
}

/// `ster calibrate`
#[derive(Debug, clap::Args)]
pub(super) struct CalibrateArgs {
    #[command(flatten)]
    read: ReadArgs,
    /// Labelled examples: requests with the correct answer to some of their
    /// questions.
    #[arg(long)]
    examples: PathBuf,
    /// Where the calibration artifact is written.
    #[arg(long)]
    output: PathBuf,
}

pub(super) fn decide(args: DecideArgs) -> Result<()> {
    let DecideArgs { read, request, calibration, output, explain } = args;
    // Both documents are read before a single weight is mapped, so a bad
    // request or a calibration for another model is refused in milliseconds
    // rather than after a full checkpoint load.
    let request = load_request(&request)?;
    let calibration = calibration
        .as_deref()
        .map(|path| Calibration::load(path).map(|document| (path, document)))
        .transpose()?;
    let (mut runtime, mut options) = read.load()?;
    if let Some((path, document)) = &calibration {
        document.check_model(path, &runtime.model_id)?;
        options.temperature = document.temperature;
    }
    options.explain = explain;
    runtime.set_chat_template(ChatChoice::parse(&read.chat_template)?);
    let response = decide::decide(
        &runtime,
        &request,
        options,
        calibration.as_ref().map(|(path, _)| path.to_string_lossy()).as_deref(),
    )?;
    let document = serde_json::to_string_pretty(&response)?;
    if let Some(output) = output {
        fs::write(&output, &document).with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{document}");
    Ok(())
}

pub(super) fn calibrate(args: CalibrateArgs) -> Result<()> {
    let CalibrateArgs { read, examples, output } = args;
    let examples = ExampleSet::load(&examples)?;
    let (mut runtime, options) = read.load()?;
    runtime.set_chat_template(ChatChoice::parse(&read.chat_template)?);
    let calibration = decide::calibrate(&runtime, &examples, options)?;
    calibration.save(&output)?;
    println!("{}", serde_json::to_string_pretty(&calibration)?);
    Ok(())
}

impl ReadArgs {
    fn load(&self) -> Result<(ster::Runtime, DecideOptions)> {
        let runtime = self.model.load_at(Precision::parse(&self.precision)?)?;
        let options = DecideOptions {
            permutations: self.permutations,
            temperature: RAW_TEMPERATURE,
            explain: false,
        };
        Ok((runtime, options))
    }
}

fn load_request(path: &Path) -> Result<Request> {
    if path.as_os_str() == "-" {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).context("failed to read the request from standard input")?;
        let request: Request =
            serde_json::from_slice(&bytes).context("invalid decide request on standard input")?;
        request.validate()?;
        return Ok(request);
    }
    Request::load(path)
}
