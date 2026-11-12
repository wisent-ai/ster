//! One typed tuning operation and execution path for CLI and native requests.
//! Each frontend renders the same actual result in its own transport.

use serde_json::json;
use ster::{
    Precision,
    failures::{lora::TARGETS_NONE, refuse},
    lora,
    workflow::parse_layers,
};

mod artifact;
mod command;
mod durable;
mod train;

pub(crate) use command::TuneCommand;

pub(super) fn run(command: TuneCommand) -> anyhow::Result<()> {
    crate::cli::answer_document(&execute(command)?)
}

pub(crate) fn execute(command: TuneCommand) -> anyhow::Result<serde_json::Value> {
    match command {
        TuneCommand::Sft(args) => train::sft(args),
        TuneCommand::Dpo(args) => train::dpo(args),
        TuneCommand::Bipo(args) => train::bipo(args),
        TuneCommand::Reward(args) => train::reward(args),
        TuneCommand::Grpo(args) => train::grpo(args),
        TuneCommand::Decide(args) => train::decide(args),
        TuneCommand::Merge(args) => artifact::merge(args),
        TuneCommand::Export(args) => artifact::export(args),
        TuneCommand::Evaluate(args) => artifact::evaluate(args),
        TuneCommand::Inspect(args) => artifact::inspect(args),
    }
}

/// Records the dtype the base weights were mapped at in a run's own report,
/// beside the chat-template decision and for the same reason.
///
/// Two runs of the same command at different precisions produce different
/// losses, and an adapter that does not say which one made it leaves an
/// operator comparing two numbers that were never comparable.
pub(super) fn note_precision(
    report: &mut serde_json::Value,
    precision: Precision,
) -> anyhow::Result<()> {
    // Every caller hands it `json::value(&report)` of a `#[derive(Serialize)]
    // pub struct ...Report` with named fields, which serde_json writes as an
    // object.
    report
        .as_object_mut()
        .expect("every run report is a struct with named fields")
        .insert("precision".to_owned(), json!(precision.name()));
    Ok(())
}

/// `--targets` is a comma-separated projection list. Repeats collapse, so
/// `query,query` builds one adapter, and the order follows the flag.
pub(super) fn parse_targets(value: &str) -> anyhow::Result<Vec<lora::Target>> {
    let mut targets = Vec::new();
    for segment in value
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        let target = lora::Target::parse(segment)?;
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    if targets.is_empty() {
        return Err(refuse(&TARGETS_NONE, &[("targets", &format!("{value:?}"))]));
    }
    Ok(targets)
}

/// `--layers` means here what it means everywhere else in Ster, with one
/// difference: `all` cannot be expanded yet. `parse_layers` needs the model's
/// layer count, and the count is only known once the weights are mapped —
/// which happens inside `Runtime::load_trainable`, after the spec exists. An
/// empty layer list is the spec's way of saying every layer, and the loader
/// resolves it against the real count before it builds any adapter.
pub(super) fn parse_adapter_layers(value: &str) -> anyhow::Result<Vec<usize>> {
    if value.trim() == "all" {
        return Ok(Vec::new());
    }
    parse_layers(value, usize::MAX)
}
