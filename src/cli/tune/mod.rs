//! `ster tune`: training adapters and working with the ones already trained.
//! The arms go through the same `tune` and `lora` functions `ster request`
//! calls and print one pretty JSON document each.

use anyhow::{bail, Context, Result};
use serde_json::json;
use ster::{lora, workflow::parse_layers, Precision};

mod artifact;
mod command;
mod train;

pub(super) use command::TuneCommand;

pub(super) fn run(command: TuneCommand) -> Result<()> {
    match command {
        TuneCommand::Sft(args) => train::sft(args),
        TuneCommand::Dpo(args) => train::dpo(args),
        TuneCommand::Reward(args) => train::reward(args),
        TuneCommand::Grpo(args) => train::grpo(args),
        TuneCommand::Decide(args) => train::decide(args),
        TuneCommand::Merge(args) => artifact::merge(args),
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
pub(super) fn note_precision(report: &mut serde_json::Value, precision: Precision) -> Result<()> {
    report
        .as_object_mut()
        .context("a run report must be a JSON object to record its precision")?
        .insert("precision".to_owned(), json!(precision.name()));
    Ok(())
}

/// `--targets` is a comma-separated projection list. Repeats collapse, so
/// `query,query` builds one adapter, and the order follows the flag.
pub(super) fn parse_targets(value: &str) -> Result<Vec<lora::Target>> {
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
pub(super) fn parse_adapter_layers(value: &str) -> Result<Vec<usize>> {
    if value.trim() == "all" {
        return Ok(Vec::new());
    }
    parse_layers(value, usize::MAX)
}
