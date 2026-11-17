//! `ster sources [--all] | ster sources discover <adapter> --path <file>`: the data other
//! tools keep that Ster reads, one adapter each (`IMP-STER-<NN>`), under the import
//! guidelines' contract. An example set another trainer published (03 OpenAI, 04 ShareGPT,
//! 05 Alpaca) is read by `tune sft --examples`; preference rows (01 Hugging Face, 06 DPO)
//! by `pairs import --benchmark preference`, which writes a pair set; and the pair set
//! Preferences exports (19: `trait_name` and `pairs` of `positive` and `negative`) is
//! already one, so `workspace import` or `--pairs` reads it as it is. `discover` reads a
//! file as that command would and says how many examples or pairs it gives, or why it is
//! not that source; nothing is written, so the writing verbs are those commands. The
//! model checkpoints `--model` loads (12, 13, 16) are in `models`.

mod models;
mod named;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use serde_json::{Value, json};
use ster::failures::{refuse, sources::{ADAPTER_UNKNOWN, SHARE_UNKEPT}};

#[derive(Debug, Args)]
pub(super) struct SourcesArgs {
    /// List every adapter (Ster reads the same files on every system).
    #[arg(long)]
    all: bool,
    #[command(subcommand)]
    command: Option<SourcesCommand>,
}

#[derive(Debug, Subcommand)]
enum SourcesCommand {
    /// How many examples or pairs a file gives as an adapter's source.
    Discover {
        adapter: String,
        /// The file to read.
        #[arg(long)]
        path: PathBuf,
    },
    /// Whether each discovery reports its adapter and counts to Wisent Integrations; alone, the question.
    Share {
        #[arg(value_parser = ["yes", "no"])]
        answer: Option<String>,
    },
}

/// What an adapter's file holds, and so the Ster command that reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// An example set, read by `tune sft --examples`.
    Examples,
    /// Preference rows of `prompt`, `chosen` and `rejected`, read by `pairs import --benchmark preference`.
    Preference,
    /// A Ster pair set (`trait_name`, `pairs` of `positive` and `negative`), read as it is.
    PairSet,
}

/// Each adapter: its id, its source and the shape of its file.
const ADAPTERS: &[(&str, &str, Shape)] = &[
    ("IMP-STER-01", "a Hugging Face preference dataset (prompt, chosen, rejected)", Shape::Preference),
    ("IMP-STER-03", "OpenAI's fine-tuning JSONL (messages, or prompt and completion)", Shape::Examples),
    ("IMP-STER-04", "ShareGPT conversations", Shape::Examples),
    ("IMP-STER-05", "Alpaca (instruction, input, output)", Shape::Examples),
    ("IMP-STER-06", "DPO rows (prompt, chosen, rejected), as TRL's DPO trainer reads them", Shape::Preference),
    ("IMP-STER-19", "Preferences' exported pair set (trait_name, pairs of positive and negative)", Shape::PairSet),
];

fn row(&(id, source, shape): &(&str, &str, Shape)) -> Value {
    let read_by = match shape {
        Shape::Examples => "ster tune sft --examples <file> …",
        Shape::Preference => "ster pairs import --benchmark preference --source <file> …",
        Shape::PairSet => "ster workspace import <file>, or --pairs <file> on a command that reads pairs",
    };
    json!({"adapter": id, "source": source, "kind": "E", "read_by": read_by, "found": false})
}

/// The pairs preference rows give and how many rows give none: a JSON list, else JSON Lines.
fn preference_pairs(text: &str) -> Result<(usize, usize), String> {
    let rows: Vec<Value> = match serde_json::from_str::<Value>(text) {
        Ok(Value::Array(rows)) => rows,
        Ok(row) => vec![row],
        Err(_) => wisent_formats::jsonl::read(text.as_bytes())
            .map(|rows| rows.into_iter().map(|row| row.value).collect())
            .map_err(|error| format!("it is neither a JSON list nor JSON Lines: {error}"))?,
    };
    let pairs = rows.iter().filter(|row| ster::pairs::benchmark::preference_pair(row).is_ok()).count();
    Ok((pairs, rows.len() - pairs))
}

pub(super) fn run(args: SourcesArgs) -> Result<()> {
    let Some(SourcesCommand::Discover { adapter, path }) = args.command else {
        if let Some(SourcesCommand::Share { answer }) = args.command {
            let kept = super::sharing::SHARING.share(answer.map(|answer| answer == "yes")).map_err(|why| refuse(&SHARE_UNKEPT, &[("why", &why)]))?;
            return super::answer_document(&kept);
        }
        let adapters: Vec<Value> = ADAPTERS
            .iter()
            .map(row)
            .chain(models::MODELS.iter().map(models::row))
            .chain(named::NAMED.iter().map(named::row))
            .collect();
        return super::answer_document(&json!({"adapters": adapters, "all": args.all, "sharing": super::sharing::SHARING.question()}));
    };
    let shown = path.display().to_string();
    if let Some(model) = models::MODELS.iter().find(|(id, _)| *id == adapter) {
        return super::answer_document(&models::discover(model, &path));
    }
    if let Some(listed) = named::NAMED.iter().find(|(id, ..)| *id == adapter) {
        return super::answer_document(&named::row(listed));
    }
    let Some(found) = ADAPTERS.iter().find(|(id, ..)| *id == adapter) else {
        let known: Vec<&str> = ADAPTERS
            .iter()
            .map(|(id, ..)| *id)
            .chain(models::MODELS.iter().map(|(id, _)| *id))
            .chain(named::NAMED.iter().map(|(id, ..)| *id))
            .collect();
        return Err(refuse(&ADAPTER_UNKNOWN, &[("adapter", &adapter), ("path", &shown), ("known", &known.join(", "))]));
    };
    let mut answer = row(found);
    answer["path"] = json!(shown);
    let read = match found.2 {
        Shape::Preference => {
            std::fs::read_to_string(&path).map_err(|error| format!("{shown}: {error}")).and_then(|text| preference_pairs(&text))
        }
        Shape::PairSet => ster::PairSet::load(&path).map(|set| (set.pairs.len(), 0)).map_err(|error| format!("{error:#}")),
        Shape::Examples => {
            ster::tune::ExampleSet::load(&path).map(|set| (set.examples.len(), 0)).map_err(|error| format!("{error:#}"))
        }
    };
    let counted = if found.2 == Shape::Examples { "examples" } else { "pairs" };
    match read {
        Ok((given, left_out)) => {
            answer["found"] = json!(given > 0);
            answer[counted] = json!(given);
            answer["left_out"] = json!(left_out);
        }
        Err(why) => answer["why_not"] = json!(why),
    }
    let given = answer[counted].as_u64().map(|given| given as usize).unwrap_or_default();
    let present = answer["found"] == json!(true);
    super::answer_document(&super::sharing::SHARING.reported(answer, found.0, present, &[(counted, given)], "discovered"))
}

/// The checks `ster doctor --check` runs alone.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub(super) enum DoctorCheck {
    /// The model checkpoints on this machine and how Ster reads them.
    ImportSources,
}

/// `ster doctor --check import-sources`: Ster moves no data in. The models the Hugging
/// Face hub cache keeps (`HF_HUB_CACHE`, else `HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`) are read where they are by `--model <org/name>`, so none
/// is outside; the example sets and preference rows are files a person names.
pub(super) fn doctor(check: DoctorCheck) -> Result<()> {
    let DoctorCheck::ImportSources = check;
    let cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/hub")));
    let in_place: Vec<Value> = cache
        .filter(|cache| cache.is_dir())
        .and_then(|cache| models::MODELS.iter().find(|(id, _)| *id == "IMP-STER-16").map(|model| models::discover(model, &cache)))
        .into_iter()
        .collect();
    super::answer_document(&json!({
        "check": "import-sources",
        "outside": [],
        "read_in_place": in_place,
        "note": "--model reads a checkpoint where it is; an example set, preference file or pair set is named to tune sft --examples, pairs import, or workspace import and --pairs",
    }))
}
