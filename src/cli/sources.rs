//! `ster sources [--all] | ster sources discover <adapter> --path <file>`: the data other
//! tools keep that Ster reads, one adapter each (`IMP-STER-<NN>`), under the import
//! guidelines' contract. An example set another trainer published (03 OpenAI, 04 ShareGPT,
//! 05 Alpaca) is read by `tune sft --examples`; preference rows (01 Hugging Face, 06 DPO, 19
//! Preferences) by `pairs import --benchmark preference`, which writes a pair set. `discover`
//! reads a file as that command would and says how many examples or pairs it gives, or why
//! it is not that source; nothing is written, so the writing verbs are those commands. The
//! model checkpoints `--model` loads (12, 13, 16) are in `models`.

mod models;
mod named;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use serde_json::{Value, json};
use ster::failures::{refuse, sources::ADAPTER_UNKNOWN};

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
}

/// Each adapter: its id, its source and whether it is preference rows (else an example set).
const ADAPTERS: &[(&str, &str, bool)] = &[
    ("IMP-STER-01", "a Hugging Face preference dataset (prompt, chosen, rejected)", true),
    ("IMP-STER-03", "OpenAI's fine-tuning JSONL (messages, or prompt and completion)", false),
    ("IMP-STER-04", "ShareGPT conversations", false),
    ("IMP-STER-05", "Alpaca (instruction, input, output)", false),
    ("IMP-STER-06", "DPO rows (prompt, chosen, rejected), as TRL's DPO trainer reads them", true),
    ("IMP-STER-19", "Preferences' exported comparisons (prompt, chosen, rejected)", true),
];

fn row(&(id, source, preference): &(&str, &str, bool)) -> Value {
    let read_by = if preference {
        "ster pairs import --benchmark preference <file> …"
    } else {
        "ster tune sft --examples <file> …"
    };
    json!({"adapter": id, "source": source, "kind": "E", "read_by": read_by, "found": false})
}

/// The pairs preference rows give and how many rows give none: a JSON list, else JSON Lines.
fn preference_pairs(text: &str) -> Result<(usize, usize), String> {
    let rows: Vec<Value> = match serde_json::from_str::<Value>(text) {
        Ok(Value::Array(rows)) => rows,
        Ok(row) => vec![row],
        Err(_) => text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()
            .map_err(|error| format!("it is neither a JSON list nor JSON Lines: {error}"))?,
    };
    let pairs = rows.iter().filter(|row| ster::pairs::benchmark::preference_pair(row).is_ok()).count();
    Ok((pairs, rows.len() - pairs))
}

pub(super) fn run(args: SourcesArgs) -> Result<()> {
    let Some(SourcesCommand::Discover { adapter, path }) = args.command else {
        let adapters: Vec<Value> = ADAPTERS
            .iter()
            .map(row)
            .chain(models::MODELS.iter().map(models::row))
            .chain(named::NAMED.iter().map(named::row))
            .collect();
        return super::answer_document(&json!({"adapters": adapters, "all": args.all}));
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
    let read = if found.2 {
        std::fs::read_to_string(&path).map_err(|error| format!("{shown}: {error}")).and_then(|text| preference_pairs(&text))
    } else {
        ster::tune::ExampleSet::load(&path).map(|set| (set.examples.len(), 0)).map_err(|error| format!("{error:#}"))
    };
    match read {
        Ok((given, left_out)) => {
            answer["found"] = json!(given > 0);
            answer[if found.2 { "pairs" } else { "examples" }] = json!(given);
            answer["left_out"] = json!(left_out);
        }
        Err(why) => answer["why_not"] = json!(why),
    }
    super::answer_document(&answer)
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
        "note": "--model reads a checkpoint where it is; an example set or preference file is named to tune sft --examples or pairs import",
    }))
}
