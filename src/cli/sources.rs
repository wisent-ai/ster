//! `ster sources [--all] | ster sources discover <adapter> --path <file>`: the data other
//! tools keep that Ster reads, one adapter each, named for the source it reads, under the
//! import guidelines' contract. An example set another trainer published
//! (`openai-fine-tuning`, `sharegpt`, `alpaca`) is read by `tune sft --examples`; preference
//! rows (`hugging-face-preferences`, `dpo`, `preferences-comparisons`) by `pairs import
//! --benchmark preference`, which writes a pair set. `discover` reads a file as that command
//! would and says how many examples or pairs it gives, or why it is not that source; nothing
//! is written, so the writing verbs are those commands. The model checkpoints `--model`
//! loads are in `models`. An adapter named by the import plan's requirement number it was
//! once listed under is refused, naming the adapter's name.

mod models;
mod named;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use serde_json::{Value, json};
use ster::failures::{refuse, sources::{ADAPTER_RENAMED, ADAPTER_UNKNOWN, SHARE_UNKEPT}};

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

/// Each adapter: its name, its source and whether it is preference rows (else an example set).
const ADAPTERS: &[(&str, &str, bool)] = &[
    ("hugging-face-preferences", "a Hugging Face preference dataset (prompt, chosen, rejected)", true),
    ("openai-fine-tuning", "OpenAI's fine-tuning JSONL (messages, or prompt and completion)", false),
    ("sharegpt", "ShareGPT conversations", false),
    ("alpaca", "Alpaca (instruction, input, output)", false),
    ("dpo", "DPO rows (prompt, chosen, rejected), as TRL's DPO trainer reads them", true),
    ("preferences-comparisons", "Preferences' exported comparisons (prompt, chosen, rejected)", true),
];

/// The import plan's requirement number each adapter was listed under before adapters had
/// names, and the adapter's name: `discover` refuses the number, naming the name.
const RETIRED: &[(&str, &str)] = &[
    ("IMP-STER-01", "hugging-face-preferences"),
    ("IMP-STER-02", "parquet-or-arrow"),
    ("IMP-STER-03", "openai-fine-tuning"),
    ("IMP-STER-04", "sharegpt"),
    ("IMP-STER-05", "alpaca"),
    ("IMP-STER-06", "dpo"),
    ("IMP-STER-07", "repeng"),
    ("IMP-STER-08", "steering-vectors"),
    ("IMP-STER-09", "transformerlens"),
    ("IMP-STER-10", "nnsight"),
    ("IMP-STER-11", "sae-lens"),
    ("IMP-STER-12", "transformers-checkpoint"),
    ("IMP-STER-13", "gguf"),
    ("IMP-STER-14", "mlx-model"),
    ("IMP-STER-15", "pytorch-checkpoint"),
    ("IMP-STER-16", "hugging-face-cache"),
    ("IMP-STER-17", "peft-lora-adapter"),
    ("IMP-STER-18", "mlx-lora-adapter"),
    ("IMP-STER-19", "preferences-comparisons"),
    ("IMP-STER-20", "evaluator-verdicts"),
];

fn row(&(name, source, preference): &(&str, &str, bool)) -> Value {
    let read_by = if preference {
        "ster pairs import --benchmark preference <file> …"
    } else {
        "ster tune sft --examples <file> …"
    };
    json!({"adapter": name, "source": source, "kind": "E", "read_by": read_by, "found": false})
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
            let kept = super::sharing::share(answer.map(|answer| answer == "yes")).map_err(|why| refuse(&SHARE_UNKEPT, &[("why", &why)]))?;
            return super::answer_document(&kept);
        }
        let adapters: Vec<Value> = ADAPTERS
            .iter()
            .map(row)
            .chain(models::MODELS.iter().map(models::row))
            .chain(named::NAMED.iter().map(named::row))
            .collect();
        return super::answer_document(&json!({"adapters": adapters, "all": args.all, "sharing": super::sharing::question()}));
    };
    let shown = path.display().to_string();
    if let Some(model) = models::MODELS.iter().find(|(name, _)| *name == adapter) {
        return super::answer_document(&models::discover(model, &path));
    }
    if let Some(listed) = named::NAMED.iter().find(|(name, ..)| *name == adapter) {
        return super::answer_document(&named::row(listed));
    }
    let Some(found) = ADAPTERS.iter().find(|(name, ..)| *name == adapter) else {
        if let Some((number, name)) = RETIRED.iter().find(|(number, _)| number.eq_ignore_ascii_case(&adapter)) {
            return Err(refuse(&ADAPTER_RENAMED, &[("adapter", &adapter), ("path", &shown), ("number", number), ("name", name)]));
        }
        let known: Vec<&str> = ADAPTERS
            .iter()
            .map(|(name, ..)| *name)
            .chain(models::MODELS.iter().map(|(name, _)| *name))
            .chain(named::NAMED.iter().map(|(name, ..)| *name))
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
        // A file that is not this adapter's source is answered with why and what file to name instead.
        Err(why) => {
            answer["why_not"] = json!(why);
            answer["remedy"] = json!(format!("name a file that exists and holds {}, written as a JSON list or JSON Lines", found.1));
        }
    }
    let given = answer["pairs"].as_u64().or(answer["examples"].as_u64()).map(|given| given as usize).unwrap_or_default();
    let present = answer["found"] == json!(true);
    super::answer_document(&super::sharing::reported(answer, found.0, present, &[(if found.2 { "pairs" } else { "examples" }, given)], "discovered"))
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
        .and_then(|cache| models::MODELS.iter().find(|(name, _)| *name == models::CACHE).map(|model| models::discover(model, &cache)))
        .into_iter()
        .collect();
    super::answer_document(&json!({
        "check": "import-sources",
        "outside": [],
        "read_in_place": in_place,
        "note": "--model reads a checkpoint where it is; an example set or preference file is named to tune sft --examples or pairs import",
    }))
}
