//! The model checkpoints Ster loads, as sources: a Transformers checkpoint directory
//! (IMP-STER-12), a llama.cpp GGUF file beside its base model's config and tokenizer (13)
//! and the Hugging Face hub cache (16). Each is read by `--model` of every command that
//! loads a model, which takes a local directory or a Hub id, reusing the files the cache
//! already holds for that id. `discover` resolves a directory as `--model` does, reading
//! its files' names only, or lists the models a hub cache keeps; nothing is written.

use std::path::Path;

use serde_json::{Value, json};
use ster::runtime::{Checkpoint, Layout};

pub(super) const MODELS: &[(&str, &str)] = &[
    ("IMP-STER-12", "a Transformers checkpoint directory (config.json, its tokenizer, *.safetensors)"),
    ("IMP-STER-13", "a llama.cpp GGUF file beside its base model's config.json and tokenizer"),
    ("IMP-STER-16", "the models the Hugging Face hub cache keeps (~/.cache/huggingface/hub)"),
];

const CACHE: &str = "IMP-STER-16";

pub(super) fn row(&(id, source): &(&str, &str)) -> Value {
    let read_by = if id == CACHE {
        "--model <org/name> of any command that loads a model, reusing the files the cache holds"
    } else {
        "--model <directory> of any command that loads a model"
    };
    json!({"adapter": id, "source": source, "kind": "F", "read_by": read_by, "found": false})
}

/// The Hub ids a hub cache directory keeps (`models--<org>--<name>`) whose snapshot holds
/// a `config.json`.
fn cached_models(cache: &Path) -> Result<Vec<String>, String> {
    let entries = std::fs::read_dir(cache).map_err(|error| format!("{}: {error}", cache.display()))?;
    let mut models = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(repo) = name.strip_prefix("models--") else { continue };
        let snapshots = entry.path().join("snapshots");
        let complete = std::fs::read_dir(&snapshots)
            .map(|revisions| revisions.flatten().any(|revision| revision.path().join("config.json").is_file()))
            .unwrap_or(false);
        if complete {
            models.push(repo.replacen("--", "/", 1));
        }
    }
    models.sort();
    Ok(models)
}

pub(super) fn discover(found: &(&str, &str), path: &Path) -> Value {
    let mut answer = row(found);
    answer["path"] = json!(path.display().to_string());
    if found.0 == CACHE {
        match cached_models(path) {
            Ok(models) => {
                answer["found"] = json!(!models.is_empty());
                answer["models"] = json!(models);
            }
            Err(why) => answer["why_not"] = json!(why),
        }
        return answer;
    }
    // Only a directory is resolved: any other name would be asked of the Hub.
    if !path.is_dir() {
        answer["why_not"] = json!("it is not a directory");
        return answer;
    }
    let shown = path.display().to_string();
    match Checkpoint::resolve(&shown, None) {
        Ok(checkpoint) => {
            let wanted = if found.0 == "IMP-STER-13" { Layout::Gguf } else { Layout::Transformers };
            let is_wanted = checkpoint.layout == wanted || (wanted == Layout::Transformers && checkpoint.layout == Layout::Mistral);
            if is_wanted {
                answer["found"] = json!(true);
                answer["weights"] = json!(checkpoint.weights.len());
            } else {
                answer["why_not"] = json!(format!("it is a {:?} checkpoint, not this adapter's", checkpoint.layout));
            }
        }
        Err(error) => answer["why_not"] = json!(format!("{error:#}")),
    }
    answer
}
