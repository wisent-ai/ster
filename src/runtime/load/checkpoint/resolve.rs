//! Resolving a checkpoint's files from a local directory or the Hugging Face
//! Hub.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use hf_hub::{
    Repo, RepoType,
    api::sync::{Api, ApiRepo},
};

use crate::{
    chat,
    failures::{
        file,
        model::{checkpoint as causes, download_refusal, repository_refusal},
        refuse,
    },
};

use super::{Checkpoint, supported};
use crate::runtime::load::{config, mistral::Layout};

mod files;

use files::require_files;

impl Checkpoint {
    /// Resolves `model` from a local directory or the Hugging Face Hub.
    pub fn resolve(model: &str, revision: Option<&str>) -> Result<Self> {
        let local = Path::new(model);
        if !local.is_dir() && names_a_path(model) {
            let state = if local.exists() {
                "it is not a directory"
            } else {
                "nothing is there"
            };
            return Err(refuse(
                &causes::DIRECTORY_MISSING,
                &[("model", model), ("state", state)],
            ));
        }
        if local.is_dir() {
            let layout =
                if !local.join("config.json").is_file() && local.join("params.json").is_file() {
                    Layout::Mistral
                } else if local_safetensors(local, Layout::Transformers)?.is_empty()
                    && !local_safetensors(local, Layout::Gguf)?.is_empty()
                {
                    // A llama.cpp fine-tune beside its base model's config and
                    // tokenizer: no safetensors, one GGUF file (more than one is
                    // refused with their count when the weights are mapped).
                    Layout::Gguf
                } else {
                    Layout::Transformers
                };
            let config = local.join(layout.config_file());
            let tokenizer = local.join(tokenizer_file(
                |name| local.join(name).is_file(),
                &local.display().to_string(),
            )?);
            let weights = local_safetensors(local, layout)?;
            require_files(
                &config,
                &tokenizer,
                &weights,
                &local.display().to_string(),
                layout,
            )?;
            let (tokenizer_config, chat_template) = chat::local_files(local);
            return Ok(Self {
                config,
                tokenizer,
                weights,
                revision: revision.map(str::to_owned),
                tokenizer_config,
                chat_template,
                layout,
            });
        }
        let api = Api::new().map_err(|error| {
            let hf_home = std::env::var_os("HF_HOME").map_or_else(
                || "unset, so the cache is ~/.cache/huggingface".to_owned(),
                |home| home.to_string_lossy().into_owned(),
            );
            refuse(
                &causes::HUB_CLIENT,
                &[
                    ("model", model),
                    ("error", &error.to_string()),
                    ("hf_home", &hf_home),
                ],
            )
        })?;
        // Without a stated revision the Hub serves its default branch,
        // which `Repo::model` names.
        let repo = match revision {
            Some(revision) => {
                Repo::with_revision(model.to_owned(), RepoType::Model, revision.to_owned())
            }
            None => Repo::model(model.to_owned()),
        };
        let remote = api.repo(repo);
        let listing = remote.info_request().url().to_owned();
        let info = stado_wait::blocking(
            stado_wait::Kind::Network,
            format!("file list of {model}"),
            HUB,
            || remote.info(),
        )
        .map_err(|error| repository_refusal(model, revision, &listing, error.into()))?;
        let tokenizer = download(
            &remote,
            tokenizer_file(|name| published(&info, name), model)?,
            model,
        )?;
        // The two template files are fetched exactly like the three required
        // ones, but only when the repository lists them: `get` on a file a
        // repository does not publish is an error, and a base model not
        // publishing a chat template is not an error.
        let has_tokenizer_config = published(&info, "tokenizer_config.json");
        let has_chat_template = published(&info, "chat_template.jinja");
        let tokenizer_config = has_tokenizer_config
            .then(|| download(&remote, "tokenizer_config.json", model))
            .transpose()?;
        let chat_template = has_chat_template
            .then(|| download(&remote, "chat_template.jinja", model))
            .transpose()?;
        // A repository that publishes Mistral's own format alone (Mistral
        // Large 3) is read from `params.json` and its consolidated weights.
        let layout = if published(&info, "config.json") || !published(&info, "params.json") {
            Layout::Transformers
        } else {
            Layout::Mistral
        };
        let config = download(&remote, layout.config_file(), model)?;
        // The architecture and quantization refusals come before the
        // weights: a checkpoint Ster cannot run is refused for the price of
        // its config, not of every shard.
        supported(&config::read(&config, layout)?.0, &config)?;
        let weight_names: Vec<String> = info
            .siblings
            .into_iter()
            .map(|file| file.rfilename)
            .filter(|name| layout.owns(name))
            .collect();
        if weight_names.is_empty() {
            return Err(refuse(
                &causes::WEIGHTS_ABSENT,
                &[
                    ("place", &format!("model {model}")),
                    ("kind", layout.weights_kind()),
                    ("layout", layout.name()),
                ],
            ));
        }
        let mut weights = Vec::with_capacity(weight_names.len());
        for name in weight_names {
            weights.push(download(&remote, &name, model)?);
        }
        require_files(&config, &tokenizer, &weights, model, layout)?;
        Ok(Self {
            config,
            tokenizer,
            weights,
            revision: Some(info.sha),
            tokenizer_config,
            chat_template,
            layout,
        })
    }
}

/// Where [`download`] fetches from, as the wait it reports names it.
const HUB: &str = "the Hugging Face Hub";

/// `file` of the Hub repository `model`, fetched (or found in the Hub cache)
/// under the one wait every network call reports through.
fn download(remote: &ApiRepo, file: &str, model: &str) -> Result<PathBuf> {
    stado_wait::blocking(
        stado_wait::Kind::Network,
        format!("{file} of {model}"),
        HUB,
        || remote.get(file),
    )
    .map_err(|error| download_refusal(file, model, &remote.url(file), error.into()))
}

/// Whether `model` names a local path rather than a Hub repository id: a
/// Hub id never starts with `/`, `.` or `~`.
fn names_a_path(model: &str) -> bool {
    model.starts_with('/')
        || model.starts_with("./")
        || model.starts_with("../")
        || model.starts_with("~/")
        || model == "."
        || model == ".."
}

/// Whether the repository lists a file, so an optional one is only fetched
/// when asking for it can succeed.
fn published(info: &hf_hub::api::RepoInfo, name: &str) -> bool {
    info.siblings.iter().any(|file| file.rfilename == name)
}

/// The tokenizer files a checkpoint may publish, in the order they are
/// looked for: Transformers' `tokenizer.json`, PLaMo's `tokenizer.jsonl`, a
/// `tokenizer.model` (GLM-4's tiktoken vocabulary) and Kimi's
/// `tiktoken.model`.
const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer.jsonl",
    "tokenizer.model",
    "tiktoken.model",
];

/// The first of [`TOKENIZER_FILES`] that `exists` finds, or the refusal of
/// a checkpoint at `place` that publishes none of them.
fn tokenizer_file(exists: impl Fn(&str) -> bool, place: &str) -> Result<&'static str> {
    match TOKENIZER_FILES.iter().copied().find(|name| exists(name)) {
        Some(name) => Ok(name),
        None => Err(refuse(
            &causes::TOKENIZER_ABSENT,
            &[("place", place), ("files", &TOKENIZER_FILES.join(", "))],
        )),
    }
}

fn local_safetensors(root: &Path, layout: Layout) -> Result<Vec<PathBuf>> {
    let unlistable = |error: std::io::Error| {
        if error.kind() == std::io::ErrorKind::PermissionDenied {
            return file::read::refusal("the checkpoint directory", root, &error);
        }
        refuse(
            &causes::DIRECTORY_UNLISTABLE,
            &[
                ("path", &root.display().to_string()),
                ("state", &file::state(root)),
                ("kind", &file::kind(&error)),
                ("error", &error.to_string()),
            ],
        )
    };
    let mut weights = Vec::new();
    for entry in fs::read_dir(root).map_err(unlistable)? {
        let path = entry.map_err(unlistable)?.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| layout.owns(name))
        {
            weights.push(path);
        }
    }
    weights.sort();
    Ok(weights)
}
