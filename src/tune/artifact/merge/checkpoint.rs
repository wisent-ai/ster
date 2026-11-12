//! Reading a checkpoint's tensors and writing them back as an ordinary
//! checkpoint directory: what a merge and an ablation both rewrite.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::failures::{file, refuse, tune::rewrite::TENSOR_TWICE};
use crate::{runtime::Checkpoint, workflow};

/// Every tensor of `source`'s weight files, by name; a name stored in two
/// shards is refused rather than letting the last shard win.
pub(crate) fn read_tensors(
    source: &Checkpoint,
    model: &str,
    device: &Device,
) -> Result<BTreeMap<String, Tensor>> {
    workflow::progress(format!(
        "reading {} weight file(s) from {model}",
        source.weights.len()
    ));
    let mut tensors: BTreeMap<String, Tensor> = BTreeMap::new();
    for shard in &source.weights {
        let path = shard.display().to_string();
        let loaded = candle_core::safetensors::load(shard, device).map_err(|error| {
            file::tensors::read_refusal(&format!("a weight file of {model}"), shard, &error)
        })?;
        for (name, tensor) in loaded {
            if tensors.insert(name.clone(), tensor).is_some() {
                // Sharded checkpoints partition the tensors; the same name in
                // two shards means the shards disagree, and silently keeping
                // the last one merges half a model.
                return Err(refuse(
                    &TENSOR_TWICE,
                    &[("model", model), ("name", &name), ("path", &path)],
                ));
            }
        }
    }
    Ok(tensors)
}

/// What [`write_checkpoint`] wrote.
pub(crate) struct Written {
    pub(crate) files: Vec<String>,
    pub(crate) parameters: usize,
    pub(crate) total: usize,
}

/// What a write that failed under `output` left there.
fn left(output: &Path) -> String {
    format!(
        "{} holds only the files written before this one, so it is no checkpoint yet",
        output.display()
    )
}

/// The refusal of creating `output` or writing `path` under it.
pub(crate) fn unwritable(output: &Path, path: &Path, error: &std::io::Error) -> anyhow::Error {
    if path == output {
        file::write::refusal("creating", "the output directory", path, &left(output), error)
    } else {
        file::write::refusal("writing", "the output file", path, &left(output), error)
    }
}

/// The refusal of writing the tensors file `path` under `output`.
pub(crate) fn tensors_unwritable(
    output: &Path,
    path: &Path,
    error: &candle_core::Error,
) -> anyhow::Error {
    file::tensors::write_refusal("the output weights", path, &left(output), error)
}

/// Copies the source's `from` to `to` under `output`: read whole, then
/// written, so a refusal says which of the two failed.
pub(crate) fn copy(output: &Path, from: &Path, to: &Path) -> Result<()> {
    let bytes = fs::read(from)
        .map_err(|error| file::read::refusal("the source checkpoint's file", from, &error))?;
    fs::write(to, bytes).map_err(|error| unwritable(output, to, &error))
}

/// `tensors` written as an ordinary checkpoint directory at `output`, beside
/// copies of `source`'s own config, tokenizer and chat template.
pub(crate) fn write_checkpoint(
    source: &Checkpoint,
    output: &Path,
    tensors: &BTreeMap<String, Tensor>,
) -> Result<Written> {
    fs::create_dir_all(output).map_err(|error| unwritable(output, output, &error))?;
    let weights_file = source.layout.merged_weights();
    let weights_path = output.join(weights_file);
    let flat: HashMap<String, Tensor> = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect();
    let parameters: usize = tensors.values().map(|tensor| tensor.elem_count()).sum();
    let total = tensors.len();
    candle_core::safetensors::save(&flat, &weights_path)
        .map_err(|error| tensors_unwritable(output, &weights_path, &error))?;

    // Copied rather than regenerated. The tokenizer and the config are the
    // source's own, and a rewritten checkpoint that tokenized differently from
    // the model it came from would be a different model wearing its name.
    //
    // `tokenizer_config.json` and `chat_template.jinja` are part of that
    // statement and not extras. The chat template lives in one or the other,
    // and an adapter trained with `--chat-template auto` was trained on the
    // markers that template renders — so a merge that dropped it produced a
    // directory whose own `tune evaluate` silently scored raw text and
    // disagreed with the same adapter attached to the source checkpoint. Both
    // are optional because plenty of checkpoints publish neither; a base model
    // with no template merges to a directory with no template, which is the
    // same statement in the other direction.
    let mut files = vec![weights_file.to_owned()];
    let optional = [
        (source.tokenizer_config.as_deref(), "tokenizer_config.json"),
        (source.chat_template.as_deref(), "chat_template.jinja"),
    ];
    // `Checkpoint::resolve`, the one place a `Checkpoint` is built, sets
    // `tokenizer` to `local.join(tokenizer_file(..)?)` or to the Hub's
    // `remote.get(file)` of that same `tokenizer_file(..)?`, a snapshot path
    // ending in the file's name; `tokenizer_file` answers one of the ASCII
    // names in `TOKENIZER_FILES`.
    let tokenizer_leaf = source
        .tokenizer
        .file_name()
        .and_then(|name| name.to_str())
        .expect("a checkpoint's tokenizer is a named file");
    let required = [
        (&source.config, source.layout.config_file()),
        (&source.tokenizer, tokenizer_leaf),
    ];
    for (from, leaf) in required
        .into_iter()
        .map(|(from, leaf)| (from.as_path(), leaf))
        .chain(
            optional
                .into_iter()
                .filter_map(|(from, leaf)| from.map(|from| (from, leaf))),
        )
    {
        copy(output, from, &output.join(leaf))?;
        files.push(leaf.to_owned());
    }
    workflow::progress(format!(
        "wrote {} to {}",
        files.join(", "),
        output.display()
    ));
    Ok(Written {
        files,
        parameters,
        total,
    })
}
