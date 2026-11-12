//! What a resolved checkpoint must hold before it is loaded: its config, its
//! tokenizer and at least one weight file.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::failures::{model::checkpoint as causes, refuse};
use crate::runtime::load::mistral::Layout;

/// Refuses a checkpoint `place` resolved without its config, tokenizer or
/// weights.
pub(super) fn require_files(
    config: &Path,
    tokenizer: &Path,
    weights: &[PathBuf],
    place: &str,
    layout: Layout,
) -> Result<()> {
    if !config.is_file() {
        let file = config.file_name().map_or_else(
            || config.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        return Err(refuse(
            &causes::CONFIG_ABSENT,
            &[
                ("place", place),
                ("file", &file),
                ("path", &config.display().to_string()),
            ],
        ));
    }
    if !tokenizer.is_file() {
        return Err(refuse(
            &causes::TOKENIZER_ABSENT,
            &[
                ("place", place),
                ("files", &tokenizer.display().to_string()),
            ],
        ));
    }
    if weights.is_empty() {
        return Err(refuse(
            &causes::WEIGHTS_ABSENT,
            &[
                ("place", place),
                ("kind", layout.weights_kind()),
                ("layout", layout.name()),
            ],
        ));
    }
    Ok(())
}
