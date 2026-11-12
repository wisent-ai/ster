//! The id a pair set takes in the workspace: the name given with --name, or
//! one derived from its file name.

use std::path::Path;

use anyhow::Result;

use crate::failures::{refuse, workspace::import};

/// A workspace name is a file name: it starts with an ASCII letter or digit and
/// holds only letters, digits, dots, underscores and hyphens. How long it may be
/// is the filesystem's to say, and it says so when the set is written.
pub(super) fn validate_name(name: &str) -> Result<()> {
    let valid = name
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphanumeric)
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(refuse(&import::NAME_INVALID, &[("name", name)]))
    }
}

/// The id a set imported without --name takes: its file name's letters and
/// digits, lowercased, each run of anything else one hyphen. A name that
/// holds none, or is not UTF-8, gives no id, and the refusal names the file
/// and the flag that supplies one.
pub(super) fn derived_name(source: &Path) -> Result<String> {
    let file = source.file_name().map_or_else(
        || "none".to_owned(),
        |file| file.to_string_lossy().into_owned(),
    );
    let refused = |cause: &'static wisent_errors::Cause| {
        refuse(
            cause,
            &[("source", &source.display().to_string()), ("file", &file)],
        )
    };
    let stem = source
        .file_stem()
        .ok_or_else(|| refused(&import::NAME_FILELESS))?
        .to_str()
        .ok_or_else(|| refused(&import::NAME_NOT_UTF8))?;
    let mut name = String::new();
    let mut separator = false;
    for character in stem.chars() {
        if character.is_ascii_alphanumeric() {
            if separator && !name.is_empty() {
                name.push('-');
            }
            separator = false;
            name.push(character.to_ascii_lowercase());
        } else {
            separator = true;
        }
    }
    if name.is_empty() {
        return Err(refused(&import::NAME_UNDERIVABLE));
    }
    Ok(name)
}
