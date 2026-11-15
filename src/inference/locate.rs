//! Where Ster finds the `stado` program it hands completions to: the order
//! Stado states for every program that runs it (`stado --help`, the README's
//! "Where another program finds Stado"), followed by `stado_program::find`:
//!
//! 1. the path in `STADO_BIN`, when it is set;
//! 2. else `stado` on `PATH`;
//! 3. else `${STADO_BIN_DIR:-$HOME/.stado/bin}/stado`, where install-stado.sh
//!    and `stado product install stado --surface cli` place it.
//!
//! An empty variable counts as unset. How a lookup or a start fails is
//! `ster.inference`'s (src/failures/inference.rs).

use std::path::{Path, PathBuf};

use stado_program::{FoundBy, Unfound};

use crate::failures::{file, inference as causes, refuse};

/// The `stado` found, and which of Stado's places it was found through.
pub(super) struct Found {
    pub path: PathBuf,
    pub through: &'static str,
}

/// A place of Stado's order, as a refusal names it.
fn through(by: FoundBy) -> &'static str {
    match by {
        FoundBy::StadoBin => "STADO_BIN",
        FoundBy::Path => "PATH",
        FoundBy::StadoBinDir => "STADO_BIN_DIR",
        FoundBy::Home => "$HOME/.stado/bin",
    }
}

/// The value of `name` as a refusal shows it, `unset` when it has none.
fn shown(name: &str) -> String {
    match std::env::var_os(name) {
        Some(value) if value.is_empty() => "empty, which counts as unset".to_owned(),
        Some(value) => value.to_string_lossy().into_owned(),
        None => "unset".to_owned(),
    }
}

/// The `stado` Stado's stated order finds. A `STADO_BIN` naming no file is
/// refused rather than passed over: it is the one place a caller names.
pub(super) fn program() -> anyhow::Result<Found> {
    match stado_program::find() {
        Ok(found) if found.by == FoundBy::StadoBin && !found.path.is_file() => Err(refuse(
            &causes::STADO_BIN_MISSING,
            &[
                ("call", &format!("stado {}", super::CALL)),
                ("stado", &found.path.display().to_string()),
                ("state", &file::state(&found.path)),
            ],
        )),
        Ok(found) => Ok(Found { path: found.path, through: through(found.by) }),
        Err(unfound) => Err(absent(&unfound)),
    }
}

/// The refusal of a `stado` found at none of Stado's three places.
fn absent(unfound: &Unfound) -> anyhow::Error {
    let (place, state) = match &unfound.looked_at {
        Some((path, by)) => (
            format!("{} (from {})", path.display(), through(*by)),
            file::state(path),
        ),
        None => (
            "${STADO_BIN_DIR:-$HOME/.stado/bin}/stado".to_owned(),
            "not looked at: STADO_BIN_DIR and HOME are both unset".to_owned(),
        ),
    };
    refuse(
        &causes::STADO_ABSENT,
        &[
            ("call", &format!("stado {}", super::CALL)),
            ("program", &shown("STADO_BIN")),
            ("path", &shown("PATH")),
            ("place", &place),
            ("state", &state),
        ],
    )
}

/// The values both start refusals name: the call, the `stado` found, the
/// place it was found through, its state and what the system answered.
fn started(found: &Found, call: &str, error: &std::io::Error) -> Vec<(&'static str, String)> {
    vec![
        ("call", call.to_owned()),
        ("stado", found.path.display().to_string()),
        ("through", found.through.to_owned()),
        ("state", file::state(&found.path)),
        ("kind", format!("{:?}", error.kind())),
        ("error", error.to_string()),
    ]
}

/// The refusal of the `stado` found, which this account may not execute.
pub(super) fn not_executable(found: &Found, call: &str, error: &std::io::Error) -> anyhow::Error {
    raised(&causes::STADO_NOT_EXECUTABLE, &started(found, call, error))
}

/// The refusal of the `stado` found, which the system says is not there
/// when it is started: it was removed after it was found, or it is a script
/// whose interpreter does not exist.
pub(super) fn unlaunchable(found: &Found, call: &str, error: &std::io::Error) -> anyhow::Error {
    let mut values = started(found, call, error);
    values.push(("interpreter", interpreter(&found.path)));
    raised(&causes::STADO_UNLAUNCHABLE, &values)
}

fn raised(cause: &'static wisent_errors::Cause, values: &[(&'static str, String)]) -> anyhow::Error {
    let values: Vec<(&str, &str)> = values.iter().map(|(name, value)| (*name, value.as_str())).collect();
    refuse(cause, &values)
}

/// How much of a script's first line a refusal reads to name its
/// interpreter: what Linux reads of it to start one, BINPRM_BUF_SIZE
/// (include/uapi/linux/binfmts.h, 256 bytes).
const FIRST_LINE: usize = 256;

/// The interpreter `path` names on its first line when it is a script
/// (`#!…`), for the refusal of a start the system answered "not found"; what
/// else it is otherwise.
fn interpreter(path: &Path) -> String {
    use std::io::Read;
    let mut head: [u8; FIRST_LINE] = [Default::default(); FIRST_LINE];
    let read = std::fs::File::open(path).and_then(|mut opened| opened.read(&mut head));
    let length = match read {
        Ok(length) => length,
        Err(error) => return format!("unknown: its first line could not be read ({:?})", error.kind()),
    };
    let Some(rest) = head[..length].strip_prefix(b"#!") else {
        return "none: it is not a script".to_owned();
    };
    let line = match rest.iter().position(|byte| *byte == b'\n') {
        Some(end) => &rest[..end],
        None => rest,
    };
    format!("#!{}", String::from_utf8_lossy(line).trim())
}
