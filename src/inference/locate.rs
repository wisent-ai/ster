//! Where Ster finds the `stado` program it hands completions to, in the order
//! Stado states for every program that runs it (`stado --help`, the README's
//! "Where another program finds Stado", stado-rs/src/binary/location.rs at
//! https://github.com/wisent-ai/stado):
//!
//! 1. the path in `STADO_BIN`, when it is set;
//! 2. else `stado` on `PATH`;
//! 3. else `${STADO_BIN_DIR:-$HOME/.stado/bin}/stado`, where install-stado.sh
//!    and `stado product install stado --surface cli` place it.
//!
//! An empty variable counts as unset, as Stado says of `STADO_BIN_DIR`. How a
//! lookup or a start fails is `ster.inference`'s (src/failures/inference.rs).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::failures::{file, inference as causes, refuse};

/// The variable holding the path of Stado's program (Stado's place 1).
const PROGRAM: &str = "STADO_BIN";
/// The directory install-stado.sh installs into when it is set (place 3).
const BIN_DIR: &str = "STADO_BIN_DIR";
/// Where Stado installs under `HOME` when `STADO_BIN_DIR` is unset (place 3).
const INSTALLED: &str = ".stado/bin";

/// The `stado` found, and which of Stado's places it was found through.
pub(super) struct Found {
    pub path: PathBuf,
    pub through: &'static str,
}

/// The value of `name`, or `None` when it is unset or empty.
fn set(name: &str) -> Option<OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

/// The value of `name` as a refusal shows it, `unset` when it has none.
fn shown(name: &str) -> String {
    match std::env::var_os(name) {
        Some(value) if value.is_empty() => "empty, which counts as unset".to_owned(),
        Some(value) => value.to_string_lossy().into_owned(),
        None => "unset".to_owned(),
    }
}

/// Stado's place 3, with the place it came from, or `None` when neither
/// `STADO_BIN_DIR` nor `HOME` is set.
fn installed() -> Option<(PathBuf, &'static str)> {
    if let Some(directory) = set(BIN_DIR) {
        return Some((PathBuf::from(directory).join("stado"), "STADO_BIN_DIR"));
    }
    set("HOME").map(|home| (PathBuf::from(home).join(INSTALLED).join("stado"), "$HOME/.stado/bin"))
}

/// The `stado` Stado's stated order finds: `STADO_BIN`, else `PATH`, else
/// `${STADO_BIN_DIR:-$HOME/.stado/bin}/stado`. A `STADO_BIN` naming no file
/// is refused rather than passed over: it is the one place a caller names.
pub(super) fn program() -> anyhow::Result<Found> {
    if let Some(named) = set(PROGRAM) {
        let path = PathBuf::from(named);
        if path.is_file() {
            return Ok(Found { path, through: PROGRAM });
        }
        return Err(refuse(
            &causes::STADO_BIN_MISSING,
            &[
                ("call", &format!("stado {}", super::CALL)),
                ("stado", &path.display().to_string()),
                ("state", &file::state(&path)),
            ],
        ));
    }
    let on_path = set("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|directory| directory.join("stado"))
            .find(|candidate| candidate.is_file())
    });
    if let Some(path) = on_path {
        return Ok(Found { path, through: "PATH" });
    }
    match installed() {
        Some((path, through)) if path.is_file() => Ok(Found { path, through }),
        _ => Err(absent()),
    }
}

/// The refusal of a `stado` found at none of Stado's three places.
fn absent() -> anyhow::Error {
    let (place, state) = match installed() {
        Some((path, through)) => (
            format!("{} (from {through})", path.display()),
            file::state(&path),
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
            ("program", &shown(PROGRAM)),
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
    let mut head = [0u8; FIRST_LINE];
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
