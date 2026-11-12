//! How reading one of Ster's files fails, split by the reason the system
//! gives: nothing there, this account kept out, a directory where a file
//! belongs, bytes that are not text, a path running through a file, and a
//! read the system broke off.

use std::io::ErrorKind;
use std::path::Path;

use wisent_errors::{Cause, Code};

use super::super::refuse;
use super::{blocker, kind, parent, state};

/// The refusal of reading `path`, which holds `role` (`the calibration
/// artifact`, `the pair set`), when the system answered `error`.
pub fn refusal(role: &str, path: &Path, error: &std::io::Error) -> anyhow::Error {
    let cause = match error.kind() {
        ErrorKind::NotFound => &ABSENT,
        ErrorKind::PermissionDenied => &DENIED,
        ErrorKind::IsADirectory => &DIRECTORY,
        ErrorKind::InvalidData => &NOT_TEXT,
        ErrorKind::NotADirectory => &BELOW_FILE,
        _ if std::fs::metadata(path).is_ok_and(|found| found.is_dir()) => &DIRECTORY,
        _ => &FAILED,
    };
    let (through, through_state) = match blocker(path) {
        Some(file) => (file.display().to_string(), state(&file)),
        None => (
            "no part of the path".to_owned(),
            "every part before the last is a directory".to_owned(),
        ),
    };
    refuse(
        cause,
        &[
            ("role", role),
            ("path", &path.display().to_string()),
            ("state", &state(path)),
            ("parent", &parent(path).display().to_string()),
            ("blocker", &through),
            ("blocker_state", &through_state),
            ("kind", &kind(error)),
            ("error", &error.to_string()),
        ],
    )
}

/// The refusal of reading `path` (`role`): the using functionality's own
/// `missing` cause when nothing is there, which names what to give instead
/// and takes `path`, `state`, `parent` and `error`; every other reason is
/// this layer's.
pub fn refusal_or(
    missing: &'static Cause,
    role: &str,
    path: &Path,
    error: &std::io::Error,
) -> anyhow::Error {
    if error.kind() != ErrorKind::NotFound {
        return refusal(role, path, error);
    }
    refuse(
        missing,
        &[
            ("role", role),
            ("path", &path.display().to_string()),
            ("state", &state(path)),
            ("parent", &parent(path).display().to_string()),
            ("error", &error.to_string()),
        ],
    )
}

const FAILED_STEP: &str = "reading {role} {path}";

pub static ABSENT: Cause = Cause {
    id: "file-absent",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "nothing is there: no file has the name {path}",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name {role} where it is; `ls -ln {parent}` shows what that directory holds",
    test: "tests/runtime/generate.sh::a_system_turn_that_cannot_be_read_is_refused",
};

pub static DENIED: Cause = Cause {
    id: "file-read-denied",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "this account may not read it: the file's permission bits or owner keep this account out",
    observed: "path: {path} ({state}); the system answered: {error}",
    remedy: "give this account read access to {path} (`chmod u+r {path}` as its owner), or name a copy it may read",
    test: "tests/pairs/file.sh::a_pair_set_this_account_may_not_read_is_refused",
};

pub static DIRECTORY: Cause = Cause {
    id: "file-is-directory",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "{path} is a directory, and {role} is one file",
    observed: "path: {path} ({state}); error kind {kind}; the system answered: {error}",
    remedy: "name the file that holds {role}, not a directory; `ls -ln {path}` shows what the directory holds",
    test: "tests/pairs/file.sh::a_directory_named_as_a_pair_set_is_refused",
};

pub static NOT_TEXT: Cause = Cause {
    id: "file-not-text",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "it holds bytes that are not UTF-8, and {role} is read as UTF-8 text",
    observed: "path: {path} ({state}); the system answered: {error}",
    remedy: "convert {path} to UTF-8 (`iconv -t UTF-8`), or name {role} as it was first written",
    test: "tests/pairs/import.sh::an_export_that_is_not_text_is_refused",
};

pub static BELOW_FILE: Cause = Cause {
    id: "file-below-file",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the path runs through a file: {blocker} is not a directory, so nothing can be found below it",
    observed: "path: {path}; {blocker}: {blocker_state}; the system answered: {error}",
    remedy: "name {role} by a path that runs through directories only; {blocker} is a file",
    test: "tests/pairs/file.sh::a_pair_set_below_a_file_is_refused",
};

pub static FAILED: Cause = Cause {
    id: "file-read-broken",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the system broke off the read although this account may read the file: it was removed or replaced while Ster read it, or the disk failed",
    observed: "path: {path} ({state}); error kind {kind}; the system answered: {error}",
    remedy: "see what is at {path} with `ls -ln {path}`, and run the command again once the file is whole",
    test: "tests/pairs/file.sh::a_read_the_system_breaks_off_is_refused",
};
