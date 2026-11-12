//! How writing, creating, linking or removing one of Ster's files fails,
//! split by the reason the system gives: this account kept out of the
//! directory, a file system mounted read-only, a full disk, a path running
//! through a file, a directory where the file belongs, and a change the
//! system broke off.

use std::io::ErrorKind;
use std::path::Path;

use wisent_errors::{Cause, Code};

use super::super::refuse;
use super::{blocker, kind, parent, state};

/// The refusal of `action` (`writing`, `creating the directory of`,
/// `removing`, `linking into place`) on `path`, which holds `role`, when the
/// system answered `error`. `left` says what the refused step left as it
/// was (`the previous artifact is unchanged`).
pub fn refusal(
    action: &str,
    role: &str,
    path: &Path,
    left: &str,
    error: &std::io::Error,
) -> anyhow::Error {
    let through = blocker(path);
    let cause = match error.kind() {
        ErrorKind::PermissionDenied => &DENIED,
        ErrorKind::ReadOnlyFilesystem => &READ_ONLY,
        ErrorKind::StorageFull | ErrorKind::FileTooLarge => &FULL,
        ErrorKind::NotADirectory => &BELOW_FILE,
        ErrorKind::AlreadyExists | ErrorKind::NotFound if through.is_some() => &BELOW_FILE,
        ErrorKind::IsADirectory => &DIRECTORY,
        _ => &FAILED,
    };
    let directory = parent(path);
    let (blocking, blocking_state) = match through {
        Some(file) => (file.display().to_string(), state(&file)),
        None => (
            "no part of the path".to_owned(),
            "every part before the last is a directory or absent".to_owned(),
        ),
    };
    refuse(
        cause,
        &[
            ("action", action),
            ("role", role),
            ("left", left),
            ("path", &path.display().to_string()),
            ("state", &state(path)),
            ("directory", &directory.display().to_string()),
            ("directory_state", &state(&directory)),
            ("blocker", &blocking),
            ("blocker_state", &blocking_state),
            ("kind", &kind(error)),
            ("error", &error.to_string()),
        ],
    )
}

const FAILED_STEP: &str = "{action} {role} {path}";

pub static DENIED: Cause = Cause {
    id: "file-write-denied",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "this account may not change {directory} or {path}: their permission bits or owner keep this account out",
    observed: "path: {path} ({state}); directory {directory} ({directory_state}); {left}; the system answered: {error}",
    remedy: "give this account write access to {directory} (`chmod u+w {directory}` as its owner), or name a path in a directory it may write",
    test: "tests/pairs/imports/file.sh::a_set_in_a_read_only_directory_is_refused",
};

pub static READ_ONLY: Cause = Cause {
    id: "file-system-read-only",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the file system holding {directory} is mounted read-only, so nothing on it can change",
    observed: "path: {path}; directory {directory} ({directory_state}); {left}; the system answered: {error}",
    remedy: "name a path on a file system mounted for writing (`mount` lists which are read-only)",
    test: "tests/pairs/imports/file.sh::a_set_on_a_read_only_file_system_is_refused",
};

pub static FULL: Cause = Cause {
    id: "file-disk-full",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the disk holding {directory} has no room left for {path}, or the file passes the largest size it allows",
    observed: "path: {path} ({state}); directory {directory}; {left}; the system answered: {error}",
    remedy: "free space on the disk holding {directory} (`df -h {directory}` shows it), then run the command again",
    test: "tests/pairs/imports/file.sh::a_set_on_a_full_disk_is_refused",
};

pub static BELOW_FILE: Cause = Cause {
    id: "file-path-through-file",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the path runs through a file: {blocker} is not a directory, so nothing can be created below it",
    observed: "path: {path}; {blocker}: {blocker_state}; {left}; the system answered: {error}",
    remedy: "name a path that runs through directories only, or move {blocker} out of the way",
    test: "tests/pairs/imports/file.sh::a_set_under_a_file_is_refused",
};

pub static DIRECTORY: Cause = Cause {
    id: "file-target-directory",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "{path} is a directory, and {role} is one file",
    observed: "path: {path} ({state}); {left}; the system answered: {error}",
    remedy: "name a file path for {role}, not the directory {path}",
    test: "tests/pairs/imports/file.sh::a_merge_into_a_directory_is_refused",
};

pub static FAILED: Cause = Cause {
    id: "file-write-broken",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the system broke off the change although this account may write {directory}: the disk failed, or another process changed {path} meanwhile",
    observed: "path: {path} ({state}); directory {directory} ({directory_state}); {left}; error kind {kind}; the system answered: {error}",
    remedy: "see {directory} with `ls -ln {directory}`, and run the command again once nothing else changes it",
    test: "tests/pairs/imports/file.sh::a_write_the_system_breaks_off_is_refused",
};
