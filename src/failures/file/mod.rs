//! How reading and writing one of Ster's files fails, whichever file it is
//! (`ster.file`, the layer every functionality that reads or writes a
//! document uses), and what a refusal says about a path: its state as the
//! system shows it now, which every cause naming a path puts in its observed
//! text so the reader sees what was there without looking.
//!
//! A file that is absent is the using functionality's own cause, since what
//! to name instead depends on what the file is; every other reason a read, a
//! JSON document or a write fails is one cause here, split by the reason.

use std::io;
use std::path::{Path, PathBuf};

use wisent_errors::Operation;

pub mod json;
pub mod read;
pub mod tensors;
pub mod write;

/// docs/guide/errors/file.md, where this layer is listed.
const DOCS: &str = page!("file");

/// Reading a file, a JSON document and writing a file, whichever role the
/// file plays.
pub static FILE: Operation = Operation {
    id: "ster.file",
    entries: &[],
    uses: &[],
    summary: "read one of Ster's files, JSON documents or safetensors files, or write, link or remove one, whichever role it plays",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &read::ABSENT,
        &read::DENIED,
        &read::DIRECTORY,
        &read::NOT_TEXT,
        &read::BELOW_FILE,
        &read::FAILED,
        &json::MALFORMED,
        &json::TRUNCATED,
        &json::SHAPE,
        &tensors::MALFORMED,
        &tensors::DTYPE,
        &tensors::UNPLACED,
        &tensors::UNSERIALIZED,
        &write::DENIED,
        &write::READ_ONLY,
        &write::FULL,
        &write::BELOW_FILE,
        &write::DIRECTORY,
        &write::FAILED,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&FILE];

/// The directory `path` sits in, as the system resolves a relative name.
pub fn parent(path: &Path) -> PathBuf {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// The nearest part of `path`, itself included, that exists and is not a
/// directory: the file a path runs through, so nothing below it can be
/// found or created, or the file standing where a directory is to be made.
pub fn blocker(path: &Path) -> Option<PathBuf> {
    path.ancestors()
        .filter(|ancestor| !ancestor.as_os_str().is_empty())
        .find(|ancestor| std::fs::metadata(ancestor).is_ok_and(|found| !found.is_dir()))
        .map(Path::to_path_buf)
}

/// `path`'s state: `absent`, or what it is (file, directory, link), its
/// size, its permission bits and its owner's user id, as `ls -ln` shows
/// them; a link is followed, and one pointing nowhere says so.
pub fn state(path: &Path) -> String {
    match std::fs::metadata(path) {
        Ok(metadata) => described(&metadata),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            match std::fs::symlink_metadata(path) {
                Ok(_) => "a link pointing to nothing".to_owned(),
                Err(_) => "absent".to_owned(),
            }
        }
        Err(error) => format!("not inspectable ({})", kind(&error)),
    }
}

#[cfg(unix)]
fn described(metadata: &std::fs::Metadata) -> String {
    use std::os::unix::fs::MetadataExt;
    format!(
        "{}, {} bytes, mode {:03o}, owner uid {}",
        what(metadata),
        metadata.len(),
        metadata.mode() & 0o777,
        metadata.uid(),
    )
}

#[cfg(not(unix))]
fn described(metadata: &std::fs::Metadata) -> String {
    format!(
        "{}, {} bytes, read-only: {}",
        what(metadata),
        metadata.len(),
        metadata.permissions().readonly()
    )
}

fn what(metadata: &std::fs::Metadata) -> &'static str {
    if metadata.is_dir() {
        "a directory"
    } else if metadata.is_file() {
        "a file"
    } else {
        special(metadata)
    }
}

/// What a path that is neither a file nor a directory is: a pipe, a device
/// or a socket, as `ls -l` tells them apart.
#[cfg(unix)]
fn special(metadata: &std::fs::Metadata) -> &'static str {
    use std::os::unix::fs::FileTypeExt;
    let kind = metadata.file_type();
    if kind.is_fifo() {
        "a pipe"
    } else if kind.is_char_device() {
        "a character device"
    } else if kind.is_block_device() {
        "a block device"
    } else if kind.is_socket() {
        "a socket"
    } else {
        "neither a file nor a directory"
    }
}

#[cfg(not(unix))]
fn special(_metadata: &std::fs::Metadata) -> &'static str {
    "neither a file nor a directory"
}

/// Whether this process's standard output reaches no reader: it was closed
/// (Rust's runtime then gives it the null device, or a write finds no
/// descriptor) or it is the null device itself. `Some` holds what the
/// stream is, for the refusal's observed text.
#[cfg(unix)]
pub fn stdout_discarded() -> Option<String> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    match (std::fs::metadata("/dev/stdout"), std::fs::metadata("/dev/null")) {
        (Err(error), _) => Some(format!("closed (the system answered {})", kind(&error))),
        (Ok(stdout), Ok(null))
            if stdout.file_type().is_char_device() && stdout.rdev() == null.rdev() =>
        {
            Some(format!("the null device /dev/null (device {})", stdout.rdev()))
        }
        // Any other stream, or a system without /dev/null to compare with:
        // a reader may be there, and a write that finds none is refused
        // where it happens (event-unwritten).
        _ => None,
    }
}

#[cfg(not(unix))]
pub fn stdout_discarded() -> Option<String> {
    None
}

/// The system's name for `error`'s kind, as `std::io::ErrorKind` spells it.
pub fn kind(error: &io::Error) -> String {
    format!("{:?}", error.kind())
}
