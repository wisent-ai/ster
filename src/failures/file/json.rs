//! How reading one of Ster's JSON documents fails once its bytes are read,
//! split by where the JSON reader stopped: not JSON, cut short, or JSON of
//! another form than the document.

use std::path::Path;

use serde_json::error::Category;
use wisent_errors::{Cause, Code};

use super::super::refuse;
use super::state;

/// The refusal of `path`, which should hold `role` in its `form` (what it
/// is and who writes it: "the JSON `ster vector train` writes"), when the
/// JSON reader stopped with `error`. A reader over a stream that broke is
/// the read's own refusal ([`super::read::refusal`]).
pub fn refusal(role: &str, form: &str, path: &Path, error: &serde_json::Error) -> anyhow::Error {
    let cause = match error.classify() {
        Category::Syntax => &MALFORMED,
        Category::Eof => &TRUNCATED,
        Category::Data => &SHAPE,
        Category::Io => {
            let broken = std::io::Error::new(
                error.io_error_kind().unwrap_or(std::io::ErrorKind::Other),
                error.to_string(),
            );
            return super::read::refusal(role, path, &broken);
        }
    };
    refuse(
        cause,
        &[
            ("role", role),
            ("form", form),
            ("path", &path.display().to_string()),
            ("state", &state(path)),
            ("line", &error.line().to_string()),
            ("column", &error.column().to_string()),
            ("error", &error.to_string()),
        ],
    )
}

const FAILED_STEP: &str = "reading {role} {path}";

pub static MALFORMED: Cause = Cause {
    id: "document-not-json",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "it is not JSON, and {role} is {form}: it is another kind of file, or one broken by hand",
    observed: "path: {path} ({state}); the JSON reader stopped at line {line}, column {column}: {error}",
    remedy: "name {role}, which is {form}, or repair the JSON of {path} at line {line}, column {column}",
    test: "tests/pairs/file.sh::a_pair_set_that_is_not_json_is_refused",
};

pub static TRUNCATED: Cause = Cause {
    id: "document-truncated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway)",
    observed: "path: {path} ({state}); the JSON reader ran out at line {line}, column {column}: {error}",
    remedy: "write {role} again; it is {form}",
    test: "tests/pairs/file.sh::a_pair_set_cut_short_is_refused",
};

pub static SHAPE: Cause = Cause {
    id: "document-shape",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "it is JSON of another form than {role}, which is {form}: a field it requires is missing, a field holds another type or value, or it is a document of another kind",
    observed: "path: {path} ({state}); at line {line}, column {column} the reader answered: {error}",
    remedy: "name {role}, which is {form}; the reader's answer names the field that differs",
    test: "tests/pairs/file.sh::a_file_that_is_not_a_pair_set_is_refused",
};
