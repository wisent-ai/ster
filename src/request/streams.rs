//! The two streams a request is carried on: standard output, where its events
//! go and which must reach a reader before anything runs, and standard input,
//! which holds its body.

use std::io::{self, Read};
use std::path::Path;

use ster::failures::{file, refuse, request};

/// Refuses `operation` when its standard output reaches no reader (closed,
/// or the null device): the refusal goes to stderr, the one stream left.
pub(super) fn answerable(operation: &str) -> anyhow::Result<()> {
    match file::stdout_discarded() {
        None => Ok(()),
        Some(stdout) => Err(refuse(
            &request::STDOUT_DISCARDED,
            &[
                ("operation", operation),
                ("stdout", &stdout),
                ("stdin", &file::state(Path::new("/dev/stdin"))),
            ],
        )),
    }
}

/// `operation`'s body, read from standard input to its end; a directory
/// redirected into it, or a stream that breaks, is refused by its own cause.
pub(super) fn body(operation: &str) -> anyhow::Result<Vec<u8>> {
    let mut body = Vec::new();
    let Err(error) = io::stdin().read_to_end(&mut body) else {
        return Ok(body);
    };
    let directory = error.kind() == io::ErrorKind::IsADirectory
        || std::fs::metadata("/dev/stdin").is_ok_and(|stdin| stdin.is_dir());
    let state = file::state(Path::new("/dev/stdin"));
    let kind = file::kind(&error);
    let error = error.to_string();
    Err(if directory {
        refuse(
            &request::BODY_DIRECTORY,
            &[
                ("operation", operation),
                ("state", &state),
                ("kind", &kind),
                ("error", &error),
            ],
        )
    } else {
        refuse(
            &request::BODY_UNREAD,
            &[
                ("operation", operation),
                ("state", &state),
                ("size", &body.len().to_string()),
                ("kind", &kind),
                ("error", &error),
            ],
        )
    })
}
