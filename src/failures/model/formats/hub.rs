//! Telling apart how a Hugging Face Hub request failed: the status the Hub
//! answered, a connection that was not made or broke, and a file that
//! arrived and could not be put into the cache.

use hf_hub::api::sync::ApiError;

use super::super::{file, refuse_from};
use super::checkpoint;

/// The status a Hub request answered with, wherever in `error`'s chain the
/// HTTP client's error sits; the Hub client keeps it boxed.
fn status(error: &anyhow::Error) -> Option<u16> {
    error.chain().find_map(|cause| {
        let client = cause
            .downcast_ref::<ureq::Error>()
            .or_else(|| cause.downcast_ref::<Box<ureq::Error>>().map(AsRef::as_ref));
        match client {
            Some(ureq::Error::Status(status, _)) => Some(*status),
            _ => None,
        }
    })
}

/// Whether the Hub client failed writing into its cache, not talking to the
/// Hub.
fn cache_write(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| matches!(cause.downcast_ref::<ApiError>(), Some(ApiError::IoError(_))))
}

/// The cache the Hub client writes into: `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`, and its state.
fn cache() -> (String, String) {
    let path = hf_hub::Cache::from_env().path().clone();
    (path.display().to_string(), file::state(&path))
}

/// The cause of a Hub answer that is neither success nor "no such
/// repository": the Hub limiting this account, the Hub failing, or a status
/// that names this request wrong.
fn answered(status: u16) -> &'static wisent_errors::Cause {
    match status {
        429 => &checkpoint::HUB_LIMITED,
        500..=599 => &checkpoint::HUB_FAILING,
        _ => &checkpoint::HUB_REFUSED,
    }
}

/// The refusal of a Hub repository listing that failed: one the Hub says
/// this account cannot see (401, 403, 404) is unknown; another status is the
/// Hub limiting, failing or refusing; no status is a connection that was not
/// made or broke.
pub fn repository_refusal(
    model: &str,
    revision: Option<&str>,
    url: &str,
    error: anyhow::Error,
) -> anyhow::Error {
    let shown = format!("{error:#}");
    // The Hub client asks for its default branch when no revision is given.
    let revision = match revision {
        Some(revision) => revision,
        None => "main",
    };
    let what = format!("the file list of {model} at revision {revision}");
    let cause = match status(&error) {
        Some(401 | 403 | 404) => &checkpoint::REPOSITORY_UNKNOWN,
        Some(code) => answered(code),
        None => &checkpoint::HUB_UNREACHABLE,
    };
    let code = match status(&error) {
        Some(code) => code.to_string(),
        None => "none".to_owned(),
    };
    refuse_from(
        cause,
        &[
            ("model", model),
            ("revision", revision),
            ("what", &what),
            ("url", url),
            ("status", &code),
            ("error", &shown),
        ],
        error,
    )
}

/// The refusal of fetching `name` of the Hub repository `model` from `url`,
/// which its file list named: the Hub refusing it, limiting or failing, a
/// connection that broke, or the file not written into the cache.
pub fn download_refusal(name: &str, model: &str, url: &str, error: anyhow::Error) -> anyhow::Error {
    let shown = format!("{error:#}");
    let what = format!("{name} of {model}");
    let (cache, cache_state) = cache();
    let cause = if cache_write(&error) {
        &checkpoint::DOWNLOAD_UNCACHED
    } else {
        match status(&error) {
            Some(code) => answered(code),
            None => &checkpoint::HUB_UNREACHABLE,
        }
    };
    let code = match status(&error) {
        Some(code) => code.to_string(),
        None => "none".to_owned(),
    };
    refuse_from(
        cause,
        &[
            ("model", model),
            ("revision", "the one listed"),
            ("file", name),
            ("what", &what),
            ("url", url),
            ("status", &code),
            ("cache", &cache),
            ("cache_state", &cache_state),
            ("error", &shown),
        ],
        error,
    )
}
