//! Running one request's job: its body decoded and checked, the operation
//! run, and what it ended as, a completed document or a refusal with the
//! status and log a CLI refusal has.

use super::*;

/// A completed job may carry a failed domain report. Keep that report while
/// giving its refusal the same failed process status and log as a CLI refusal.
pub(super) enum JobOutcome {
    Completed(Value),
    Failed {
        document: Value,
        refusal: anyhow::Error,
    },
}

impl From<Value> for JobOutcome {
    fn from(document: Value) -> Self {
        Self::Completed(document)
    }
}

impl Job<'_> {
    /// Parse and validate the body, then run the operation. Failures before
    /// it starts (a body that does not parse, a missing field) are refusals;
    /// a failure while it runs mirrors the CLI instead: the refusal sentence
    /// on a log event and one failed result.
    pub(super) fn run<R, F, O>(&self, run: F) -> i32
    where
        R: DeserializeOwned + Validate,
        F: FnOnce(R) -> Result<O>,
        O: Into<JobOutcome>,
    {
        run_decoded_job(
            self.operation,
            self.body.len(),
            serde_json::from_slice::<R>(request_document(self.body)),
            self.counts,
            run,
        )
    }
}

pub(super) fn request_document(body: &[u8]) -> &[u8] {
    if body.iter().all(u8::is_ascii_whitespace) {
        b"{}"
    } else {
        body
    }
}

pub(super) fn run_decoded_job<R, F, O>(
    operation: &str,
    size: usize,
    request: std::result::Result<R, serde_json::Error>,
    counts: bool,
    run: F,
) -> i32
where
    R: Validate,
    F: FnOnce(R) -> Result<O>,
    O: Into<JobOutcome>,
{
    let request = match request {
        Ok(request) => request,
        Err(error) => {
            return refuse(&ster::failures::request::body_refusal(
                operation, size, &error,
            ));
        }
    };
    if let Err(refusal) = request.validate() {
        return refuse(&refusal);
    }
    match run(request).map(Into::into) {
        Ok(JobOutcome::Completed(document)) => {
            if counts {
                star_request_event();
            }
            emit_result(COMPLETED, document);
            COMPLETED
        }
        Ok(JobOutcome::Failed { document, refusal }) => failed(document, &refusal),
        Err(error) => {
            let error = ster::failures::tensor::settle(error, &format!("ster request {operation}"));
            failed(json!({"error": format!("{error:#}")}), &error)
        }
    }
}
