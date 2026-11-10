//! The `ster` binary. Everything it owns lives in `cli` and in `request`, the
//! desktop's request protocol, which decodes a tuning request straight into the
//! CLI's own command; the work itself is the library's.

mod cli;
mod request;

use std::process::ExitCode;

/// Runs the command and exits with the status its failure declares: 2 for a
/// usage refusal, 69 when retrying unchanged can help, 1 for every other
/// failure (`ster::failures`). The refusal is printed as `Error: ` and the
/// chain, as before any cause was declared. Once the status is known, the run
/// is queued for telemetry as the user agreed (`ster telemetry`), which never
/// changes the output or the status.
fn main() -> ExitCode {
    let result = cli::run();
    let status = match &result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Error: {error:?}");
            ster::failures::exit_code(error)
        }
    };
    cli::finish(result.as_ref().err(), status);
    if status == 0 {
        return ExitCode::SUCCESS;
    }
    u8::try_from(status).map_or(ExitCode::FAILURE, ExitCode::from)
}
