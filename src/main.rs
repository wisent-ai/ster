//! The `ster` binary. Everything it owns lives in `cli` and in `request`, the
//! desktop's request protocol, which decodes a tuning request straight into the
//! CLI's own command; the work itself is the library's.

mod cli;
mod request;

use std::process::ExitCode;

/// Runs the command and exits with the status its failure declares: 2 for a
/// usage refusal, 69 when retrying unchanged can help, 1 for every other
/// failure (`ster::failures`). The refusal is printed as `Error: ` and the
/// chain, as before any cause was declared.
fn main() -> ExitCode {
    match cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:?}");
            let status = ster::failures::exit_code(&error);
            u8::try_from(status).map_or(ExitCode::FAILURE, ExitCode::from)
        }
    }
}
