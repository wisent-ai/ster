//! The `ster` binary. Everything it owns lives in `cli` and in `request`, the
//! desktop's request protocol, which decodes a tuning request straight into the
//! CLI's own command; the work itself is the library's.

mod cli;
mod request;

use anyhow::Result;

fn main() -> Result<()> {
    cli::run()
}
