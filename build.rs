//! The commit this binary was built from, for `ster --version`. Stado's
//! builds (kompilator, install recipes) set WISENT_SOURCE_COMMIT; a build
//! without it says the revision is unknown rather than guessing one. Before,
//! the version named only the package version, so whether a repair was
//! installed could not be read from the binary.

fn main() {
    println!("cargo:rerun-if-env-changed=WISENT_SOURCE_COMMIT");
    let revision = match std::env::var("WISENT_SOURCE_COMMIT") {
        Ok(commit) if !commit.trim().is_empty() => commit.trim().to_owned(),
        // fallback: a build no Stado started (a checkout built by hand) names
        // no commit, and the version says the revision is unknown rather
        // than guessing one.
        Ok(_) | Err(std::env::VarError::NotPresent) => "unknown".to_owned(),
        Err(std::env::VarError::NotUnicode(raw)) => {
            panic!("WISENT_SOURCE_COMMIT is not UTF-8 ({raw:?}): set it to the commit's hex id")
        }
    };
    println!(
        "cargo:rustc-env=STER_VERSION_LONG={} (revision {revision})",
        env!("CARGO_PKG_VERSION")
    );
}
