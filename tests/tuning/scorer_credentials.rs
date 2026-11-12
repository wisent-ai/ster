//! Real CLI credential refusals before policy loading or scorer requests.
//! STER_TEST_GRPO_ARGS names a JSON array of the required GRPO declarations,
//! without command words, --reward or --output. No training run is attempted.
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn malformed_scorer_bearer_is_not_dropped() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let declaration = std::env::var_os("STER_TEST_GRPO_ARGS")
        .expect("STER_TEST_GRPO_ARGS must name the GRPO argument declaration file");
    let args: Vec<String> =
        serde_json::from_slice(&std::fs::read(&declaration).expect("read declared GRPO arguments"))
            .expect("GRPO arguments must be a JSON array of strings");
    assert!(
        args.iter().all(|arg| arg != "--reward"
            && !arg.starts_with("--reward=")
            && arg != "--output"
            && !arg.starts_with("--output=")),
        "the runner owns --reward and --output"
    );
    let reward = std::env::var("STER_TEST_REWARD_URL")
        .expect("STER_TEST_REWARD_URL must name the real scorer URL and JSON pointer");
    let run = root
        .join("target/real-tests/scorer-credentials")
        .join(format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos(),
        ));
    std::fs::create_dir_all(&run).expect("create retained report directory");
    let revision = Command::new("git")
        .args(["--no-optional-locks", "rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .expect("read source revision");
    assert!(revision.status.success());
    let changes = Command::new("git")
        .args(["--no-optional-locks", "diff", "--binary", "HEAD"])
        .current_dir(&root)
        .output()
        .expect("read source modifications");
    assert!(changes.status.success());
    std::fs::write(run.join("source.patch"), &changes.stdout).expect("retain source modifications");
    let output_path = run.join("adapter.safetensors");
    let cases = [
        ("empty", OsString::new(), "present but empty"),
        ("blank", OsString::from(" \t\n"), "present but empty"),
        (
            "non-unicode",
            OsString::from_vec(vec![u8::MAX]),
            "not valid Unicode",
        ),
    ];
    for (name, bearer, cause) in cases {
        let output = Command::new(env!("CARGO_BIN_EXE_ster"))
            .args(["tune", "grpo"])
            .args(&args)
            .arg("--reward")
            .arg(&reward)
            .arg("--output")
            .arg(&output_path)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").expect("PATH declaration"))
            .env("HOME", &run)
            .env("STER_REWARD_BEARER", bearer)
            .current_dir(&root)
            .output()
            .expect("run actual Ster CLI");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let report = serde_json::json!({
            "source_revision": String::from_utf8_lossy(&revision.stdout).trim(),
            "binary": env!("CARGO_BIN_EXE_ster"),
            "command": ["tune", "grpo"], "arguments": args,
            "reward": reward, "output": output_path,
            "case": name, "exit": output.status.to_string(),
            "stdout": String::from_utf8_lossy(&output.stdout), "stderr": stderr,
        });
        std::fs::write(
            run.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&report).expect("encode report"),
        )
        .expect("retain report");
        assert!(
            !output.status.success(),
            "{name}: credential declaration was accepted"
        );
        assert!(
            stderr.contains(&format!("STER_REWARD_BEARER {cause}")),
            "{name}: {stderr}"
        );
        assert!(
            !output_path.exists(),
            "{name}: credential refusal wrote an adapter"
        );
    }
}
