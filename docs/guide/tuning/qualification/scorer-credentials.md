# Scorer credential refusals

`tests/tuning/scorer_credentials.rs` drives the built Ster CLI through
`tune grpo` with an empty, whitespace-only and non-Unicode
`STER_REWARD_BEARER`. Each must fail with the credential declaration named
before policy loading or a scorer request, and leave no adapter output.
It does not qualify successful scoring or training.

Declare the qualification inputs in the test host's environment:

- `STER_TEST_GRPO_ARGS`: path to a JSON array of the required GRPO argument
  strings for that host, including model, prompts and training declarations.
  Do not include the command words, `--reward` or `--output`; the runner owns
  the latter arguments. Use the actual `ster tune grpo --help` contract for
  the required declarations, not an assumed training profile.
- `STER_TEST_REWARD_URL`: the real scorer's URL with the JSON-pointer fragment
  naming its reward field. Malformed credentials must prevent contacting it.

Run `kompilator test --manifest-path Cargo.toml --test scorer-credentials` on the qualification host.
Reports are retained under `target/real-tests/scorer-credentials/`, one per
refusal, with source revision, source patch, command arguments, exit status,
stdout and stderr. The invalid credential values are not written into reports.
This runner supplies only invalid credentials and never reads a real bearer.

A pass proves only the local refusal boundary. Successful GRPO qualification
still needs the actual checkpoint, prompt set, authenticated scorer, persisted
adapter and training report; it cannot be replaced by these refusals.
