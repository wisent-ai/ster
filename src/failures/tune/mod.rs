//! How training, scoring and rewriting adapters fails: the settings every
//! trainer and each objective takes, the example sets, the shared run, its
//! checkpoints, the reward policy optimization reads, and the operations
//! that read or rewrite a trained adapter.

use wisent_errors::Operation;

/// docs/guide/errors/tune.md, where these functionalities are listed.
const DOCS: &str = page!("tune");

#[path = "edits/ablate.rs"]
pub mod ablate;
#[path = "runs/checkpoint.rs"]
pub mod checkpoint;
#[path = "edits/commands.rs"]
pub mod commands;
#[path = "edits/examples.rs"]
pub mod examples;
#[path = "policies/objectives.rs"]
pub mod objectives;
#[path = "policies/policy.rs"]
pub mod policy;
#[path = "runs/resume.rs"]
pub mod resume;
#[path = "edits/rewrite.rs"]
pub mod rewrite;
#[path = "runs/rollout.rs"]
pub mod rollout;
#[path = "runs/run.rs"]
pub mod run;
#[path = "policies/settings.rs"]
pub mod settings;
#[path = "policies/strength.rs"]
pub mod strength;

pub(super) static OPERATIONS: &[&Operation] = &[
    &settings::SETTINGS,
    &objectives::OBJECTIVES,
    &examples::EXAMPLES,
    &run::RUN,
    &checkpoint::LATEST,
    &checkpoint::SAVE,
    &resume::RESUME,
    &policy::REWARD,
    &rollout::ROLLOUT,
    &rollout::CONVERSE,
    &strength::STRENGTH,
    &rewrite::FIT,
    &rewrite::WRITE,
    &rewrite::MERGE,
    &rewrite::EXPORT,
    &ablate::ABLATE,
    &commands::SFT,
    &commands::PREFERENCE,
    &commands::GRPO,
    &commands::DECIDE,
    &commands::EVALUATE,
];
