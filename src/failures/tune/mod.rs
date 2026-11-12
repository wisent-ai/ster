//! How training, scoring and rewriting adapters fails: the settings every
//! trainer and each objective takes, the example sets, the shared run, its
//! checkpoints, the reward policy optimization reads, and the operations
//! that read or rewrite a trained adapter.

use wisent_errors::Operation;

/// docs/guide/errors/tune.md, where these functionalities are listed.
const DOCS: &str = page!("tune");

pub mod ablate;
pub mod checkpoint;
pub mod commands;
pub mod examples;
pub mod objectives;
pub mod policy;
pub mod resume;
pub mod rewrite;
pub mod rollout;
pub mod run;
pub mod settings;
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
