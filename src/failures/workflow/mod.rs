//! How Ster's steering workflows fail (src/workflow, src/representation.rs):
//! choosing the layers a run reads, reading a prompt set, fitting and
//! scoring a direction, holding pairs out to rank on, comparing artifacts,
//! projecting and scoring along a direction, exporting representations and
//! measuring parity with a reference implementation.

use wisent_errors::Operation;

pub mod compare;
pub mod direction;
pub mod holdout;
pub mod layers;
pub mod parity;
pub mod project;
pub mod prompts;

/// docs/guide/errors/workflow.md, where these functionalities are listed.
const DOCS: &str = page!("workflow");

pub(super) static OPERATIONS: &[&Operation] = &[
    &layers::LAYERS,
    &prompts::PROMPTS,
    &direction::DIRECTION,
    &holdout::HOLDOUT,
    &holdout::TRAIN,
    &holdout::OPTIMIZE,
    &holdout::CURVE,
    &holdout::EVALUATE,
    &compare::COMPARE,
    &project::PROJECT,
    &project::SCORE,
    &prompts::EXTRACT,
    &parity::PARITY,
];
