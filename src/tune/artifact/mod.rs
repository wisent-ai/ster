//! What can be done with an adapter once it exists: measure it on held-out
//! text, or fold it into the base weights it was trained beside.

mod evaluate;
mod merge;

pub use evaluate::{
    EvaluateOptions, EvaluateReport, EvaluatedExample, evaluate, warn_on_provenance,
};
pub use merge::{MergeReport, merge};
