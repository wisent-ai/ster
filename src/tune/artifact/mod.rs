//! What can be done with an adapter once it exists: measure it on held-out
//! text, fold it into the base weights it was trained beside, or write it as
//! a PEFT adapter the serving ecosystem loads.

mod evaluate;
mod export;
mod merge;

pub use evaluate::{
    EvaluateOptions, EvaluateReport, EvaluatedExample, evaluate, warn_on_provenance,
};
pub use export::{ExportReport, export_peft};
pub use merge::{MergeReport, merge};
