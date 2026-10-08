//! What can be done with an adapter or a direction once it exists: measure
//! an adapter on held-out text, fold it into the base weights it was trained
//! beside, write it as a PEFT adapter the serving ecosystem loads, or write a
//! steering direction out of the weights for good.

mod ablate;
mod evaluate;
mod export;
mod merge;

pub use ablate::{AblationReport, ablate};
pub use evaluate::{
    EvaluateOptions, EvaluateReport, EvaluatedExample, evaluate, warn_on_provenance,
};
pub use export::{ExportReport, export_peft};
pub use merge::{MergeReport, merge};
