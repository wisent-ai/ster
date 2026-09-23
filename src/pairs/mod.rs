//! Authoring, inspecting, and synthesizing contrastive pair sets.
//!
//! Ster could previously only consume a `pairs.json` someone else produced.
//! This package closes that gap with two surfaces, one per module:
//!
//! * `inspect` — a model-free audit of an existing set. It answers the three
//!   questions that make a pair set silently useless: are pairs duplicated,
//!   did the generating model refuse instead of answering, and is one side
//!   systematically longer than the other (a length confound trains a
//!   "verbosity" direction rather than the trait).
//! * `synthesis` — a faithful port of Wisent's
//!   `SyntheticContrastivePairsGenerator.generate`, driven by whichever
//!   `Generator` the caller picked: Ster's own local `Runtime`, or a hosted
//!   route reached through Brama. Steering itself still needs hidden states
//!   and therefore a local model, but writing pair text needs no activations
//!   at all, so the generator model and the steered model are two different
//!   roles and only the writer may be hosted.
//!
//! Both are the single implementation behind the CLI arms and the
//! `pairs/*` operations of `ster request`.

pub mod quality;

mod generator;
mod inspect;
mod synthesis;

pub use generator::Generator;
pub use inspect::{inspect, EntryReport, InspectOptions, RefusalFlag, SetReport, UNBALANCED_RATIO};
pub use synthesis::{synthesize, SynthesisOptions, SynthesisReport};
