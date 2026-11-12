//! A posttraining run's durable state, so a run that stops loses nothing.
//!
//! Every objective steps the same [`optimizer::AdamW`], and every objective
//! can write, at the steps the operator names, what it needs to continue:
//! the trained tensors, both AdamW moments of each, the learning rate of
//! every step taken, the position in the run, the objective's running report
//! fields, and a fingerprint of everything the run was started from (model,
//! revision, objective, its options and adapter spec, a digest of its data,
//! the seed). A resumed run reads them back and refuses a fingerprint that
//! differs from the run being started, naming each field with both values:
//! continuing a run from another run's state would produce an adapter no
//! single run produced.
//!
//! A checkpoint is written under a unique attempt name and only then renamed
//! into place; `latest` names the newest finished one and is itself replaced
//! by rename. A stop after the directory rename but before pointer publication
//! may leave an unreferenced attempt, which must not prevent the same step
//! from being saved again. Once the new one is named, its predecessor is removed.

pub(crate) mod optimizer;
mod resume;
mod save;

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::Result;
use blake2::{Blake2b512, Digest};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::failures::{
    file, refuse,
    tune::{
        checkpoint::LATEST_INVALID,
        resume::{EPOCH_LOSSLESS, POSITION_PAST_PASS, RUN_LOSSLESS, RUNNING_FOREIGN},
        run::DATA_UNREADABLE,
    },
};

/// What a checkpoint's state file says it is, so a directory of something
/// else is refused rather than read.
const KIND: &str = "ster tune checkpoint";
const STATE: &str = "state.json";
const TENSORS: &str = "tensors.safetensors";
const LATEST: &str = "latest";

/// Where and how often the operator asked for checkpoints, and the
/// checkpoint to continue from, as `ster tune train` was given them.
#[derive(Debug, Clone, Default)]
pub struct Durable {
    pub checkpoint: Option<(PathBuf, NonZeroUsize)>,
    pub resume: Option<PathBuf>,
}

/// Where a run stands after a step: the pass it is in, the steps taken in
/// all, the steps taken in this pass, and what the objective keeps running
/// for its report (loss sums, the first loss).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub pass: usize,
    pub step: usize,
    pub within: usize,
    pub running: Value,
}

impl Position {
    /// The refusal `cause` of a resumed `objective` run whose checkpoint
    /// records this position, with `values` its sentence names beside the
    /// objective and the position.
    fn refusal(
        &self,
        cause: &'static wisent_errors::Cause,
        objective: &str,
        values: &[(&str, &str)],
    ) -> anyhow::Error {
        let position = format!(
            "pass {}, step {}, {} steps into the pass",
            self.pass, self.step, self.within
        );
        let mut all = vec![("objective", objective), ("position", position.as_str())];
        all.extend_from_slice(values);
        refuse(cause, &all)
    }

    /// The steps of this pass a resumed run still has to take, of `plans`.
    pub(crate) fn remaining<'a, T>(&self, plans: &'a [T], objective: &str) -> Result<&'a [T]> {
        plans.get(self.within..).ok_or_else(|| {
            self.refusal(
                &POSITION_PAST_PASS,
                objective,
                &[
                    ("within", &self.within.to_string()),
                    ("pass", &self.pass.to_string()),
                    ("steps", &plans.len().to_string()),
                ],
            )
        })
    }

    /// The running report `objective` keeps, as the checkpoint carries it.
    pub(crate) fn running<T: DeserializeOwned>(&self, objective: &str) -> Result<T> {
        serde_json::from_value(self.running.clone()).map_err(|error| {
            self.refusal(&RUNNING_FOREIGN, objective, &[("error", &error.to_string())])
        })
    }

    /// The refusal of a resumed run at the end of `epoch` whose running
    /// report records no step of it, which only a checkpoint can bring about.
    pub(crate) fn epoch_lossless(&self, objective: &str, epoch: usize) -> anyhow::Error {
        self.refusal(&EPOCH_LOSSLESS, objective, &[("epoch", &epoch.to_string())])
    }

    /// The refusal of a run that ends with no loss to report, which only a
    /// resumed run can: one that was not resumed takes a step in every pass,
    /// because `Preflight::check` refuses `self.epochs == 0`, the encoders
    /// refuse `if encoded.is_empty()` (src/tune/encode.rs) and `batch::plan`
    /// cuts a non-empty order with `order.chunks(batch * accumulation)`,
    /// which yields at least one step and never an empty one.
    pub(crate) fn unreported(resumed: Option<&Self>, objective: &str) -> anyhow::Error {
        resumed
            .expect("a run that was not resumed takes a step in every pass")
            .refusal(&RUN_LOSSLESS, objective, &[])
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    kind: String,
    written_by: String,
    fingerprint: BTreeMap<String, Value>,
    position: Position,
    rates: Vec<f64>,
}

/// A digest of a data file a run trains on, for its fingerprint.
pub fn digest(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|error| {
        refuse(
            &DATA_UNREADABLE,
            &[
                ("path", &path.display().to_string()),
                ("state", &crate::failures::file::state(path)),
                ("kind", &format!("{:?}", error.kind())),
                ("error", &error.to_string()),
            ],
        )
    })?;
    Ok(Blake2b512::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// What a run was started from beyond its objective's options: the model,
/// its revision, the digest of every data file it trains on, and how its
/// text was encoded and its weights mapped. The CLI and the request API
/// build it the same way, so a run started from one resumes from the other.
pub fn identity(
    model: &str,
    revision: Option<&str>,
    data: &[(&str, &Path)],
    encoding: Value,
    precision: Value,
) -> Result<BTreeMap<String, Value>> {
    let mut identity = BTreeMap::from([
        ("model".to_owned(), Value::from(model)),
        ("revision".to_owned(), Value::from(revision)),
        ("encoding".to_owned(), encoding),
        ("precision".to_owned(), precision),
    ]);
    for (name, path) in data {
        identity.insert(format!("{name}_digest"), Value::from(digest(path)?));
    }
    Ok(identity)
}

/// The step directory `directory`'s `latest` names, or `None` when nothing
/// is there to read. `latest` names one checkpoint beside it, never a path
/// outside the run's output; the same check protects both restoration and
/// predecessor removal.
fn latest(directory: &Path) -> Result<Option<PathBuf>> {
    let pointer = directory.join(LATEST);
    let shown = pointer.display().to_string();
    let name = match fs::read_to_string(&pointer) {
        Ok(name) => name,
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => {
            return Ok(None);
        }
        Err(error) => {
            return Err(file::read::refusal("the latest pointer", &pointer, &error));
        }
    };
    let name = name.trim();
    let mut components = Path::new(name).components();
    if !name.starts_with("step-")
        || !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(refuse(
            &LATEST_INVALID,
            &[
                ("pointer", &shown),
                ("name", &format!("{name:?}")),
                ("directory", &directory.display().to_string()),
            ],
        ));
    }
    Ok(Some(directory.join(name)))
}

impl Durable {
    /// Whether a checkpoint is due after step `step`.
    pub fn due(&self, step: usize) -> bool {
        self.checkpoint
            .as_ref()
            .is_some_and(|(_, every)| step.is_multiple_of(every.get()))
    }
}
