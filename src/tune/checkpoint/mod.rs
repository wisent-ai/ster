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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use blake2::{Blake2b512, Digest};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use optimizer::{AdamW, Tracked};

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
    let bytes = fs::read(path).with_context(|| {
        format!(
            "{} could not be read to fingerprint the run",
            path.display()
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

/// One fingerprint field as a refusal names it.
fn described(value: Option<&Value>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "not recorded".to_owned(),
    }
}

/// `latest` names one checkpoint beside it, never a path outside the run's
/// output. The same check protects both restoration and predecessor removal.
fn checkpoint_path(directory: &Path, name: &str) -> Result<PathBuf> {
    let name = name.trim();
    let mut components = Path::new(name).components();
    if !name.starts_with("step-")
        || !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        bail!(
            "{} names invalid checkpoint {name:?}; expected one step directory inside {}",
            directory.join(LATEST).display(),
            directory.display()
        );
    }
    Ok(directory.join(name))
}

impl Durable {
    /// Whether a checkpoint is due after step `step`.
    pub fn due(&self, step: usize) -> bool {
        self.checkpoint
            .as_ref()
            .is_some_and(|(_, every)| step.is_multiple_of(every.get()))
    }

    /// Write the checkpoint of `optimizer` at `position`.
    pub(crate) fn save(
        &self,
        optimizer: &AdamW,
        fingerprint: &BTreeMap<String, Value>,
        position: &Position,
    ) -> Result<()> {
        let Some((directory, _)) = &self.checkpoint else {
            bail!("no --checkpoint directory was given, so no checkpoint can be written");
        };
        fs::create_dir_all(directory).with_context(|| {
            format!(
                "the checkpoint directory {} could not be created",
                directory.display()
            )
        })?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("the operating system clock cannot name this checkpoint attempt")?
            .as_nanos();
        let name = format!("step-{}-{}-{nonce}", position.step, std::process::id());
        let partial = directory.join(format!(".{name}.partial"));
        fs::create_dir(&partial)
            .with_context(|| format!("{} could not be created", partial.display()))?;
        let mut tensors = HashMap::new();
        for (tensor, Tracked { var, first, second }) in &optimizer.tracked {
            tensors.insert(format!("var/{tensor}"), var.as_tensor().clone());
            tensors.insert(format!("first/{tensor}"), first.as_tensor().clone());
            tensors.insert(format!("second/{tensor}"), second.as_tensor().clone());
        }
        candle_core::safetensors::save(&tensors, partial.join(TENSORS))
            .with_context(|| format!("the tensors of checkpoint {name} could not be written"))?;
        let state = State {
            kind: KIND.to_owned(),
            written_by: format!("ster {}", env!("CARGO_PKG_VERSION")),
            fingerprint: fingerprint.clone(),
            position: position.clone(),
            rates: optimizer.rates.clone(),
        };
        fs::write(partial.join(STATE), serde_json::to_vec_pretty(&state)?)
            .with_context(|| format!("the state of checkpoint {name} could not be written"))?;
        let finished = directory.join(&name);
        fs::rename(&partial, &finished).with_context(|| {
            format!(
                "checkpoint {name} could not be moved into {}",
                finished.display()
            )
        })?;
        let latest = directory.join(LATEST);
        let previous = match fs::read_to_string(&latest) {
            Ok(previous) => Some(checkpoint_path(directory, &previous)?),
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Err(error) => bail!(
                "{} could not be read to replace it: {error}",
                latest.display()
            ),
        };
        let pointer = directory.join(format!(".{name}.{LATEST}.partial"));
        fs::write(&pointer, &name)
            .with_context(|| format!("{} could not be written", pointer.display()))?;
        fs::rename(&pointer, &latest)
            .with_context(|| format!("{} could not name checkpoint {name}", latest.display()))?;
        if let Some(previous) = previous.filter(|previous| *previous != finished) {
            fs::remove_dir_all(&previous).with_context(|| {
                format!(
                    "the replaced checkpoint {} could not be removed",
                    previous.display()
                )
            })?;
        }
        crate::workflow::progress(format!(
            "checkpoint {} written at step {}",
            finished.display(),
            position.step
        ));
        Ok(())
    }

    /// Read the checkpoint `--resume` names into `optimizer` and answer where
    /// the run stands, refusing a checkpoint of another run.
    pub(crate) fn load(
        &self,
        optimizer: &mut AdamW,
        fingerprint: &BTreeMap<String, Value>,
    ) -> Result<Option<Position>> {
        let Some(directory) = &self.resume else {
            return Ok(None);
        };
        let pointer = directory.join(LATEST);
        let name = fs::read_to_string(&pointer).with_context(|| {
            format!(
                "--resume {} holds no checkpoint: {} could not be read",
                directory.display(),
                pointer.display()
            )
        })?;
        let checkpoint = checkpoint_path(directory, &name)?;
        let state_path = checkpoint.join(STATE);
        let state: State = serde_json::from_slice(
            &fs::read(&state_path)
                .with_context(|| format!("{} could not be read", state_path.display()))?,
        )
        .with_context(|| format!("{} is not a Ster checkpoint state", state_path.display()))?;
        if state.kind != KIND {
            bail!(
                "{} says it is {:?}, not a {KIND}",
                state_path.display(),
                state.kind
            );
        }
        let fields: BTreeSet<&String> =
            fingerprint.keys().chain(state.fingerprint.keys()).collect();
        let differing: Vec<String> = fields
            .into_iter()
            .filter(|field| fingerprint.get(*field) != state.fingerprint.get(*field))
            .map(|field| {
                format!(
                    "{field} was {} and is {}",
                    described(state.fingerprint.get(field)),
                    described(fingerprint.get(field))
                )
            })
            .collect();
        if !differing.is_empty() {
            bail!(
                "checkpoint {} belongs to another run, so continuing it would train an adapter no single run produced: {}",
                checkpoint.display(),
                differing.join("; ")
            );
        }
        let tensors_path = checkpoint.join(TENSORS);
        let Some(device) = optimizer
            .tracked
            .values()
            .next()
            .map(|tracked| tracked.var.device().clone())
        else {
            bail!("this run trains no tensor, so no checkpoint can be read into it");
        };
        let mut stored = candle_core::safetensors::load(&tensors_path, &device)
            .with_context(|| format!("{} could not be read", tensors_path.display()))?;
        for (tensor, Tracked { var, first, second }) in &optimizer.tracked {
            for (part, target) in [("var", var), ("first", first), ("second", second)] {
                let key = format!("{part}/{tensor}");
                let Some(value) = stored.remove(&key) else {
                    bail!(
                        "{} holds no {key}, which this run trains",
                        tensors_path.display()
                    );
                };
                target.set(&value).with_context(|| {
                    format!("{key} of {} does not fit this run", tensors_path.display())
                })?;
            }
        }
        if let Some(extra) = stored.keys().next() {
            bail!(
                "{} holds {extra}, which this run does not train",
                tensors_path.display()
            );
        }
        optimizer.rates = state.rates;
        crate::workflow::progress(format!(
            "resuming from {} at step {}",
            checkpoint.display(),
            state.position.step
        ));
        Ok(Some(state.position))
    }
}
