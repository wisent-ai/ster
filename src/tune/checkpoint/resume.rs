//! Reading the checkpoint `--resume` names back into a run, refusing one
//! another run wrote.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use anyhow::Result;
use serde_json::Value;

use super::optimizer::{AdamW, Tracked};
use super::{Durable, KIND, LATEST, Position, STATE, State, TENSORS, latest};
use crate::failures::{
    file, refuse,
    tune::resume::{
        FOREIGN_RUN, KIND_FOREIGN, NO_CHECKPOINT, STATE_FORM, STATE_MISSING, TENSOR_ABSENT,
        TENSOR_DTYPE, TENSOR_EXTRA, TENSOR_UNFIT,
    },
};

/// One fingerprint field as a refusal names it.
fn described(value: Option<&Value>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "not recorded".to_owned(),
    }
}

impl Durable {
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
        let Some(checkpoint) = latest(directory)? else {
            let pointer = directory.join(LATEST);
            return Err(refuse(
                &NO_CHECKPOINT,
                &[
                    ("directory", &directory.display().to_string()),
                    ("directory_state", &file::state(directory)),
                    ("pointer", &pointer.display().to_string()),
                    ("state", &file::state(&pointer)),
                ],
            ));
        };
        let shown = checkpoint.display().to_string();
        let state_path = checkpoint.join(STATE);
        let bytes = fs::read(&state_path).map_err(|error| {
            file::read::refusal_or(&STATE_MISSING, "the checkpoint state", &state_path, &error)
        })?;
        let state: State = serde_json::from_slice(&bytes).map_err(|error| {
            file::json::refusal("the checkpoint state", STATE_FORM, &state_path, &error)
        })?;
        if state.kind != KIND {
            return Err(refuse(
                &KIND_FOREIGN,
                &[
                    ("path", &state_path.display().to_string()),
                    ("kind", &format!("{:?}", state.kind)),
                ],
            ));
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
            return Err(refuse(
                &FOREIGN_RUN,
                &[("checkpoint", &shown), ("differing", &differing.join("; "))],
            ));
        }
        let tensors_path = checkpoint.join(TENSORS);
        let tensors_shown = tensors_path.display().to_string();
        // `AdamW::over` tracks every float variable of the run's `VarMap`,
        // and `Adapters::fresh` registers an `a` and a `b` for each of the
        // targets on each of the layers `Spec::validate` refuses to see
        // empty (`self.targets.is_empty()`, `layer_count == 0`).
        let device = optimizer
            .tracked
            .values()
            .next()
            .map(|tracked| tracked.var.device().clone())
            .expect("every trainer's optimizer tracks the tensors the run trains");
        let mut stored = candle_core::safetensors::load(&tensors_path, &device).map_err(|error| {
            file::tensors::read_refusal("the checkpoint's tensors", &tensors_path, &error)
        })?;
        for (tensor, Tracked { var, first, second }) in &optimizer.tracked {
            for (part, target) in [("var", var), ("first", first), ("second", second)] {
                let key = format!("{part}/{tensor}");
                let values = [
                    ("checkpoint", shown.as_str()),
                    ("path", tensors_shown.as_str()),
                    ("key", key.as_str()),
                ];
                let Some(value) = stored.remove(&key) else {
                    return Err(refuse(&TENSOR_ABSENT, &values));
                };
                target.set(&value).map_err(|error| {
                    let stored = format!("{:?} {:?}", value.dims(), value.dtype());
                    let trained = format!("{:?} {:?}", target.dims(), target.dtype());
                    let mut values = values.to_vec();
                    values.extend([("stored", stored.as_str()), ("trained", trained.as_str())]);
                    if value.dims() != target.dims() {
                        refuse(&TENSOR_UNFIT, &values)
                    } else if value.dtype() != target.dtype() {
                        refuse(&TENSOR_DTYPE, &values)
                    } else {
                        file::tensors::read_refusal("the checkpoint's tensors", &tensors_path, &error)
                    }
                })?;
            }
        }
        if let Some(extra) = stored.keys().next() {
            return Err(refuse(
                &TENSOR_EXTRA,
                &[
                    ("checkpoint", &shown),
                    ("path", &tensors_shown),
                    ("key", extra),
                ],
            ));
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
