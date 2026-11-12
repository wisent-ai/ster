//! What a parity comparison reads before the model runs: the parity file,
//! each record's token ids and the reference states file. Every refusal is
//! `ster.parity`'s (src/failures/workflow/parity.rs) or the `ster.file`
//! layer's.

use std::{fs, io::ErrorKind, path::Path};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use wisent_errors::Cause;

use crate::failures::{file, refuse, workflow::parity as causes};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ParitySet {
    pub(super) records: Vec<ParityInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ParityInput {
    #[serde(default)]
    token_ids: Option<Vec<u32>>,
    #[serde(default)]
    token_ids_file: Option<String>,
    pub(super) hidden: String,
    #[serde(default)]
    pub(super) tensor: Option<String>,
}

/// The parity file at `input`, holding at least one record.
pub(super) fn read_set(input: &Path) -> Result<ParitySet> {
    let role = "the parity file";
    let bytes = fs::read(input)
        .map_err(|error| file::read::refusal_or(&causes::INPUT_MISSING, role, input, &error))?;
    let set: ParitySet = serde_json::from_slice(&bytes)
        .map_err(|error| file::json::refusal(role, causes::FORM, input, &error))?;
    if set.records.is_empty() {
        return Err(refuse(
            &causes::RECORDS_NONE,
            &[("input", &input.display().to_string()), ("state", &file::state(input))],
        ));
    }
    Ok(set)
}

/// Record `index`'s token ids, from the record or the file it names under
/// `base`, each one the model's `vocabulary` holds.
pub(super) fn token_ids(
    record: &ParityInput,
    base: &Path,
    index: usize,
    input: &Path,
    vocabulary: usize,
) -> Result<Vec<u32>> {
    let refused = |cause: &'static Cause, extra: &[(&str, String)]| {
        let (index, input) = (index.to_string(), input.display().to_string());
        let mut values = vec![("index", index.as_str()), ("input", input.as_str())];
        values.extend(extra.iter().map(|(name, value)| (*name, value.as_str())));
        refuse(cause, &values)
    };
    let (ids, source) = match (&record.token_ids, &record.token_ids_file) {
        (Some(ids), None) => (ids.clone(), "tokenIds".to_owned()),
        (None, Some(named)) => {
            let path = base.join(named);
            let role = format!("record {index}'s token id file");
            let bytes = fs::read(&path).map_err(|error| {
                file::read::refusal_or(&causes::IDS_FILE_MISSING, &role, &path, &error)
            })?;
            let ids = serde_json::from_slice(&bytes)
                .map_err(|error| file::json::refusal(&role, causes::IDS_FORM, &path, &error))?;
            (ids, format!("tokenIdsFile {}", path.display()))
        }
        (Some(ids), Some(named)) => {
            return Err(refused(
                &causes::IDS_BOTH,
                &[("count", ids.len().to_string()), ("file", named.clone())],
            ));
        }
        (None, None) => {
            return Err(refused(&causes::IDS_NEITHER, &[("hidden", record.hidden.clone())]));
        }
    };
    if ids.is_empty() {
        return Err(refused(&causes::IDS_NONE, &[("source", source)]));
    }
    if let Some((position, id)) = ids.iter().enumerate().find(|(_, id)| **id as usize >= vocabulary) {
        return Err(refused(
            &causes::TOKEN_PAST,
            &[
                ("id", id.to_string()),
                ("position", position.to_string()),
                ("vocabulary", vocabulary.to_string()),
                ("last", vocabulary.saturating_sub(1).to_string()),
            ],
        ));
    }
    Ok(ids)
}

/// Record `index`'s reference states at `path`, `[positions, hidden]` in
/// F32. A three-dimensional tensor is accepted only when its leading batch
/// holds a single sequence.
pub(super) fn reference_states(path: &Path, tensor: &str, index: usize, input: &Path) -> Result<Tensor> {
    let role = format!("record {index}'s reference states");
    let tensors = candle_core::safetensors::load(path, &Device::Cpu).map_err(|error| {
        match file::tensors::io(&error).filter(|io| io.kind() == ErrorKind::NotFound) {
            Some(io) => file::read::refusal_or(&causes::STATES_MISSING, &role, path, io),
            None => file::tensors::read_refusal(&role, path, &error),
        }
    })?;
    let (shown_index, shown_input, shown_path) =
        (index.to_string(), input.display().to_string(), path.display().to_string());
    let at = [
        ("index", shown_index.as_str()),
        ("input", shown_input.as_str()),
        ("path", shown_path.as_str()),
        ("tensor", tensor),
    ];
    let Some(states) = tensors.get(tensor) else {
        let mut names: Vec<&str> = tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        let names = names.join(", ");
        let mut values = at.to_vec();
        values.push(("names", names.as_str()));
        return Err(refuse(&causes::TENSOR_ABSENT, &values));
    };
    let shape = format!("{:?}", states.dims());
    let dtype = format!("{:?}", states.dtype());
    let mut values = at.to_vec();
    values.extend([("shape", shape.as_str()), ("dtype", dtype.as_str())]);
    let states = match *states.dims() {
        [_, _] => states.clone(),
        [1, positions, width] => states.reshape((positions, width))?,
        _ => return Err(refuse(&causes::TENSOR_SHAPE, &values)),
    };
    if states.dims()[0] == 0 {
        return Err(refuse(&causes::POSITIONS_NONE, &values));
    }
    Ok(states.to_dtype(DType::F32)?)
}
