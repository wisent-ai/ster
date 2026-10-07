//! Comparing this model's forward pass with hidden states another
//! implementation recorded for the same tokens: the measure of whether Ster
//! computes a family the way its reference implementation does.
//!
//! The input is `{"records": [{"tokenIds": [...] | "tokenIdsFile":
//! "ids.json", "hidden": "states.safetensors", "tensor": "hidden_states"}]}`,
//! paths relative to the input file. Each reference tensor holds the
//! residual stream after the final norm for the first `P` positions, `[P,
//! hidden]` or `[1, P, hidden]`. Ster reads the same token ids — no
//! tokenizer, no chat template — and reports how far its states lie from the
//! reference ones: the relative error over every compared position, the
//! worst position's, the largest single difference, and the cosine of each
//! position's pair. A reference computed at a narrower precision than
//! Ster's sets the floor these can reach.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::runtime::Runtime;

use super::progress;

/// The tensor a reference file holds its states under when the record names
/// none, as Transformers' `output_hidden_states` captures name them.
const DEFAULT_TENSOR: &str = "hidden_states";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ParitySet {
    records: Vec<ParityInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ParityInput {
    #[serde(default)]
    token_ids: Option<Vec<u32>>,
    #[serde(default)]
    token_ids_file: Option<String>,
    hidden: String,
    #[serde(default)]
    tensor: Option<String>,
}

/// How far Ster's states lie from one record's reference states, or from
/// every record's together.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityMeasure {
    pub positions: usize,
    pub relative_error: f64,
    pub worst_position_relative_error: f64,
    pub largest_difference: f64,
    pub mean_cosine: f64,
    pub smallest_cosine: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityRecord {
    pub hidden: String,
    pub tokens: usize,
    #[serde(flatten)]
    pub measure: ParityMeasure,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityReport {
    pub model: String,
    pub model_revision: Option<String>,
    pub records: Vec<ParityRecord>,
    pub overall: ParityMeasure,
}

pub fn parity(runtime: &Runtime, input: &Path) -> Result<ParityReport> {
    let set: ParitySet = serde_json::from_slice(
        &fs::read(input).with_context(|| format!("failed to read {}", input.display()))?,
    )
    .with_context(|| format!("invalid parity JSON in {}", input.display()))?;
    if set.records.is_empty() {
        bail!("{} lists no records to compare", input.display());
    }
    let base = input.parent().map(Path::to_path_buf).unwrap_or_default();
    let vocabulary = runtime.vocab_size();
    let mut records = Vec::with_capacity(set.records.len());
    let mut totals = Totals::default();
    for (index, record) in set.records.iter().enumerate() {
        progress(format!(
            "comparing record {}/{}",
            index + 1,
            set.records.len()
        ));
        let ids = token_ids(record, &base, index)?;
        if let Some(id) = ids.iter().find(|id| **id as usize >= vocabulary) {
            bail!("record {index} holds token id {id}, past the model's {vocabulary} tokens");
        }
        let tensor = record.tensor.as_deref().unwrap_or(DEFAULT_TENSOR);
        let reference = reference_states(&base.join(&record.hidden), tensor)?;
        let (positions, width) = reference.dims2()?;
        if positions > ids.len() {
            bail!(
                "record {index}'s reference holds {positions} positions for {} tokens",
                ids.len()
            );
        }
        let ours = runtime
            .forward_hidden_scored(&ids)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let (_, ours_width) = ours.dims2()?;
        if ours_width != width {
            bail!(
                "record {index}'s reference states are {width} wide and this model's {ours_width}"
            );
        }
        let ours = ours.narrow(0, 0, positions)?.to_vec2::<f32>()?;
        let theirs = reference.to_vec2::<f32>()?;
        let mut record_totals = Totals::default();
        for (mine, reference) in ours.iter().zip(&theirs) {
            record_totals.add(mine, reference);
        }
        totals.merge(&record_totals);
        records.push(ParityRecord {
            hidden: record.hidden.clone(),
            tokens: ids.len(),
            measure: record_totals.measure(),
        });
    }
    Ok(ParityReport {
        model: runtime.model_id.clone(),
        model_revision: runtime.revision.clone(),
        records,
        overall: totals.measure(),
    })
}

fn token_ids(record: &ParityInput, base: &Path, index: usize) -> Result<Vec<u32>> {
    let ids = match (&record.token_ids, &record.token_ids_file) {
        (Some(ids), None) => ids.clone(),
        (None, Some(file)) => {
            let path: PathBuf = base.join(file);
            serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
            )
            .with_context(|| format!("{} is not a JSON list of token ids", path.display()))?
        }
        (Some(_), Some(_)) => {
            bail!("record {index} names both tokenIds and tokenIdsFile; give one")
        }
        (None, None) => bail!("record {index} names neither tokenIds nor tokenIdsFile"),
    };
    if ids.is_empty() {
        bail!("record {index} holds no token ids");
    }
    Ok(ids)
}

/// The reference states, `[positions, hidden]` in F32.
fn reference_states(path: &Path, tensor: &str) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let Some(states) = tensors.get(tensor) else {
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        bail!(
            "{} holds no tensor {tensor:?}; it holds {names:?}",
            path.display()
        );
    };
    let states = states.to_dtype(DType::F32)?;
    match states.dims() {
        [_, _] => Ok(states),
        [1, _, _] => Ok(states.squeeze(0)?),
        other => bail!(
            "{}'s {tensor:?} is {other:?}; Ster compares [positions, hidden] states",
            path.display()
        ),
    }
}

#[derive(Debug, Default)]
struct Totals {
    positions: usize,
    difference_squares: f64,
    reference_squares: f64,
    worst_relative: f64,
    largest: f64,
    cosine_sum: f64,
    smallest_cosine: Option<f64>,
}

impl Totals {
    fn add(&mut self, mine: &[f32], reference: &[f32]) {
        let (mut difference, mut norm_mine, mut norm_reference, mut dot) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in mine.iter().zip(reference) {
            let (a, b) = (f64::from(*a), f64::from(*b));
            difference += (a - b) * (a - b);
            norm_mine += a * a;
            norm_reference += b * b;
            dot += a * b;
            self.largest = self.largest.max((a - b).abs());
        }
        self.positions += 1;
        self.difference_squares += difference;
        self.reference_squares += norm_reference;
        self.worst_relative = self
            .worst_relative
            .max(ratio(difference.sqrt(), norm_reference.sqrt()));
        let cosine = ratio(dot, (norm_mine * norm_reference).sqrt());
        self.cosine_sum += cosine;
        self.smallest_cosine = Some(
            self.smallest_cosine
                .map_or(cosine, |smallest| smallest.min(cosine)),
        );
    }

    fn merge(&mut self, other: &Self) {
        self.positions += other.positions;
        self.difference_squares += other.difference_squares;
        self.reference_squares += other.reference_squares;
        self.worst_relative = self.worst_relative.max(other.worst_relative);
        self.largest = self.largest.max(other.largest);
        self.cosine_sum += other.cosine_sum;
        self.smallest_cosine = match (self.smallest_cosine, other.smallest_cosine) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
    }

    fn measure(&self) -> ParityMeasure {
        ParityMeasure {
            positions: self.positions,
            relative_error: ratio(
                self.difference_squares.sqrt(),
                self.reference_squares.sqrt(),
            ),
            worst_position_relative_error: self.worst_relative,
            largest_difference: self.largest,
            mean_cosine: if self.positions == 0 {
                0.0
            } else {
                self.cosine_sum / self.positions as f64
            },
            smallest_cosine: self.smallest_cosine.unwrap_or(0.0),
        }
    }
}

/// `numerator / denominator`, zero when both are zero, infinite when only
/// the denominator is.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        if numerator == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        numerator / denominator
    }
}
