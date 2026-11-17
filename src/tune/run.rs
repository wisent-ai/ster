//! The steps every objective's training loop takes the same way: starting the
//! optimizer, from the checkpoint `--resume` names when it names one; setting
//! each step's learning rate from the run's schedule; and writing a
//! checkpoint after a step when one is due. What a step's loss is, the order
//! the data is taken in and what a run reports stay each objective's own.

use std::collections::BTreeMap;

use anyhow::Result;
use candle_nn::{ParamsAdamW, VarMap};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::checkpoint::{Durable, Position, optimizer::AdamW};
use super::schedule;

/// The optimizer over `varmap` at `learning_rate`, where the run stands when
/// `durable` resumes one, and the running report `subject` keeps: the
/// checkpoint's when the run is resumed, an empty one otherwise. A
/// checkpoint of a run whose fingerprint is not `fingerprint` is refused.
pub(crate) fn start<R: DeserializeOwned + Default>(
    varmap: &VarMap,
    learning_rate: f64,
    durable: &Durable,
    fingerprint: &BTreeMap<String, Value>,
    subject: &str,
) -> Result<(AdamW, Option<Position>, R)> {
    let mut optimizer = AdamW::over(
        varmap,
        ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        },
    )?;
    let resumed = durable.load(&mut optimizer, fingerprint)?;
    let running = match &resumed {
        Some(position) => position.running(subject)?,
        None => R::default(),
    };
    Ok((optimizer, resumed, running))
}

/// Sets the learning rate of the step `optimizer` is about to take from the
/// schedule of a run of `total` steps at `base`, `warmup` of them warming up.
pub(crate) fn schedule_step(optimizer: &mut AdamW, base: f64, total: usize, warmup: usize) {
    optimizer.set_learning_rate(schedule(base, optimizer.rates.len(), total, warmup));
}

/// Writes the checkpoint of the step `optimizer` has just taken, `within`
/// steps into pass `pass` and with `running` as the objective's running
/// report, when `durable` says one is due after it, and adds that step to
/// `checkpoints`.
pub(crate) fn keep(
    durable: &Durable,
    optimizer: &AdamW,
    fingerprint: &BTreeMap<String, Value>,
    pass: usize,
    within: usize,
    running: &impl Serialize,
    checkpoints: &mut Vec<usize>,
) -> Result<()> {
    let step = optimizer.rates.len();
    if !durable.due(step) {
        return Ok(());
    }
    let position = Position {
        pass,
        step,
        within,
        running: crate::json::value(running),
    };
    durable.save(optimizer, fingerprint, &position)?;
    checkpoints.push(step);
    Ok(())
}
