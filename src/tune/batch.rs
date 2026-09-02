//! batch.rs — how many sequences share a forward, and where each row's real
//! positions are afterwards.
//!
//! Every objective that batches needs the same two things, and neither is
//! objective-specific: an arrangement of this epoch's shuffled order into
//! forwards and steps, and a way to read one row back out of a padded result.
//! Four copies of that slicing would be four chances for one of them to read
//! the wrong row, so both live here once.
//!
//! Three properties hold, and they are the whole reason the shape below is
//! what it is.
//!
//! * **Grouping never crosses a step boundary.** A step's slots are the
//!   `accumulation * batch` the epoch's shuffle put there, and [`plan`] only
//!   decides how they are arranged into forwards *inside* that step. The
//!   gradient of a step is a sum over its slots, so the arrangement cannot
//!   change which examples share a gradient — only which ones share a kernel
//!   launch.
//! * **Length grouping consumes no randomness.** It is a stable sort of a
//!   slice the trainer's per-epoch RNG already shuffled, so a run is still
//!   reproducible from `--seed` alone, and at `batch == 1` the sort is skipped
//!   entirely and the order is handed back untouched — which is what makes an
//!   omitted `--batch-size` reproduce every run recorded before batching
//!   existed.
//! * **The short tail is kept.** The last step of an epoch may hold fewer
//!   slots, and the last forward of a step fewer rows. Neither is dropped and
//!   neither is padded with repeats: [`divisor`] is the constant
//!   `accumulation * batch` whatever the step actually held, so a short tail
//!   steps proportionally smaller — exactly as a short accumulation group does
//!   today.
//!
//! The grouping rule above is what the design says; this is what the product
//! measured. `ster tune evaluate` over 64 held-out examples on a
//! grouped-query toy checkpoint, at `--batch-size 4` and `--batch-size 8`
//! against `--batch-size 1`, agreed on 63 of the 64 examples *bitwise* and
//! differed on one, at the eighth significant digit
//! (`4.175486373901367` against `4.175485992431641`); the corpus loss moved at
//! the tenth. That asymmetry is the evidence the padding mask holds: a row
//! that could see another row's filler would be wrong in proportion to how
//! much filler it was given, so a leak moves *every* short row and never one
//! of them. One row differing at the last few bits is float summation
//! associating differently inside a wider kernel, which is the one difference
//! a correct batch is allowed to make.

use anyhow::Result;
use candle_core::Tensor;

/// One optimizer step: the forwards it takes, in the order it takes them.
pub(crate) struct Step {
    /// Each entry is one forward pass — the slots whose rows are padded to a
    /// common width and put through the model together. At `batch == 1` every
    /// entry holds exactly one slot, which is the unbatched pass.
    pub forwards: Vec<Vec<usize>>,
    /// How many units this step covers, summed over its forwards. Reported
    /// rather than used arithmetically: the loss divisor is [`divisor`], not
    /// this, which is what makes a short tail smaller instead of louder.
    pub units: usize,
}

/// Arranges one epoch's shuffled `order` into steps and forwards.
///
/// `lengths[slot]` is the token count of the sequence at that slot — for a
/// preference objective, the longer of the pair's two sides, since the pair is
/// what shares a forward and the wider side is what sets the padding.
///
/// `batch == 1` returns the slots in exactly the order given, one per forward.
/// Above one, each step's slots are sorted by descending length before being
/// cut into forwards, so the rows that share a padded width are the rows that
/// were already nearly the same length; the padding a batch wastes is the
/// spread within a forward, and this is the cheapest way to make that spread
/// small without moving an example into a different step.
pub(crate) fn plan(
    order: &[usize],
    lengths: &[usize],
    batch: usize,
    accumulation: usize,
) -> Vec<Step> {
    let batch = batch.max(1);
    let accumulation = accumulation.max(1);
    order
        .chunks(batch * accumulation)
        .map(|slots| {
            let mut slots = slots.to_vec();
            if batch > 1 {
                // Stable, and keyed on length alone: slots of equal length
                // keep the relative order the shuffle gave them, so the plan
                // is a function of the seed and nothing else.
                slots.sort_by_key(|&slot| std::cmp::Reverse(lengths[slot]));
            }
            Step {
                units: slots.len(),
                forwards: slots.chunks(batch).map(<[usize]>::to_vec).collect(),
            }
        })
        .collect()
}

/// How many optimizer steps one pass over `units` units takes.
///
/// A step covers `accumulation` forwards of `batch` units each, and the last
/// one is allowed to be short, which is what `div_ceil` says.
pub(crate) fn steps_per_epoch(units: usize, batch: usize, accumulation: usize) -> usize {
    units.div_ceil(batch.max(1) * accumulation.max(1))
}

/// What every unit's loss is divided by before it is accumulated.
///
/// A constant, deliberately: dividing by what a step actually held would
/// rescale a short tail up to the weight of a full step, and a tail that is
/// half the data would then move the weights as far as a full step does.
pub(crate) fn divisor(batch: usize, accumulation: usize) -> f64 {
    (batch.max(1) * accumulation.max(1)) as f64
}

/// Row `index` of a batched forward, sliced back to its own real positions.
///
/// `output` is `[batch, width, trailing]` — logits or a residual stream — and
/// the padding is on the right, so row `index`'s real values are its first
/// `length` positions. The result is `[1, length, trailing]`, which is the
/// exact shape a single-sequence forward returns, so every reader downstream —
/// `token_logprobs`, the completion window, the reward head — takes it
/// unchanged and cannot tell which pass produced it.
pub(crate) fn row(output: &Tensor, index: usize, length: usize) -> Result<Tensor> {
    Ok(output.narrow(0, index, 1)?.narrow(1, 0, length)?)
}

/// Puts every row of one planned forward through `run`, and hands each row's
/// own output back sliced to its own length.
///
/// `rows_per_unit` is how many sequences one unit of the objective is: one for
/// an example, two for a preference pair. At `batch == 1` the stride is one
/// row whatever that is, so a pair's two sides go through the model in two
/// separate passes — which is the unbatched shape, and the reason a run at the
/// default reproduces every number recorded before batching existed, down to
/// the bytes of the adapter it writes. Above one, the whole planned forward is
/// one pass.
pub(crate) fn read_rows(
    rows: &[&[u32]],
    batch: usize,
    rows_per_unit: usize,
    run: impl Fn(&[&[u32]]) -> Result<Tensor>,
) -> Result<Vec<Tensor>> {
    let stride = if batch <= 1 { 1 } else { batch * rows_per_unit.max(1) };
    let mut read = Vec::with_capacity(rows.len());
    for pass in rows.chunks(stride) {
        let output = run(pass)?;
        for (index, ids) in pass.iter().enumerate() {
            read.push(row(&output, index, ids.len())?);
        }
    }
    Ok(read)
}
