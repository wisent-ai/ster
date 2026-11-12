//! The optimizer every objective steps: AdamW exactly as candle-nn 0.11
//! computes it (Loshchilov and Hutter's decoupled weight decay with Adam's
//! bias correction, over candle's own `ParamsAdamW`), owned here so that its
//! moments and its steps can be written to a checkpoint and read back.
//! candle's AdamW keeps both private, and a run resumed with fresh moments is
//! not the run that was stopped: its next steps would be larger than the
//! ones the uninterrupted run takes.

use std::collections::BTreeMap;
use std::sync::PoisonError;

use anyhow::Result;
use candle_core::{Tensor, Var, backprop::GradStore};
use candle_nn::{ParamsAdamW, VarMap};

use crate::failures::{
    amend, refuse, tensor,
    tune::run::{STEP_EMPTY, unallocatable},
};

/// One trained tensor and the two running moments AdamW keeps for it.
pub(crate) struct Tracked {
    pub var: Var,
    pub first: Var,
    pub second: Var,
}

pub(crate) struct AdamW {
    /// Keyed by the name the run's `VarMap` gives the tensor, so a checkpoint
    /// names each moment by the tensor it belongs to.
    pub(crate) tracked: BTreeMap<String, Tracked>,
    /// The learning rate of every step taken, in order: the run's schedule as
    /// it was applied, and its length is the `t` of the bias correction.
    pub(crate) rates: Vec<f64>,
    params: ParamsAdamW,
}

impl AdamW {
    /// An optimizer over every floating-point tensor `varmap` holds, with
    /// both moments at zero and no step taken, as candle's starts.
    pub(crate) fn over(varmap: &VarMap, params: ParamsAdamW) -> Result<Self> {
        // A poisoned lock only says another thread panicked holding it; the
        // map it guards is still whole, and nothing here writes to it.
        let named = varmap
            .data()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut tracked = BTreeMap::new();
        for (name, var) in named.iter().filter(|(_, var)| var.dtype().is_float()) {
            let moment = || {
                Var::zeros(var.shape(), var.dtype(), var.device()).map_err(|error| {
                    unallocatable(&format!("the AdamW moments of {name}"), var.dims(), error)
                })
            };
            let first = moment()?;
            let second = moment()?;
            tracked.insert(
                name.clone(),
                Tracked {
                    var: var.clone(),
                    first,
                    second,
                },
            );
        }
        Ok(Self {
            tracked,
            rates: Vec::new(),
            params,
        })
    }

    pub(crate) fn set_learning_rate(&mut self, learning_rate: f64) {
        self.params.lr = learning_rate;
    }

    /// Backpropagate the loss one step `summed` over `forwards` forwards
    /// holding `units` `unit`s, and take the step; refused as `step-empty`
    /// when the step summed no loss.
    pub(crate) fn backward_summed(
        &mut self,
        summed: Option<Tensor>,
        forwards: usize,
        units: usize,
        unit: &str,
    ) -> Result<()> {
        let Some(loss) = summed else {
            return Err(refuse(
                &STEP_EMPTY,
                &[
                    ("step", &(self.rates.len() + 1).to_string()),
                    ("forwards", &forwards.to_string()),
                    ("units", &units.to_string()),
                    ("unit", unit),
                    ("tensors", &self.tracked.len().to_string()),
                ],
            ));
        };
        self.backward_step(&loss)
    }

    /// Backpropagate `loss` and take one step.
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        // The step's number before `step` records its rate.
        let number = self.rates.len() + 1;
        let taken = loss.backward().and_then(|grads| self.step(&grads));
        taken.map_err(|error| {
            let failed = tensor::computing(&Self::taking(number, "gradients and AdamW update"), error);
            self.amended(failed, number)
        })
    }

    /// The refusal of the step being taken when its forward pass or its loss
    /// fails on the device: `error` itself when a cause already declares
    /// it, otherwise the cause `ster.tensor` declares for its Candle error.
    pub(crate) fn failed(&self, error: anyhow::Error) -> anyhow::Error {
        let number = self.rates.len() + 1;
        let failed = tensor::within(error, &Self::taking(number, "forward pass and loss"));
        self.amended(failed, number)
    }

    /// The step a refusal names: `taking optimizer step 3 (forward pass and
    /// loss)`.
    fn taking(number: usize, phase: &str) -> String {
        format!("taking optimizer step {number} ({phase})")
    }

    /// `failed` with what the step was: its learning rate and how many
    /// tensors it trains, and that no earlier checkpoint was touched.
    fn amended(&self, failed: anyhow::Error, number: usize) -> anyhow::Error {
        amend(
            failed,
            format!(
                "step {number} at learning rate {} over {} trained tensors; the run's earlier checkpoints are left as they were",
                self.params.lr,
                self.tracked.len()
            ),
        )
    }

    /// One AdamW step. `1 - b^t` is written `-expm1(t ln b)`, which is the
    /// same quantity computed without cancellation, and `1 - b` likewise.
    fn step(&mut self, grads: &GradStore) -> candle_core::Result<()> {
        let ParamsAdamW {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        } = self.params;
        self.rates.push(lr);
        let elapsed = self.rates.len() as f64;
        let complement = |beta: f64| -beta.ln().exp_m1();
        let corrected = |beta: f64| -(beta.ln() * elapsed).exp_m1();
        let (keep1, keep2) = (complement(beta1), complement(beta2));
        let (scale_m, scale_v) = (corrected(beta1).recip(), corrected(beta2).recip());
        for Tracked { var, first, second } in self.tracked.values() {
            let Some(gradient) = grads.get(var) else {
                continue;
            };
            let next_first = ((first.as_tensor() * beta1)? + (gradient * keep1)?)?;
            let next_second = ((second.as_tensor() * beta2)? + (gradient.sqr()? * keep2)?)?;
            let first_hat = (&next_first * scale_m)?;
            let second_hat = (&next_second * scale_v)?;
            let decayed = (var.as_tensor() - (var.as_tensor() * (lr * weight_decay))?)?;
            let adjusted = (first_hat / (second_hat.sqrt()? + eps)?)?;
            let next = (decayed - (adjusted * lr)?)?;
            first.set(&next_first)?;
            second.set(&next_second)?;
            var.set(&next)?;
        }
        Ok(())
    }
}
