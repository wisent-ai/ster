//! The optimizer every objective steps: AdamW exactly as candle-nn 0.11
//! computes it (Loshchilov and Hutter's decoupled weight decay with Adam's
//! bias correction, over candle's own `ParamsAdamW`), owned here so that its
//! moments and its steps can be written to a checkpoint and read back.
//! candle's AdamW keeps both private, and a run resumed with fresh moments is
//! not the run that was stopped: its next steps would be larger than the
//! ones the uninterrupted run takes.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow};
use candle_core::{Tensor, Var, backprop::GradStore};
use candle_nn::{ParamsAdamW, VarMap};

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
        let named = varmap
            .data()
            .lock()
            .map_err(|poisoned| anyhow!("the run's tensor map is unreadable: {poisoned}"))?;
        let mut tracked = BTreeMap::new();
        for (name, var) in named.iter().filter(|(_, var)| var.dtype().is_float()) {
            let first = Var::zeros(var.shape(), var.dtype(), var.device())
                .with_context(|| format!("the first moment of {name} could not be allocated"))?;
            let second = Var::zeros(var.shape(), var.dtype(), var.device())
                .with_context(|| format!("the second moment of {name} could not be allocated"))?;
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

    /// Backpropagate `loss` and take one step.
    pub(crate) fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    /// One AdamW step. `1 - b^t` is written `-expm1(t ln b)`, which is the
    /// same quantity computed without cancellation, and `1 - b` likewise.
    fn step(&mut self, grads: &GradStore) -> Result<()> {
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
