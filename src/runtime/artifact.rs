//! Collecting what a run trained into the document that carries it: the
//! adapter factors, an optional reward head, and the record of the run.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;

use crate::lora;

use super::Runtime;

impl Runtime {
    /// Collects this runtime's trained adapters into a durable document.
    ///
    /// `train` is whatever the caller wants recorded about the run that
    /// produced them; `tune::sft` passes its serialized report so the artifact
    /// carries the losses and hyperparameters that made it.
    pub fn adapter_artifact(
        &self,
        spec: &lora::Spec,
        train: serde_json::Value,
    ) -> Result<lora::Artifact> {
        self.artifact(spec, lora::Kind::Adapter, None, train)
    }

    /// The same document plus the scalar head trained on top of these adapters.
    ///
    /// The head goes in the same safetensors file rather than beside it,
    /// because a head and the adapters that shaped the residual stream it
    /// reads are one model: separating them would let an operator pair a head
    /// with adapters it never saw and get scores that mean nothing.
    pub fn reward_artifact(
        &self,
        spec: &lora::Spec,
        head: &Tensor,
        train: serde_json::Value,
    ) -> Result<lora::Artifact> {
        self.artifact(spec, lora::Kind::Reward, Some(head), train)
    }

    fn artifact(
        &self,
        spec: &lora::Spec,
        kind: lora::Kind,
        head: Option<&Tensor>,
        train: serde_json::Value,
    ) -> Result<lora::Artifact> {
        spec.validate(self.layer_count())?;
        let spec = spec.resolved(self.layer_count());
        let adapters = self.model.adapters();
        if adapters.is_empty() {
            bail!("this runtime carries no adapters to write");
        }
        let mut tensors = BTreeMap::new();
        for &layer in &spec.layers {
            for &target in &spec.targets {
                let adapter = adapters.get(layer, target).with_context(|| {
                    format!("layer {layer} carries no {} adapter", target.name())
                })?;
                let (a, b) = lora::Adapter::tensor_names(layer, target);
                tensors.insert(a, adapter.a.clone());
                tensors.insert(b, adapter.b.clone());
            }
        }
        if let Some(head) = head {
            tensors.insert(lora::REWARD_HEAD_TENSOR.to_owned(), head.clone());
        }
        let artifact = lora::Artifact {
            schema_version: lora::ARTIFACT_SCHEMA_VERSION,
            product: "ster".to_owned(),
            kind,
            model: self.model_id.clone(),
            model_revision: self.revision.clone(),
            rank: spec.rank,
            alpha: spec.alpha,
            targets: spec.targets.clone(),
            layers: spec.layers.clone(),
            hidden_size: self.hidden_size(),
            train,
            tensors,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}
