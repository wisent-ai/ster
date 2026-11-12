//! Fitting a trained artifact to the safetensors checkpoint a merge, an
//! export or an ablation writes it into, refusing through `ster.tune.fit`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::llama::Config;
use wisent_errors::Cause;

use crate::failures::{
    refuse, refuse_from,
    tune::rewrite::{
        LAYER_PAST, MODEL_OTHER, PROJECTION_ABSENT, TENSOR_MISSING, UPDATE_UNCOMPUTED, UPDATE_UNFIT,
        WIDTH_OTHER,
    },
};
use crate::{
    lora::{self, Target},
    model::{Architecture, Placement},
    runtime::{Checkpoint, Layout},
};

/// What one rewrite is doing, as each of its refusals names it.
pub(crate) struct Fit<'a> {
    /// `merging`, `exporting` or `ablating`.
    pub step: &'a str,
    pub path: &'a Path,
    pub model: &'a str,
}

/// The checkpoint an artifact is fitted to, its decoder config read.
pub(crate) struct Source {
    pub checkpoint: Checkpoint,
    pub config: Config,
    pub architecture: Architecture,
}

impl Fit<'_> {
    fn refuse(&self, cause: &'static Cause, values: &[(&str, &str)]) -> anyhow::Error {
        let path = self.path.display().to_string();
        let mut all = vec![("step", self.step), ("path", path.as_str()), ("model", self.model)];
        all.extend_from_slice(values);
        refuse(cause, &all)
    }

    /// Refuses a reward model's artifact through `refusal`, the cause of this
    /// rewrite (`REWARD_UNMERGEABLE`, `REWARD_UNEXPORTABLE`) that says why it
    /// has no place for the head.
    pub fn adapter(&self, artifact: &lora::Artifact, refusal: &'static Cause) -> Result<()> {
        if artifact.kind == lora::Kind::Adapter {
            return Ok(());
        }
        Err(self.refuse(refusal, &[("kind", artifact.kind.name())]))
    }

    /// Refuses an artifact trained for another model than this one.
    pub fn trained_for(&self, trained: &str) -> Result<()> {
        if trained == self.model {
            return Ok(());
        }
        Err(self.refuse(&MODEL_OTHER, &[("trained", &format!("{trained:?}"))]))
    }

    /// The safetensors checkpoint `revision` of the model names, its config
    /// read; a GGUF checkpoint is refused through `gguf`, the cause of this
    /// rewrite (`GGUF_UNMERGEABLE`, `GGUF_UNEXPORTABLE`, `GGUF_UNABLATABLE`).
    pub fn source(&self, revision: Option<&str>, gguf: &'static Cause) -> Result<Source> {
        let checkpoint = Checkpoint::resolve(self.model, revision)?;
        if checkpoint.layout == Layout::Gguf {
            return Err(self.refuse(gguf, &[]));
        }
        let (config, architecture, _) = checkpoint.decoder_config()?;
        Ok(Source {
            checkpoint,
            config,
            architecture,
        })
    }

    /// Refuses an artifact `trained` wide on a model `width` wide.
    pub fn width(&self, trained: usize, width: usize) -> Result<()> {
        if trained == width {
            return Ok(());
        }
        Err(self.refuse(
            &WIDTH_OTHER,
            &[
                ("trained", &trained.to_string()),
                ("width", &width.to_string()),
            ],
        ))
    }

    /// Refuses an artifact naming a layer past the model's `count` layers.
    pub fn layers(&self, layers: &[usize], count: usize) -> Result<()> {
        let Some(layer) = layers.iter().find(|layer| **layer >= count) else {
            return Ok(());
        };
        let listed: Vec<String> = layers.iter().map(usize::to_string).collect();
        Err(self.refuse(
            &LAYER_PAST,
            &[
                ("layer", &layer.to_string()),
                ("layers", &count.to_string()),
                ("last", &count.saturating_sub(1).to_string()),
                ("artifact_layers", &listed.join(", ")),
            ],
        ))
    }

    /// Where `target` of `layer` sits in the checkpoint.
    pub fn placement(&self, source: &Source, target: Target, layer: usize) -> Result<Placement> {
        source
            .architecture
            .placement(target, layer, &source.config)
            .ok_or_else(|| {
                self.refuse(
                    &PROJECTION_ABSENT,
                    &[
                        ("target", target.name()),
                        ("layer", &layer.to_string()),
                        ("targets", target.name()),
                    ],
                )
            })
    }

    /// Refuses a projection stored fused with others through `fused`, and,
    /// when this rewrite cannot write one stored transposed either, a
    /// transposed one through `transposed`: the causes of this rewrite
    /// (`FUSED_UNEXPORTABLE` and `TRANSPOSED_UNEXPORTABLE`, or
    /// `FUSED_UNABLATABLE`) that say why it cannot write them.
    pub fn whole(
        &self,
        placement: &Placement,
        (target, layer): (Target, usize),
        fused: &'static Cause,
        transposed: Option<&'static Cause>,
    ) -> Result<()> {
        let refusal = match (placement.blocks.is_some(), transposed) {
            (true, _) => fused,
            (false, Some(transposed)) if placement.transposed => transposed,
            (false, _) => return Ok(()),
        };
        Err(self.refuse(
            refusal,
            &[
                ("target", target.name()),
                ("layer", &layer.to_string()),
                ("tensor", &placement.tensor),
            ],
        ))
    }

    /// The tensor `name` of `tensors`, which `holder` (the checkpoint or
    /// the adapter artifact) must hold for `target` of `layer`.
    pub fn tensor<'t>(
        &self,
        tensors: &'t BTreeMap<String, Tensor>,
        holder: &str,
        name: &str,
        target: Target,
        layer: usize,
    ) -> Result<&'t Tensor> {
        tensors.get(name).ok_or_else(|| {
            self.refuse(
                &TENSOR_MISSING,
                &[
                    ("target", target.name()),
                    ("layer", &layer.to_string()),
                    ("holder", holder),
                    ("name", name),
                    ("count", &tensors.len().to_string()),
                ],
            )
        })
    }

    /// The refusal of an update to `name` of `shape` the arithmetic could
    /// not apply.
    pub fn unfit(
        &self,
        target: Target,
        layer: usize,
        name: &str,
        shape: &[usize],
        error: candle_core::Error,
    ) -> anyhow::Error {
        let path = self.path.display().to_string();
        let detail = error.to_string();
        let cause = if crate::failures::shapes_differ(&error) {
            &UPDATE_UNFIT
        } else {
            &UPDATE_UNCOMPUTED
        };
        refuse_from(
            cause,
            &[
                ("step", self.step),
                ("path", &path),
                ("model", self.model),
                ("target", target.name()),
                ("layer", &layer.to_string()),
                ("name", name),
                ("shape", &format!("{shape:?}")),
                ("error", &detail),
            ],
            error.into(),
        )
    }
}
