//! Caller-owned inputs for an untrained local checkpoint, not a model preset.

use std::{
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use anyhow::Result;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use crate::failures::{
    fixture::{declaration as causes, json_refusal, read_refusal},
    refuse,
};

use super::tokenizer;

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Spec {
    pub config: PathBuf,
    pub tokenizer: TokenizerInput,
    pub initialization: Initialization,
    pub verification: Verification,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum TokenizerInput {
    File(PathBuf),
    Declaration(TokenizerDeclaration),
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TokenizerDeclaration {
    pub(super) from: PathBuf,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Initialization {
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed_why: Option<String>,
    pub embedding: Gaussian,
    pub projection: Gaussian,
    pub normalization: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalization_why: Option<String>,
    pub bias: Option<Gaussian>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Gaussian {
    pub mean: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_why: Option<String>,
    pub scale: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_why: Option<String>,
}

impl Gaussian {
    /// The normal distribution `field` of the declaration at `path`
    /// declares.
    pub fn distribution(&self, field: &str, path: &Path) -> Result<Normal<f32>> {
        let shown = path.display().to_string();
        for (key, value) in [("mean", self.mean), ("scale", self.scale)] {
            if !value.is_finite() {
                return Err(refuse(
                    &causes::NOT_FINITE,
                    &[
                        ("role", "declaration"),
                        ("path", &shown),
                        ("key", &format!("{field}.{key}")),
                        ("value", &value.to_string()),
                    ],
                ));
            }
        }
        Normal::new(self.mean, self.scale).map_err(|error| {
            refuse(
                &causes::SCALE_NEGATIVE,
                &[
                    ("field", field),
                    ("mean", &self.mean.to_string()),
                    ("scale", &self.scale.to_string()),
                    ("error", &error.to_string()),
                    ("path", &shown),
                ],
            )
        })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Verification {
    pub prompt: String,
    pub layers: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers_why: Option<String>,
}

// These are native Hugging Face Llama configuration identities, not a word
// classifier: https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig
// Require the declared values instead of manufacturing a fixture architecture.
#[derive(Deserialize, Serialize)]
pub(super) struct Dimensions {
    pub model_type: String,
    pub hidden_size: NonZeroUsize,
    pub intermediate_size: NonZeroUsize,
    pub vocab_size: NonZeroUsize,
    pub num_hidden_layers: NonZeroUsize,
    pub num_attention_heads: NonZeroUsize,
    pub num_key_value_heads: NonZeroUsize,
    pub head_dim: NonZeroUsize,
    pub max_position_embeddings: NonZeroUsize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
}

pub(super) struct Resolved {
    pub spec: Spec,
    /// The declaration's path, resolved.
    pub declaration: PathBuf,
    pub dimensions: Dimensions,
    pub config_bytes: Vec<u8>,
    pub tokenizer_bytes: Vec<u8>,
    pub tokenizer_verification: Option<crate::tokenizer::VerificationReport>,
    pub query_width: usize,
    pub kv_width: usize,
}

impl Resolved {
    pub fn read(path: &Path) -> Result<Self> {
        let path = path
            .canonicalize()
            .map_err(|error| read_refusal("declaration", path, &error))?;
        let bytes = fs::read(&path).map_err(|error| read_refusal("declaration", &path, &error))?;
        let spec: Spec = serde_json::from_slice(&bytes)
            .map_err(|error| json_refusal("declaration", &path, &error))?;
        // A canonical path to a file always has a parent; the root itself is
        // a directory, which reading it as a declaration refused above.
        let parent = path.parent().map_or_else(|| PathBuf::from("/"), Path::to_path_buf);
        let config_path = parent.join(&spec.config);
        let config_bytes =
            fs::read(&config_path).map_err(|error| read_refusal("config", &config_path, &error))?;
        let dimensions: Dimensions = serde_json::from_slice(&config_bytes)
            .map_err(|error| json_refusal("config", &config_path, &error))?;
        check(&dimensions, &config_path, &spec, &path)?;
        let (tokenizer_bytes, tokenizer_verification) =
            tokenizer::resolve(&spec.tokenizer, &parent, &dimensions)?;
        let width = |heads: NonZeroUsize, what: &str| {
            dimensions.head_dim.get().checked_mul(heads.get()).ok_or_else(|| {
                refuse(
                    &causes::WIDTH_OVERFLOW,
                    &[
                        ("what", what),
                        ("path", &config_path.display().to_string()),
                        ("head_dim", &dimensions.head_dim.to_string()),
                        ("heads", &heads.to_string()),
                        ("bits", &usize::BITS.to_string()),
                    ],
                )
            })
        };
        let query_width = width(dimensions.num_attention_heads, "query")?;
        let kv_width = width(dimensions.num_key_value_heads, "key/value")?;
        Ok(Self {
            spec,
            declaration: path,
            dimensions,
            config_bytes,
            tokenizer_bytes,
            tokenizer_verification,
            query_width,
            kv_width,
        })
    }
}

/// Refuses what the config at `config` and the declaration at `path`
/// declare that the fixture cannot write or verify.
fn check(dimensions: &Dimensions, config: &Path, spec: &Spec, path: &Path) -> Result<()> {
    let config_shown = config.display().to_string();
    let shown = path.display().to_string();
    if dimensions.model_type != "llama" {
        let model_type = format!("{:?}", dimensions.model_type);
        return Err(refuse(
            &causes::MODEL_TYPE,
            &[("path", &config_shown), ("model_type", &model_type)],
        ));
    }
    if dimensions.mlp_bias {
        return Err(refuse(&causes::MLP_BIAS, &[("path", &config_shown)]));
    }
    // Llama grouped-query attention requires whole groups of heads:
    // https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig
    let (heads, kv) = (dimensions.num_attention_heads.get(), dimensions.num_key_value_heads.get());
    if heads % kv != 0 {
        return Err(refuse(
            &causes::HEADS_UNGROUPED,
            &[
                ("path", &config_shown),
                ("heads", &heads.to_string()),
                ("kv", &kv.to_string()),
                ("remainder", &(heads % kv).to_string()),
            ],
        ));
    }
    let normalization = f64::from(spec.initialization.normalization);
    let settings = [
        ("config", &config_shown, "rms_norm_eps", dimensions.rms_norm_eps),
        ("config", &config_shown, "rope_theta", dimensions.rope_theta),
        ("declaration", &shown, "initialization.normalization", normalization),
    ];
    for (role, at, key, value) in settings {
        if !value.is_finite() {
            return Err(refuse(
                &causes::NOT_FINITE,
                &[("role", role), ("path", at.as_str()), ("key", key), ("value", &value.to_string())],
            ));
        }
    }
    if spec.initialization.bias.is_none() && dimensions.attention_bias {
        return Err(refuse(&causes::BIAS_MISSING, &[("path", &shown), ("config", &config_shown)]));
    }
    let prompt = &spec.verification.prompt;
    if prompt.trim().is_empty() {
        return Err(refuse(
            &causes::PROMPT_BLANK,
            &[("path", &shown), ("prompt", &format!("{prompt:?}"))],
        ));
    }
    let count = dimensions.num_hidden_layers.get();
    let (count_shown, last) = (count.to_string(), (count - 1).to_string());
    let layers = format!("{:?}", spec.verification.layers);
    if spec.verification.layers.is_empty() {
        return Err(refuse(
            &causes::LAYERS_NONE,
            &[("path", &shown), ("count", &count_shown), ("last", &last)],
        ));
    }
    if let Some(layer) = spec.verification.layers.iter().find(|layer| **layer >= count) {
        return Err(refuse(
            &causes::LAYER_OUTSIDE,
            &[
                ("path", &shown),
                ("layer", &layer.to_string()),
                ("layers", &layers),
                ("count", &count_shown),
                ("last", &last),
            ],
        ));
    }
    Ok(())
}
