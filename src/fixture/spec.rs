//! Caller-owned inputs for an untrained local checkpoint, not a model preset.

use std::{
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

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
    from: PathBuf,
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
    pub fn distribution(&self, field: &str) -> Result<Normal<f32>> {
        ensure!(
            self.mean.is_finite() && self.scale.is_finite(),
            "fixture {field} requires finite mean and scale; observed mean={}, scale={}",
            self.mean,
            self.scale
        );
        Normal::new(self.mean, self.scale)
            .with_context(|| format!("construct fixture {field} normal distribution"))
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
            .with_context(|| format!("resolve fixture declaration {}", path.display()))?;
        let bytes = fs::read(&path)
            .with_context(|| format!("read fixture declaration {}", path.display()))?;
        let spec: Spec = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse fixture declaration {}", path.display()))?;
        let parent = path
            .parent()
            .context("fixture declaration has no parent directory")?;
        let config_path = parent.join(&spec.config);
        let config_bytes = fs::read(&config_path)
            .with_context(|| format!("read fixture config {}", config_path.display()))?;
        let dimensions: Dimensions = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("parse fixture config {}", config_path.display()))?;
        ensure!(
            dimensions.model_type == "llama",
            "fixture tensor layout requires model_type=llama; {} declares {:?}",
            config_path.display(),
            dimensions.model_type
        );
        ensure!(
            !dimensions.mlp_bias,
            "fixture config declares mlp_bias=true, but Ster's Llama consumer does not apply feed-forward biases; no checkpoint was written"
        );
        // Llama grouped-query attention requires whole groups of heads:
        // https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig
        ensure!(
            dimensions.num_attention_heads.get() % dimensions.num_key_value_heads.get() == 0,
            "fixture attention heads {} must be divisible by key/value heads {}",
            dimensions.num_attention_heads,
            dimensions.num_key_value_heads
        );
        ensure!(
            dimensions.rms_norm_eps.is_finite() && dimensions.rope_theta.is_finite(),
            "fixture config requires finite rms_norm_eps and rope_theta"
        );
        ensure!(
            spec.initialization.normalization.is_finite(),
            "fixture initialization.normalization must be finite; observed {}",
            spec.initialization.normalization
        );
        if spec.initialization.bias.is_none() && dimensions.attention_bias {
            bail!(
                "fixture initialization.bias is required because config declares attention_bias=true"
            );
        }
        ensure!(
            !spec.verification.prompt.trim().is_empty(),
            "fixture verification.prompt is required"
        );
        ensure!(
            !spec.verification.layers.is_empty(),
            "fixture verification.layers is required"
        );
        for layer in &spec.verification.layers {
            ensure!(
                *layer < dimensions.num_hidden_layers.get(),
                "fixture verification layer {layer} lies outside {} declared layers",
                dimensions.num_hidden_layers
            );
        }
        let (tokenizer_bytes, tokenizer_verification) = match &spec.tokenizer {
            TokenizerInput::File(source) => {
                let source = parent.join(source);
                let bytes = fs::read(&source)
                    .with_context(|| format!("read fixture tokenizer {}", source.display()))?;
                let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|error| {
                    anyhow::anyhow!("parse fixture tokenizer {}: {error}", source.display())
                })?;
                let vocab = tokenizer.get_vocab(true);
                ensure!(
                    !vocab.is_empty(),
                    "fixture tokenizer {} has no vocabulary",
                    source.display()
                );
                for (token, id) in vocab {
                    ensure!(
                        usize::try_from(id).context("token id does not fit this host")?
                            < dimensions.vocab_size.get(),
                        "fixture tokenizer token {token:?} has id {id}, outside config vocab_size {}",
                        dimensions.vocab_size
                    );
                }
                (bytes, None)
            }
            TokenizerInput::Declaration(source) => {
                let source = parent.join(&source.from);
                let compiled = crate::tokenizer::compile(&source).with_context(|| {
                    format!("compile fixture tokenizer declaration {}", source.display())
                })?;
                // This producer assigns contiguous model ids by declared order
                // and native added ids after them, so its measured unique count
                // bounds every vocabulary id without parsing the same bytes twice.
                ensure!(
                    compiled.vocabulary_size <= dimensions.vocab_size.get(),
                    "compiled fixture tokenizer {} has {} token identities, exceeding config vocab_size {}",
                    source.display(),
                    compiled.vocabulary_size,
                    dimensions.vocab_size
                );
                (compiled.document.into_bytes(), Some(compiled.verification))
            }
        };
        let query_width = dimensions
            .head_dim
            .get()
            .checked_mul(dimensions.num_attention_heads.get())
            .context("fixture head_dim times num_attention_heads overflows this host")?;
        let kv_width = dimensions
            .head_dim
            .get()
            .checked_mul(dimensions.num_key_value_heads.get())
            .context("fixture head_dim times num_key_value_heads overflows this host")?;
        Ok(Self {
            spec,
            dimensions,
            config_bytes,
            tokenizer_bytes,
            tokenizer_verification,
            query_width,
            kv_width,
        })
    }
}
