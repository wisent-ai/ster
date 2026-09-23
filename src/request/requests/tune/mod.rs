//! The adapter-training operations' requests: supervised fine-tuning,
//! preference optimization, reward modelling, policy optimization, and the
//! merge, evaluate and inspect surfaces beside them.

use serde::Deserialize;

use super::defaults::*;
use super::{require, ModelRequest, Validate};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneSftRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) examples: String,
    #[serde(default)]
    pub(in crate::request) output: String,
    #[serde(default = "default_rank")]
    pub(in crate::request) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::request) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::request) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    #[serde(default = "default_epochs")]
    pub(in crate::request) epochs: usize,
    #[serde(default = "default_learning_rate")]
    pub(in crate::request) learning_rate: f64,
    #[serde(default = "default_accumulation")]
    pub(in crate::request) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::request) warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    pub(in crate::request) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::request) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    /// Rows folded into one forward pass — examples here, pairs on the
    /// preference operations, where a pair is two rows. One is the unbatched
    /// pass every run recorded so far took.
    #[serde(default = "default_batch_size")]
    pub(in crate::request) batch_size: usize,
    /// The dtype the frozen base weights are mapped at: `f32`, `f16`, or
    /// `bf16`. Adapters, any head, and every optimizer moment stay in f32
    /// whatever this says. `bf16` needs the `metal` device.
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for TuneSftRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune sft")?;
        require(&self.examples, "tune sft requires an example set".to_owned())?;
        require(&self.output, "tune sft requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneDpoRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// A contrastive pair set. The positive side is the chosen response and
    /// the negative side the rejected one, so the file Train already reads is
    /// the file this reads.
    #[serde(default)]
    pub(in crate::request) pairs: String,
    #[serde(default)]
    pub(in crate::request) output: String,
    #[serde(default = "default_rank")]
    pub(in crate::request) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::request) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::request) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    #[serde(default = "default_beta")]
    pub(in crate::request) beta: f64,
    #[serde(default = "default_preference_loss")]
    pub(in crate::request) loss: String,
    #[serde(default = "default_epochs")]
    pub(in crate::request) epochs: usize,
    #[serde(default = "default_learning_rate")]
    pub(in crate::request) learning_rate: f64,
    #[serde(default = "default_accumulation")]
    pub(in crate::request) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::request) warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    pub(in crate::request) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::request) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_batch_size")]
    pub(in crate::request) batch_size: usize,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for TuneDpoRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune dpo")?;
        require(&self.pairs, "tune dpo requires a pairs file".to_owned())?;
        require(&self.output, "tune dpo requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneRewardRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// A contrastive pair set. The positive side is the response the head
    /// learns to score higher.
    #[serde(default)]
    pub(in crate::request) pairs: String,
    #[serde(default)]
    pub(in crate::request) output: String,
    #[serde(default = "default_rank")]
    pub(in crate::request) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::request) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::request) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    #[serde(default = "default_epochs")]
    pub(in crate::request) epochs: usize,
    #[serde(default = "default_learning_rate")]
    pub(in crate::request) learning_rate: f64,
    #[serde(default = "default_accumulation")]
    pub(in crate::request) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::request) warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    pub(in crate::request) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::request) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_batch_size")]
    pub(in crate::request) batch_size: usize,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for TuneRewardRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune reward")?;
        require(&self.pairs, "tune reward requires a pairs file".to_owned())?;
        require(&self.output, "tune reward requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneGrpoRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// A prompt set, `{"prompts": ["…"]}` — the shape extract already takes.
    #[serde(default)]
    pub(in crate::request) prompts: String,
    #[serde(default)]
    pub(in crate::request) output: String,
    /// The keyword `length`, or the path to a reward artifact.
    #[serde(default = "default_reward")]
    pub(in crate::request) reward: String,
    #[serde(default = "default_group")]
    pub(in crate::request) group: usize,
    #[serde(default = "default_iterations")]
    pub(in crate::request) iterations: usize,
    #[serde(default = "default_kl_beta")]
    pub(in crate::request) beta: f64,
    #[serde(default = "default_rank")]
    pub(in crate::request) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::request) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::request) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    #[serde(default = "default_learning_rate")]
    pub(in crate::request) learning_rate: f64,
    /// One group is already `group` sequences, so a step per group is the
    /// natural unit and the default is one rather than eight.
    #[serde(default = "default_group_accumulation")]
    pub(in crate::request) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::request) warmup_steps: usize,
    #[serde(default = "default_grpo_max_new_tokens")]
    pub(in crate::request) max_new_tokens: usize,
    #[serde(default = "default_grpo_temperature")]
    pub(in crate::request) temperature: f64,
    #[serde(default = "default_top_p")]
    pub(in crate::request) top_p: f64,
    #[serde(default = "default_max_sequence")]
    pub(in crate::request) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::request) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for TuneGrpoRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune grpo")?;
        require(&self.prompts, "tune grpo requires a prompt set".to_owned())?;
        require(&self.output, "tune grpo requires an output path".to_owned())?;
        require(&self.reward, "tune grpo requires a reward source".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneMergeRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// The adapter to fold in; it must be a generation adapter trained for
    /// this exact model.
    #[serde(default)]
    pub(in crate::request) adapter: String,
    /// Directory to write: model.safetensors beside the source's own
    /// config.json and tokenizer.json, plus whichever of
    /// tokenizer_config.json and chat_template.jinja the source published,
    /// which together are what `model` accepts. Those last two are where a
    /// chat template lives, so a source that published one merges to a
    /// checkpoint that still reports `applied` rather than `absent`.
    #[serde(default)]
    pub(in crate::request) output: String,
}

impl Validate for TuneMergeRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune merge")?;
        require(&self.adapter, "tune merge requires an adapter".to_owned())?;
        require(&self.output, "tune merge requires an output directory".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneEvaluateRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) examples: String,
    /// A frozen adapter to attach before scoring; omit or leave empty to score
    /// the bare checkpoint, which is the run an adapter is compared against.
    #[serde(default)]
    pub(in crate::request) adapter: Option<String>,
    #[serde(default = "default_max_sequence")]
    pub(in crate::request) max_sequence: usize,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_batch_size")]
    pub(in crate::request) batch_size: usize,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for TuneEvaluateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune evaluate")?;
        require(&self.examples, "tune evaluate requires an example set".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneInspectRequest {
    #[serde(default)]
    pub(in crate::request) artifact: String,
}

impl Validate for TuneInspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.artifact, "tune inspect requires an adapter artifact".to_owned())
    }
}
