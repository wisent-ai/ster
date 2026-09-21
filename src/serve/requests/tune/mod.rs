//! The adapter-training endpoints' requests: supervised fine-tuning,
//! preference optimization, reward modelling, policy optimization, and the
//! merge, evaluate and inspect surfaces beside them.

use serde::Deserialize;

use super::defaults::*;
use super::{require, ModelRequest, Validate};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::serve) struct TuneSftRequest {
    #[serde(flatten)]
    pub(in crate::serve) model: ModelRequest,
    #[serde(default)]
    pub(in crate::serve) examples: String,
    #[serde(default)]
    pub(in crate::serve) output: String,
    #[serde(default = "default_rank")]
    pub(in crate::serve) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::serve) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::serve) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::serve) layers: String,
    #[serde(default = "default_epochs")]
    pub(in crate::serve) epochs: usize,
    #[serde(default = "default_learning_rate")]
    pub(in crate::serve) learning_rate: f64,
    #[serde(default = "default_accumulation")]
    pub(in crate::serve) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::serve) warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    pub(in crate::serve) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::serve) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::serve) chat_template: String,
    /// Rows folded into one forward pass — examples here, pairs on the
    /// preference endpoints, where a pair is two rows. One is the unbatched
    /// pass every run recorded so far took.
    #[serde(default = "default_batch_size")]
    pub(in crate::serve) batch_size: usize,
    /// The dtype the frozen base weights are mapped at: `f32`, `f16`, or
    /// `bf16`. Adapters, any head, and every optimizer moment stay in f32
    /// whatever this says. `bf16` needs the `metal` device.
    #[serde(default = "default_precision")]
    pub(in crate::serve) precision: String,
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
pub(in crate::serve) struct TuneDpoRequest {
    #[serde(flatten)]
    pub(in crate::serve) model: ModelRequest,
    /// A contrastive pair set. The positive side is the chosen response and
    /// the negative side the rejected one, so the file Train already reads is
    /// the file this reads.
    #[serde(default)]
    pub(in crate::serve) pairs: String,
    #[serde(default)]
    pub(in crate::serve) output: String,
    #[serde(default = "default_rank")]
    pub(in crate::serve) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::serve) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::serve) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::serve) layers: String,
    #[serde(default = "default_beta")]
    pub(in crate::serve) beta: f64,
    #[serde(default = "default_preference_loss")]
    pub(in crate::serve) loss: String,
    #[serde(default = "default_epochs")]
    pub(in crate::serve) epochs: usize,
    #[serde(default = "default_learning_rate")]
    pub(in crate::serve) learning_rate: f64,
    #[serde(default = "default_accumulation")]
    pub(in crate::serve) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::serve) warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    pub(in crate::serve) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::serve) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::serve) chat_template: String,
    #[serde(default = "default_batch_size")]
    pub(in crate::serve) batch_size: usize,
    #[serde(default = "default_precision")]
    pub(in crate::serve) precision: String,
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
pub(in crate::serve) struct TuneRewardRequest {
    #[serde(flatten)]
    pub(in crate::serve) model: ModelRequest,
    /// A contrastive pair set. The positive side is the response the head
    /// learns to score higher.
    #[serde(default)]
    pub(in crate::serve) pairs: String,
    #[serde(default)]
    pub(in crate::serve) output: String,
    #[serde(default = "default_rank")]
    pub(in crate::serve) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::serve) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::serve) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::serve) layers: String,
    #[serde(default = "default_epochs")]
    pub(in crate::serve) epochs: usize,
    #[serde(default = "default_learning_rate")]
    pub(in crate::serve) learning_rate: f64,
    #[serde(default = "default_accumulation")]
    pub(in crate::serve) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::serve) warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    pub(in crate::serve) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::serve) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::serve) chat_template: String,
    #[serde(default = "default_batch_size")]
    pub(in crate::serve) batch_size: usize,
    #[serde(default = "default_precision")]
    pub(in crate::serve) precision: String,
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
pub(in crate::serve) struct TuneGrpoRequest {
    #[serde(flatten)]
    pub(in crate::serve) model: ModelRequest,
    /// A prompt set, `{"prompts": ["…"]}` — the shape extract already takes.
    #[serde(default)]
    pub(in crate::serve) prompts: String,
    #[serde(default)]
    pub(in crate::serve) output: String,
    /// The keyword `length`, or the path to a reward artifact.
    #[serde(default = "default_reward")]
    pub(in crate::serve) reward: String,
    #[serde(default = "default_group")]
    pub(in crate::serve) group: usize,
    #[serde(default = "default_iterations")]
    pub(in crate::serve) iterations: usize,
    #[serde(default = "default_kl_beta")]
    pub(in crate::serve) beta: f64,
    #[serde(default = "default_rank")]
    pub(in crate::serve) rank: usize,
    #[serde(default = "default_alpha")]
    pub(in crate::serve) alpha: f64,
    #[serde(default = "default_targets")]
    pub(in crate::serve) targets: String,
    #[serde(default = "default_layers")]
    pub(in crate::serve) layers: String,
    #[serde(default = "default_learning_rate")]
    pub(in crate::serve) learning_rate: f64,
    /// One group is already `group` sequences, so a step per group is the
    /// natural unit and the default is one rather than eight.
    #[serde(default = "default_group_accumulation")]
    pub(in crate::serve) accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    pub(in crate::serve) warmup_steps: usize,
    #[serde(default = "default_grpo_max_new_tokens")]
    pub(in crate::serve) max_new_tokens: usize,
    #[serde(default = "default_grpo_temperature")]
    pub(in crate::serve) temperature: f64,
    #[serde(default = "default_top_p")]
    pub(in crate::serve) top_p: f64,
    #[serde(default = "default_max_sequence")]
    pub(in crate::serve) max_sequence: usize,
    #[serde(default = "default_seed")]
    pub(in crate::serve) seed: u64,
    #[serde(default = "default_chat_template")]
    pub(in crate::serve) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::serve) precision: String,
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
pub(in crate::serve) struct TuneMergeRequest {
    #[serde(flatten)]
    pub(in crate::serve) model: ModelRequest,
    /// The adapter to fold in; it must be a generation adapter trained for
    /// this exact model.
    #[serde(default)]
    pub(in crate::serve) adapter: String,
    /// Directory to write: model.safetensors beside the source's own
    /// config.json and tokenizer.json, plus whichever of
    /// tokenizer_config.json and chat_template.jinja the source published,
    /// which together are what `model` accepts. Those last two are where a
    /// chat template lives, so a source that published one merges to a
    /// checkpoint that still reports `applied` rather than `absent`.
    #[serde(default)]
    pub(in crate::serve) output: String,
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
pub(in crate::serve) struct TuneEvaluateRequest {
    #[serde(flatten)]
    pub(in crate::serve) model: ModelRequest,
    #[serde(default)]
    pub(in crate::serve) examples: String,
    /// A frozen adapter to attach before scoring; omit or leave empty to score
    /// the bare checkpoint, which is the run an adapter is compared against.
    #[serde(default)]
    pub(in crate::serve) adapter: Option<String>,
    #[serde(default = "default_max_sequence")]
    pub(in crate::serve) max_sequence: usize,
    #[serde(default = "default_chat_template")]
    pub(in crate::serve) chat_template: String,
    #[serde(default = "default_batch_size")]
    pub(in crate::serve) batch_size: usize,
    #[serde(default = "default_precision")]
    pub(in crate::serve) precision: String,
}

impl Validate for TuneEvaluateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune evaluate")?;
        require(&self.examples, "tune evaluate requires an example set".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::serve) struct TuneInspectRequest {
    #[serde(default)]
    pub(in crate::serve) artifact: String,
}

impl Validate for TuneInspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.artifact, "tune inspect requires an adapter artifact".to_owned())
    }
}
