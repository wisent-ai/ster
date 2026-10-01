//! The adapter-training operations' requests: supervised fine-tuning,
//! preference optimization, reward modelling, policy optimization; the
//! merge, evaluate and inspect surfaces beside them live in `artifacts`.

use serde::Deserialize;

use super::defaults::*;
use super::{ModelRequest, Validate, require};

mod artifacts;

pub(in crate::request) use artifacts::{TuneEvaluateRequest, TuneInspectRequest, TuneMergeRequest};

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
        require(
            &self.examples,
            "tune sft requires an example set".to_owned(),
        )?;
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
        require(
            &self.output,
            "tune reward requires an output path".to_owned(),
        )
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
    /// Assistant turns per sampled conversation; above one, `userModel`
    /// writes the user's turns in between.
    #[serde(default = "default_turns")]
    pub(in crate::request) turns: usize,
    /// The model that plays the user when `turns` is above one.
    #[serde(default)]
    pub(in crate::request) user_model: Option<String>,
    #[serde(default)]
    pub(in crate::request) user_revision: Option<String>,
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
        require(
            &self.reward,
            "tune grpo requires a reward source".to_owned(),
        )
    }
}
