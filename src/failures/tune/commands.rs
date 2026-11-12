//! The `ster tune` commands and their `ster request tune/*` twins: each is
//! the functionalities it runs, in the order a run meets them, and declares
//! no cause of its own.

use wisent_errors::Operation;

use super::super::lora::{SPEC, artifact::ARTIFACT};
use super::super::model::weights::LOAD;
use super::super::pairs::FILE;
use super::super::workspace::ACTIVE;
use super::super::workflow::prompts::PROMPTS;
use super::DOCS;
use super::checkpoint::SAVE;
use super::examples::EXAMPLES;
use super::objectives::OBJECTIVES;
use super::policy::REWARD;
use super::resume::RESUME;
use super::rollout::ROLLOUT;
use super::run::RUN;
use super::settings::SETTINGS;

/// `ster tune sft`: supervised fine-tuning on prompt and completion examples.
pub static SFT: Operation = Operation {
    id: "ster.tune.sft",
    entries: &["ster tune sft", "ster request tune/sft"],
    uses: &[
        &SPEC, &SETTINGS, &EXAMPLES, &LOAD, &RUN, &RESUME, &SAVE, &ARTIFACT,
    ],
    summary: "train LoRA adapters on prompt and completion examples, the loss taken over each completion only",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// `ster tune dpo`, `bipo` and `reward`: training on preference pairs.
pub static PREFERENCE: Operation = Operation {
    id: "ster.tune.preference",
    entries: &[
        "ster tune dpo",
        "ster tune bipo",
        "ster tune reward",
        "ster request tune/dpo",
        "ster request tune/bipo",
        "ster request tune/reward",
    ],
    uses: &[
        &SPEC,
        &SETTINGS,
        &OBJECTIVES,
        &ACTIVE,
        &FILE,
        &LOAD,
        &RUN,
        &RESUME,
        &SAVE,
        &ARTIFACT,
    ],
    summary: "train on contrastive pairs: LoRA adapters that prefer each positive side (DPO, IPO), one steering vector both ways (BiPO), or a scalar reward head that ranks the two sides",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// `ster tune grpo`: policy optimization against a reward.
pub static GRPO: Operation = Operation {
    id: "ster.tune.grpo",
    entries: &["ster tune grpo", "ster request tune/grpo"],
    uses: &[
        &SPEC, &SETTINGS, &OBJECTIVES, &REWARD, &PROMPTS, &LOAD, &ROLLOUT, &RUN, &RESUME, &SAVE,
        &ARTIFACT,
    ],
    summary: "optimize LoRA adapters against a reward, each prompt's sampled group its own baseline",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// `ster tune decide`: calibrated decision training on labelled decisions.
pub static DECIDE: Operation = Operation {
    id: "ster.tune.decide",
    entries: &["ster tune decide", "ster request tune/decide"],
    uses: &[
        &SPEC,
        &SETTINGS,
        &super::super::decide::EXAMPLES,
        &LOAD,
        &RUN,
        &RESUME,
        &SAVE,
        &ARTIFACT,
    ],
    summary: "train LoRA adapters to answer typed decisions with calibrated probabilities, every option order of each labelled question one row",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// `ster tune evaluate`: held-out loss and perplexity, nothing trained.
pub static EVALUATE: Operation = Operation {
    id: "ster.tune.evaluate",
    entries: &["ster tune evaluate", "ster request tune/evaluate"],
    uses: &[&SETTINGS, &EXAMPLES, &LOAD, &RUN],
    summary: "score a checkpoint, with or without an adapter, on held-out examples: loss and perplexity over each completion, nothing trained or written",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};
