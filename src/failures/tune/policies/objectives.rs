//! How the settings only one objective takes fail: the preference loss and
//! its beta, the steering vector BiPO learns, and the groups and
//! conversations policy optimization samples. Each is refused before any
//! model or data is read, except the layer BiPO trains, which only the model
//! can answer.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

pub static LOSS_UNKNOWN: Cause = Cause {
    id: "loss-unknown",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the preference loss",
    cause: "unknown preference loss {value}; expected dpo or ipo",
    observed: "--loss {value}",
    remedy: "give --loss dpo for the sigmoid loss of DPO or ipo for IPO's squared loss",
    test: "tests/tuning/runs/objectives.sh::an_unknown_preference_loss_is_refused",
};

pub static BETA_UNUSABLE: Cause = Cause {
    id: "beta-unusable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting {subject}",
    cause: "{subject} requires a finite beta above zero; at zero every pair's loss is the constant log 2 and no gradient reaches the model",
    observed: "--beta {value}",
    remedy: "give --beta a finite number above zero; 0.1 is the value the DPO paper reports across its settings",
    test: "tests/tuning/runs/objectives.sh::a_beta_of_zero_is_refused_before_the_model_loads",
};

pub static STRENGTH_UNUSABLE: Cause = Cause {
    id: "strength-unusable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting bi-directional preference optimization",
    cause: "bi-directional preference optimization requires a finite strength above zero, the scale +v and -v are added at",
    observed: "--strength {value}",
    remedy: "give --strength a finite number above zero, such as 1",
    test: "tests/tuning/runs/objectives.sh::a_strength_of_zero_is_refused_before_the_model_loads",
};

pub static LAYER_OUTSIDE: Cause = Cause {
    id: "layer-outside",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "placing the steering vector bi-directional preference optimization learns",
    cause: "layer {layer} is outside the model's {layers} layers",
    observed: "--layer {layer}; the model has layers 0 to {last}",
    remedy: "give --layer a layer from 0 to {last}",
    test: "tests/tuning/runs/objectives.sh::a_layer_past_the_model_is_refused",
};

pub static GROUP_TOO_SMALL: Cause = Cause {
    id: "group-too-small",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting group-relative policy optimization",
    cause: "group-relative policy optimization requires a group of at least two completions, because the group is the baseline",
    observed: "--group {group}",
    remedy: "give --group at least 2",
    test: "tests/tuning/runs/objectives.sh::a_group_of_one_is_refused_before_the_model_loads",
};

pub static TURNS_NONE: Cause = Cause {
    id: "turns-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting group-relative policy optimization",
    cause: "group-relative policy optimization requires at least one assistant turn",
    observed: "--turns {turns}; --group {group}",
    remedy: "give --turns at least 1",
    test: "tests/tuning/runs/objectives.sh::no_assistant_turn_is_refused_before_the_model_loads",
};

pub static USER_UNNEEDED: Cause = Cause {
    id: "user-unneeded",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting group-relative policy optimization",
    cause: "a simulated user writes turns between assistant turns, so it needs turns above one; this run asked for one",
    observed: "--turns 1 --user-model {user_model}",
    remedy: "give --turns 2 or more for a conversation, or leave out --user-model",
    test: "tests/tuning/runs/objectives.sh::a_simulated_user_for_one_turn_is_refused",
};

pub static USER_MISSING: Cause = Cause {
    id: "user-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting group-relative policy optimization",
    cause: "a conversation of {turns} assistant turns needs a simulated user model to write the turns in between",
    observed: "--turns {turns}, no --user-model",
    remedy: "name the model that writes the user's turns with --user-model, or give --turns 1",
    test: "tests/tuning/runs/objectives.sh::a_conversation_without_a_simulated_user_is_refused",
};

pub static POLICY_BETA_NEGATIVE: Cause = Cause {
    id: "policy-beta-negative",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting group-relative policy optimization",
    cause: "group-relative policy optimization requires a finite beta of zero or more",
    observed: "--beta {value}",
    remedy: "give --beta 0 to switch the KL term off, or a finite number above zero",
    test: "tests/tuning/runs/objectives.sh::a_negative_policy_beta_is_refused_before_the_model_loads",
};

pub static TEMPERATURE_UNUSABLE: Cause = Cause {
    id: "temperature-unusable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting group-relative policy optimization",
    cause: "group-relative policy optimization requires a temperature above zero; argmax sampling would draw one identical completion per group",
    observed: "--temperature {value}",
    remedy: "give --temperature a number above zero, such as 1",
    test: "tests/tuning/runs/objectives.sh::a_temperature_of_zero_is_refused_before_the_model_loads",
};

pub static NEW_TOKENS_NONE: Cause = Cause {
    id: "new-tokens-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting group-relative policy optimization",
    cause: "max_new_tokens must be greater than zero, or no completion has a token to reward",
    observed: "--max-new-tokens {tokens}; --max-sequence {max_sequence}",
    remedy: "give --max-new-tokens at least 1",
    test: "tests/tuning/runs/objectives.sh::no_new_token_is_refused_before_the_model_loads",
};

/// The settings one objective takes beside the shared ones.
pub static OBJECTIVES: Operation = Operation {
    id: "ster.tune.objective",
    entries: &[],
    uses: &[],
    summary: "check the settings only one objective takes: DPO's loss and beta, the strength and layer of BiPO's vector, and the group, turns, simulated user, beta, temperature and token budget of policy optimization",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &LOSS_UNKNOWN,
        &BETA_UNUSABLE,
        &STRENGTH_UNUSABLE,
        &LAYER_OUTSIDE,
        &GROUP_TOO_SMALL,
        &TURNS_NONE,
        &USER_UNNEEDED,
        &USER_MISSING,
        &POLICY_BETA_NEGATIVE,
        &TEMPERATURE_UNUSABLE,
        &NEW_TOKENS_NONE,
    ],
};
