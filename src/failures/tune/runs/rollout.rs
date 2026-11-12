//! How sampling the conversations of a policy-optimization group fails
//! (src/tune/objective/grpo/rollout.rs): the simulated user that writes the
//! user's turns, a conversation that outgrows the sequence limit, and an
//! assistant turn that ends before its first token.

use wisent_errors::{Cause, Code, Operation};

use super::super::file::FILE;
use super::super::model::weights::LOAD;
use super::super::workflow::prompts::PROMPTS;
use super::DOCS;

pub static USER_TEMPLATE_ABSENT: Cause = Cause {
    id: "user-template-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "loading the simulated user {model}",
    cause: "the simulated user {model} publishes no chat template, so it cannot write a user's turn in a conversation",
    observed: "chat template of {model}: {status}",
    remedy: "give --user-model a chat model whose tokenizer_config.json carries a chat_template",
    test: "tests/tuning/runs/objectives.sh::a_simulated_user_without_a_chat_template_is_refused",
};

pub static CONVERSATION_TOO_LONG: Cause = Cause {
    id: "conversation-too-long",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "sampling assistant turn {turn} of a conversation",
    cause: "turn {turn} needs {context} conversation tokens plus {sampled} sampled tokens, past the {limit} token limit",
    observed: "turn {turn}: {context} + {sampled} tokens; limit {limit}",
    remedy: "lower --turns or --max-new-tokens, or raise --max-sequence",
    test: "tests/tuning/runs/objectives.sh::a_conversation_past_the_limit_is_refused",
};

pub static TURN_EMPTY: Cause = Cause {
    id: "turn-empty",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "sampling assistant turn {turn} of a conversation",
    cause: "assistant turn {turn} of a sampled conversation was empty: its first token ended the sequence, so it has nothing to take a gradient through, and dropping it would bias the group's baseline",
    observed: "turn {turn}: 0 tokens",
    remedy: "raise --temperature so the policy draws other first tokens, or train the policy with ster tune sft first so it answers these prompts",
    test: "tests/tuning/runs/objectives.sh::an_empty_assistant_turn_is_refused",
};

/// Sampling the conversations one group is scored on.
pub static ROLLOUT: Operation = Operation {
    id: "ster.tune.rollout",
    entries: &[],
    uses: &[],
    summary: "sample the conversations of one policy-optimization group: the policy's assistant turns and, above one turn, the simulated user's turns between them",
    docs: DOCS,
    warnings: &[],
    causes: &[&USER_TEMPLATE_ABSENT, &CONVERSATION_TOO_LONG, &TURN_EMPTY],
};

pub static TURNS_FEW: Cause = Cause {
    id: "turns-few",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "playing conversations of --turns {turns} with the simulated user {user}",
    cause: "a conversation with a simulated user needs --turns of at least 2; one turn is ster generate, since the user answers only after the model's first turn",
    observed: "--turns {turns}",
    remedy: "give --turns 2 or more, or answer each prompt once with ster generate --prompts",
    test: "tests/tuning/runs/objectives.sh::a_conversation_of_one_turn_is_refused",
};

/// `ster converse`: conversations played between a model and a simulated
/// user, with no gradient, as `tune grpo --turns` plays them.
pub static CONVERSE: Operation = Operation {
    id: "ster.converse",
    entries: &["ster converse"],
    uses: &[&PROMPTS, &LOAD, &ROLLOUT, &FILE],
    summary: "read the prompt set, load the model (with its adapter) and the simulated user, and write each conversation they play as one JSON line",
    docs: DOCS,
    warnings: &[],
    causes: &[&TURNS_FEW],
};
