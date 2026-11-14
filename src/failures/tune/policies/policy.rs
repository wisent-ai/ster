//! How policy optimization's reward and its sampled conversations fail
//! (src/tune/objective/grpo): the source `--reward` names, the outside
//! scorer it may name and what that scorer answers, the reward model's
//! head, and the conversations a group is sampled as.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

const FIX_SOURCE: &str = "give --reward length, the path of an artifact ster tune reward wrote, or https://<host>/<path>#/<json pointer> of a scorer";

pub static SOURCE_BLANK: Cause = Cause {
    id: "source-blank",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing where the reward comes from",
    cause: "group-relative policy optimization requires a reward source; pass length or a reward artifact",
    observed: "--reward {value}",
    remedy: FIX_SOURCE,
    test: "tests/tuning/runs/reward.sh::a_blank_reward_source_is_refused_before_the_model_loads",
};

pub static SOURCE_UNKNOWN: Cause = Cause {
    id: "source-unknown",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "choosing where the reward comes from",
    cause: "reward source {value} is neither the keyword length nor a file that exists",
    observed: "--reward {value}",
    remedy: FIX_SOURCE,
    test: "tests/tuning/runs/reward.sh::a_reward_source_that_names_nothing_is_refused",
};

pub static SCORER_FIELD_ABSENT: Cause = Cause {
    id: "scorer-field-absent",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the reward scorer --reward names",
    cause: "reward scorer {value} names no field: end it with #/<json pointer> to the reward, for example #/score",
    observed: "--reward {value}",
    remedy: "end --reward with #/ and the JSON pointer of the number the scorer answers, such as #/score",
    test: "tests/tuning/runs/reward.sh::a_scorer_without_a_field_is_refused",
};

pub static SCORER_POINTER_INVALID: Cause = Cause {
    id: "scorer-pointer-invalid",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the reward scorer --reward names",
    cause: "reward scorer field {pointer} is not a JSON pointer; it starts with /",
    observed: "--reward {value}",
    remedy: "write the field after # as a JSON pointer, such as #/score or #/result/0",
    test: "tests/tuning/runs/reward.sh::a_scorer_field_that_is_no_pointer_is_refused",
};

pub static BEARER_EMPTY: Cause = Cause {
    id: "bearer-empty",
    code: Code::Auth,
    usage: false,
    exit: None,
    failed: "reading STER_REWARD_BEARER for the reward scorer",
    cause: "STER_REWARD_BEARER is present but empty",
    observed: "STER_REWARD_BEARER: {bytes} bytes, all whitespace; the scorer: {url}",
    remedy: "set STER_REWARD_BEARER to the scorer's bearer token, or unset it for a scorer that needs none",
    test: "tests/tuning/runs/reward.sh::an_empty_scorer_bearer_is_refused",
};

pub static BEARER_NOT_UNICODE: Cause = Cause {
    id: "bearer-not-unicode",
    code: Code::Auth,
    usage: false,
    exit: None,
    failed: "reading STER_REWARD_BEARER for the reward scorer",
    cause: "STER_REWARD_BEARER is not valid Unicode, so it cannot be sent as a header",
    observed: "STER_REWARD_BEARER holds {bytes} bytes that are not UTF-8; the scorer: {url}",
    remedy: "set STER_REWARD_BEARER to the scorer's bearer token as text",
    test: "tests/tuning/runs/reward.sh::a_scorer_bearer_that_is_not_text_is_refused",
};

pub static SCORER_UNREACHABLE: Cause = Cause {
    id: "scorer-unreachable",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "asking the reward scorer {url} for the score of one completion",
    cause: "no answer came from the reward scorer {url}: nothing listens there, its name does not resolve, or the connection broke",
    observed: "url: {url}; the client's error kind {kind}; it answered: {error}",
    remedy: "start the scorer at {url} or name one that answers in --reward, then run again",
    test: "tests/tuning/runs/reward.sh::a_scorer_nothing_listens_at_is_refused",
};

pub static SCORER_REFUSED: Cause = Cause {
    id: "scorer-refused",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "asking the reward scorer {url} for the score of one completion",
    cause: "the reward scorer {url} answered HTTP {status}, which says the request itself is wrong: it takes another body than an object of `prompt` and `text` or another address",
    observed: "HTTP {status}; STER_REWARD_BEARER {bearer}; the scorer's body: {body}",
    remedy: "name in --reward the scorer address that takes an object of `prompt` and `text`; the scorer's body says what it expected",
    test: "tests/tuning/runs/reward.sh::a_scorer_answering_an_error_status_is_refused",
};

pub static SCORER_UNAUTHORIZED: Cause = Cause {
    id: "scorer-unauthorized",
    code: Code::Auth,
    usage: false,
    exit: None,
    failed: "asking the reward scorer {url} for the score of one completion",
    cause: "the reward scorer {url} answered HTTP {status}: it does not accept the bearer token sent, or wants one and none was sent",
    observed: "HTTP {status}; STER_REWARD_BEARER {bearer}; the scorer's body: {body}",
    remedy: "set STER_REWARD_BEARER to a token the scorer at {url} accepts, then run again",
    test: "tests/tuning/runs/reward.sh::a_scorer_refusing_the_bearer_is_refused",
};

pub static SCORER_LIMITED: Cause = Cause {
    id: "scorer-limited",
    code: Code::RateLimit,
    usage: false,
    exit: None,
    failed: "asking the reward scorer {url} for the score of one completion",
    cause: "the reward scorer {url} answered HTTP 429: it limits how often it is asked, and this run passed the limit",
    observed: "HTTP {status}; the scorer's body: {body}",
    remedy: "run again after a while, or score fewer completions per prompt (--group)",
    test: "tests/tuning/runs/reward.sh::a_scorer_limiting_requests_is_refused",
};

pub static SCORER_FAILING: Cause = Cause {
    id: "scorer-failing",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "asking the reward scorer {url} for the score of one completion",
    cause: "the reward scorer {url} answered HTTP {status}: it failed on its side, so the request was right and the scorer could not score it",
    observed: "HTTP {status}; the scorer's body: {body}",
    remedy: "run again once the scorer at {url} serves; its own log says why it failed",
    test: "tests/tuning/runs/reward.sh::a_scorer_failing_on_its_side_is_refused",
};

pub static SCORER_ANSWER_UNREADABLE: Cause = Cause {
    id: "scorer-answer-unreadable",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "reading the reward scorer's answer from {url}",
    cause: "the answer broke off before it was read whole: the scorer closed the connection midway",
    observed: "url: {url}; error kind {kind}; the client answered: {error}",
    remedy: "run again; when it repeats, check that the scorer at {url} finishes its answers",
    test: "tests/tuning/runs/reward.sh::a_scorer_answer_that_breaks_off_is_refused",
};

pub static SCORER_NOT_TEXT: Cause = Cause {
    id: "scorer-not-text",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "reading the reward scorer's answer from {url}",
    cause: "the scorer {url} answered bytes that are not UTF-8 text, and a score is read from a JSON document",
    observed: "url: {url}; the client answered: {error}",
    remedy: "name a scorer address in --reward that answers a JSON document",
    test: "tests/tuning/runs/reward.sh::a_scorer_answer_that_is_not_text_is_refused",
};

pub static SCORER_NOT_JSON: Cause = Cause {
    id: "scorer-not-json",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "reading the reward scorer's answer from {url}",
    cause: "the reward scorer {url} answered text that is not JSON, and a score is read from a JSON document",
    observed: "answer ({size} bytes): {body}; the JSON reader stopped at line {line}, column {column}: {error}",
    remedy: "name a scorer address in --reward that answers a JSON document",
    test: "tests/tuning/runs/reward.sh::a_scorer_answer_that_is_not_json_is_refused",
};

pub static SCORER_NO_NUMBER: Cause = Cause {
    id: "scorer-no-number",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "reading the reward scorer's answer from {url}",
    cause: "the reward scorer {url} answered without a number at {pointer}",
    observed: "answer: {body}",
    remedy: "point the #/ fragment of --reward at the number the scorer answers",
    test: "tests/tuning/runs/reward.sh::a_scorer_answer_without_the_number_is_refused",
};

pub static REWARD_NOT_FINITE: Cause = Cause {
    id: "reward-not-finite",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "summing the rewards of conversation {conversation} of the {group} sampled for one prompt",
    cause: "the conversation's reward is not a finite number: the reward source answered an infinite or NaN reward for a turn, or the turns' rewards summed past the largest finite number, so no advantage can be taken from it; the group is refused before a policy update",
    observed: "after turn {turn}: this turn's reward {score}, the sum {total}; reward source: {source}",
    remedy: "use a reward source that answers finite numbers: a scorer whose number stays below 1e300, or a reward artifact trained without diverging",
    test: "tests/tuning/runs/reward.sh::rewards_that_are_not_finite_are_refused",
};

pub static GROUP_SPREAD_NOT_FINITE: Cause = Cause {
    id: "group-spread-not-finite",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "normalizing the rewards of the group sampled for one prompt",
    cause: "every conversation's reward is finite, and the group's mean or variance is not: the rewards lie so far apart that their squared spread overflows, so no advantage can be taken from them; the group is refused before a policy update",
    observed: "{count} conversations, rewards {rewards}: mean {mean}, variance {variance}; reward source: {source}",
    remedy: "use a reward source whose numbers stay below 1e150, so their squares stay finite: rescale the scorer's answer",
    test: "tests/tuning/runs/reward.sh::rewards_too_far_apart_are_refused",
};

/// Choosing the reward and scoring each sampled completion with it.
pub static REWARD: Operation = Operation {
    id: "ster.tune.reward",
    entries: &[],
    uses: &[],
    summary: "resolve --reward to the length reward, an outside scorer posted the prompt and text over HTTP, or a reward artifact, and score each sampled completion with it",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &SOURCE_BLANK,
        &SOURCE_UNKNOWN,
        &SCORER_FIELD_ABSENT,
        &SCORER_POINTER_INVALID,
        &BEARER_EMPTY,
        &BEARER_NOT_UNICODE,
        &SCORER_UNREACHABLE,
        &SCORER_REFUSED,
        &SCORER_UNAUTHORIZED,
        &SCORER_LIMITED,
        &SCORER_FAILING,
        &SCORER_ANSWER_UNREADABLE,
        &SCORER_NOT_TEXT,
        &SCORER_NOT_JSON,
        &SCORER_NO_NUMBER,
        &REWARD_NOT_FINITE,
        &GROUP_SPREAD_NOT_FINITE,
    ],
};
