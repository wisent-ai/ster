//! How writing a pair set with a model fails: the settings, checked before
//! any model loads or any route is asked, and what the writing model answers.

use wisent_errors::{Cause, Code, Operation};

use super::FILE;
use super::inspect::DEDUPE;

/// docs/guide/errors/pairs-synthesize.md, where this functionality is listed.
const DOCS: &str = page!("pairs-synthesize");

pub static TRAIT_BLANK: Cause = Cause {
    id: "trait-blank",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "writing pairs",
    cause: "pairs synthesize needs a description of the trait the positive side shows",
    observed: "--trait \"{trait}\"",
    remedy: "describe the trait in one sentence with --trait",
    test: "tests/pairs/synthesize.sh::a_blank_trait_is_refused_before_any_model_loads",
};

pub static TEMPERATURE_NOT_POSITIVE: Cause = Cause {
    id: "temperature-not-positive",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the sampling settings",
    cause: "synthesis requires a temperature above zero; argmax generation repeats a single prompt",
    observed: "--temperature {temperature}",
    remedy: "give --temperature a number above zero",
    test: "tests/pairs/synthesize.sh::a_temperature_at_or_below_zero_is_refused_before_any_model_loads",
};

pub static GENERATOR_UNKNOWN: Cause = Cause {
    id: "generator-unknown",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: "choosing the model that writes the pairs",
    cause: "pairs synthesize writes with local or brama, and \"{generator}\" is neither",
    observed: "generator: \"{generator}\"",
    remedy: "give generator local, with a model, or brama, with generatorModel",
    test: "tests/pairs/synthesize.sh::an_unknown_generator_is_refused",
};

pub static MODEL_MISSING: Cause = Cause {
    id: "model-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the model that writes the pairs",
    cause: "pairs synthesize with --generator local requires --model",
    observed: "--generator local without --model; writing pairs for \"{trait}\"",
    remedy: "name the Hugging Face model id or local model directory that writes the pairs with --model, or write them through Stado's inference alternatives with --generator brama --generator-model <route>",
    test: "tests/pairs/synthesize.sh::a_local_generator_without_a_model_is_refused",
};

pub static ROUTE_MISSING: Cause = Cause {
    id: "route-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the model that writes the pairs",
    cause: "pairs synthesize with --generator brama requires --generator-model",
    observed: "--generator brama without --generator-model; writing pairs for \"{trait}\"",
    remedy: "name the route that writes the pairs with --generator-model: a Brama alias, a provider/model route or a selector",
    test: "tests/pairs/synthesize.sh::a_brama_generator_without_a_route_is_refused",
};

pub static COUNT_MISSING: Cause = Cause {
    id: "count-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the pairs/synthesize request",
    cause: "pairs synthesize requires count, a pair count above zero",
    observed: "the request names no count; writing pairs for \"{trait}\"",
    remedy: "give count, the number of pairs to keep",
    test: "tests/pairs/synthesize.sh::a_request_without_a_count_is_refused",
};

pub static OPPOSITE_EMPTY: Cause = Cause {
    id: "opposite-empty",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "asking {generator} for the opposite of {trait}",
    cause: "{generator} returned no opposite trait for {trait}",
    observed: "its answer was empty, sampled with at most {tokens} new tokens at temperature {temperature}",
    remedy: "state the opposite trait with --opposite",
    test: "tests/pairs/synthesize.sh::a_generator_naming_no_opposite_is_refused",
};

pub static VERDICT_MALFORMED: Cause = Cause {
    id: "verdict-malformed",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "reading {generator}'s refusal judgement",
    cause: "{generator} answered the refusal judgement with {text}, which is not JSON, where the judgement is exactly one object whose only key `refuses` is true or false",
    observed: "{length} characters; the JSON reader stopped at column {column}: {error}",
    remedy: "write the pairs with a generator that answers a one-line JSON instruction exactly: an instruction-tuned --model, or another --generator-model route",
    test: "tests/pairs/synthesize.sh::a_judgement_that_is_not_the_json_form_is_refused",
};

pub static VERDICT_SHAPE: Cause = Cause {
    id: "verdict-shape",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "reading {generator}'s refusal judgement",
    cause: "{generator} answered the refusal judgement with {text}, JSON of another form than exactly one object whose only key `refuses` is true or false: another key, a key beside refuses, or a value that is not true or false",
    observed: "{length} characters; at column {column} the reader answered: {error}",
    remedy: "write the pairs with a generator that answers a one-line JSON instruction exactly: an instruction-tuned --model, or another --generator-model route",
    test: "tests/pairs/synthesize.sh::a_judgement_of_another_json_form_is_refused",
};

pub static NO_PAIRS: Cause = Cause {
    id: "no-pairs",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "writing pairs with {generator}",
    cause: "none of {attempts} attempts kept a pair, and a set without pairs is refused",
    observed: "rejected as empty: {empty}, as refusals: {refusals}, as duplicates: {duplicates}",
    remedy: "raise --retry-multiplier for more attempts, state --opposite, or write with another generator; the counts say which step rejected the attempts",
    test: "tests/pairs/synthesize.sh::a_run_that_keeps_no_pair_is_refused",
};

/// A pair set written by a local model or a route Stado's inference
/// alternatives answer.
pub static SYNTHESIZE: Operation = Operation {
    id: "ster.pairs.synthesize",
    entries: &["ster pairs synthesize", "ster request pairs/synthesize"],
    uses: &[
        &crate::failures::inference::INFERENCE,
        &crate::failures::model::weights::LOAD,
        &DEDUPE,
        &FILE,
    ],
    summary: "write a contrastive pair set with a local model or a route Stado's inference alternatives answer: the opposite trait, then a question and two answers per attempt, each answer judged for refusal and each pair checked for a near-duplicate as it lands",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &TRAIT_BLANK,
        &TEMPERATURE_NOT_POSITIVE,
        &GENERATOR_UNKNOWN,
        &MODEL_MISSING,
        &ROUTE_MISSING,
        &COUNT_MISSING,
        &OPPOSITE_EMPTY,
        &VERDICT_MALFORMED,
        &VERDICT_SHAPE,
        &NO_PAIRS,
    ],
};
