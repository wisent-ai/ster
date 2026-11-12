//! How sampling a continuation fails (src/runtime/infer/generate.rs): the
//! token budget, a prompt the model's context cannot hold, and the sampled
//! tokens the tokenizer cannot spell back.

use wisent_errors::{Cause, Code, Operation};

use super::super::model::weights::LOAD;
use super::super::file::FILE;
use super::super::steering::ARTIFACT;
use super::super::workflow::prompts::PROMPTS;
use super::DOCS;
use super::encode::ENCODE;
use super::steer::STEER;

pub static NEW_TOKENS_NONE: Cause = Cause {
    id: "new-tokens-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "sampling a continuation",
    cause: "max_new_tokens must be greater than zero, or there is nothing to sample",
    observed: "--max-new-tokens {tokens}; the prompt is {prompt} tokens of {model}",
    remedy: "give --max-new-tokens at least 1",
    test: "tests/runtime/generate.sh::no_new_token_is_refused",
};

pub static PROMPT_TOO_LONG: Cause = Cause {
    id: "prompt-too-long",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "sampling a continuation",
    cause: "the prompt is {tokens} tokens, and this model's context holds fewer than {context}",
    observed: "prompt: {tokens} tokens; max_position_embeddings: {context}",
    remedy: "shorten the prompt below {context} tokens, or use a model with a longer context",
    test: "tests/runtime/generate.sh::a_prompt_past_the_context_is_refused",
};

pub static UNSPELLABLE: Cause = Cause {
    id: "unspellable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "spelling the sampled tokens {tokens}",
    cause: "the tokenizers library cannot turn the sampled ids back into text: an id lies outside the vocabulary its tokenizer.json holds, or its decoder raised on them",
    observed: "token ids: {tokens}; the tokenizer holds {vocabulary} tokens; the library answered: {error}",
    remedy: "use the tokenizer.json the checkpoint was published with, unedited",
    test: "tests/runtime/generate.sh::tokens_the_tokenizer_cannot_spell_are_refused",
};

/// Sampling one continuation, steered or not.
pub static GENERATE: Operation = Operation {
    id: "ster.runtime.generate",
    entries: &[],
    uses: &[&ENCODE, &STEER],
    summary: "sample a continuation of an encoded prompt within the model's context, with steering artifacts added when given, and spell the sampled tokens",
    docs: DOCS,
    warnings: &[],
    causes: &[&NEW_TOKENS_NONE, &PROMPT_TOO_LONG, &UNSPELLABLE],
};

pub static STRENGTHS_UNPAIRED: Cause = Cause {
    id: "strengths-unpaired",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "pairing each --vector with its --strength",
    cause: "every --vector needs its own --strength, and this run gave {vectors} vector(s) and {strengths} strength(s); Ster assumes no steering scale",
    observed: "--vector given {vectors} time(s), --strength {strengths} time(s)",
    remedy: "give one --strength after each --vector, in the same order",
    test: "tests/runtime/generate.sh::a_vector_without_its_strength_is_refused",
};

pub static SYSTEM_EMPTY: Cause = Cause {
    id: "system-empty",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the system turn {path}",
    cause: "the file holds no text, so there is no system turn to give the model",
    observed: "--system {path}: {bytes} bytes of whitespace",
    remedy: "write the system turn into {path}, or leave --system out",
    test: "tests/runtime/generate.sh::an_empty_system_turn_is_refused",
};

/// `ster generate`: a model's continuation of one prompt or of each of a
/// prompt set, steered, adapted or neither.
pub static COMMAND: Operation = Operation {
    id: "ster.generate",
    entries: &["ster generate", "ster request generate"],
    uses: &[&ARTIFACT, &PROMPTS, &LOAD, &ENCODE, &GENERATE, &FILE],
    summary: "read every steering artifact and the system turn before any weight, load the model with its adapter when one is named, and sample a continuation of each prompt",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &STRENGTHS_UNPAIRED,
        &SYSTEM_EMPTY,
    ],
};