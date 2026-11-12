//! How running a T5 encoder-decoder network fails (src/model/t5,
//! src/model/network, src/runtime/infer/encoder_decoder.rs). No loader builds
//! one yet: choosing the decoder refuses `model_type: "t5"` before any weight
//! is read (`ster.model.config.family-unknown`), so none of these causes can
//! be met until a loader constructs the network.

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/model.md, beside the decoder's own causes.
const DOCS: &str = page!("model");

/// What a defect of Ster's own T5 code tells its reader to do.
const REPORT: &str = "report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt";

/// One way the inputs Ster's own caller builds for a T5 pass disagree: each
/// a defect of Ster, its observed the values the check compared.
macro_rules! inputs_cause {
    ($name:ident, $id:literal, $cause:literal, $test:literal) => {
        pub static $name: Cause = Cause {
            id: $id,
            code: Code::Config,
            usage: false,
            exit: None,
            failed: "running the T5 network's {stack}",
            cause: $cause,
            observed: "{check}",
            remedy: REPORT,
            test: concat!("tests/models/t5.sh::", $test),
        };
    };
}

inputs_cause!(CROSS_MISPLACED, "t5-cross-attention-misplaced",
    "an encoder block was asked for or handed cross-attention memory, and only a decoder block attends to the encoder",
    "an_encoder_block_with_cross_attention_is_refused");
inputs_cause!(MEMORY_MISSING, "t5-encoder-memory-missing",
    "the decoder was run without the encoder memory its cross-attention reads: no source tokens were given and the cache holds none",
    "a_decoder_without_encoder_memory_is_refused");
inputs_cause!(MEMORY_HELD, "t5-encoder-memory-held",
    "source tokens were given to a cache that already holds the encoder memory of an earlier source",
    "a_second_source_for_one_cache_is_refused");
inputs_cause!(CACHE_EMPTY, "t5-cache-empty",
    "the key-value cache holds no layer, so it has no position to continue from",
    "a_cache_of_no_layers_is_refused");
inputs_cause!(CACHE_UNEVEN, "t5-cache-uneven",
    "the key-value cache's layers hold different numbers of positions: an earlier pass ended half done and its cache was reused",
    "a_cache_of_uneven_layers_is_refused");
inputs_cause!(POSITION_MISMATCH, "t5-position-mismatch",
    "the pass was given a starting position other than the number of positions its cache holds",
    "a_position_the_cache_does_not_hold_is_refused");
inputs_cause!(CACHE_LAYERS_MISMATCH, "t5-cache-layers-mismatch",
    "the key-value cache holds another number of layer slots than the decoder has blocks",
    "a_cache_for_other_blocks_is_refused");
inputs_cause!(CACHE_UNSUPPORTED, "t5-cache-unsupported",
    "a key-value cache was handed to a padded batch or a differentiable pass, and batched training and scoring run without one",
    "a_cache_for_a_batch_or_training_is_refused");
inputs_cause!(LAST_POSITION_PADDED, "t5-last-position-padded",
    "a padded batch asked for one common last position, and its rows end at different positions",
    "a_padded_batch_read_at_one_position_is_refused");
inputs_cause!(CAPTURE_BATCHED, "t5-capture-batched",
    "states were asked to be captured from several rows or a padded row, and a capture reads one unpadded sequence",
    "a_capture_of_a_batch_is_refused");
inputs_cause!(CAPTURE_OUTSIDE, "t5-capture-outside",
    "a capture names a layer past the decoder's blocks",
    "a_capture_past_the_blocks_is_refused");
inputs_cause!(STEERING_WIDTH, "t5-steering-width",
    "the steering plan is of another width than the decoder's hidden states it is added to",
    "steering_of_another_width_is_refused");
inputs_cause!(LENGTHS_WITHOUT_TOKENS, "t5-lengths-without-tokens",
    "encoder lengths were given without the encoder token rows they measure",
    "lengths_without_tokens_are_refused");
inputs_cause!(ROWS_MISMATCH, "t5-rows-mismatch",
    "the encoder memory holds another number of rows than the decoder is run on, and both describe the same examples",
    "encoder_and_decoder_rows_that_differ_are_refused");
inputs_cause!(ROWS_EMPTY, "t5-rows-empty",
    "the {stack} was handed no token row or rows of no token, and a pass reads at least one token",
    "rows_of_no_token_are_refused");
inputs_cause!(LENGTHS_COUNT, "t5-lengths-count",
    "the {stack} was handed another number of lengths than token rows, one length per row",
    "lengths_for_other_rows_are_refused");
inputs_cause!(PADDING_NONE, "t5-padding-none",
    "a padding mask was asked for rows whose lengths list is empty",
    "a_padding_mask_of_no_rows_is_refused");
inputs_cause!(PADDING_OUTSIDE, "t5-padding-outside",
    "a row's valid length is zero or past the width of its tensor, so its attention would see no key or keys that are not there",
    "a_padding_length_past_the_row_is_refused");

pub static SIZE_OVERFLOW: Cause = Cause {
    id: "t5-size-overflow",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "sizing {what} of the T5 network",
    cause: "a position, shape or bucket count the pass computes is past this platform's sizes",
    observed: "{check}",
    remedy: "run on a shorter prompt; a sequence that long is past what any T5 checkpoint is trained at",
    test: "tests/models/t5.sh::a_size_past_the_platform_is_refused",
};

pub static MEMORY_UNRESERVED: Cause = Cause {
    id: "t5-memory-unreserved",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "allocating {what} of the T5 network",
    cause: "the host has no memory left for {what}",
    observed: "{cells} entries; the allocator answered: {error}",
    remedy: "run on a shorter prompt, or with fewer other programs holding the host's memory",
    test: "tests/models/t5.sh::a_table_past_the_host_memory_is_refused",
};

pub static CONTEXT_EXCEEDED: Cause = Cause {
    id: "t5-sequence-past-context",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the T5 {stack}'s sequence",
    cause: "the {stack} sequence reaches position {end}, past the {limit} positions the checkpoint's n_positions says it was trained at",
    observed: "{stack}: {end} positions; n_positions: {limit}",
    remedy: "shorten the prompt below {limit} tokens, or lower --max-new-tokens",
    test: "tests/models/t5.sh::a_prompt_past_n_positions_is_refused",
};

pub static CONTEXT_UNSTATED: Cause = Cause {
    id: "t5-context-unstated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "bounding the sequences of the T5 checkpoint",
    cause: "the checkpoint's config.json states no n_positions, so Ster has no sequence length to bound its prompts and training rows by",
    observed: "n_positions: absent",
    remedy: "add n_positions, the length the checkpoint was trained at, to its config.json",
    test: "tests/models/t5.sh::a_checkpoint_without_n_positions_is_refused",
};

pub static PASS_CAUSAL_ONLY: Cause = Cause {
    id: "t5-pass-causal-only",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "running {command} on the T5 checkpoint",
    cause: "{command} scores one token sequence from its first position, and a T5 model reads the prompt in its encoder and writes the answer in its decoder, a split that pass does not carry",
    observed: "network: encoder-decoder (T5)",
    remedy: "run {command} on a causal-decoder checkpoint",
    test: "tests/models/t5.sh::a_whole_sequence_pass_is_refused",
};

/// Running a T5 checkpoint's encoder and decoder.
pub static T5: Operation = Operation {
    id: "ster.model.t5",
    entries: &[],
    uses: &[],
    summary: "run a T5 encoder-decoder network: the encoder reads the prompt, the decoder writes the answer, and steering, capture and adapters address the decoder's blocks",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &CROSS_MISPLACED,
        &MEMORY_MISSING,
        &MEMORY_HELD,
        &CACHE_EMPTY,
        &CACHE_UNEVEN,
        &POSITION_MISMATCH,
        &CACHE_LAYERS_MISMATCH,
        &CACHE_UNSUPPORTED,
        &LAST_POSITION_PADDED,
        &CAPTURE_BATCHED,
        &CAPTURE_OUTSIDE,
        &STEERING_WIDTH,
        &LENGTHS_WITHOUT_TOKENS,
        &ROWS_MISMATCH,
        &ROWS_EMPTY,
        &LENGTHS_COUNT,
        &PADDING_NONE,
        &PADDING_OUTSIDE,
        &SIZE_OVERFLOW,
        &MEMORY_UNRESERVED,
        &CONTEXT_EXCEEDED,
        &CONTEXT_UNSTATED,
        &PASS_CAUSAL_ONLY,
    ],
};

/// The refusal of a pass handed inputs that disagree as `cause` says,
/// `check` the values it compared, as an error of `stack` (`encoder`,
/// `decoder`).
pub fn inconsistent(cause: &'static Cause, stack: &str, check: &str) -> anyhow::Error {
    super::refuse(cause, &[("stack", stack), ("check", check)])
}

/// [`inconsistent`] where Candle's `Result` is the type.
pub fn inconsistent_candle(cause: &'static Cause, stack: &str, check: &str) -> candle_core::Error {
    super::super::refuse_candle(cause, &[("stack", stack), ("check", check)])
}

/// The refusal of a size past the platform's, `what` the step sizing it.
pub fn overflow(what: &str, check: &str) -> anyhow::Error {
    super::refuse(&SIZE_OVERFLOW, &[("what", what), ("check", check)])
}

/// The refusal of `what` (`the encoder's padding mask`) of `cells` entries,
/// which the host did not reserve memory for.
pub fn unreserved(
    what: &str,
    cells: usize,
    error: &std::collections::TryReserveError,
) -> anyhow::Error {
    super::refuse(
        &MEMORY_UNRESERVED,
        &[("what", what), ("cells", &cells.to_string()), ("error", &error.to_string())],
    )
}
