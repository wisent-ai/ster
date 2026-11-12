//! How building a family's own blocks from a checkpoint fails where the
//! weights hold values no config can describe (src/model/), raised through
//! `refuse_candle` inside the build and lifted by `weights_unfit`.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

pub static ROUTE_OUTSIDE: Cause = Cause {
    id: "route-outside",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the hash routing table of layer {layer}",
    cause: "layer {layer}'s mlp.gate.tid2eid sends token {token} to expert {expert}, and the layer holds experts 0 to {count} exclusive",
    observed: "tid2eid row {token}: expert {expert}; experts: {count}",
    remedy: "fetch the checkpoint's weights again from where it was published; a routing table names only the experts its layer stores",
    test: "tests/models/decoder.sh::a_routing_table_naming_an_expert_past_the_layer_is_refused",
};

pub static KEYS_UNSHARED: Cause = Cause {
    id: "keys-unshared",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "attending at layer {layer} over the keys and values layer {source} leaves for it",
    cause: "layer {layer} reuses the keys and values layer {source} stores earlier in the same forward pass (Gemma 4's key-value sharing, IQuest-LoopCoder's later loops), and layer {source} stored none in this pass: the decoder ran layer {layer} without running layer {source} first",
    observed: "layer {layer} reads layer {source}'s keys; layer {source}'s slot: empty",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder runs every layer in order, so this is a defect of Ster, not of the checkpoint",
    test: "tests/models/decoder.sh::a_layer_reusing_keys_its_source_did_not_store_is_refused",
};

pub static SHORTCUT_UNOPENED: Cause = Cause {
    id: "shortcut-unopened",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "adding LongCat-Flash's shortcut experts at layer {layer}",
    cause: "the second half-layer {layer} adds the shortcut experts' output its first half, layer {first}, leaves in the same forward pass, and layer {first} left none: the decoder ran layer {layer} without running layer {first} first",
    observed: "layer {layer} closes the shortcut; the output layer {first} keeps: absent",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder runs every layer in order, so this is a defect of Ster, not of the checkpoint",
    test: "tests/models/decoder.sh::a_shortcut_no_earlier_layer_opened_is_refused",
};

pub static LAYER_FORM_UNBUILT: Cause = Cause {
    id: "layer-form-unbuilt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "running layer {layer} over {form}",
    cause: "the decoder ran layer {layer} over {form}, and the layer was built without the weights that run them: the decoder and its layers were built from different readings of the config",
    observed: "layer {layer}; residual form: {form}; the layer's parts for it: absent",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder and every layer are built from one config, so this is a defect of Ster, not of the checkpoint",
    test: "tests/models/decoder.sh::a_layer_run_in_a_form_it_was_not_built_for_is_refused",
};

pub static NGRAM_ROWS: Cause = Cause {
    id: "ngram-rows",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "hashing LongCat-Flash-Lite's n-grams of row {row}",
    cause: "this forward pass continues a decode whose cache kept the n-gram context of {kept} rows, and it runs row {row}: a cache continues only the batch that filled it",
    observed: "rows the cache kept: {kept}; row asked: {row}",
    remedy: "start a new cache (Runtime::cache) for a batch of another size; `ster` itself decodes one sequence per cache, so through the CLI report it at https://github.com/wisent-ai/ster/issues with the model and the command",
    test: "tests/models/decoder.sh::a_cache_continued_by_a_wider_batch_is_refused",
};

pub static TOKEN_UNROUTED: Cause = Cause {
    id: "token-unrouted",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "routing token id {id} through a hash layer's mlp.gate.tid2eid",
    cause: "the tokenizer spelled token id {id}, and the checkpoint's routing table names experts for token ids 0 to {rows} exclusive: the tokenizer is not the one the table was written for",
    observed: "token id: {id}; rows of tid2eid: {rows}",
    remedy: "load the checkpoint with the tokenizer it was published with; a hash-routed layer routes only the ids its tid2eid has a row for",
    test: "tests/models/decoder.sh::a_token_past_the_routing_table_is_refused",
};

pub static TOKEN_IDS_UNSET: Cause = Cause {
    id: "token-ids-unset",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "routing {tokens} tokens by token id",
    cause: "a hash-routed layer chooses its experts by the ids of the tokens it runs, and the forward pass gave it {given} ids for {tokens} tokens: the decoder records every call's ids before its first layer",
    observed: "token ids given: {given}; tokens: {tokens}",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder records the ids of every call that reaches a hash layer, so this is a defect of Ster, not of the checkpoint",
    test: "tests/models/decoder.sh::a_hash_layer_without_token_ids_is_refused",
};

pub static XIELU_UNHELD: Cause = Cause {
    id: "xielu-unheld",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "applying xIELU outside a dense feed-forward",
    cause: "xIELU takes its parameters from the dense feed-forward that holds them (mlp.act_fn), and a part that holds none applied it: the config reading accepts xielu only on Apertus, whose dense feed-forward applies it itself",
    observed: "activation: xielu, applied to a tensor of shape {shape}; parameters: none held",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint",
    test: "tests/models/decoder.sh::xielu_outside_a_dense_feed_forward_is_refused",
};

pub static ENTRIES_UNMATCHED: Cause = Cause {
    id: "entries-unmatched",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "attending over layer {layer}'s compressed entries",
    cause: "DeepSeek-V4's indexer ranked {ranked} compressed entries of layer {layer} and the compressor holds {held}: both pool the same positions at the same ratio, so the cache they continue from was filled by another sequence",
    observed: "entries ranked: {ranked}; entries held: {held}",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint",
    test: "tests/models/decoder.sh::compressed_entries_the_indexer_did_not_rank_are_refused",
};

pub static LOGITS_UNREAD: Cause = Cause {
    id: "logits-unread",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the logits of a forward pass whose residual stream is {hidden}",
    cause: "the pass was run with the readout {readout}, which projects nothing onto the vocabulary, and its caller reads the vocabulary's logits: every reader of logits runs a readout of the last position or of every position",
    observed: "readout: {readout}; logits: none; residual stream shape {hidden}",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint",
    test: "tests/models/decoder.sh::logits_read_from_a_hidden_readout_are_refused",
};

pub static ACTIVATIONS_UNREQUESTED: Cause = Cause {
    id: "activations-unrequested",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the activations of one forward pass",
    cause: "the pass returned activations at another set of layers than it was asked for, or an empty one: the decoder keeps exactly the requested layers' last-position states, so a run would fit or export a direction on fewer reads than prompts",
    observed: "requested layers: {requested}; returned layers: {returned}; empty at: {empty}",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint or the prompts",
    test: "tests/models/decoder.sh::activations_of_other_layers_are_refused",
};

pub static ROWS_EMPTY: Cause = Cause {
    id: "rows-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{pass}",
    cause: "the pass was handed no token rows, or a row of no tokens: every row is an encode's output, and encoding refuses a text the tokenizer turns into no tokens",
    observed: "row lengths: {lengths}",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the prompts",
    test: "tests/models/decoder.sh::a_pass_over_no_tokens_is_refused",
};

pub static POSITIONS_UNKEPT: Cause = Cause {
    id: "positions-unkept",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading every token's state at layer {layer}",
    cause: "the pass asked to keep each position's state at the layer kept another count of them than the text has tokens: the decoder keeps one state per token at every layer it is asked for",
    observed: "layer {layer}: {kept} states for {tokens} tokens",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the text",
    test: "tests/models/decoder.sh::states_of_another_count_are_refused",
};

/// Building a family's own blocks: the routing tables and per-layer values
/// its weights store beside the projections.
pub static DECODER: Operation = Operation {
    id: "ster.model.decoder",
    entries: &[],
    uses: &[],
    summary: "build a family's own blocks from the weights and run them: the expert routing tables and scalars a checkpoint stores beside its projections, and what one layer leaves in a forward pass for a later one",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &ROUTE_OUTSIDE,
        &KEYS_UNSHARED,
        &SHORTCUT_UNOPENED,
        &LAYER_FORM_UNBUILT,
        &NGRAM_ROWS,
        &TOKEN_UNROUTED,
        &TOKEN_IDS_UNSET,
        &XIELU_UNHELD,
        &ENTRIES_UNMATCHED,
        &LOGITS_UNREAD,
        &ACTIVATIONS_UNREQUESTED,
        &ROWS_EMPTY,
        &POSITIONS_UNKEPT,
    ],
};
