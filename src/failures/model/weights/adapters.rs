//! How placing LoRA adapters on a model fails (src/model/architecture/
//! targets.rs): a family whose layers hold no projection an adapter attaches
//! to on every layer, one cause per reason, and a target projection the
//! family does not store as one tensor, one cause per reason.

use wisent_errors::{Cause, Code, Operation};

use super::LOAD_DOCS;

/// What every refusal of a family that holds no adapters tells its reader.
const STEER_INSTEAD: &str = "steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer";

/// Each family reason's cause: what placing adapters met, and why that
/// family holds no adapter on every layer.
macro_rules! unadaptable {
    ($name:ident, $id:literal, $why:literal, $test:literal) => {
        pub static $name: Cause = Cause {
            id: $id,
            code: Code::Config,
            usage: false,
            exit: None,
            failed: "placing LoRA adapters on {model}",
            cause: concat!($why, "; Ster steers it but trains no adapters on it"),
            observed: "the family's {feature}; targets asked for: {targets}",
            remedy: STEER_INSTEAD,
            test: $test,
        };
    };
}

unadaptable!(
    SHORT_CONVOLUTION,
    "adapters-short-convolution",
    "this model's blocks include short-convolution mixers (LFM2) with no attention projection to adapt on every layer",
    "tests/models/loading/targets.sh::a_short_convolution_model_is_refused"
);
unadaptable!(
    LAYER_PLANS,
    "adapters-layer-plans",
    "this model's layers differ in key-value heads and feed-forward width, and some have neither (DeciLM's block_configs), so no projection has one shape on every layer",
    "tests/models/loading/targets.sh::a_model_of_per_layer_plans_is_refused"
);
unadaptable!(
    SHORTCUT_EXPERTS,
    "adapters-shortcut-experts",
    "this model's layers each hold two attention and feed-forward pairs beside shortcut-connected experts (LongCat-Flash), so no projection sits at one name per Ster layer",
    "tests/models/loading/targets.sh::a_shortcut_experts_model_is_refused"
);
unadaptable!(
    LAYERS_REPEATED,
    "adapters-layers-repeated",
    "this model runs its layers more than once (Nanbeige's num_loops, IQuest-LoopCoder's loop_num, HRM-Text's H_cycles and L_cycles), so one adapter would correct every pass of its layer",
    "tests/models/loading/targets.sh::a_model_running_its_layers_twice_is_refused"
);
unadaptable!(
    STREAMS_HYPER_CONNECTED,
    "adapters-streams-hyper-connected",
    "this model's residual is several hyper-connected streams (DeepSeek-V4's and GLM-5-Next's hc_mult), which no adapter on one sublayer's projection sees whole",
    "tests/models/loading/targets.sh::a_hyper_connected_model_is_refused"
);
unadaptable!(
    INKLING_PROJECTIONS,
    "adapters-inkling-projections",
    "this model's attention and feed-forward are Inkling's own (wq_du, wk_dv, wv_dv, wo_ud, w13_dn, w2_md behind short convolutions), which Ster's adapters do not attach to",
    "tests/models/loading/targets.sh::an_inkling_model_is_refused"
);
unadaptable!(
    LIGHTNING_ATTENTION,
    "adapters-lightning-attention",
    "this model's blocks include lightning-attention mixers (MiniMax-Text-01's and Ling 2.5's linear attention) with no attention projection to adapt on every layer",
    "tests/models/loading/targets.sh::a_lightning_attention_model_is_refused"
);
unadaptable!(
    DELTA_RULE,
    "adapters-delta-rule",
    "this model's blocks include gated delta-rule mixers (Qwen3-Next's and Kimi-Linear's linear attention) with no attention projection to adapt on every layer",
    "tests/models/loading/targets.sh::a_delta_rule_model_is_refused"
);
unadaptable!(
    STATE_SPACE,
    "adapters-state-space",
    "this model's blocks are state-space mixers (Mamba) with no attention or feed-forward projection to adapt",
    "tests/models/loading/targets.sh::a_model_that_trains_no_adapters_is_refused"
);

/// Each target reason's cause: the projection asked for, and why the family
/// stores no single tensor for it.
macro_rules! target_absent {
    ($name:ident, $id:literal, $why:literal, $test:literal) => {
        pub static $name: Cause = Cause {
            id: $id,
            code: Code::Config,
            usage: true,
            exit: None,
            failed: "placing LoRA adapters on {model}",
            cause: concat!("this model has no single {target} projection to adapt because ", $why),
            observed: "targets this model has: {choices}; targets asked for: {targets}",
            remedy: "choose adapter targets among {choices} with --targets",
            test: $test,
        };
    };
}

target_absent!(
    QUERY_BOTTLENECKED,
    "target-query-bottlenecked",
    "its query passes a bottleneck and a norm (Step3's q_proj and inter_norm) before wq",
    "tests/models/loading/targets.sh::a_bottlenecked_query_is_refused"
);
target_absent!(
    HEADS_DIFFER,
    "target-heads-differ",
    "its full-attention layers' heads differ from its sliding-window layers' in count or width (Gemma 4's global_head_dim, MiMo-V2's swa_num_key_value_heads, Step 3.5's attention_other_setting, Dots3-Note's swa_* latent attention)",
    "tests/models/loading/targets.sh::an_attention_target_of_two_head_shapes_is_refused"
);
target_absent!(
    VALUE_EXPERTS,
    "target-value-experts",
    "its routed layers take their value from value experts (K2-Horizon's v_experts) with no v_proj",
    "tests/models/loading/targets.sh::a_value_of_value_experts_is_refused"
);
target_absent!(
    ATTENTION_LATENT,
    "target-attention-latent",
    "its attention is latent (DeepSeek's low-rank query and key-value), with no query, key or value projection of the head width",
    "tests/models/loading/targets.sh::a_latent_attention_target_is_refused"
);
target_absent!(
    FEED_FORWARD_EXPERTS,
    "target-feed-forward-experts",
    "its feed-forward is a mixture of experts, one gate, up and down projection per expert",
    "tests/models/loading/targets.sh::a_feed_forward_target_of_experts_is_refused"
);
target_absent!(
    FEED_FORWARD_UNGATED,
    "target-feed-forward-ungated",
    "its feed-forward has no gate",
    "tests/models/loading/targets.sh::a_target_the_model_lacks_is_refused"
);

/// Placing fresh or trained LoRA adapters on a model's projections.
pub static ADAPTERS: Operation = Operation {
    id: "ster.model.adapters",
    entries: &[],
    uses: &[],
    summary: "place LoRA adapters on the projections a model's family stores as one tensor on every layer, refusing a family that holds none and a target it does not store",
    docs: LOAD_DOCS,
    warnings: &[],
    causes: &[
        &SHORT_CONVOLUTION,
        &LAYER_PLANS,
        &SHORTCUT_EXPERTS,
        &LAYERS_REPEATED,
        &STREAMS_HYPER_CONNECTED,
        &INKLING_PROJECTIONS,
        &LIGHTNING_ATTENTION,
        &DELTA_RULE,
        &STATE_SPACE,
        &QUERY_BOTTLENECKED,
        &HEADS_DIFFER,
        &VALUE_EXPERTS,
        &ATTENTION_LATENT,
        &FEED_FORWARD_EXPERTS,
        &FEED_FORWARD_UNGATED,
    ],
};
