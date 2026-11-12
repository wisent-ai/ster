#!/bin/sh
# Real test of settings a config states that the decoder cannot run, refused
# while the config is read and before a weight is mapped, through the built
# binary: one test per reason (`ster.model.config.settings-incomplete`,
# `-contradictory`, `-out-of-range` and `-malformed`; `-unimplemented` is
# tests/models/load.sh::a_setting_ster_does_not_implement_is_refused).
#
# Each test copies the four-layer Llama fixture `ster model fixture create`
# writes, rewrites its config.json into another family's with jq, and asks
# `ster generate` to load the copy: the refusal comes from the config alone,
# so the fixture's Llama weights are never read.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: tests/models/settings.sh   (STER selects the binary, default target/debug/ster)
AREA=models-settings . "$(dirname "$0")/../runtime/harness.sh"

# config NAME JQ: a copy of the fixture whose config.json is the fixture's
# through JQ, its path in $COPY.
config() {
  COPY="$ROOT/$1"
  cp -R "$MODEL" "$COPY"
  jq "$2" "$MODEL/config.json" >"$COPY/config.json"
}

# DeepSeek-V3.2's latent attention with the parts the test states.
LATENT='.model_type = "deepseek_v32" | .architectures = ["DeepseekV32ForCausalLM"]
  | .kv_lora_rank = 16 | .qk_nope_head_dim = 8 | .qk_rope_head_dim = 8 | .v_head_dim = 16'

an_indexer_without_a_latent_query_is_refused() {
  config no-query-rank "$LATENT | del(.q_lora_rank)"
  generate 1 "$COPY" "describe the sea ."
  said "reading the decoder's settings from the checkpoint's config: the config turns on a part of the decoder"
  said "observed: $COPY/config.json declares DeepSeek Sparse Attention's indexer (index_topk) without latent attention under q_lora_rank; the indexer reads the latent query q_lora_rank sizes"
  said "; the config states kv_lora_rank: 16, q_lora_rank: absent; fix: state the settings the observed sentence names"
  passed an_indexer_without_a_latent_query_is_refused
}

an_indexer_narrower_than_the_rotation_is_refused() {
  config narrow-index "$LATENT | .q_lora_rank = 16 | .index_head_dim = 4"
  generate 1 "$COPY" "describe the sea ."
  said "reading the decoder's settings from the checkpoint's config: two settings of the config disagree"
  said "observed: $COPY/config.json rotates 8 components of each latent head and the indexer's heads are 4 wide; the indexer rotates as many of its own components"
  said "; the config states qk_rope_head_dim: 8, index_head_dim: 4; fix: correct the key the observed sentence names"
  passed an_indexer_narrower_than_the_rotation_is_refused
}

# A rotated share three times the head: more components than a head holds.
a_rotation_wider_than_the_head_is_refused() {
  config wide-rotation '.partial_rotary_factor = 3'
  generate 1 "$COPY" "describe the sea ."
  said "reading the decoder's settings from the checkpoint's config: a setting of the config holds a value outside the range the decoder runs"
  said "components per head (partial_rotary_factor, rotary_pct or rotary_dim); the rotated width must be even, above zero and at most the head"
  said "partial_rotary_factor: 3, rotary_pct: absent, rotary_dim: absent; fix: state the setting within the range the observed sentence names"
  passed a_rotation_wider_than_the_head_is_refused
}

# SmolLM3 marks each layer 1 (rotates) or 0 (does not) in no_rope_layers.
a_layer_mark_out_of_form_is_refused() {
  config mark-two '.model_type = "smollm3" | .architectures = ["SmolLM3ForCausalLM"] | .no_rope_layers = [1, 1, 2, 1]'
  generate 1 "$COPY" "describe the sea ."
  said "reading the decoder's settings from the checkpoint's config: a setting of the config is not in the form its family spells it"
  said "observed: $COPY/config.json marks layer 2 in no_rope_layers as 2; expected 0 or 1; the config states no_rope_layers: [1,1,2,1]; fix: state the setting in the form the observed sentence names"
  passed a_layer_mark_out_of_form_is_refused
}

an_indexer_without_a_latent_query_is_refused
an_indexer_narrower_than_the_rotation_is_refused
a_rotation_wider_than_the_head_is_refused
a_layer_mark_out_of_form_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
