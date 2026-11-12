#!/bin/sh
# Real test of the sidecar declarations no Ster trainer writes
# (`ster.lora.artifact`: rank-zero, alpha-invalid, hidden-size-zero,
# targets-none, layers-none, target-repeated, layer-repeated), through the
# built binary on the four-layer fixture.
#
# One real adapter is trained with `ster tune sft`; each test copies it,
# rewrites one field of its sidecar through jq and reads the copy with `ster
# tune inspect`, which refuses it before any tensor is trusted. Every
# refusal names the sidecar and the weights it read.
#
# Usage: tests/tuning/scales/sidecar.sh   (STER selects the binary, default target/debug/ster)
AREA=sidecar
. "$(dirname "$0")/../harness.sh"

sft_adapter trained
TRAINED="$ADAPTER"
# rewritten NAME JQ: a copy of the trained adapter whose sidecar is the
# trained one through JQ, as $ROOT/NAME.safetensors and $ROOT/NAME.json.
rewritten() {
  ADAPTER="$ROOT/$1.safetensors"
  SIDECAR="$ROOT/$1.json"
  cp "$TRAINED" "$ADAPTER"
  jq "$2" "$ROOT/trained/adapter.json" >"$SIDECAR"
}
inspect() {
  run "$1" tune inspect "$ADAPTER"
}

a_sidecar_declaring_rank_zero_is_refused() {
  rewritten rank-zero '.rank = 0'
  inspect 1
  said "Error: checking the adapter artifact: the sidecar declares rank 0, and an adapter of rank 0 stores no update"
  said "; observed: rank: 0; sidecar $SIDECAR, weights $ADAPTER; fix: use the sidecar the ster tune command wrote beside these weights"
  passed a_sidecar_declaring_rank_zero_is_refused
}

a_sidecar_declaring_alpha_zero_is_refused() {
  rewritten alpha-zero '.alpha = 0'
  inspect 1
  said "Error: checking the adapter artifact: the sidecar declares an alpha that is not a positive finite number"
  said "; observed: alpha: 0; rank: "
  passed a_sidecar_declaring_alpha_zero_is_refused
}

a_sidecar_declaring_hidden_size_zero_is_refused() {
  rewritten width-zero '.hidden_size = 0'
  inspect 1
  said "Error: checking the adapter artifact: the sidecar declares hidden size 0"
  said "; observed: hidden_size: 0; model: $MODEL"
  passed a_sidecar_declaring_hidden_size_zero_is_refused
}

a_sidecar_naming_no_target_is_refused() {
  rewritten no-target '.targets = []'
  inspect 1
  said "Error: checking the adapter artifact: the sidecar names no target projection, so the adapter adapts nothing"
  said "; observed: targets: []; layers: "
  passed a_sidecar_naming_no_target_is_refused
}

a_sidecar_naming_no_layer_is_refused() {
  rewritten no-layer '.layers = []'
  inspect 1
  said "Error: checking the adapter artifact: the sidecar names no layer, so the adapter adapts nothing"
  said "; observed: layers: []; targets: "
  passed a_sidecar_naming_no_layer_is_refused
}

a_sidecar_naming_a_target_twice_is_refused() {
  rewritten target-twice '.targets = ["query", "query"]'
  inspect 1
  said "Error: checking the adapter artifact: the sidecar names target query twice"
  said "; observed: targets: query,query"
  passed a_sidecar_naming_a_target_twice_is_refused
}

a_sidecar_naming_a_layer_twice_is_refused() {
  rewritten layer-twice '.layers = [0, 0]'
  inspect 1
  said "Error: checking the adapter artifact: the sidecar names layer 0 twice"
  said "; observed: layers: 0,0"
  passed a_sidecar_naming_a_layer_twice_is_refused
}

a_sidecar_declaring_rank_zero_is_refused
a_sidecar_declaring_alpha_zero_is_refused
a_sidecar_declaring_hidden_size_zero_is_refused
a_sidecar_naming_no_target_is_refused
a_sidecar_naming_no_layer_is_refused
a_sidecar_naming_a_target_twice_is_refused
a_sidecar_naming_a_layer_twice_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
