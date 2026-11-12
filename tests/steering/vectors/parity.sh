#!/bin/sh
# Real test of `ster parity` and how it fails (`ster.parity`), through the
# built binary on the four-layer fixture (hidden_size 64, 67 tokens). The
# reference states files are the fixture's own model.safetensors: its
# tensors are real safetensors of the shapes each test needs, an up
# projection [128, 64] standing in for 128 positions of states. The run's
# home and fixture are tests/runtime/harness.sh's.
#
# Usage: tests/steering/vectors/parity.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=steering-parity . "$(dirname "$0")/../../runtime/harness.sh"
# parity EXIT RECORDS: ster parity over a file holding RECORDS (JSON).
parity() {
  echo "{\"records\": $2}" >"$ROOT/parity.json"
  run "$1" parity --model "$ROOT/base" --input "$ROOT/parity.json"
}
ids() { jq -nc "[range($1) | . % 67]"; }
UP=model.layers.0.mlp.up_proj.weight

cp "$ROOT/base/model.safetensors" "$ROOT/states.safetensors"
parity 0 "[{\"tokenIds\": $(ids 128), \"hidden\": \"states.safetensors\", \"tensor\": \"$UP\"}]"
[ "$(echo "$out" | jq '.records[0].positions')" = "128" ] || fail "parity compared other than 128 positions: $out"
passed "128 positions of a [128, 64] reference are compared"

a_missing_parity_file_is_refused() {
  run 1 parity --model "$ROOT/base" --input "$ROOT/absent.json"
  said "reading the parity file $ROOT/absent.json: no file is there"
  passed a_missing_parity_file_is_refused
}

a_parity_file_without_records_is_refused() {
  parity 1 "[]"
  said "$ROOT/parity.json lists no records to compare"
  passed a_parity_file_without_records_is_refused
}

a_record_naming_two_id_sources_is_refused() {
  parity 1 '[{"tokenIds": [1, 2], "tokenIdsFile": "ids.json", "hidden": "states.safetensors"}]'
  said "record 0 names both tokenIds and tokenIdsFile; give one"
  said "observed: record 0: tokenIds holds 2 ids; tokenIdsFile: ids.json"
  passed a_record_naming_two_id_sources_is_refused
}

a_record_naming_no_ids_is_refused() {
  parity 1 '[{"hidden": "states.safetensors"}]'
  said "record 0 names neither tokenIds nor tokenIdsFile"
  passed a_record_naming_no_ids_is_refused
}

a_missing_token_id_file_is_refused() {
  parity 1 '[{"tokenIdsFile": "absent-ids.json", "hidden": "states.safetensors"}]'
  said "reading record 0's token id file $ROOT/absent-ids.json: no file is there"
  passed a_missing_token_id_file_is_refused
}

a_record_without_ids_is_refused() {
  echo '[]' >"$ROOT/ids.json"
  parity 1 '[{"tokenIdsFile": "ids.json", "hidden": "states.safetensors"}]'
  said "record 0 holds no token ids"
  said "observed: record 0: tokenIdsFile $ROOT/ids.json holds []"
  passed a_record_without_ids_is_refused
}

a_token_past_the_vocabulary_is_refused() {
  parity 1 '[{"tokenIds": [1, 2, 999], "hidden": "states.safetensors"}]'
  said "record 0 holds token id 999, past the model's 67 tokens"
  said "id 999 at position 2; the model's embedding table holds ids 0 to 66"
  passed a_token_past_the_vocabulary_is_refused
}

a_missing_states_file_is_refused() {
  parity 1 '[{"tokenIds": [1, 2], "hidden": "absent.safetensors"}]'
  said "reading record 0's reference states $ROOT/absent.safetensors: no file is there"
  passed a_missing_states_file_is_refused
}

a_tensor_the_file_lacks_is_refused() {
  parity 1 '[{"tokenIds": [1, 2], "hidden": "states.safetensors", "tensor": "nothing"}]'
  said "$ROOT/states.safetensors holds no tensor \"nothing\""
  said "$UP"
  passed a_tensor_the_file_lacks_is_refused
}

a_tensor_of_another_shape_is_refused() {
  parity 1 '[{"tokenIds": [1, 2], "hidden": "states.safetensors", "tensor": "model.norm.weight"}]'
  said "\"model.norm.weight\" is [64]; Ster compares [positions, hidden] states"
  passed a_tensor_of_another_shape_is_refused
}

more_positions_than_tokens_are_refused() {
  parity 1 '[{"tokenIds": [1, 2, 3], "hidden": "states.safetensors", "tensor": "model.embed_tokens.weight"}]'
  said "record 0's reference holds 67 positions for 3 tokens"
  passed more_positions_than_tokens_are_refused
}

states_of_another_width_are_refused() {
  parity 1 "[{\"tokenIds\": $(ids 64), \"hidden\": \"states.safetensors\", \"tensor\": \"model.layers.0.mlp.down_proj.weight\"}]"
  said "record 0's reference states are 128 wide and this model's 64"
  passed states_of_another_width_are_refused
}

a_missing_parity_file_is_refused
a_parity_file_without_records_is_refused
a_record_naming_two_id_sources_is_refused
a_record_naming_no_ids_is_refused
a_missing_token_id_file_is_refused
a_record_without_ids_is_refused
a_token_past_the_vocabulary_is_refused
a_missing_states_file_is_refused
a_tensor_the_file_lacks_is_refused
a_tensor_of_another_shape_is_refused
more_positions_than_tokens_are_refused
states_of_another_width_are_refused
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
