#!/bin/sh
# Real test of how LoRA adapters fail: the spec a trainer is given
# (`ster.lora.spec`) and the artifact a trained adapter travels in
# (`ster.lora.artifact`), through the built binary on the four-layer
# fixture.
#
# The spec settings are refused with exit status 2 before anything is
# read: the runs name a model that does not exist. One real adapter is
# trained with `ster tune sft`; each artifact test copies it and breaks one
# thing (the sidecar removed, unreadable, rewritten through jq, or the
# weights replaced), then reads it with `ster tune inspect`. The adapters
# no trainer writes (a NaN factor, a head on an adapter, a key projection
# of another width) are written as safetensors by hand.
#
# Usage: tests/tuning/adapters.sh   (STER selects the binary, default target/debug/ster)
AREA=adapters
. "$(dirname "$0")/harness.sh"

ABSENT="$ROOT/no-such-model"
sft_adapter trained
TRAINED="$ADAPTER"
# copied NAME: a copy of the trained adapter and its sidecar as
# $ROOT/NAME.safetensors and $ROOT/NAME.json.
copied() {
  ADAPTER="$ROOT/$1.safetensors"
  SIDECAR="$ROOT/$1.json"
  cp "$TRAINED" "$ADAPTER"
  cp "$ROOT/trained/adapter.json" "$SIDECAR"
}
# rewritten NAME JQ: a copy whose sidecar is the trained one through JQ.
rewritten() {
  copied "$1"
  jq "$2" "$ROOT/trained/adapter.json" >"$SIDECAR"
}
inspect() {
  run "$1" tune inspect "$ADAPTER"
}
# handmade NAME SIDECAR HEADER DATA: an adapter written by hand, its sidecar
# the JSON SIDECAR with this run's model, its weights HEADER over DATA.
handmade() {
  ADAPTER="$ROOT/$1.safetensors"
  SIDECAR="$ROOT/$1.json"
  printf '%s' "$2" | jq --arg model "$MODEL" '. + {schema_version: 1, product: "ster", model: $model,
    model_revision: null, alpha: 1, layers: [0], hidden_size: 64, train: {}}' >"$SIDECAR"
  safetensors "$ADAPTER" "$3" "$4"
}

an_unknown_target_is_refused_before_the_model_loads() {
  tune 2 sft "$SFT" --model "$ABSENT" --targets query,nope
  said "Error: reading the adapter targets: unknown adapter target \"nope\"; expected one of query, key, value, output, gate, up, down; observed: --targets nope"
  unsaid "finding model"
  passed an_unknown_target_is_refused_before_the_model_loads
}

no_target_is_refused_before_the_model_loads() {
  tune 2 sft "$SFT" --model "$ABSENT" --targets ,
  said "Error: reading the adapter targets: adapter spec names no targets; choose at least one of query, key, value, output, gate, up, down; observed: --targets \",\""
  passed no_target_is_refused_before_the_model_loads
}

a_rank_of_zero_is_refused_before_the_model_loads() {
  tune 2 dpo "$DPO" --model "$ABSENT" --rank 0
  said "Error: checking the adapter spec: adapter rank 0 trains nothing; choose a rank of at least 1; observed: --rank 0"
  unsaid "finding model"
  passed a_rank_of_zero_is_refused_before_the_model_loads
}

an_alpha_of_zero_is_refused_before_the_model_loads() {
  tune 2 sft "$SFT" --model "$ABSENT" --alpha 0
  said "Error: checking the adapter spec: adapter alpha 0 is not a positive finite number; observed: --alpha 0"
  passed an_alpha_of_zero_is_refused_before_the_model_loads
}

a_layer_past_the_model_is_refused() {
  tune 2 sft "$SFT" --layers 9 --output "$ROOT/past/adapter.safetensors"
  said "Error: placing the adapters on the model's layers: adapter spec names layer 9, but the model has 4 layers; observed: --layers 9; fix: give --layers indexes from 0 to 3, or all"
  [ ! -e "$ROOT/past/adapter.safetensors" ] || fail "a refused run wrote an adapter"
  passed a_layer_past_the_model_is_refused
}

a_model_without_layers_is_refused() {
  cp -R "$MODEL" "$ROOT/layerless"
  jq '.num_hidden_layers = 0' "$MODEL/config.json" >"$ROOT/layerless/config.json"
  tune 1 sft "$SFT" --model "$ROOT/layerless" --output "$ROOT/layerless-run/adapter.safetensors"
  said "Error: placing the adapters on the model's layers: the model's config states 0 layers, so it holds no projection an adapter can be placed on; observed: num_hidden_layers: 0"
  [ ! -e "$ROOT/layerless-run/adapter.safetensors" ] || fail "a refused run wrote an adapter"
  passed a_model_without_layers_is_refused
}

a_missing_adapter_is_refused() {
  ADAPTER="$ROOT/absent.safetensors"
  inspect 1
  said "Error: reading adapter $ADAPTER: no file is there: nothing at $ADAPTER is the adapter named; observed: path: $ADAPTER (absent); its directory $ROOT"
  passed a_missing_adapter_is_refused
}

an_unreadable_sidecar_is_refused() {
  copied closed
  chmod 000 "$SIDECAR"
  inspect 1
  chmod 644 "$SIDECAR"
  said "Error: reading the adapter's sidecar $SIDECAR: this account may not read it: the file's permission bits or owner keep this account out; observed: path: $SIDECAR (a file, "
  passed an_unreadable_sidecar_is_refused
}

an_adapter_without_its_sidecar_is_refused() {
  copied alone
  rm "$SIDECAR"
  inspect 1
  said "Error: reading adapter $ADAPTER: adapter $ADAPTER has no sidecar at $SIDECAR; the pair is written together and must travel together"
  passed an_adapter_without_its_sidecar_is_refused
}

a_steering_artifact_given_as_an_adapter_is_refused() {
  copied steering
  printf '%s' '{"trait_name": "calm", "vectors": []}' >"$SIDECAR"
  inspect 2
  said "Error: reading adapter $ADAPTER: $SIDECAR is a steering artifact, not a LoRA adapter sidecar"
  passed a_steering_artifact_given_as_an_adapter_is_refused
}

a_sidecar_that_is_not_one_is_refused() {
  copied not-json
  echo nope >"$SIDECAR"
  inspect 1
  said "Error: reading the adapter's sidecar $SIDECAR: it is not JSON, and the adapter's sidecar is the JSON sidecar a \`ster tune\` command writes beside the adapter's .safetensors file: it is another kind of file, or one broken by hand; observed: path: $SIDECAR (a file, 5 bytes, "
  passed a_sidecar_that_is_not_one_is_refused
}

adapter_weights_that_are_not_safetensors_are_refused() {
  copied junk
  echo junk >"$ADAPTER"
  inspect 1
  said "Error: reading the adapter's weights $ADAPTER: $ADAPTER is not a safetensors file: its header or its table of tensors does not parse, so it was cut short while written or is a file of another format; observed: path: $ADAPTER (a file, 5 bytes, "
  passed adapter_weights_that_are_not_safetensors_are_refused
}

# A safetensors file whose one tensor is stored as bool: its header names the
# dtype, eight little-endian bytes give the header's length.
adapter_weights_of_a_dtype_candle_does_not_load_are_refused() {
  copied boolean
  header='{"w":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}}'
  length=$(printf '%s' "$header" | wc -c | tr -d ' ')
  { printf "$(printf '\\%03o' "$length")"; printf '\000\000\000\000\000\000\000'; printf '%s' "$header"; printf '\001'; } >"$ADAPTER"
  inspect 1
  said "Error: reading the adapter's weights $ADAPTER: a tensor of $ADAPTER is stored in a dtype Candle does not load, such as bool, i8 or u64"
  passed adapter_weights_of_a_dtype_candle_does_not_load_are_refused
}

a_sidecar_of_another_schema_is_refused() {
  rewritten schema '.schema_version = 2'
  inspect 1
  said "Error: checking the adapter artifact: adapter artifact schema 2 is unsupported; this Ster build reads schema 1"
  passed a_sidecar_of_another_schema_is_refused
}

a_sidecar_of_another_product_is_refused() {
  rewritten product '.product = "other"'
  inspect 1
  said "Error: checking the adapter artifact: adapter artifact belongs to product \"other\", not Ster; observed: product: \"other\""
  passed a_sidecar_of_another_product_is_refused
}

a_sidecar_naming_a_layer_the_weights_lack_is_refused() {
  rewritten layer '.layers = [7]'
  inspect 1
  said "Error: checking the adapter artifact: adapter artifact is missing tensor layers.7.query.a, which its sidecar's targets and layers need"
  passed a_sidecar_naming_a_layer_the_weights_lack_is_refused
}

a_sidecar_of_another_rank_is_refused() {
  rewritten rank '.rank = 3'
  inspect 1
  said "Error: checking the adapter artifact: adapter tensor layers.0.query.a has 2 rows, expected 3; observed: layers.0.query.a: [2, 64]"
  passed a_sidecar_of_another_rank_is_refused
}

a_non_finite_factor_is_refused() {
  floats "$ROOT/nan.a" 63 "$ZERO"
  floats "$ROOT/nan.one" 1 "$NAN"
  floats "$ROOT/nan.b" 64 "$ZERO"
  cat "$ROOT/nan.a" "$ROOT/nan.one" "$ROOT/nan.b" >"$ROOT/nan.data"
  handmade nan '{"kind": "adapter", "rank": 1, "targets": ["query"]}' \
    '{"layers.0.query.a":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"layers.0.query.b":{"dtype":"F32","shape":[64,1],"data_offsets":[256,512]}}' \
    "$ROOT/nan.data"
  inspect 1
  said "Error: checking the adapter artifact: adapter tensor layers.0.query.a contains a non-finite value, which would carry NaN through every forward pass; observed: layers.0.query.a: 1 non-finite values"
  passed a_non_finite_factor_is_refused
}

a_reward_sidecar_without_its_head_is_refused() {
  rewritten reward '.kind = "reward"'
  inspect 1
  said "Error: checking the adapter artifact: reward artifact is missing tensor reward.head, which is the head it exists to carry; observed: kind: reward, tensors: 16"
  passed a_reward_sidecar_without_its_head_is_refused
}

an_adapter_sidecar_over_reward_weights_is_refused() {
  floats "$ROOT/head.data" 192 "$ZERO"
  handmade head '{"kind": "adapter", "rank": 1, "targets": ["query"]}' \
    '{"layers.0.query.a":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"layers.0.query.b":{"dtype":"F32","shape":[64,1],"data_offsets":[256,512]},"reward.head":{"dtype":"F32","shape":[1,64],"data_offsets":[512,768]}}' \
    "$ROOT/head.data"
  inspect 1
  said "Error: checking the adapter artifact: adapter artifact carries a reward.head tensor but declares kind adapter"
  passed an_adapter_sidecar_over_reward_weights_is_refused
}

# The fixture's key projection writes 2 heads of 16, 32 outputs; this one
# writes 48.
an_adapter_of_another_projection_width_is_refused() {
  floats "$ROOT/key.data" 112 "$ZERO"
  handmade key '{"kind": "adapter", "rank": 1, "targets": ["key"]}' \
    '{"layers.0.key.a":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"layers.0.key.b":{"dtype":"F32","shape":[48,1],"data_offsets":[256,448]}}' \
    "$ROOT/key.data"
  inspect 0
  run 1 generate --model "$MODEL" --adapter "$ADAPTER" --prompt "describe the sea ." --max-new-tokens 1 --temperature 0 --seed 1
  said "fitting the adapter to the model's projections: adapter tensor layers.0.key.b has shape [48, 1], expected [32, 1] for the model projection"
  passed an_adapter_of_another_projection_width_is_refused
}

an_output_under_a_file_is_refused() {
  tune 1 sft "$SFT" --output "$ROOT/examples.json/adapter.safetensors"
  said "Error: creating the directory of the adapter's weights $ROOT/examples.json/adapter.safetensors: the path runs through a file: $ROOT/examples.json is not a directory, so nothing can be created below it; observed: path: $ROOT/examples.json/adapter.safetensors; $ROOT/examples.json: a file, "
  said "; no adapter was written; the system answered: "
  passed an_output_under_a_file_is_refused
}

an_unknown_target_is_refused_before_the_model_loads
no_target_is_refused_before_the_model_loads
a_rank_of_zero_is_refused_before_the_model_loads
an_alpha_of_zero_is_refused_before_the_model_loads
a_layer_past_the_model_is_refused
a_model_without_layers_is_refused
a_missing_adapter_is_refused
an_unreadable_sidecar_is_refused
an_adapter_without_its_sidecar_is_refused
a_steering_artifact_given_as_an_adapter_is_refused
a_sidecar_that_is_not_one_is_refused
adapter_weights_that_are_not_safetensors_are_refused
adapter_weights_of_a_dtype_candle_does_not_load_are_refused
a_sidecar_of_another_schema_is_refused
a_sidecar_of_another_product_is_refused
a_sidecar_naming_a_layer_the_weights_lack_is_refused
a_sidecar_of_another_rank_is_refused
a_non_finite_factor_is_refused
a_reward_sidecar_without_its_head_is_refused
an_adapter_sidecar_over_reward_weights_is_refused
an_adapter_of_another_projection_width_is_refused
an_output_under_a_file_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
