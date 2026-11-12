#!/bin/sh
# Real test of how rewriting a checkpoint with a trained artifact fails:
# fitting it to the checkpoint (`ster.tune.fit`), reading and writing the
# checkpoint (`ster.tune.write`), and removing a steering direction
# (`ster.tune.ablate`), through `ster tune merge`, `ster tune export` and
# `ster vector ablate` on the four-layer fixture.
#
# The artifacts are a real adapter `ster tune sft` trains, a real reward
# model `ster tune reward` trains, and adapters and directions written by
# hand where no trainer writes one (another width, a layer past the model,
# a direction of length zero). The checkpoints are copies of the fixture
# with one thing broken, and one written by hand whose query projection is
# 8 by 8 and whose attention output writes 32 rows, which the config it
# carries does not describe.
#
# Usage: tests/tuning/scales/rewrite.sh   (STER selects the binary, default target/debug/ster)
AREA=rewrite
. "$(dirname "$0")/../harness.sh"

sft_adapter trained
TRAINED="$ADAPTER"
OUTPUTS=0
# merge EXPECTED MODEL ADAPTER [OUTPUT]: ster tune merge, into OUTPUT or a
# fresh directory of this run.
merge() {
  OUTPUTS=$((OUTPUTS + 1))
  run "$1" tune merge --model "$2" --adapter "$3" --output "${4:-$ROOT/merged-$OUTPUTS}"
}
# copy NAME: a copy of the fixture at $ROOT/NAME, its path in $COPY, and in
# $ADAPTER a copy of the trained adapter whose sidecar names it.
copy() {
  COPY="$ROOT/$1"
  cp -R "$MODEL" "$COPY"
  ADAPTER="$ROOT/$1-adapter.safetensors"
  cp "$TRAINED" "$ADAPTER"
  jq --arg model "$COPY" '.model = $model' "$ROOT/trained/adapter.json" >"$ROOT/$1-adapter.json"
}
# handmade NAME MODEL SIDECAR HEADER DATA: an adapter for MODEL written by
# hand, rank 1, its sidecar SIDECAR completed, its weights HEADER over DATA.
handmade() {
  ADAPTER="$ROOT/$1.safetensors"
  printf '%s' "$3" | jq --arg model "$2" '. + {schema_version: 1, product: "ster", kind: "adapter",
    model: $model, model_revision: null, rank: 1, alpha: 1, train: {}}' >"$ROOT/$1.json"
  safetensors "$ADAPTER" "$4" "$5"
}
floats "$ROOT/rank-one.data" 128 "$ZERO"

# A checkpoint written by hand: the fixture's config and tokenizer over a
# query projection 8 by 8 and an attention output of 32 rows.
HAND="$ROOT/hand"
mkdir -p "$HAND"
cp "$MODEL/config.json" "$MODEL/tokenizer.json" "$HAND/"
floats "$ROOT/hand.data" 2112 "$ZERO"
safetensors "$HAND/model.safetensors" \
  '{"model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]},"model.layers.0.self_attn.o_proj.weight":{"dtype":"F32","shape":[32,64],"data_offsets":[256,8448]}}' \
  "$ROOT/hand.data"
# direction NAME MODEL LAYER VALUE: a steering artifact for MODEL whose one
# vector, at LAYER, holds VALUE 64 times.
direction() {
  VECTOR="$ROOT/$1.json"
  jq -n --arg model "$2" --argjson layer "$3" --argjson value "$4" '{schema_version: 1,
    product: "ster", model: $model, model_revision: null, trait_name: "calm", method: "handmade",
    hidden_size: 64, vectors: [{layer: $layer, values: [range(64) | $value], train_margin: 0,
    train_accuracy: 0}]}' >"$VECTOR"
}
# ablate EXPECTED MODEL STRENGTH: ster vector ablate of $VECTOR.
ablate() {
  OUTPUTS=$((OUTPUTS + 1))
  run "$1" vector ablate --model "$2" --vector "$VECTOR" --strength "$3" --output "$ROOT/ablated-$OUTPUTS"
}

a_reward_model_is_refused_by_merge() {
  tune 0 reward "$REWARD"
  merge 1 "$MODEL" "$ROOT/reward/adapter.safetensors"
  said "Error: merging adapter $ROOT/reward/adapter.safetensors: the artifact is a reward model, not a generation adapter: a merge folds an adapter's low-rank updates into the base and writes no head"
  said "observed: kind: reward; model: $MODEL"
  passed a_reward_model_is_refused_by_merge
}

# Runs after a_reward_model_is_refused_by_merge, which trains the reward model.
a_reward_model_is_refused_by_export() {
  run 1 tune export --model "$MODEL" --adapter "$ROOT/reward/adapter.safetensors" --format peft --output "$ROOT/exported-reward"
  said "Error: exporting adapter $ROOT/reward/adapter.safetensors: the artifact is a reward model, not a generation adapter: a PEFT adapter directory holds low-rank updates only and has no place for the reward model's head"
  passed a_reward_model_is_refused_by_export
}

an_artifact_for_another_model_is_refused() {
  cp -R "$MODEL" "$ROOT/other"
  merge 1 "$ROOT/other" "$TRAINED"
  said "Error: merging artifact $TRAINED: the artifact was trained for model \"$MODEL\", current model is $ROOT/other"
  passed an_artifact_for_another_model_is_refused
}

# gguf NAME: a copy of the fixture whose weights are one empty .gguf file.
gguf() {
  copy "$1"
  rm "$COPY/model.safetensors"
  : >"$COPY/model.gguf"
}

a_gguf_checkpoint_is_refused_by_merge() {
  gguf gguf
  merge 1 "$COPY" "$ADAPTER"
  said "Error: merging artifact $ADAPTER: $COPY is a GGUF checkpoint, and an adapter is merged into safetensors weights only"
  said "fix: run the adapter on it unmerged (\`ster generate --model $COPY --adapter $ADAPTER --prompt …\`)"
  passed a_gguf_checkpoint_is_refused_by_merge
}

a_gguf_checkpoint_is_refused_by_export() {
  gguf gguf-export
  run 1 tune export --model "$COPY" --adapter "$ADAPTER" --format peft --output "$ROOT/exported-gguf"
  said "Error: exporting artifact $ADAPTER: $COPY is a GGUF checkpoint, and a PEFT adapter names the modules of the base model's safetensors release"
  [ ! -e "$ROOT/exported-gguf" ] || fail "a refused export wrote $ROOT/exported-gguf"
  passed a_gguf_checkpoint_is_refused_by_export
}

a_gguf_checkpoint_is_refused_by_ablate() {
  gguf gguf-ablate
  direction gguf-direction "$COPY" 0 1
  ablate 1 "$COPY" 1
  said "Error: ablating artifact $VECTOR: $COPY is a GGUF checkpoint, and a direction is projected out of safetensors weights only"
  passed a_gguf_checkpoint_is_refused_by_ablate
}

an_artifact_of_another_width_is_refused() {
  floats "$ROOT/narrow.data" 96 "$ZERO"
  handmade narrow "$MODEL" '{"targets": ["query"], "layers": [0], "hidden_size": 32}' \
    '{"layers.0.query.a":{"dtype":"F32","shape":[1,32],"data_offsets":[0,128]},"layers.0.query.b":{"dtype":"F32","shape":[64,1],"data_offsets":[128,384]}}' \
    "$ROOT/narrow.data"
  merge 1 "$MODEL" "$ADAPTER"
  said "Error: merging artifact $ADAPTER: artifact width 32 does not match model width 64; observed: hidden_size: 64"
  passed an_artifact_of_another_width_is_refused
}

an_artifact_past_the_model_layers_is_refused() {
  handmade past "$MODEL" '{"targets": ["query"], "layers": [7], "hidden_size": 64}' \
    '{"layers.7.query.a":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"layers.7.query.b":{"dtype":"F32","shape":[64,1],"data_offsets":[256,512]}}' \
    "$ROOT/rank-one.data"
  merge 1 "$MODEL" "$ADAPTER"
  said "Error: merging artifact $ADAPTER: layer 7 is outside the model's 4 layers; observed: artifact layers: 7"
  passed an_artifact_past_the_model_layers_is_refused
}

# The fixture's value projection writes 2 heads of 16, 32 outputs.
a_checkpoint_without_the_projection_is_refused() {
  floats "$ROOT/value.data" 96 "$ZERO"
  handmade value "$HAND" '{"targets": ["value"], "layers": [0], "hidden_size": 64}' \
    '{"layers.0.value.a":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"layers.0.value.b":{"dtype":"F32","shape":[32,1],"data_offsets":[256,384]}}' \
    "$ROOT/value.data"
  merge 1 "$HAND" "$ADAPTER"
  said "Error: merging the value projection of layer 0: the checkpoint holds no tensor "
  said "v_proj.weight; observed: the checkpoint: 2 tensors"
  passed a_checkpoint_without_the_projection_is_refused
}

an_update_that_does_not_fit_is_refused() {
  handmade query "$HAND" '{"targets": ["query"], "layers": [0], "hidden_size": 64}' \
    '{"layers.0.query.a":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"layers.0.query.b":{"dtype":"F32","shape":[64,1],"data_offsets":[256,512]}}' \
    "$ROOT/rank-one.data"
  merge 1 "$HAND" "$ADAPTER"
  said "Error: merging the query projection of layer 0: the update the artifact carries has another shape than model.layers.0.self_attn.q_proj.weight as the checkpoint stores it: the artifact was trained on another model or width"
  said "observed: model.layers.0.self_attn.q_proj.weight: [8, 8] in "
  passed an_update_that_does_not_fit_is_refused
}

a_weight_file_that_is_not_safetensors_is_refused() {
  copy junk
  echo junk >"$COPY/model.safetensors"
  merge 1 "$COPY" "$ADAPTER"
  said "Error: reading a weight file of $COPY $COPY/model.safetensors: $COPY/model.safetensors is not a safetensors file: its header or its table of tensors does not parse"
  passed a_weight_file_that_is_not_safetensors_is_refused
}

a_tensor_in_two_shards_is_refused() {
  copy shards
  cp "$COPY/model.safetensors" "$COPY/second.safetensors"
  merge 1 "$COPY" "$ADAPTER"
  said "Error: reading the weights of $COPY: model weight "
  said " appears in more than one weight file, so which one the checkpoint means is unknown"
  passed a_tensor_in_two_shards_is_refused
}

an_output_under_a_file_is_refused() {
  merge 1 "$MODEL" "$TRAINED" "$ROOT/examples.json/merged"
  said "Error: creating the output directory $ROOT/examples.json/merged: the path runs through a file: $ROOT/examples.json is not a directory, so nothing can be created below it; observed: path: $ROOT/examples.json/merged; $ROOT/examples.json: a file, "
  said "; $ROOT/examples.json/merged holds only the files written before this one, so it is no checkpoint yet; the system answered: "
  passed an_output_under_a_file_is_refused
}

a_tokenizer_that_cannot_be_copied_is_refused() {
  copy closed-tokenizer
  chmod 000 "$COPY/tokenizer.json"
  merge 1 "$COPY" "$ADAPTER" "$ROOT/merged-closed"
  chmod 644 "$COPY/tokenizer.json"
  said "Error: reading the source checkpoint's file $COPY/tokenizer.json: this account may not read it: the file's permission bits or owner keep this account out; observed: path: $COPY/tokenizer.json (a file, "
  passed a_tokenizer_that_cannot_be_copied_is_refused
}

an_ablation_strength_of_zero_is_refused() {
  direction calm "$MODEL" 1 0.5
  ablate 2 "$ROOT/no-such-model" 0
  said "Error: removing the direction of $VECTOR from $ROOT/no-such-model: ablation strength 0 changes nothing or nothing sensible; it must be a finite, non-zero number; observed: --strength 0"
  unsaid "finding model"
  passed an_ablation_strength_of_zero_is_refused
}

a_direction_of_length_zero_is_refused() {
  direction zero "$MODEL" 1 0
  ablate 1 "$MODEL" 1
  said "Error: removing the direction at layer 1 of $VECTOR: the direction at layer 1 has length 0, so there is nothing to remove; observed: length of layer 1: 0"
  passed a_direction_of_length_zero_is_refused
}

a_write_of_another_width_is_refused() {
  direction hand "$HAND" 0 0.5
  ablate 1 "$HAND" 1
  said "Error: removing the direction at layer 0 of $VECTOR: model.layers.0.self_attn.o_proj.weight writes 32 outputs where the residual stream is 64 wide"
  passed a_write_of_another_width_is_refused
}

a_reward_model_is_refused_by_merge
a_reward_model_is_refused_by_export
an_artifact_for_another_model_is_refused
a_gguf_checkpoint_is_refused_by_merge
a_gguf_checkpoint_is_refused_by_export
a_gguf_checkpoint_is_refused_by_ablate
an_artifact_of_another_width_is_refused
an_artifact_past_the_model_layers_is_refused
a_checkpoint_without_the_projection_is_refused
an_update_that_does_not_fit_is_refused
a_weight_file_that_is_not_safetensors_is_refused
a_tensor_in_two_shards_is_refused
an_output_under_a_file_is_refused
a_tokenizer_that_cannot_be_copied_is_refused
an_ablation_strength_of_zero_is_refused
a_direction_of_length_zero_is_refused
a_write_of_another_width_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
