#!/bin/sh
# Real test of how steering a loaded model with an artifact fails
# (`ster.runtime.steer`, `ster.runtime.layers`), through `ster generate
# --vector` on the four-layer fixture. The artifacts are written by hand
# with jq, each wrong in one way: trained for another model, at another
# width, or at a layer the fixture lacks; the strengths are refused before
# any vector is added.
#
# Usage: tests/runtime/steer.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=steer
. "$(dirname "$0")/harness.sh"

# direction NAME MODEL LAYER WIDTH: a steering artifact for MODEL whose one
# vector, at LAYER, holds WIDTH values of 0.5.
direction() {
  VECTOR="$ROOT/$1.json"
  jq -n --arg model "$2" --argjson layer "$3" --argjson width "$4" '{schema_version: 1,
    product: "ster", model: $model, model_revision: null, trait_name: "calm", method: "handmade",
    hidden_size: $width, vectors: [{layer: $layer, values: [range($width) | 0.5], train_margin: 0,
    train_accuracy: 0}]}' >"$VECTOR"
}
direction calm "$MODEL" 1 64
generate 0 "$MODEL" "describe the sea ." --vector "$VECTOR" --strength 2
passed "a vector trained for the fixture steers it"

a_vector_for_another_model_is_refused() {
  direction other "$ROOT/other" 1 64
  generate 1 "$MODEL" "describe the sea ." --vector "$VECTOR" --strength 2
  said "Error: steering $MODEL with the calm artifact: the artifact was trained for model \"$ROOT/other\", current model is $MODEL"
  passed a_vector_for_another_model_is_refused
}

a_vector_of_another_width_is_refused() {
  direction narrow "$MODEL" 1 32
  generate 1 "$MODEL" "describe the sea ." --vector "$VECTOR" --strength 2
  said "Error: steering $MODEL with the calm artifact: artifact width 32 does not match model width 64; observed: artifact hidden_size: 32; model hidden_size: 64"
  passed a_vector_of_another_width_is_refused
}

a_vector_past_the_model_layers_is_refused() {
  direction past "$MODEL" 7 64
  generate 1 "$MODEL" "describe the sea ." --vector "$VECTOR" --strength 2
  said "Error: reading layer 7 of the model: layer 7 is outside the model's 4 layers; observed: layers named: 7; the model has layers 0 to 3"
  passed a_vector_past_the_model_layers_is_refused
}

a_strength_that_is_not_finite_is_refused() {
  direction calm "$MODEL" 1 64
  generate 2 "$MODEL" "describe the sea ." --vector "$VECTOR" --strength inf
  said "Error: scaling the steering vectors: steering strength inf is not finite; observed: --strength inf"
  passed a_strength_that_is_not_finite_is_refused
}

strengths_of_zero_are_refused() {
  direction calm "$MODEL" 1 64
  generate 2 "$MODEL" "describe the sea ." --vector "$VECTOR" --strength 0
  said "Error: scaling the steering vectors: every steering strength is zero, so the vectors would steer nothing; observed: --strength 0"
  passed strengths_of_zero_are_refused
}

a_vector_for_another_model_is_refused
a_vector_of_another_width_is_refused
a_vector_past_the_model_layers_is_refused
a_strength_that_is_not_finite_is_refused
strengths_of_zero_are_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
