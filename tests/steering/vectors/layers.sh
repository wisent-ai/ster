#!/bin/sh
# Real test of how choosing the layers a run reads fails
# (`ster.vector.layers`) and how naming the direction estimator fails
# (`ster.vector.direction` method-unknown), through `ster vector train` on
# the four-layer fixture. Each refusal is a usage refusal (exit 2) that
# names the selection, the segment at fault and the model's layers, and
# writes no artifact. The run's home and fixture are tests/runtime/harness.sh's.
#
# Usage: tests/steering/vectors/layers.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=steering-layers . "$(dirname "$0")/../../runtime/harness.sh"
# train EXIT SELECTION [FLAG...]: vector train at --layers SELECTION.
train() {
  expected=$1
  selection=$2
  shift 2
  rm -f "$ROOT/out.json"
  run "$expected" vector train --model "$ROOT/base" --pairs docs/examples/pairs.json \
    --output "$ROOT/out.json" --layers "$selection" "$@"
}
unwritten() {
  [ ! -e "$ROOT/out.json" ] || fail "a refused run wrote $ROOT/out.json"
}

train 0 "1,2..4"
[ "$(echo "$out" | jq -c '[.artifact.layers[].layer]')" = "[1,2,3]" ] \
  || fail "--layers 1,2..4 trained layers other than 1, 2 and 3: $out"
passed "a comma list and a half-open range select layers 1, 2 and 3"

a_range_bound_that_is_not_a_number_is_refused() {
  train 2 "1..x"
  said "choosing the layers --layers 1..x names: the range 1..x has a bound, \"x\", that is not a whole number"
  said "the model has 4 layers, 0 to 3"
  unwritten
  passed a_range_bound_that_is_not_a_number_is_refused
}

a_range_that_holds_no_layer_is_refused() {
  train 2 "3..1"
  said "layer range 3..1 must have start < end"
  said "fix: give the smaller layer first, such as 1..4, or name one layer as 3"
  unwritten
  passed a_range_that_holds_no_layer_is_refused
}

a_layer_that_is_not_a_number_is_refused() {
  train 2 "1,two"
  said "\"two\" is not a layer number"
  said "observed: --layers 1,two; segment \"two\": invalid digit found in string; the model has 4 layers, 0 to 3"
  unwritten
  passed a_layer_that_is_not_a_number_is_refused
}

a_selection_naming_no_layer_is_refused() {
  train 2 " , "
  said "no layers selected"
  unwritten
  passed a_selection_naming_no_layer_is_refused
}

a_layer_past_the_model_is_refused() {
  train 2 "1,9,2..6"
  said "layer 9 is outside the model's 4 layers, counted from the first"
  said "every layer named past the model: 9, 4..6"
  said "fix: name layers from 0 to 3, such as 0..4, or all"
  unwritten
  passed a_layer_past_the_model_is_refused
}

an_unknown_method_is_refused() {
  train 2 "1" --method lda
  said "unknown training method \"lda\""
  said "expected caa, pca, or logistic"
  unwritten
  passed an_unknown_method_is_refused
}

a_range_bound_that_is_not_a_number_is_refused
a_range_that_holds_no_layer_is_refused
a_layer_that_is_not_a_number_is_refused
a_selection_naming_no_layer_is_refused
a_layer_past_the_model_is_refused
an_unknown_method_is_refused
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
