#!/bin/sh
# Real test of how choosing the layers a run reads fails
# (`ster.vector.layers`) and how naming the direction estimator fails
# (`ster.vector.direction` method-unknown, seed-missing, seed-unused), through
# `ster vector train` on the four-layer fixture. Each refusal is a usage
# refusal (exit 2) that names the selection, the segment at fault and the
# model's layers, and writes no artifact; a seeded random direction is
# written like a fitted one and drawn again by its seed. The run's home and
# fixture are tests/runtime/harness.sh's.
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
  said "expected caa, pca, logistic, or random"
  unwritten
  passed an_unknown_method_is_refused
}

a_random_method_without_a_seed_is_refused() {
  train 2 "1" --method random
  said "Error: drawing a random direction: --method random draws its direction from a seed, and none was given; Ster assumes no seed"
  said "fix: give ster vector train --method random a --seed, such as --seed 1, or fit the direction with caa, pca or logistic"
  unwritten
  passed a_random_method_without_a_seed_is_refused
}

a_seed_for_a_fitted_method_is_refused() {
  train 2 "1" --method caa --seed 7
  said "Error: choosing the direction estimator --method caa: --seed 7 is read only by --method random, and caa fits its direction from the pairs with no seed; observed: --method caa --seed 7"
  unwritten
  passed a_seed_for_a_fitted_method_is_refused
}

# A random direction is written as every fitted one is: one unit vector per
# layer, the same for one seed and one layer whichever other layers are
# named, another for another layer or another seed, and the seed kept.
a_seeded_random_direction_is_drawn_again_by_its_seed() {
  train 0 "1,2" --method random --seed 7
  first=$(jq -c '[.vectors[].values]' "$ROOT/out.json")
  [ "$(jq -r .method "$ROOT/out.json")" = random ] || fail "the artifact names another method: $(jq -r .method "$ROOT/out.json")"
  [ "$(jq -r .metadata.seed "$ROOT/out.json")" = 7 ] || fail "the artifact does not keep its seed"
  # Each value is a quotient by the vector's norm rounded once to binary32,
  # and the norm a sum of n squares and a square root, so to first order the
  # sum of squares is 1 within (n + 2) times binary32's machine epsilon 2^-23
  # (precision p = 24, IEEE 754-2019 §3.6; summation bound γ_n in Higham,
  # Accuracy and Stability of Numerical Algorithms, 2nd ed., §3.1).
  jq -e 'all(.vectors[]; (.values | length) as $n
    | ([.values[] | . * .] | add | . - 1 | fabs) <= ($n + 2) * pow(2; -23))' "$ROOT/out.json" >/dev/null \
    || fail "a random direction is not of unit length: $first"
  jq -e '.vectors[0].values != .vectors[1].values' "$ROOT/out.json" >/dev/null \
    || fail "layers 1 and 2 drew one direction"
  train 0 "1,2" --method random --seed 7
  [ "$(jq -c '[.vectors[].values]' "$ROOT/out.json")" = "$first" ] || fail "seed 7 drew another direction the second time"
  train 0 "2" --method random --seed 7
  [ "$(jq -c '.vectors[0].values' "$ROOT/out.json")" = "$(printf '%s' "$first" | jq -c '.[1]')" ] \
    || fail "layer 2's random direction changed when layer 1 was not named"
  train 0 "1,2" --method random --seed 8
  [ "$(jq -c '[.vectors[].values]' "$ROOT/out.json")" != "$first" ] || fail "seeds 7 and 8 drew one direction"
  passed a_seeded_random_direction_is_drawn_again_by_its_seed
}

a_range_bound_that_is_not_a_number_is_refused
a_range_that_holds_no_layer_is_refused
a_layer_that_is_not_a_number_is_refused
a_selection_naming_no_layer_is_refused
a_layer_past_the_model_is_refused
an_unknown_method_is_refused
a_random_method_without_a_seed_is_refused
a_seed_for_a_fitted_method_is_refused
a_seeded_random_direction_is_drawn_again_by_its_seed
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
