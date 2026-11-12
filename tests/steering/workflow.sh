#!/bin/sh
# Real test of how reading a prompt set (`ster.prompts`), exporting states
# (`ster.extract`), comparing artifacts (`ster.vector.compare`) and
# projecting a pair set (`ster.vector.project`) fail, through the built
# binary on the four-layer fixture. Two artifacts are trained at layers 1
# and 2 and one at layer 3; each test names an input that is wrong in one
# way and reads what was refused and what was written. The run's home and
# fixture are tests/runtime/harness.sh's.
#
# Usage: tests/steering/workflow.sh   (STER selects the binary, default target/debug/ster)
AREA=steering-workflow . "$(dirname "$0")/../runtime/harness.sh"
train() {
  run 0 vector train --model "$ROOT/base" --pairs docs/examples/pairs.json --output "$1" --layers "$2"
}

CALM="$ROOT/calm.json"
LATE="$ROOT/late.json"
train "$CALM" 1,2
train "$LATE" 3
jq '.trait_name = "formal"' "$CALM" >"$ROOT/formal.json"
run 0 vector compare "$CALM" "$ROOT/formal.json" --clusters 1
passed "two artifacts sharing layers 1 and 2 are compared"

a_missing_prompt_set_is_refused() {
  run 1 extract --model "$ROOT/no-model" --input "$ROOT/absent.json" --output "$ROOT/x.json"
  said "reading the prompt set $ROOT/absent.json: no file is there: nothing at $ROOT/absent.json is the prompt set named"
  passed a_missing_prompt_set_is_refused
}

an_empty_prompt_set_is_refused() {
  echo '{"prompts": []}' >"$ROOT/empty.json"
  run 1 extract --model "$ROOT/no-model" --input "$ROOT/empty.json" --output "$ROOT/x.json"
  said "prompt set contains no prompts"
  [ ! -e "$ROOT/x.json" ] || fail "a refused extract wrote $ROOT/x.json"
  passed an_empty_prompt_set_is_refused
}

states_that_are_not_finite_are_refused() {
  # Weights of 1e30 are finite in F32 and past F16's largest value.
  jq '.initialization.embedding.scale = 1e30 | .initialization.projection.scale = 1e30' \
    docs/examples/model/fixture.json >"$ROOT/huge.json"
  "$BIN" model fixture create --from "$ROOT/huge.json" --output "$ROOT/huge" >>"$REPORT" 2>&1 \
    || fail "the overflowing fixture could not be written"
  run 1 extract --model "$ROOT/huge" --input docs/examples/prompts.json --output "$ROOT/x.json" \
    --layers 1 --precision f16
  said "reading prompt 0 of 2 at layer 1: the state the model reached holds a value that is not finite"
  said "precision f16; nothing was written to $ROOT/x.json"
  [ ! -e "$ROOT/x.json" ] || fail "an extract of states that are not finite wrote $ROOT/x.json"
  passed states_that_are_not_finite_are_refused
}

one_artifact_to_compare_is_refused() {
  run 2 vector compare "$CALM" --clusters 1
  said "vector compare needs at least two artifacts and got 1"
  passed one_artifact_to_compare_is_refused
}

more_groups_than_artifacts_are_refused() {
  run 2 vector compare "$CALM" "$ROOT/formal.json" --clusters 3
  said "--clusters 3 cannot group 2 artifacts: at most one group per artifact"
  said "fix: give --clusters from 1 to 2"
  passed more_groups_than_artifacts_are_refused
}

artifacts_of_two_models_are_refused() {
  jq '.trait_name = "other" | .model = "elsewhere"' "$CALM" >"$ROOT/other.json"
  run 1 vector compare "$CALM" "$ROOT/other.json" --clusters 1
  said "vector compare compares directions of one model: calm is $ROOT/base at width 64, other is elsewhere at width 64"
  passed artifacts_of_two_models_are_refused
}

a_layer_an_artifact_lacks_is_refused() {
  run 2 vector compare "$CALM" "$ROOT/formal.json" --clusters 1 --layers 1,3
  said "layer 3 is not in calm's artifact, which carries layers 1, 2"
  said "observed: --layers 1,3; layers every artifact carries: 1, 2"
  passed a_layer_an_artifact_lacks_is_refused
}

artifacts_sharing_no_layer_are_refused() {
  run 1 vector compare "$CALM" "$LATE" --clusters 1
  said "the artifacts share no layer to compare at"
  said "layers each artifact carries: calm [1, 2]; calm [3]"
  passed artifacts_sharing_no_layer_are_refused
}

a_direction_without_length_is_refused() {
  jq '.trait_name = "flat" | .vectors[0].values |= map(0)' "$CALM" >"$ROOT/flat.json"
  run 1 vector compare "$CALM" "$ROOT/flat.json" --clusters 1
  said "flat's direction at layer 1 has length 0, so it has no angle to compare"
  passed a_direction_without_length_is_refused
}

reads_that_do_not_vary_are_refused() {
  jq '.pairs = [.pairs[] | .positive = "the sea" | .negative = "the sea"]' docs/examples/pairs.json >"$ROOT/same.json"
  run 1 vector project --model "$ROOT/base" --pairs "$ROOT/same.json" --vector "$CALM" --layer 1 \
    --strength 1 --components 2
  said "the unsteered reads at layer 1 do not vary, so they have no component to project onto"
  said "16 unsteered reads at layer 1, 2 component(s) asked for"
  passed reads_that_do_not_vary_are_refused
}

a_missing_prompt_set_is_refused
an_empty_prompt_set_is_refused
states_that_are_not_finite_are_refused
one_artifact_to_compare_is_refused
more_groups_than_artifacts_are_refused
artifacts_of_two_models_are_refused
a_layer_an_artifact_lacks_is_refused
artifacts_sharing_no_layer_are_refused
a_direction_without_length_is_refused
reads_that_do_not_vary_are_refused
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
