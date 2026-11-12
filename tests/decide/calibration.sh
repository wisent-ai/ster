#!/bin/sh
# Real test of how reading and writing a calibration fails
# (`ster.decide.calibration`, `ster.decide.calibrate`) through the built
# binary, on the lettered fixture tests/decide/harness.sh writes.
#
# `ster calibrate` is checked to fit and write a calibration on two labelled
# examples first, and `ster decide --calibration` to read it back. Each test
# then names a missing, unreadable or broken calibration, one fitted for
# another model, or an output that cannot be written, and checks the
# declared sentence and exit status.
#
# Usage: tests/decide/calibration.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=calibration . "$(dirname "$0")/harness.sh"

jq '{examples: [(. + {answers: {weather: "calm"}}),
  (.state = "a storm and loud waves" | . + {answers: {weather: "storm", loud: true}})]}' \
  "$ROOT/ok.json" >"$ROOT/examples.json"
CALIBRATE="calibrate --model $MODEL --chat-template off --examples $ROOT/examples.json --ece-bins 10"
run 0 $CALIBRATE --output "$ROOT/calibration.json"
jq -e '.schema == "ster-calibration/1" and .temperature > 0' "$ROOT/calibration.json" >/dev/null \
  || fail "calibrate wrote no calibration: $(cat "$ROOT/calibration.json")"
run 0 $DECIDE --request "$ROOT/ok.json" --calibration "$ROOT/calibration.json"
passed "a calibration is fitted, written and read back"

# calibration NAME JQ: a copy of the written calibration through JQ, its path
# in $CALIBRATION.
calibration() {
  CALIBRATION="$ROOT/$1.json"
  jq "$2" "$ROOT/calibration.json" >"$CALIBRATION"
}

a_missing_calibration_is_refused() {
  run 1 $DECIDE --request "$ROOT/ok.json" --calibration "$ROOT/absent.json"
  said "Error: reading calibration $ROOT/absent.json: no file is there: nothing at $ROOT/absent.json is the calibration named; observed: path: $ROOT/absent.json (absent); its directory $ROOT; the system answered: "
  passed a_missing_calibration_is_refused
}

a_directory_named_as_the_calibration_is_refused() {
  run 1 $DECIDE --request "$ROOT/ok.json" --calibration "$ROOT/spec"
  said "Error: reading the calibration $ROOT/spec: $ROOT/spec is a directory, and the calibration is one file; observed: path: $ROOT/spec (a directory, "
  passed a_directory_named_as_the_calibration_is_refused
}

a_calibration_that_is_not_json_is_refused() {
  printf 'temperature = 1' >"$ROOT/broken.json"
  run 1 $DECIDE --request "$ROOT/ok.json" --calibration "$ROOT/broken.json"
  said "Error: reading the calibration $ROOT/broken.json: it is not JSON, and the calibration is the JSON \`ster calibrate --output <file>\` writes: it is another kind of file, or one broken by hand; observed: path: $ROOT/broken.json (a file, 15 bytes, "
  passed a_calibration_that_is_not_json_is_refused
}

a_calibration_of_another_schema_is_refused() {
  calibration old '.schema = "ster-calibration/0"'
  run 1 $DECIDE --request "$ROOT/ok.json" --calibration "$CALIBRATION"
  said "Error: reading calibration $CALIBRATION: it declares schema \"ster-calibration/0\", and this Ster reads ster-calibration/1"
  passed a_calibration_of_another_schema_is_refused
}

a_calibration_with_a_temperature_of_zero_is_refused() {
  calibration cold '.temperature = 0'
  run 1 $DECIDE --request "$ROOT/ok.json" --calibration "$CALIBRATION"
  said "Error: reading calibration $CALIBRATION: its temperature is 0, and the logits are divided by it"
  passed a_calibration_with_a_temperature_of_zero_is_refused
}

a_calibration_for_another_model_is_refused() {
  calibration other '.model = "another/model"'
  run 1 $DECIDE --request "$ROOT/ok.json" --calibration "$CALIBRATION"
  said "Error: applying calibration $CALIBRATION: it was fitted for model another/model"
  said "observed: calibration model: another/model; loaded model: $MODEL"
  passed a_calibration_for_another_model_is_refused
}

a_calibration_under_a_file_is_refused() {
  run 1 $CALIBRATE --output "$ROOT/ok.json/calibration.json"
  said "Error: creating the directory of the calibration $ROOT/ok.json/calibration.json: the path runs through a file: $ROOT/ok.json is not a directory, so nothing can be created below it; observed: path: $ROOT/ok.json/calibration.json; $ROOT/ok.json: a file, "
  said "; nothing was written; the system answered: "
  passed a_calibration_under_a_file_is_refused
}

a_calibration_in_a_read_only_directory_is_refused() {
  mkdir -p "$ROOT/closed"
  chmod 555 "$ROOT/closed"
  run 1 $CALIBRATE --output "$ROOT/closed/calibration.json"
  chmod 755 "$ROOT/closed"
  said "Error: writing the calibration $ROOT/closed/calibration.json: this account may not change $ROOT/closed or $ROOT/closed/calibration.json: their permission bits or owner keep this account out; observed: path: $ROOT/closed/calibration.json (absent); directory $ROOT/closed (a directory, "
  [ ! -e "$ROOT/closed/calibration.json" ] || fail "a calibration was written into the read-only directory"
  passed a_calibration_in_a_read_only_directory_is_refused
}

zero_confidence_bins_are_refused() {
  run 2 calibrate --model "$ROOT/absent-model" --examples "$ROOT/absent.json" \
    --output "$ROOT/never.json" --ece-bins 0
  said "Error: checking the calibration settings: the expected calibration error is measured over equal-width confidence bins, and zero bins measure nothing; observed: --ece-bins 0"
  [ ! -e "$ROOT/never.json" ] || fail "a calibration was written with zero bins"
  passed zero_confidence_bins_are_refused
}

a_missing_calibration_is_refused
a_directory_named_as_the_calibration_is_refused
a_calibration_that_is_not_json_is_refused
a_calibration_of_another_schema_is_refused
a_calibration_with_a_temperature_of_zero_is_refused
a_calibration_for_another_model_is_refused
a_calibration_under_a_file_is_refused
a_calibration_in_a_read_only_directory_is_refused
zero_confidence_bins_are_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
