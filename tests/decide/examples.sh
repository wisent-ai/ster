#!/bin/sh
# Real test of how reading labelled examples fails (`ster.decide.examples`)
# through the built binary: `ster calibrate` on the lettered fixture
# tests/decide/harness.sh writes, given a missing, unreadable or broken
# example set, or one whose labels break a rule. Every refusal is made before
# the checkpoint loads, and no calibration is written.
#
# Usage: tests/decide/examples.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=examples . "$(dirname "$0")/harness.sh"

jq '{examples: [(. + {answers: {weather: "calm"}})]}' "$ROOT/ok.json" >"$ROOT/examples.json"
# examples NAME JQ: a copy of examples.json through JQ, its path in $EXAMPLES.
examples() {
  EXAMPLES="$ROOT/$1.json"
  jq "$2" "$ROOT/examples.json" >"$EXAMPLES"
}
calibrate() {
  run 1 calibrate --model "$MODEL" --chat-template off --examples "$1" --ece-bins 10 \
    --output "$ROOT/never.json"
  [ ! -e "$ROOT/never.json" ] || fail "a calibration was written from $1"
}

a_missing_example_set_is_refused() {
  calibrate "$ROOT/absent.json"
  said "Error: reading labelled examples $ROOT/absent.json: no file is there: nothing at $ROOT/absent.json is the labelled example set named; observed: path: $ROOT/absent.json (absent); its directory $ROOT; the system answered: "
  passed a_missing_example_set_is_refused
}

a_directory_named_as_the_examples_is_refused() {
  calibrate "$ROOT/spec"
  said "Error: reading the labelled example set $ROOT/spec: $ROOT/spec is a directory, and the labelled example set is one file; observed: path: $ROOT/spec (a directory, "
  passed a_directory_named_as_the_examples_is_refused
}

an_example_set_that_is_not_json_is_refused() {
  printf '[' >"$ROOT/broken.json"
  calibrate "$ROOT/broken.json"
  said "Error: reading the labelled example set $ROOT/broken.json: it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway); observed: path: $ROOT/broken.json (a file, 1 bytes, "
  said "; fix: write the labelled example set again; it is {\"examples\": "
  passed an_example_set_that_is_not_json_is_refused
}

an_empty_example_set_is_refused() {
  examples none '.examples = []'
  calibrate "$EXAMPLES"
  said "Error: checking the labelled examples: the set holds no example, and a temperature is fitted on labelled examples; observed: $EXAMPLES (a file"
  said "\"examples\": []"
  passed an_empty_example_set_is_refused
}

a_label_for_a_question_not_asked_is_refused() {
  examples unasked '.examples[0].answers = {sky: "clear"}'
  calibrate "$EXAMPLES"
  said "Error: checking labelled example 0: it labels question 'sky', which its request does not ask; observed: questions asked: loud, weather"
  passed a_label_for_a_question_not_asked_is_refused
}

a_label_that_is_not_an_option_is_refused() {
  examples stranger '.examples[0].answers = {weather: "fog"}'
  calibrate "$EXAMPLES"
  said "Error: checking labelled example 0: it answers question 'weather' with \"fog\", which is not one of its options; observed: answer: \"fog\"; options: \"calm\", \"storm\""
  passed a_label_that_is_not_an_option_is_refused
}

an_example_set_without_labels_is_refused() {
  examples unlabelled '.examples[0].answers = {}'
  calibrate "$EXAMPLES"
  said "Error: checking the labelled examples: none of its 1 examples labels a question, so there is nothing to fit on"
  passed an_example_set_without_labels_is_refused
}

a_missing_example_set_is_refused
a_directory_named_as_the_examples_is_refused
an_example_set_that_is_not_json_is_refused
an_empty_example_set_is_refused
a_label_for_a_question_not_asked_is_refused
a_label_that_is_not_an_option_is_refused
an_example_set_without_labels_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
