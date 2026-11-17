#!/bin/sh
# Real test of reading decisions under steering artifacts (`ster.decide` and
# `ster.decisions.benchmark` through `ster.runtime.vectors` and
# `ster.runtime.steer`), through the built binary on the lettered fixture
# tests/decide/harness.sh writes, with a direction `ster vector train` fits
# on it.
#
# `ster decide --vector --strength` answers the same request with other
# probabilities and names the artifact and its strength; a --vector without
# its --strength is refused before the checkpoint loads. `ster decisions
# benchmark` lists every labelled question with its gold, prediction and
# option probabilities, and under --vector reads each one steered as well,
# counting the questions the steering turned right and wrong from those
# rows.
#
# Usage: tests/decide/steering.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=steering . "$(dirname "$0")/harness.sh"

VECTOR="$ROOT/calm.json"
run 0 vector train --model "$MODEL" --chat-template off --pairs docs/examples/pairs.json \
  --layers 1,2 --output "$VECTOR"
jq '{examples: [(. + {answers: {weather: "calm", loud: false}}),
  (. + {state: "the sea is wild", answers: {weather: "storm", loud: true}})]}' \
  "$ROOT/ok.json" >"$ROOT/examples.json"

a_steered_decision_reads_other_probabilities() {
  run 0 $DECIDE --request "$ROOT/ok.json"
  base=$(printf '%s' "$out" | jq -c '.answers.weather.probabilities')
  printf '%s' "$out" | jq -e 'has("steering") | not' >/dev/null || fail "an unsteered decision names steering: $out"
  run 0 $DECIDE --request "$ROOT/ok.json" --vector "$VECTOR" --strength 8
  steered=$(printf '%s' "$out" | jq -c '.answers.weather.probabilities')
  [ "$steered" != "$base" ] || fail "steering at strength 8 left the probabilities at $base"
  [ "$(printf '%s' "$out" | jq -c '[.steering[] | [.vector, .strength]]')" = "[[\"$VECTOR\",8.0]]" ] \
    || [ "$(printf '%s' "$out" | jq -c '[.steering[] | [.vector, .strength]]')" = "[[\"$VECTOR\",8]]" ] \
    || fail "the response does not name the artifact and its strength: $out"
  passed a_steered_decision_reads_other_probabilities
}

a_vector_without_its_strength_is_refused() {
  run 2 $DECIDE --request "$ROOT/ok.json" --vector "$VECTOR" --vector "$VECTOR" --strength 8
  said "Error: pairing each --vector with its --strength: every --vector needs its own --strength, and this run gave 2 vector(s) and 1 strength(s); Ster assumes no steering scale"
  passed a_vector_without_its_strength_is_refused
}

a_benchmark_lists_every_question() {
  run 0 decisions benchmark --model "$MODEL" --chat-template off --examples "$ROOT/examples.json" \
    --ece-bins 5 --output "$ROOT/base.json"
  jq -e '(.rows | length) == .questions
    and all(.rows[]; has("gold") and has("base") and (has("steered") | not)
      and (.base.probabilities | length) == (.options | length))
    and (has("steered") | not)' "$ROOT/base.json" >/dev/null \
    || fail "an unsteered benchmark does not list each question once, unsteered: $(cat "$ROOT/base.json")"
  jq -e 'all(.rows[]; .base.correct | type == "boolean")' "$ROOT/base.json" >/dev/null \
    || fail "the rows carry no correctness"
  passed a_benchmark_lists_every_question
}

a_steered_benchmark_counts_what_the_steering_changed() {
  run 0 decisions benchmark --model "$MODEL" --chat-template off --examples "$ROOT/examples.json" \
    --ece-bins 5 --output "$ROOT/steered.json" --vector "$VECTOR" --strength 8
  jq -e --slurpfile base "$ROOT/base.json" --arg vector "$VECTOR" '
    (.rows | length) == .questions
    and all(.rows[]; has("steered"))
    and .metrics == $base[0].metrics
    and [.rows[].base] == [$base[0].rows[].base]
    and .steered.improved == ([.rows[] | select((.base.correct | not) and .steered.correct)] | length)
    and .steered.worsened == ([.rows[] | select(.base.correct and (.steered.correct | not))] | length)
    and .steered.metrics.accuracy == (([.rows[] | select(.steered.correct)] | length) / .questions)
    and .steered.vectors[0].vector == $vector' "$ROOT/steered.json" >/dev/null \
    || fail "a steered benchmark does not report both reads and what changed: $(cat "$ROOT/steered.json")"
  passed a_steered_benchmark_counts_what_the_steering_changed
}

a_steered_decision_reads_other_probabilities
a_vector_without_its_strength_is_refused
a_benchmark_lists_every_question
a_steered_benchmark_counts_what_the_steering_changed

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
