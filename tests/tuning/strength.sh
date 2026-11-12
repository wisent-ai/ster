#!/bin/sh
# Real test of how choosing the strength a steering artifact is added at
# fails (`ster.tune.strength`), through `ster vector evaluate --strengths`.
# Every refusal exits 2 before the model, the artifact or the pairs are
# read: the runs name a model and an artifact that do not exist, so a
# refusal that came after either was looked for would say so instead.
#
# Usage: tests/tuning/strength.sh   (STER selects the binary, default target/debug/ster)
AREA=strength
. "$(dirname "$0")/harness.sh"

EVALUATE="--model $ROOT/no-such-model --vector $ROOT/no-such-vector.json --pairs $ROOT/pairs.json"

a_strength_that_is_not_finite_is_refused() {
  # shellcheck disable=SC2086
  run 2 vector evaluate $EVALUATE --strengths 1,inf --batch-size 1 --max-sequence 64
  said "Error: choosing the strength to add the steering artifact at: strength selection needs finite strengths, not inf; observed: --strengths 1,inf"
  unsaid "no-such-model"
  unsaid "no-such-vector"
  passed a_strength_that_is_not_finite_is_refused
}

strengths_without_a_batch_are_refused() {
  # shellcheck disable=SC2086
  run 2 vector evaluate $EVALUATE --strengths 0.5,1 --max-sequence 64
  said "Error: choosing the strength to add the steering artifact at: --strengths needs --batch-size; Ster assumes none; observed: --strengths given without --batch-size; fix: give --batch-size beside --strengths"
  # shellcheck disable=SC2086
  run 2 vector evaluate $EVALUATE --strengths 0.5,1 --batch-size 1
  said "--strengths needs --max-sequence; Ster assumes none"
  passed strengths_without_a_batch_are_refused
}

a_batch_of_zero_is_refused() {
  # shellcheck disable=SC2086
  run 2 vector evaluate $EVALUATE --strengths 0.5,1 --batch-size 0 --max-sequence 64
  said "Error: choosing the strength to add the steering artifact at: strength selection requires batch size of at least one; observed: --batch-size 0"
  # shellcheck disable=SC2086
  run 2 vector evaluate $EVALUATE --strengths 0.5,1 --batch-size 1 --max-sequence 0
  said "strength selection requires sequence limit of at least one; observed: --max-sequence 0"
  passed a_batch_of_zero_is_refused
}

a_strength_that_is_not_finite_is_refused
strengths_without_a_batch_are_refused
a_batch_of_zero_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
