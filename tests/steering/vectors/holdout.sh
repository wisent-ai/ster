#!/bin/sh
# Real test of how holding pairs out fails (`ster.vector.holdout`) and how
# naming the sizes of a curve fails (`ster.vector.curve`), through `ster
# vector optimize` and `ster vector curve` on the eight example pairs. Each
# names a checkpoint that does not exist and is refused by its own cause
# all the same, which proves it was checked before any weight was read; and
# nothing is written. The run's home is tests/runtime/harness.sh's.
#
# Usage: tests/steering/vectors/holdout.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=steering-holdout . "$(dirname "$0")/../../runtime/harness.sh"
# A model directory that does not exist: a refusal that names the holdout
# rather than the missing checkpoint was raised before any load.
ABSENT_MODEL="$ROOT/no-model"

a_fraction_outside_zero_and_one_is_refused() {
  run 2 vector optimize --model "$ABSENT_MODEL" --pairs docs/examples/pairs.json \
    --output "$ROOT/out.json" --holdout 1.5
  said "holding out --holdout 1.5 of the 8 pairs of the \"calm\" pair set: the held-out fraction must be above zero and below one"
  said "fix: give --holdout a fraction above 0 and below 1 that leaves a pair on each side, such as 0.5"
  [ ! -e "$ROOT/out.json" ] || fail "a refused optimize wrote $ROOT/out.json"
  passed a_fraction_outside_zero_and_one_is_refused
}

a_fraction_that_empties_a_side_is_refused() {
  run 2 vector optimize --model "$ABSENT_MODEL" --pairs docs/examples/pairs.json \
    --output "$ROOT/out.json" --holdout 0.01
  said "a held-out fraction of 0.01 over 8 contrastive pairs leaves 8 pairs to fit and 0 to rank on; both need at least one"
  passed a_fraction_that_empties_a_side_is_refused
}

a_size_past_the_fitting_part_is_refused() {
  run 2 vector curve --model "$ABSENT_MODEL" --pairs docs/examples/pairs.json --method caa \
    --holdout 0.5 --sizes 1,5,9
  said "a size of 5 pairs does not fit in the 4 pairs left to fit on after holding out 4"
  said "sizes past the fitting part: 5, 9"
  passed a_size_past_the_fitting_part_is_refused
}

a_fraction_outside_zero_and_one_is_refused
a_fraction_that_empties_a_side_is_refused
a_size_past_the_fitting_part_is_refused
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
