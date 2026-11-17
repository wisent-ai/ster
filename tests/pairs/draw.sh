#!/bin/sh
# Real test of `ster pairs split` and `ster pairs sample` (`ster.pairs.split`,
# `ster.pairs.sample`, `ster.pairs.output`), through the built binary, on a
# pair set this run writes under its own directory.
#
# A split parts the set into two sides that hold every pair once, each in
# the set's order, and the same seed parts it the same way again while
# another seed parts it another way; a sample draws the asked number of the
# set's pairs, again the same for the same seed. Then the refusals, each with
# its declared sentence and exit status and nothing written: an output where
# a file is (before the set is read), one path for both sides, a count the
# set cannot give, and a held-out fraction outside zero and one or one that
# empties a side (`ster.vector.holdout`, as `vector optimize` refuses it).
#
# Usage: tests/pairs/draw.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-draw/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
SET="$ROOT/calm.json"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1  # https://pubs.opengroup.org/onlinepubs/9799919799/functions/exit.html
}
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ ster %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster $* exited $status, expected $expected: $err"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
passed() { echo "ok: $1" >>"$REPORT"; }
# positives FILE: the positive sides of FILE's pairs, one JSON list.
positives() { jq -c '[.pairs[].positive]' "$1"; }
absent() { [ ! -e "$1" ] || fail "a refused run wrote $1"; }

for word in dawn noon dusk night tide; do
  run 0 pairs add --pairs "$SET" --positive "Calm at $word." --negative "Storm at $word." --trait calm
done

splitting_is_seeded_and_keeps_every_pair_once() {
  run 0 pairs split --pairs "$SET" --holdout 0.4 --seed 3 --train "$ROOT/train.json" --test "$ROOT/test.json"
  check "the held-out side holds the rounded fraction" "$(jq '.pairs | length' "$ROOT/test.json")" "2"
  check "the training side holds the rest" "$(jq '.pairs | length' "$ROOT/train.json")" "3"
  check "the report counts both sides" "$(echo "$out" | jq -c '[.train.pairs, .test.pairs, .pair_count, .seed]')" "[3,2,5,3]"
  check "both sides keep the trait" "$(jq -r .trait_name "$ROOT/test.json")" "calm"
  # Every pair lands on exactly one side, and each side lists its pairs in
  # the set's own order.
  check "the two sides hold every pair once" \
    "$(jq -sc '[.[].pairs[].positive] | sort' "$ROOT/train.json" "$ROOT/test.json")" \
    "$(jq -c '[.pairs[].positive] | sort' "$SET")"
  for side in train test; do
    jq -e --slurpfile set "$SET" '[.pairs[].positive] as $kept
      | ([$set[0].pairs[].positive | select(IN($kept[]))]) == $kept' "$ROOT/$side.json" >/dev/null \
      || fail "the $side side does not keep the set's order"
  done
  run 0 pairs split --pairs "$SET" --holdout 0.4 --seed 3 --train "$ROOT/train-again.json" --test "$ROOT/test-again.json"
  check "the same seed holds out the same pairs" "$(positives "$ROOT/test-again.json")" "$(positives "$ROOT/test.json")"
  # Two of five pairs can be held out ten ways; nine other seeds all drawing
  # the one seed 3 drew is a chance of one in a billion, and means the seed
  # is not read.
  differs=no
  for seed in 4 5 6 7 8 9 10 11 12; do
    rm -f "$ROOT/train-other.json" "$ROOT/test-other.json"
    run 0 pairs split --pairs "$SET" --holdout 0.4 --seed "$seed" --train "$ROOT/train-other.json" --test "$ROOT/test-other.json"
    [ "$(positives "$ROOT/test-other.json")" = "$(positives "$ROOT/test.json")" ] || differs=yes
  done
  check "another seed holds out other pairs" "$differs" "yes"
  passed splitting_is_seeded_and_keeps_every_pair_once
}

sampling_is_seeded_and_draws_pairs_of_the_set() {
  run 0 pairs sample --pairs "$SET" --count 2 --seed 3 --output "$ROOT/sample.json"
  check "the sample holds the count asked" "$(jq '.pairs | length' "$ROOT/sample.json")" "2"
  check "the report names the sample" "$(echo "$out" | jq -c '[.sample.pairs, .pair_count, .seed]')" "[2,5,3]"
  jq -e --slurpfile set "$SET" 'all(.pairs[]; . as $pair | any($set[0].pairs[]; . == $pair))' "$ROOT/sample.json" >/dev/null \
    || fail "the sample holds a pair the set does not"
  run 0 pairs sample --pairs "$SET" --count 2 --seed 3 --output "$ROOT/sample-again.json"
  check "the same seed draws the same pairs" "$(positives "$ROOT/sample-again.json")" "$(positives "$ROOT/sample.json")"
  passed sampling_is_seeded_and_draws_pairs_of_the_set
}

an_output_that_exists_is_refused() {
  printf 'kept\n' >"$ROOT/taken.json"
  run 2 pairs split --pairs "$ROOT/absent.json" --holdout 0.4 --seed 3 --train "$ROOT/taken.json" --test "$ROOT/fresh.json"
  said "Error: choosing --train $ROOT/taken.json: $ROOT/taken.json already holds a file, and a drawn pair set is written only where none is; nothing was written; observed: --train $ROOT/taken.json ("
  check "the file there is kept" "$(cat "$ROOT/taken.json")" "kept"
  absent "$ROOT/fresh.json"
  run 2 pairs sample --pairs "$SET" --count 2 --seed 3 --output "$ROOT/taken.json"
  said "Error: choosing --output $ROOT/taken.json: $ROOT/taken.json already holds a file"
  check "the file there is kept" "$(cat "$ROOT/taken.json")" "kept"
  passed an_output_that_exists_is_refused
}

# A path the system will not say anything about is not taken for a free one.
an_output_this_account_may_not_look_at_is_refused() {
  mkdir -p "$ROOT/shut"
  chmod 000 "$ROOT/shut"  # https://pubs.opengroup.org/onlinepubs/9799919799/utilities/chmod.html
  run 1 pairs sample --pairs "$SET" --count 2 --seed 3 --output "$ROOT/shut/sample.json"
  chmod 755 "$ROOT/shut"  # https://pubs.opengroup.org/onlinepubs/9799919799/utilities/chmod.html
  said "Error: checking the --output pair set $ROOT/shut/sample.json: this account may not change $ROOT/shut or $ROOT/shut/sample.json"
  said "nothing was written; the system answered: "
  absent "$ROOT/shut/sample.json"
  passed an_output_this_account_may_not_look_at_is_refused
}

one_path_for_both_sides_is_refused() {
  run 2 pairs split --pairs "$SET" --holdout 0.4 --seed 3 --train "$ROOT/both.json" --test "$ROOT/both.json"
  said "Error: choosing where pairs split writes: --train and --test both name $ROOT/both.json, and the two sides are two files; observed: --train $ROOT/both.json --test $ROOT/both.json; fix: give --train and --test two different paths"
  absent "$ROOT/both.json"
  # The same file written two ways, and through a link to its directory, is
  # still one file.
  run 2 pairs split --pairs "$SET" --holdout 0.4 --seed 3 --train "$ROOT/both.json" --test "$ROOT/./new/../both.json"
  said "--train and --test both name $ROOT/both.json, and the two sides are two files"
  ln -s "$ROOT" "$ROOT/linked"
  run 2 pairs split --pairs "$SET" --holdout 0.4 --seed 3 --train "$ROOT/both.json" --test "$ROOT/linked/both.json"
  said "--train and --test both name $ROOT/both.json, and the two sides are two files"
  absent "$ROOT/both.json"
  passed one_path_for_both_sides_is_refused
}

a_count_past_the_set_is_refused() {
  run 2 pairs sample --pairs "$SET" --count 9 --seed 3 --output "$ROOT/too-many.json"
  said "Error: drawing 9 pairs from $SET: --count 9 is not between 1 and the 5 pair(s) the set holds, and a sample repeats no pair; observed: --count 9; $SET holds 5 pair(s); fix: give --count a number from 1 to 5"
  absent "$ROOT/too-many.json"
  run 2 pairs sample --pairs "$SET" --count 0 --seed 3 --output "$ROOT/too-many.json"
  said "--count 0 is not between 1 and the 5 pair(s) the set holds"
  absent "$ROOT/too-many.json"
  passed a_count_past_the_set_is_refused
}

a_fraction_that_parts_no_pair_is_refused() {
  run 2 pairs split --pairs "$SET" --holdout 1.5 --seed 3 --train "$ROOT/train-none.json" --test "$ROOT/test-none.json"
  said "holding out --holdout 1.5 of the 5 pairs of the \"calm\" pair set: the held-out fraction must be above zero and below one"
  run 2 pairs split --pairs "$SET" --holdout 0.05 --seed 3 --train "$ROOT/train-none.json" --test "$ROOT/test-none.json"
  said "a held-out fraction of 0.05 over 5 contrastive pairs leaves 5 pairs to fit and 0 to rank on; both need at least one"
  absent "$ROOT/train-none.json"
  absent "$ROOT/test-none.json"
  passed a_fraction_that_parts_no_pair_is_refused
}

splitting_is_seeded_and_keeps_every_pair_once
sampling_is_seeded_and_draws_pairs_of_the_set
an_output_that_exists_is_refused
an_output_this_account_may_not_look_at_is_refused
one_path_for_both_sides_is_refused
a_count_past_the_set_is_refused
a_fraction_that_parts_no_pair_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
