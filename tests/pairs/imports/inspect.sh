#!/bin/sh
# Real test of `ster pairs inspect` and `ster request pairs/inspect`
# (`ster.pairs.inspect` and the near-duplicate settings `ster.pairs.dedupe`),
# through the built binary, on a pair set this run writes under its own
# directory.
#
# It inspects a set holding a repeated pair and an unbalanced one and reads
# the counts. Then the refusals, each with its declared sentence and exit
# status 2: a band count that does not divide 64, a threshold wider than the
# 64-bit fingerprint, and a balance ratio below 1 or not a number. Each names
# a set that does not exist, which shows the settings are checked before any
# set is read; the request refuses the same settings before it runs.
#
# Usage: tests/pairs/imports/inspect.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-inspect/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
SET="$ROOT/set.json"
ABSENT="$ROOT/absent.json"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
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
request() {
  expected=$1
  set +e
  out=$(printf '%s' "$3" | "$BIN" request "$2" 2>"$ROOT/stderr")
  status=$?
  set -e
  printf '$ ster request %s <<< %s\nexit: %s\nstdout: %s\n\n' "$2" "$3" "$status" "$out" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster request $2 exited $status, expected $expected: $out"
  result=$(printf '%s\n' "$out" | jq -c 'select(.type == "result")')
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
inspect() { run "$1" pairs inspect --pairs "$2" --dedupe-bits "$3" --dedupe-bands "$4" --unbalanced-ratio "$5"; }

inspecting_counts_duplicates_and_unbalanced_pairs() {
  run 0 pairs add --pairs "$SET" --positive "I am glad to help with that." --negative "No." --trait helpfulness
  run 0 pairs add --pairs "$SET" --positive "Here is how it works." --negative "Look it up."
  run 0 pairs add --pairs "$SET" --positive "Here is how it works." --negative "Look it up."
  inspect 0 "$SET" 3 4 3
  check "the set holds three pairs" "$(echo "$out" | jq .pair_count)" "3"
  check "the repeated pair is a duplicate" "$(echo "$out" | jq .duplicate_count)" "1"
  check "the long-against-short pair is unbalanced" "$(echo "$out" | jq .unbalanced_count)" "1"
  passed inspecting_counts_duplicates_and_unbalanced_pairs
}

a_band_count_that_does_not_divide_64_is_refused() {
  inspect 2 "$ABSENT" 3 5 3
  said "Error: checking the near-duplicate settings: dedupe band count 5 must divide the 64-bit simhash evenly, such as 4, 8, 16, or 32; observed: --dedupe-bands 5; fix: give --dedupe-bands a divisor of 64: 1, 2, 4, 8, 16, 32 or 64"
  inspect 2 "$ABSENT" 3 0 3
  said "dedupe band count 0 must divide the 64-bit simhash evenly"
  request 2 pairs/inspect "{\"pairs\": \"$SET\", \"dedupeBits\": 3, \"dedupeBands\": 5, \"unbalancedRatio\": 3}"
  check "the request refuses the same settings" "$(echo "$result" | jq -r .json.error)" \
    "checking the near-duplicate settings: dedupe band count 5 must divide the 64-bit simhash evenly, such as 4, 8, 16, or 32; observed: --dedupe-bands 5; fix: give --dedupe-bands a divisor of 64: 1, 2, 4, 8, 16, 32 or 64"
  passed a_band_count_that_does_not_divide_64_is_refused
}

a_threshold_wider_than_the_fingerprint_is_refused() {
  inspect 2 "$ABSENT" 65 4 3
  said "Error: checking the near-duplicate settings: dedupe threshold 65 exceeds the 64-bit simhash width; observed: --dedupe-bits 65; fix: give --dedupe-bits a Hamming distance from 0 to 64"
  passed a_threshold_wider_than_the_fingerprint_is_refused
}

a_ratio_below_one_is_refused() {
  inspect 2 "$ABSENT" 3 4 0.5
  said "Error: checking the length balance setting: a pair's longer side over its shorter side is 1 or more, so a ratio below 1, or one that is not a number, counts every pair or none as unbalanced; observed: --unbalanced-ratio 0.5; fix: give --unbalanced-ratio a number of 1 or more"
  inspect 2 "$ABSENT" 3 4 NaN
  said "observed: --unbalanced-ratio NaN;"
  request 2 pairs/inspect "{\"pairs\": \"$SET\", \"dedupeBits\": 3, \"dedupeBands\": 4, \"unbalancedRatio\": 0.5}"
  check "the request refuses the same ratio" "$(echo "$result" | jq -r .json.error | cut -d';' -f2)" \
    " observed: --unbalanced-ratio 0.5"
  passed a_ratio_below_one_is_refused
}

inspecting_counts_duplicates_and_unbalanced_pairs
a_band_count_that_does_not_divide_64_is_refused
a_threshold_wider_than_the_fingerprint_is_refused
a_ratio_below_one_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
