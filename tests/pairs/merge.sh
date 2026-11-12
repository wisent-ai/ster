#!/bin/sh
# Real test of `ster pairs merge` and `ster request pairs/merge`
# (`ster.pairs.merge`), through the built binary, on pair sets this run writes
# under its own directory.
#
# It merges two sets and reads the merged file back: every pair of each, in
# the order named, under the trait given. Then the refusals, each with its
# declared sentence and exit status, and no merged file written: fewer than
# two sets and a blank trait, both refused before any set is read (the sets
# named do not exist), and a set that is missing. The request answers the
# same refusal with its declared cause in json.cause.
#
# Usage: tests/pairs/merge.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-merge/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
MERGED="$ROOT/merged.json"
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
nothing_merged() { [ ! -e "$MERGED" ] || fail "a refused merge wrote $MERGED"; }

merging_two_sets_keeps_every_pair_in_order() {
  run 0 pairs add --pairs "$ROOT/calm.json" --positive "Still water." --negative "Storm." --trait calm
  run 0 pairs add --pairs "$ROOT/kind.json" --positive "Thank you." --negative "Whatever." --trait kindness
  run 0 pairs add --pairs "$ROOT/kind.json" --positive "Take care." --negative "Leave."
  run 0 pairs merge --pairs "$ROOT/calm.json" --pairs "$ROOT/kind.json" --trait goodness --output "$ROOT/all.json"
  check "the merged set holds every pair" "$(jq '.pairs | length' "$ROOT/all.json")" "3"
  check "the first set's pair comes first" "$(jq -r '.pairs[0].positive' "$ROOT/all.json")" "Still water."
  check "the second set's last pair comes last" "$(jq -r '.pairs[2].positive' "$ROOT/all.json")" "Take care."
  check "the trait is the one given" "$(jq -r .trait_name "$ROOT/all.json")" "goodness"
  check "the report counts each source" "$(echo "$out" | jq -c '[.report.sources[].pairs]')" "[1,2]"
  passed merging_two_sets_keeps_every_pair_in_order
}

fewer_than_two_sets_are_refused_before_any_is_read() {
  run 2 pairs merge --pairs "$ROOT/absent.json" --trait goodness --output "$MERGED"
  said "Error: merging pair sets: pairs merge needs at least two pair sets and got 1; observed: pair sets named: $ROOT/absent.json; fix: name every set to merge with its own --pairs, two or more: one set is already itself"
  nothing_merged
  request 1 pairs/merge "{\"sources\": [], \"traitName\": \"goodness\", \"output\": \"$MERGED\"}"
  check "the request names the cause" "$(echo "$result" | jq -r .json.cause.id)" "too-few-sources"
  check "the request carries the CLI's status" "$(echo "$result" | jq -r .json.cause.exit)" "2"
  check "the request shows no set named" "$(echo "$result" | jq -r .json.error)" \
    "merging pair sets: pairs merge needs at least two pair sets and got 0; observed: pair sets named: none; fix: name every set to merge with its own --pairs, two or more: one set is already itself"
  nothing_merged
  passed fewer_than_two_sets_are_refused_before_any_is_read
}

a_blank_trait_is_refused_before_any_set_is_read() {
  run 2 pairs merge --pairs "$ROOT/absent.json" --pairs "$ROOT/gone.json" --trait " " --output "$MERGED"
  said "Error: merging pair sets: pairs merge needs the trait name the merged set is fitted for; observed: --trait \" \"; fix: give --trait the name of the trait every merged set expresses"
  nothing_merged
  passed a_blank_trait_is_refused_before_any_set_is_read
}

a_missing_set_is_refused_and_nothing_written() {
  run 1 pairs merge --pairs "$ROOT/calm.json" --pairs "$ROOT/absent.json" --trait goodness --output "$MERGED"
  said "Error: reading pair set $ROOT/absent.json: no file is there: "
  nothing_merged
  passed a_missing_set_is_refused_and_nothing_written
}

merging_two_sets_keeps_every_pair_in_order
fewer_than_two_sets_are_refused_before_any_is_read
a_blank_trait_is_refused_before_any_set_is_read
a_missing_set_is_refused_and_nothing_written

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
