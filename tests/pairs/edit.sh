#!/bin/sh
# Real test of `ster pairs edit` through the built binary, on a pair-set file
# it creates under this run's directory.
#
# It writes a set of two pairs with `pairs add`, edits one side of the second
# pair, then both sides of the first together with the set's trait, reads the
# file back after each edit, and checks the refusals: an edit that names no
# change and an index past the end. Every command, its exit status and output
# go to the run's report.txt.
#
# Usage: tests/pairs/edit.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs/$RUN"
REPORT="$ROOT/report.txt"
SET="$ROOT/set.json"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" pairs "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ ster pairs %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  if [ "$status" -ne "$expected" ]; then
    echo "FAIL: pairs $* exited $status, expected $expected: $err" | tee -a "$REPORT" >&2
    exit 1
  fi
}
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL: $1: got '$2', expected '$3'" | tee -a "$REPORT" >&2
    exit 1
  fi
  echo "ok: $1 = $2" >>"$REPORT"
}
refused() {
  case "$err" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) echo "FAIL: expected a refusal containing '$1', got: $err" | tee -a "$REPORT" >&2; exit 1 ;;
  esac
}

run 0 add --pairs "$SET" --positive "I will help." --negative "I refuse." --trait helpfulness
run 0 add --pairs "$SET" --positive "Here is how." --negative "Figure it out."

run 1 edit --pairs "$SET" --index 1
refused "pairs edit changes nothing without --positive, --negative or --trait"
run 1 edit --pairs "$SET" --index 2 --positive "Too far."
refused "pair index 2 is outside the set: $SET holds 2 pair(s), indexed from 0"

run 0 edit --pairs "$SET" --index 1 --negative "Work it out yourself."
check "the answer names the edited pair" "$(echo "$out" | jq -c .edited)" \
  '{"index":1,"negative":"Work it out yourself.","positive":"Here is how."}'
check "the second pair's negative is stored" "$(jq -r '.pairs[1].negative' "$SET")" "Work it out yourself."
check "its positive is kept" "$(jq -r '.pairs[1].positive' "$SET")" "Here is how."

run 0 edit --pairs "$SET" --index 0 --positive "Glad to help." --negative "No." --trait warmth
check "the first pair's positive is stored" "$(jq -r '.pairs[0].positive' "$SET")" "Glad to help."
check "the first pair's negative is stored" "$(jq -r '.pairs[0].negative' "$SET")" "No."
check "the trait is renamed" "$(jq -r .trait_name "$SET")" "warmth"
check "the set still holds two pairs" "$(jq '.pairs | length' "$SET")" "2"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
