#!/usr/bin/env bash
# Real test of `ster pairs merge` and `ster request pairs/merge` through the
# built binary, on pair sets it writes under this run's directory.
#
# It merges the checked-in calm set with a second set of the same pairs
# sides swapped, and checks the merged file: every pair of both in order,
# under the trait name given, with one source row per set; then fits a
# direction on it, as wisent's train-unified-goodness did across benchmarks.
# Then the refusals: one set, and an empty trait name. The request path must
# write the same pairs. Every command, whether it was accepted, and its answer
# go to the run's report.txt.
#
# Usage: STER=target/debug/ster tests/pairs/merge.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs/merge-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
run() {
  expected=$1
  shift
  if [ "$expected" = accepted ]; then
    if "$BIN" "$@" >"$ROOT/output" </dev/null; then outcome=accepted; else outcome=refused; fi
  else
    if "$BIN" "$@" &>"$ROOT/output" </dev/null; then outcome=accepted; else outcome=refused; fi
  fi
  out=$(cat "$ROOT/output")
  printf '$ ster %s\noutcome: %s\noutput: %s\n\n' "$*" "$outcome" "$out" >>"$REPORT"
  [ "$outcome" = "$expected" ] || fail "$* was $outcome, expected $expected: $out"
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
refused() {
  case "$out" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) fail "expected a refusal containing '$1', got: $out" ;;
  esac
}

CALM=docs/examples/pairs.json
SWAPPED="$ROOT/swapped.json"
MERGED="$ROOT/merged.json"
jq '.trait_name = "restless" | .pairs |= map({positive: .negative, negative: .positive})' "$CALM" >"$SWAPPED"
COUNT=$(jq '.pairs | length' "$CALM")
ONE=$(jq '.pairs | length / length' "$CALM")

run accepted pairs merge --pairs "$CALM" --pairs "$SWAPPED" --trait weather --output "$MERGED"
check "every pair of both sets is written" "$(jq '.pairs | length' "$MERGED")" "$((COUNT + COUNT))"
check "the calm pairs come first" "$(jq -c '.pairs[0]' "$MERGED")" "$(jq -c '.pairs[0]' "$CALM")"
check "the swapped pairs follow" "$(jq -c ".pairs[$COUNT]" "$MERGED")" "$(jq -c '.pairs[0]' "$SWAPPED")"
check "the merged set carries the trait given" "$(jq -r .trait_name "$MERGED")" "weather"
check "one source row per set" "$(echo "$out" | jq -c '[.report.sources[].trait_name]')" '["calm","restless"]'

TOY="$ROOT/toy-model"
run accepted toy-model "$TOY"
run accepted vector train --model "$TOY" --pairs "$MERGED" --output "$ROOT/weather.ster.json"
check "a direction is fitted on the merged set" "$(jq -r .trait_name "$ROOT/weather.ster.json")" "weather"

run refused pairs merge --pairs "$CALM" --trait weather --output "$ROOT/single.json"
refused "pairs merge needs at least two pair sets and got 1"
run refused pairs merge --pairs "$CALM" --pairs "$SWAPPED" --trait " " --output "$ROOT/unnamed.json"
refused "pairs merge needs the trait name"

# The desktop's path: the same merge through `ster request pairs/merge`.
BODY=$(jq -cn --arg calm "$CALM" --arg swapped "$SWAPPED" --arg output "$ROOT/requested.json" \
  '{sources: [$calm, $swapped], traitName: "weather", output: $output}')
if echo "$BODY" | "$BIN" request pairs/merge >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request pairs/merge <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(tail -n "$ONE" "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request writes the CLI's pairs" "$(jq -c . "$ROOT/requested.json")" "$(jq -c . "$MERGED")"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
