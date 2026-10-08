#!/usr/bin/env bash
# Real test of `ster vector optimize` through the built binary, on a toy
# checkpoint it writes under this run's directory, for the answer wisent's
# check-linearity gave: how far apart a trait's sides lie along a linear
# direction, on pairs the direction was not fitted on.
#
# It holds out a quarter of the checked-in pair set and checks the answer:
# one candidate per layer and method, exactly one selected, every candidate
# carrying a held-out effect size that is a number or null, and every number
# pointing the same way as its candidate's held-out margin, since both
# measure the positive sides minus the negative ones along the direction.
# Then the refusal of a fraction that empties the holdout. Every command,
# whether it was accepted, and its answer go to the run's report.txt.
#
# Usage: STER=target/debug/ster tests/vectors/optimize.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/vectors/optimize-$RUN"
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

TOY="$ROOT/toy-model"
PAIRS=docs/examples/pairs.json
run accepted toy-model "$TOY"
# Settings are read from the run's inputs so none is a chosen number.
ONE=$(jq '.pairs | length / length' "$PAIRS")
QUARTER=$(jq '.pairs | (length / (length + length)) / ((length + length) / length)' "$PAIRS")
LAYERS=$(jq .num_hidden_layers "$TOY/config.json")
METHODS=$(jq -n '["caa", "pca", "logistic"] | length')

run accepted vector optimize --model "$TOY" --pairs "$PAIRS" --holdout "$QUARTER" --output "$ROOT/best.ster.json"
cp "$ROOT/output" "$ROOT/selection.json"
check "one candidate per layer and method" "$(jq '.selection.candidates | length' "$ROOT/selection.json")" "$((LAYERS * METHODS))"
check "exactly one candidate is selected" "$(jq '[.selection.candidates[] | select(.selected)] | length' "$ROOT/selection.json")" "$ONE"
check "every candidate carries an effect size, a number or null" \
  "$(jq '[.selection.candidates[] | .holdout_effect_size | type] | all(. == "number" or . == "null")' "$ROOT/selection.json")" "true"
check "every effect size points the way its margin does" \
  "$(jq '[.selection.candidates[] | select(.holdout_effect_size != null and .holdout_margin != (.holdout_margin - .holdout_margin)) | (.holdout_effect_size * .holdout_margin) > (.holdout_margin - .holdout_margin)] | all' "$ROOT/selection.json")" "true"

run refused vector optimize --model "$TOY" --pairs "$PAIRS" --holdout "$ONE" --output "$ROOT/none.ster.json"
refused "the held-out fraction must be above zero and below one"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
