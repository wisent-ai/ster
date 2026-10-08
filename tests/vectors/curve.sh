#!/usr/bin/env bash
# Real test of `ster vector curve` and `ster request vector/curve` through the
# built binary, on a toy checkpoint it writes under this run's directory.
#
# It fits caa at the last layer of the toy model on one pair, two pairs and
# every pair left after holding out a quarter of the checked-in set, scores
# each on the same held-out pairs, and checks the answer: one point per size,
# in the order given, the holdout it was cut with, and a smallest size per
# layer that is one of the sizes and reached that layer's best accuracy. Then
# the refusals: a size larger than the pairs left to fit on, a held-out
# fraction of the whole set, and a request without sizes. Every command,
# whether it was accepted, and its answer go to the run's report.txt.
#
# Usage: STER=target/debug/ster tests/vectors/curve.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/vectors/curve-$RUN"
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
# Settings are read from the run's inputs so none is a chosen number: a
# quarter of the set held out, and the sizes one, two and all that remain.
COUNT=$(jq '.pairs | length' "$PAIRS")
ONE=$(jq '.pairs | length / length' "$PAIRS")
TWO=$(jq '.pairs | (length + length) / length' "$PAIRS")
QUARTER=$(jq '.pairs | (length / (length + length)) / ((length + length) / length)' "$PAIRS")
HELD=$(jq --argjson quarter "$QUARTER" '.pairs | length * $quarter | round' "$PAIRS")
FIT=$((COUNT - HELD))
LAST=$(jq '.num_hidden_layers - (.num_hidden_layers / .num_hidden_layers)' "$TOY/config.json")
SIZES="$ONE,$TWO,$FIT"

run accepted vector curve --model "$TOY" --pairs "$PAIRS" --layers "$LAST" --method caa \
  --holdout "$QUARTER" --sizes "$SIZES"
cp "$ROOT/output" "$ROOT/curve.json"
check "one point per size, in the order given" \
  "$(jq -r '[.points[].size | tostring] | join(",")' "$ROOT/curve.json")" "$SIZES"
check "the holdout is cut as stated" \
  "$(jq -c '[.holdout.fit_pairs, .holdout.holdout_pairs]' "$ROOT/curve.json")" "[$FIT,$HELD]"
check "the smallest size is one of the sizes" \
  "$(jq --arg sizes "$SIZES" '.enough[0].smallest_size | tostring | IN($sizes | split(",")[])' "$ROOT/curve.json")" "true"
check "the smallest size reached the layer's best accuracy" \
  "$(jq '.enough[0] as $enough | [.points[] | select(.size == $enough.smallest_size) | .holdout_accuracy == $enough.best_accuracy] | all' "$ROOT/curve.json")" "true"

run refused vector curve --model "$TOY" --pairs "$PAIRS" --layers "$LAST" --method caa \
  --holdout "$QUARTER" --sizes "$COUNT"
refused "a size of $COUNT pairs does not fit in the $FIT pairs left to fit on"
run refused vector curve --model "$TOY" --pairs "$PAIRS" --layers "$LAST" --method caa \
  --holdout "$ONE" --sizes "$ONE"
refused "the held-out fraction must be above zero and below one"

# The desktop's path: the same curve through `ster request vector/curve`.
BODY=$(jq -cn --arg model "$TOY" --arg pairs "$PAIRS" --arg layers "$LAST" --argjson holdout "$QUARTER" \
  --argjson sizes "[$SIZES]" '{model: $model, pairs: $pairs, layers: $layers, method: "caa", holdout: $holdout, sizes: $sizes}')
if echo "$BODY" | "$BIN" request vector/curve >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request vector/curve <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(tail -n "$ONE" "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request reports the CLI's points" \
  "$(tail -n "$ONE" "$ROOT/request.ndjson" | jq -c .json.points)" "$(jq -c .points "$ROOT/curve.json")"
UNSIZED=$(echo "$BODY" | jq -c 'del(.sizes)')
if echo "$UNSIZED" | "$BIN" request vector/curve >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
out=$(cat "$ROOT/request.ndjson")
printf '$ ster request vector/curve <<< %s\noutcome: %s\noutput: %s\n\n' "$UNSIZED" "$outcome" "$out" >>"$REPORT"
check "a body without sizes is refused" "$outcome" "refused"
refused "missing field \`sizes\`"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
