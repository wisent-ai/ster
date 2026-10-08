#!/usr/bin/env bash
# Real test of `ster vector project` and `ster request vector/project`
# through the built binary, on a toy checkpoint it writes under this run's
# directory.
#
# It fits calm at every layer from the checked-in pair set and projects
# where adding it moves every pair side read at the last layer: one point
# per side, unsteered and steered; at most as many components as asked for,
# each carrying a share of the variance and giving every point a coordinate;
# a shift measured along calm's own direction there; steered negatives at
# another distance from the positive sides than unsteered ones, since the
# vectors added at the layers before the read move them; and the share of
# negatives brought toward the positives lying between none and all. Then
# the refusals: a layer past the model's last, no components, and the
# request without a strength. Every command, whether it was accepted, and
# its answer go to the run's report.txt.
#
# Usage: STER=target/debug/ster tests/vectors/project.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/vectors/project-$RUN"
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
ARTIFACT="$ROOT/calm.ster.json"
run accepted toy-model "$TOY"
# Settings are read from the run's inputs so none is a chosen number.
COUNT=$(jq '.pairs | length' "$PAIRS")
NONE=$(jq '.pairs | length - length' "$PAIRS")
ONE=$(jq '.pairs | length / length' "$PAIRS")
TWO=$(jq '.pairs | (length + length) / length' "$PAIRS")
LAST=$(jq '.num_hidden_layers - (.num_hidden_layers / .num_hidden_layers)' "$TOY/config.json")
PAST=$(jq .num_hidden_layers "$TOY/config.json")

run accepted vector train --model "$TOY" --pairs "$PAIRS" --output "$ARTIFACT"
run accepted vector project --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" \
  --layer "$LAST" --strength "$TWO" --components "$TWO"
cp "$ROOT/output" "$ROOT/projection.json"
check "one point per side, unsteered and steered" \
  "$(jq '.points | length' "$ROOT/projection.json")" "$((COUNT * TWO * TWO))"
check "no more components than asked for" \
  "$(jq --argjson asked "$TWO" '.explained | length <= $asked' "$ROOT/projection.json")" "true"
check "every component carries a share of the variance" \
  "$(jq --argjson none "$NONE" --argjson one "$ONE" '[.explained[] | . > $none and . <= $one] | all' "$ROOT/projection.json")" "true"
check "every point has a coordinate per component" \
  "$(jq '(.explained | length) as $components | [.points[].coordinates | length] | unique == [$components]' "$ROOT/projection.json")" "true"
check "the shift is measured along calm's own direction at the read layer" \
  "$(jq '.shift_along_direction | type' "$ROOT/projection.json")" "number"
check "steering moved the negatives" \
  "$(jq '.negative_distance_to_positive | .steered != .unsteered' "$ROOT/projection.json")" "true"
check "the share of negatives moved toward the positives is a share" \
  "$(jq --argjson none "$NONE" --argjson one "$ONE" '.negatives_moved_toward_positive >= $none and .negatives_moved_toward_positive <= $one' "$ROOT/projection.json")" "true"

run refused vector project --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" \
  --layer "$PAST" --strength "$TWO" --components "$TWO"
refused "layer $PAST is outside the model's"
run refused vector project --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" \
  --layer "$LAST" --strength "$TWO" --components "$NONE"
refused "--components"

# The desktop's path: the same projection through `ster request vector/project`.
BODY=$(jq -cn --arg model "$TOY" --arg pairs "$PAIRS" --arg vector "$ARTIFACT" \
  --argjson layer "$LAST" --argjson two "$TWO" \
  '{model: $model, pairs: $pairs, vector: $vector, layer: $layer, strength: $two, components: $two}')
if echo "$BODY" | "$BIN" request vector/project >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request vector/project <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(tail -n "$ONE" "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request reports the CLI's shift" \
  "$(tail -n "$ONE" "$ROOT/request.ndjson" | jq .json.shift_along_direction)" "$(jq .shift_along_direction "$ROOT/projection.json")"
UNSCALED=$(echo "$BODY" | jq -c 'del(.strength)')
if echo "$UNSCALED" | "$BIN" request vector/project >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
out=$(cat "$ROOT/request.ndjson")
printf '$ ster request vector/project <<< %s\noutcome: %s\noutput: %s\n\n' "$UNSCALED" "$outcome" "$out" >>"$REPORT"
check "a body without a strength is refused" "$outcome" "refused"
refused "missing field \`strength\`"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
