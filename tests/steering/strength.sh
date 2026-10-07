#!/usr/bin/env bash
# Real test of `ster evaluate --strengths` and `ster request evaluate` with
# strengths, through the built binary, on a toy checkpoint it writes under this
# run's directory.
#
# It writes the toy model, fits a direction from the checked-in pair set, and
# measures it at three strengths: every strength is reported in the order
# given, exactly one is selected and it is the one the report names, every
# pair is either scored or counted as skipped, and every share lies between
# none and all. Then the refusals: a missing batch size and sequence limit, a
# batch size without strengths, a non-finite strength, a zero batch size, a
# sequence limit no pair fits, and the request body's missing batchSize. Every
# command, whether it was accepted, and its answer go to the run's report.txt;
# a failed check stops the run (set -e) after naming itself there.
#
# Usage: STER=target/debug/ster tests/steering/strength.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/steering/strength-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
# run accepted|refused ARGS...: the command must end the way named. An accepted
# run's answer is its stdout; a refused run's answer is everything it wrote.
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
# Every setting is read from the run's inputs so none is a chosen number: the
# last layer, the set's size, one, none, the toy context.
COUNT=$(jq '.pairs | length' "$PAIRS")
ONE=$(jq '.pairs | length / length' "$PAIRS")
NONE=$(jq '.pairs | length - length' "$PAIRS")
CONTEXT=$(jq .max_position_embeddings "$TOY/config.json")
LAST=$(jq '.num_hidden_layers - (.num_hidden_layers / .num_hidden_layers)' "$TOY/config.json")
STRENGTHS="-$ONE,$ONE,$COUNT"

run accepted train --model "$TOY" --pairs "$PAIRS" --output "$ARTIFACT" --layers "$LAST"
run accepted evaluate --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" \
  --strengths "$STRENGTHS" --batch-size "$COUNT" --max-sequence "$CONTEXT"
cp "$ROOT/output" "$ROOT/selection.json"
check "every strength is reported in the order given" \
  "$(jq -r '[.strength.candidates[].strength | tostring] | join(",")' "$ROOT/selection.json")" "$STRENGTHS"
check "exactly one strength is selected" \
  "$(jq '[.strength.candidates[] | select(.selected)] | length' "$ROOT/selection.json")" "$ONE"
check "the selected row is the strength the report names" \
  "$(jq '(.strength.candidates[] | select(.selected) | .strength) == .strength.selected_strength' "$ROOT/selection.json")" "true"
check "every pair is scored or counted as skipped" \
  "$(jq '.strength.scored_pairs + .strength.skipped_long' "$ROOT/selection.json")" "$COUNT"
check "every share lies between none and all" \
  "$(jq --argjson none "$NONE" --argjson one "$ONE" '[.strength.candidates[].ordered | . >= $none and . <= $one] | all' "$ROOT/selection.json")" "true"
check "the representation report is still there" \
  "$(jq 'has("layers")' "$ROOT/selection.json")" "true"

run refused evaluate --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" --strengths "$STRENGTHS" --max-sequence "$CONTEXT"
refused "--strengths needs --batch-size; Ster assumes none"
run refused evaluate --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" --strengths "$STRENGTHS" --batch-size "$COUNT"
refused "--strengths needs --max-sequence; Ster assumes none"
run refused evaluate --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" --batch-size "$COUNT"
refused "--strengths"
run refused evaluate --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" \
  --strengths "$ONE,inf" --batch-size "$COUNT" --max-sequence "$CONTEXT"
refused "strength selection needs finite strengths, not inf"
run refused evaluate --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" \
  --strengths "$STRENGTHS" --batch-size "$NONE" --max-sequence "$CONTEXT"
refused "strength selection requires batch size of at least one"
run refused evaluate --model "$TOY" --pairs "$PAIRS" --vector "$ARTIFACT" \
  --strengths "$STRENGTHS" --batch-size "$COUNT" --max-sequence "$ONE"
refused "every pair is longer than the sequence limit"

# The desktop's path: the same measurement through `ster request evaluate`,
# and its refusal before anything runs.
BODY=$(jq -cn --arg model "$TOY" --arg pairs "$PAIRS" --arg vector "$ARTIFACT" \
  --argjson strengths "[$STRENGTHS]" --argjson batch "$COUNT" --argjson sequence "$CONTEXT" \
  '{model: $model, pairs: $pairs, vector: $vector, strengths: $strengths, batchSize: $batch, maxSequence: $sequence}')
if echo "$BODY" | "$BIN" request evaluate >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request evaluate <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(tail -n "$ONE" "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request selects the strength the CLI selected" \
  "$(tail -n "$ONE" "$ROOT/request.ndjson" | jq .json.strength.selected_strength)" \
  "$(jq .strength.selected_strength "$ROOT/selection.json")"
UNSIZED=$(echo "$BODY" | jq -c 'del(.batchSize)')
if echo "$UNSIZED" | "$BIN" request evaluate >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
out=$(cat "$ROOT/request.ndjson")
printf '$ ster request evaluate <<< %s\noutcome: %s\noutput: %s\n\n' "$UNSIZED" "$outcome" "$out" >>"$REPORT"
check "a body without batchSize is refused" "$outcome" "refused"
refused "evaluate with strengths requires batchSize; Ster assumes none"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
