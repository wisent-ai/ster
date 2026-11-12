#!/usr/bin/env bash
# Real test of `ster pairs synthesize` with the refusal judgement the writing
# model makes, and of the model-free `ster pairs inspect`, through the built
# binary and a real instruct checkpoint.
#
# Synthesis asks the writing model, once per answer, whether the answer
# declined its role, and reads only {"refuses": true|false}; every attempt then
# ends kept or counted under exactly one rejection, so the report's counts add
# up to its attempts and the written set holds the kept pairs. No refusal
# score is taken: the flag that used to set one is refused as an unknown
# argument, and inspect reports no refusal finding, because judging a refusal
# needs a model and inspect loads none. The request path must agree. Every
# command, whether it was accepted, and its answer go to the run's report.txt.
#
# Ster assumes no dedupe, balance or sampling setting, so the caller states
# every one in STER_SYNTHESIS_SETTINGS, a JSON object with count,
# retry_multiplier, dedupe_bits, dedupe_bands, unbalanced_ratio,
# max_new_tokens, temperature, top_p and seed; a missing key is refused by name.
#
# Usage: STER=.build/kompilator/release/ster STER_SYNTHESIS_MODEL=<instruct model id> \
#   STER_SYNTHESIS_DEVICE=<cpu|metal|cuda> STER_SYNTHESIS_SETTINGS='{...}' \
#   tests/synthesis/judged.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=.build/kompilator/release/ster}
MODEL=${STER_SYNTHESIS_MODEL:?set STER_SYNTHESIS_MODEL to the instruct checkpoint that writes and judges the pairs}
DEVICE=${STER_SYNTHESIS_DEVICE:?set STER_SYNTHESIS_DEVICE to the device the checkpoint runs on}
SETTINGS=${STER_SYNTHESIS_SETTINGS:?set STER_SYNTHESIS_SETTINGS to the JSON object of the run settings}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/synthesis/judged-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT/home"
# The binary runs under this run's own home, never the operator's. A Hub
# model id is fetched into the Hub cache STER_TEST_HF_HOME names, or into
# one of this run's own.
export HOME="$ROOT/home" HF_HOME="${STER_TEST_HF_HOME:-$ROOT/home/huggingface}"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
echo "model: $MODEL on $DEVICE" >>"$REPORT"
echo "settings: $SETTINGS" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
setting() {
  echo "$SETTINGS" | jq -e --arg key "$1" 'has($key)' >/dev/null ||
    fail "STER_SYNTHESIS_SETTINGS has no $1"
  echo "$SETTINGS" | jq -r --arg key "$1" '.[$key]'
}
run() {
  expected=$1
  shift
  if [ "$expected" = accepted ]; then
    if "$BIN" "$@" >"$ROOT/output" 2>>"$ROOT/progress.txt" </dev/null; then outcome=accepted; else outcome=refused; fi
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

BITS=$(setting dedupe_bits)
BANDS=$(setting dedupe_bands)
RATIO=$(setting unbalanced_ratio)
TEMPERATURE=$(setting temperature)
SET="$ROOT/calm.json"
DEDUPE=(--dedupe-bits "$BITS" --dedupe-bands "$BANDS")
SYNTHESIS=(--trait "speaks with calm, measured patience" --count "$(setting count)"
  --retry-multiplier "$(setting retry_multiplier)" "${DEDUPE[@]}"
  --max-new-tokens "$(setting max_new_tokens)" --temperature "$TEMPERATURE"
  --top-p "$(setting top_p)" --seed "$(setting seed)")

run refused pairs synthesize --generator local --model "$MODEL" --device "$DEVICE" \
  --output "$SET" "${SYNTHESIS[@]}" --refusal-threshold "$TEMPERATURE"
refused "unexpected argument '--refusal-threshold'"
run refused pairs synthesize --generator cloud --output "$SET" "${SYNTHESIS[@]}"
refused "invalid value 'cloud' for '--generator <GENERATOR>'"
run refused pairs synthesize --generator brama --output "$SET" "${SYNTHESIS[@]}"
refused "pairs synthesize with --generator brama requires --generator-model"

run accepted pairs synthesize --generator local --model "$MODEL" --device "$DEVICE" \
  --output "$SET" "${SYNTHESIS[@]}"
check "every attempt ends kept or under one rejection" \
  "$(echo "$out" | jq '.report | .kept + .rejected_empty + .rejected_refusals + .rejected_duplicates')" \
  "$(echo "$out" | jq '.report.attempts')"
check "the set holds the kept pairs" "$(jq '.pairs | length' "$SET")" "$(echo "$out" | jq '.report.kept')"
check "the report names the writing model" "$(echo "$out" | jq -r '.report.generator')" "local:$MODEL"

CALM=docs/examples/pairs.json
INSPECT=(--pairs "$CALM" "${DEDUPE[@]}" --unbalanced-ratio "$RATIO")
run accepted pairs inspect "${INSPECT[@]}"
check "inspect reports no refusal finding" \
  "$(echo "$out" | jq '[paths | map(tostring) | join(".") | select(test("refusal"))] | length')" \
  "$(echo "$out" | jq '[] | length')"
check "inspect audits every pair" "$(echo "$out" | jq '.pair_count')" "$(jq '.pairs | length' "$CALM")"
CLI_INSPECT=$(echo "$out" | jq -c .)
run refused pairs inspect "${INSPECT[@]}" --refusal-threshold "$TEMPERATURE"
refused "unexpected argument '--refusal-threshold'"

# The desktop's path: the same audit through `ster request pairs/inspect`.
BODY=$(jq -cn --arg pairs "$CALM" --argjson bits "$BITS" --argjson bands "$BANDS" --argjson ratio "$RATIO" \
  '{pairs: $pairs, dedupeBits: $bits, dedupeBands: $bands, unbalancedRatio: $ratio}')
if echo "$BODY" | "$BIN" request pairs/inspect >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request pairs/inspect <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(cat "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request answers the CLI's audit" \
  "$(jq -c 'select(.type == "result") | .json' "$ROOT/request.ndjson")" "$CLI_INSPECT"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
