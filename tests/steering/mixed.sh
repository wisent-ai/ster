#!/usr/bin/env bash
# Real test of `ster generate` with several steering vectors at once (wisent's
# multi-steer) and of `ster request generate` with its `steering` list,
# through the built binary, on a toy checkpoint it writes under this run's
# directory.
#
# It fits calm from the checked-in pair set and restless from the same pairs
# with their sides swapped, which is exactly calm's negation, then generates
# greedily: calm and restless at one strength each cancel and must answer
# what the unsteered model answers; calm twice at one strength must answer
# what calm once at twice that strength answers. The request path must
# answer as the CLI does. Then the refusals: a vector without a strength,
# strengths that are all zero, and a request part without a strength. Every
# command, whether it was accepted, and its answer go to the run's report.txt.
#
# Usage: STER=target/debug/ster tests/steering/mixed.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/steering/mixed-$RUN"
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
CALM=docs/examples/pairs.json
RESTLESS="$ROOT/restless.json"
jq '.trait_name = "restless" | .pairs |= map({positive: .negative, negative: .positive})' "$CALM" >"$RESTLESS"
run accepted toy-model "$TOY"
# Settings are read from the run's inputs so none is a chosen number.
COUNT=$(jq '.pairs | length' "$CALM")
ONE=$(jq '.pairs | length / length' "$CALM")
NONE=$(jq '.pairs | length - length' "$CALM")
TWICE=$(jq '.pairs | (length + length) / length' "$CALM")
LAST=$(jq '.num_hidden_layers - (.num_hidden_layers / .num_hidden_layers)' "$TOY/config.json")
PROMPT="describe the sea ."
GREEDY=(--max-new-tokens "$COUNT" --temperature "$NONE" --seed "$COUNT")

run accepted vector train --model "$TOY" --pairs "$CALM" --output "$ROOT/calm.ster.json" --layers "$LAST"
run accepted vector train --model "$TOY" --pairs "$RESTLESS" --output "$ROOT/restless.ster.json" --layers "$LAST"

run accepted generate --model "$TOY" --prompt "$PROMPT" "${GREEDY[@]}"
PLAIN=$out
run accepted generate --model "$TOY" --prompt "$PROMPT" "${GREEDY[@]}" \
  --vector "$ROOT/calm.ster.json" --strength "$TWICE" --vector "$ROOT/restless.ster.json" --strength "$TWICE"
check "calm and its negation at one strength cancel to the unsteered answer" "$out" "$PLAIN"
run accepted generate --model "$TOY" --prompt "$PROMPT" "${GREEDY[@]}" \
  --vector "$ROOT/calm.ster.json" --strength "$TWICE"
DOUBLE=$out
run accepted generate --model "$TOY" --prompt "$PROMPT" "${GREEDY[@]}" \
  --vector "$ROOT/calm.ster.json" --strength "$ONE" --vector "$ROOT/calm.ster.json" --strength "$ONE"
check "calm twice at one strength answers as calm once at twice it" "$out" "$DOUBLE"

run refused generate --model "$TOY" --prompt "$PROMPT" "${GREEDY[@]}" \
  --vector "$ROOT/calm.ster.json" --vector "$ROOT/restless.ster.json" --strength "$ONE"
refused "every --vector needs its own --strength: got 2 vector(s) and 1 strength(s)"
run refused generate --model "$TOY" --prompt "$PROMPT" "${GREEDY[@]}" \
  --vector "$ROOT/calm.ster.json" --strength "$NONE" --vector "$ROOT/restless.ster.json" --strength "$NONE"
refused "every steering strength is zero"

# The desktop's path: the same mix through `ster request generate`.
BODY=$(jq -cn --arg model "$TOY" --arg prompt "$PROMPT" --arg calm "$ROOT/calm.ster.json" \
  --argjson one "$ONE" --argjson tokens "$COUNT" --argjson none "$NONE" \
  '{model: $model, prompt: $prompt, steering: [{vector: $calm, strength: $one}, {vector: $calm, strength: $one}], maxNewTokens: $tokens, temperature: $none, seed: $tokens}')
if echo "$BODY" | "$BIN" request generate >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request generate <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(tail -n 1 "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request answers as the CLI's mix" "$(tail -n 1 "$ROOT/request.ndjson" | jq -r .json.text)" "$DOUBLE"
UNSCALED=$(echo "$BODY" | jq -c '.steering[0] |= del(.strength)')
if echo "$UNSCALED" | "$BIN" request generate >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
out=$(cat "$ROOT/request.ndjson")
printf '$ ster request generate <<< %s\noutcome: %s\noutput: %s\n\n' "$UNSCALED" "$outcome" "$out" >>"$REPORT"
check "a steering part without a strength is refused" "$outcome" "refused"
refused "missing field \`strength\`"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
