#!/usr/bin/env bash
# Real test of `ster vector ablate` and `ster request vector/ablate` through
# the built binary, on a toy checkpoint it writes under this run's directory.
#
# It fits calm at every layer of the toy model, writes the direction out of
# the weights at full strength, and checks the rewritten checkpoint: both
# residual-stream writes of every layer rewritten, every other tensor copied,
# the weights, config and tokenizer written, and the directory loads and
# generates like any model. Then the refusals: a zero strength, and an
# artifact fitted on another model. The request path must rewrite the same
# tensors. Every command, whether it was accepted, and its answer go to the
# run's report.txt.
#
# Usage: STER=target/debug/ster tests/vectors/ablate.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/vectors/ablate-$RUN"
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
OTHER="$ROOT/other-model"
PAIRS=docs/examples/pairs.json
ARTIFACT="$ROOT/calm.ster.json"
run accepted toy-model "$TOY"
run accepted toy-model "$OTHER"
# Settings are read from the run's inputs so none is a chosen number.
COUNT=$(jq '.pairs | length' "$PAIRS")
NONE=$(jq '.pairs | length - length' "$PAIRS")
ONE=$(jq '.pairs | length / length' "$PAIRS")
LAYERS=$(jq .num_hidden_layers "$TOY/config.json")

run accepted vector train --model "$TOY" --pairs "$PAIRS" --output "$ARTIFACT"
run accepted vector ablate --model "$TOY" --vector "$ARTIFACT" --strength "$ONE" --output "$ROOT/ablated"
cp "$ROOT/output" "$ROOT/ablation.json"
check "both residual-stream writes of every layer are rewritten" \
  "$(jq '.rewritten | length' "$ROOT/ablation.json")" "$((LAYERS + LAYERS))"
check "every other tensor is copied" \
  "$(jq '.copied_tensors + (.rewritten | length) == .total_tensors' "$ROOT/ablation.json")" "true"
check "the weights, config and tokenizer are written" \
  "$(jq '[.files[] | select(. == "model.safetensors" or . == "config.json" or . == "tokenizer.json")] | length' "$ROOT/ablation.json")" "3"
run accepted generate --model "$ROOT/ablated" --prompt "describe the sea ." \
  --max-new-tokens "$COUNT" --temperature "$NONE" --seed "$COUNT"
echo "ok: the rewritten checkpoint loads and generates" >>"$REPORT"

run refused vector ablate --model "$TOY" --vector "$ARTIFACT" --strength "$NONE" --output "$ROOT/unchanged"
refused "changes nothing or nothing sensible"
run refused vector ablate --model "$OTHER" --vector "$ARTIFACT" --strength "$ONE" --output "$ROOT/elsewhere"
refused "artifact was trained for model"

# The desktop's path: the same rewrite through `ster request vector/ablate`.
BODY=$(jq -cn --arg model "$TOY" --arg vector "$ARTIFACT" --arg output "$ROOT/requested" --argjson one "$ONE" \
  '{model: $model, vector: $vector, strength: $one, output: $output}')
if echo "$BODY" | "$BIN" request vector/ablate >"$ROOT/request.ndjson"; then outcome=accepted; else outcome=refused; fi
printf '$ ster request vector/ablate <<< %s\noutcome: %s\noutput: %s\n\n' "$BODY" "$outcome" "$(tail -n "$ONE" "$ROOT/request.ndjson")" >>"$REPORT"
check "the request completes" "$outcome" "accepted"
check "the request rewrites the CLI's tensors" \
  "$(tail -n "$ONE" "$ROOT/request.ndjson" | jq -c .json.rewritten)" "$(jq -c .rewritten "$ROOT/ablation.json")"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
