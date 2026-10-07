#!/usr/bin/env bash
# Real test of `ster tune bipo` through the built binary, on a toy checkpoint
# it writes under this run's directory.
#
# It writes the toy model, learns a steering vector from the checked-in pair
# set, reads the artifact back (method bipo, one vector at the asked layer, as
# wide as the model, moved off zero), and steers a generation with it. Then
# the refusal of a layer past the model's last. Every command, whether it was
# accepted, and its answer go to the run's report.txt; a failed check stops
# the run (set -e) after naming itself there.
#
# Usage: tests/tuning/bipo.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/tuning/bipo-$RUN"
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
ARTIFACT="$ROOT/calm.bipo.json"
run accepted toy-model "$TOY"
LAYERS=$(jq .num_hidden_layers "$TOY/config.json")
# The settings only need to move the vector: the checks read its shape, its
# layer and that it left zero, which no particular setting decides. Each is
# read from the run's inputs so none is a chosen number: the last layer, the
# set's size, one pass, the toy context.
COUNT=$(jq '.pairs | length' "$PAIRS")
ONE=$(jq '.pairs | length / length' "$PAIRS")
SETTINGS=(--strength "$ONE" --beta "$ONE" --epochs "$ONE" --learning-rate "$(jq '1 / (.pairs | length)' "$PAIRS")"
  --accumulation "$ONE" --warmup-steps "$(jq '.pairs | length - length' "$PAIRS")"
  --max-sequence "$(jq .max_position_embeddings "$TOY/config.json")" --batch-size "$COUNT" --seed "$COUNT")
LAST=$(jq '.num_hidden_layers - (.num_hidden_layers / .num_hidden_layers)' "$TOY/config.json")

run accepted tune bipo --model "$TOY" --pairs "$PAIRS" --output "$ARTIFACT" --layer "$LAST" "${SETTINGS[@]}"
check "the artifact names its method" "$(jq -r .method "$ARTIFACT")" "bipo"
check "it holds one vector at the asked layer" "$(jq -c '[.vectors[].layer]' "$ARTIFACT")" "[$LAST]"
check "the vector is as wide as the model" "$(jq '.vectors[0].values | length' "$ARTIFACT")" "$(jq .hidden_size "$TOY/config.json")"
check "the vector moved off zero" "$(jq '.report.vector_norm > (.report.vector_norm - .report.vector_norm)' "$ROOT/output")" "true"
check "every pair was trained" "$(jq .report.trained_pairs "$ROOT/output")" "$COUNT"

run accepted generate --model "$TOY" --vector "$ARTIFACT" --prompt "describe the sea ." --max-new-tokens "$COUNT"
echo "ok: the learned vector steers a generation" >>"$REPORT"

run refused tune bipo --model "$TOY" --pairs "$PAIRS" --output "$ROOT/outside.json" --layer "$LAYERS" "${SETTINGS[@]}"
refused "layer $LAYERS is outside the model's $LAYERS layers"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
