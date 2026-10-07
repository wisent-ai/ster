#!/usr/bin/env bash
# Real test of `ster generate --prompts` through the built binary, on a toy
# checkpoint it writes under this run's directory.
#
# It answers the checked-in prompt set with the model loaded once, reads the
# written file back (one entry per prompt, in order, each carrying its prompt
# and an answer), checks that an answer equals the one `--prompt` gives for
# the same prompt and settings, and checks the refusal of an empty set. Every
# command, whether it was accepted, and its answer go to the run's
# report.txt; a failed check stops the run (set -e) after naming itself there.
#
# Usage: STER=target/debug/ster tests/generate/prompts.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/generate/prompts-$RUN"
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
PROMPTS=docs/examples/prompts.json
ANSWERS="$ROOT/answers.json"
EMPTY="$ROOT/empty.json"
run accepted toy-model "$TOY"
# Greedy decoding so the batch and the single answer can be compared; the
# token budget and seed are read from the set, so none is a chosen number.
COUNT=$(jq '.prompts | length' "$PROMPTS")
SETTINGS=(--max-new-tokens "$COUNT" --temperature "$(jq '.prompts | length - length' "$PROMPTS")" --seed "$COUNT")

run accepted generate --model "$TOY" --prompts "$PROMPTS" --output "$ANSWERS" "${SETTINGS[@]}"
check "the run names the file it wrote" "$out" "$ANSWERS"
check "one answer per prompt" "$(jq length "$ANSWERS")" "$COUNT"
check "the answers keep the set's order" "$(jq -c '[.[].prompt]' "$ANSWERS")" "$(jq -c .prompts "$PROMPTS")"
check "every entry carries an answer" "$(jq '[.[] | has("model_output")] | all' "$ANSWERS")" "true"

FIRST=$(jq -r '.prompts | first' "$PROMPTS")
run accepted generate --model "$TOY" --prompt "$FIRST" "${SETTINGS[@]}"
check "a batch answer is the single-prompt answer" "$(jq -r 'first | .model_output' "$ANSWERS")" "$out"

jq '.prompts = []' "$PROMPTS" >"$EMPTY"
run refused generate --model "$TOY" --prompts "$EMPTY" --output "$ROOT/none.json" "${SETTINGS[@]}"
refused "prompt set contains no prompts"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
