#!/bin/sh
# Real test of `ster vector score` through the built binary
# (docs/guide/steering/geometry.md, "Scoring a text token by token").
#
# It creates the reference checkpoint with `model fixture create`, fits a
# direction on the example pairs with `vector train`, scores a text with
# `vector score` and the same text through `ster request vector/score`, and
# checks what was answered: one score per token, the minimum, maximum and
# mean the scores have, the layer and trait, and the same scores on both
# surfaces. It then checks the refusals: a layer the artifact carries no
# vector at, a layer the model lacks, an empty text, no text, a text and a
# text file both given, and a request with no model. Every command, its exit
# status and output go to the run's report.txt.
#
# Usage: tests/steering/score.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/steering/score-$RUN"
REPORT="$ROOT/report.txt"
MODEL="$ROOT/model"
VECTOR="$ROOT/calm.ster.json"
TEXT="the sea is calm but the storm waves crash ."
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ ster %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  if [ "$status" -ne "$expected" ]; then
    echo "FAIL: $* exited $status, expected $expected: $err" | tee -a "$REPORT" >&2
    exit 1
  fi
}
request() {
  set +e
  out=$(printf '%s' "$1" | "$BIN" request vector/score 2>"$ROOT/stderr")
  status=$?
  set -e
  printf '$ ster request vector/score <<< %s\nexit: %s\nstdout: %s\n\n' "$1" "$status" "$out" >>"$REPORT"
  out=$(printf '%s\n' "$out" | tail -n 1)
}
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL: $1: got '$2', expected '$3'" | tee -a "$REPORT" >&2
    exit 1
  fi
  echo "ok: $1 = $2" >>"$REPORT"
}
refused() {
  case "$err" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) echo "FAIL: expected a refusal containing '$1', got: $err" | tee -a "$REPORT" >&2; exit 1 ;;
  esac
}

run 0 model fixture create --from docs/examples/model/fixture.json --output "$MODEL"
run 0 vector train --model "$MODEL" --pairs docs/examples/pairs.json --output "$VECTOR" --layers 1,2

run 0 vector score --model "$MODEL" --vector "$VECTOR" --layer 2 --input "$TEXT"
SCORED="$out"
check "the layer scored" "$(echo "$SCORED" | jq .layer)" "2"
check "the trait scored" "$(echo "$SCORED" | jq -r .trait_name)" "calm"
check "one score per token, in order" "$(echo "$SCORED" | jq '[.tokens[].index] == [range(.tokens | length)]')" "true"
check "the tokens spell the text" "$(echo "$SCORED" | jq '.tokens | length > 0')" "true"
check "min is the smallest score" "$(echo "$SCORED" | jq '.min == ([.tokens[].score] | min)')" "true"
check "max is the largest score" "$(echo "$SCORED" | jq '.max == ([.tokens[].score] | max)')" "true"
check "mean is the scores' mean" "$(echo "$SCORED" | jq '((.mean - ([.tokens[].score] | add / length)) | fabs) < 1e-9')" "true"

echo "$TEXT" >"$ROOT/text.txt"
run 0 vector score --model "$MODEL" --vector "$VECTOR" --layer 2 --input-file "$ROOT/text.txt"
check "a text file scores its text" "$(echo "$out" | jq '.tokens | length > 0')" "true"

request "{\"model\":\"$MODEL\",\"vector\":\"$VECTOR\",\"layer\":2,\"text\":\"$TEXT\"}"
check "the request ends in a result" "$(echo "$out" | jq -r .type)" "result"
check "the request answers the CLI's scores" "$(echo "$out" | jq -c '[.json.tokens[].score]')" "$(echo "$SCORED" | jq -c '[.tokens[].score]')"

run 1 vector score --model "$MODEL" --vector "$VECTOR" --layer 3 --input "$TEXT"
refused "the artifact carries no vector at layer 3; it carries layers 1, 2"
run 1 vector score --model "$MODEL" --vector "$VECTOR" --layer 9 --input "$TEXT"
refused "the artifact carries no vector at layer 9"
run 1 vector score --model "$MODEL" --vector "$VECTOR" --layer 2 --input "   "
refused "the text is empty; give the text to score"
run 1 vector score --model "$MODEL" --vector "$VECTOR" --layer 2
refused "give the text to score with --input or --input-file"
run 2 vector score --model "$MODEL" --vector "$VECTOR" --layer 2 --input "$TEXT" --input-file "$ROOT/text.txt"
refused "cannot be used with"

request '{"layer":2,"text":"x"}'
check "a request with no model is refused" "$(echo "$out" | jq -r .json.error)" "vector/score requires a model"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
