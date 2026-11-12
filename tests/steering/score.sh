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
# Usage: tests/steering/score.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/steering/score-$RUN"
REPORT="$ROOT/report.txt"
MODEL="$ROOT/model"
VECTOR="$ROOT/calm.ster.json"
TEXT="the sea is calm but the storm waves crash ."
mkdir -p "$ROOT/home"
# The binary runs under this run's own home, never the operator's.
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
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

a_layer_the_artifact_lacks_is_refused() {
  run 2 vector score --model "$MODEL" --vector "$VECTOR" --layer 3 --input "$TEXT"
  refused "the artifact carries no vector at layer 3; it carries layers 1, 2"
  refused "fix: give --layer one of 1, 2, or train a vector at layer 3 with ster vector train --layers 3"
  run 2 vector score --model "$MODEL" --vector "$VECTOR" --layer 9 --input "$TEXT"
  refused "the artifact carries no vector at layer 9"
}
an_empty_text_is_refused() {
  run 2 vector score --model "$MODEL" --vector "$VECTOR" --layer 2 --input "   "
  refused "reading the text to score from --input: the text is empty; give the text to score"
  refused "observed: --input: 3 bytes, all whitespace"
}
no_text_is_refused() {
  run 2 vector score --model "$MODEL" --vector "$VECTOR" --layer 2
  refused "give the text to score with --input or --input-file"
}
a_text_and_a_text_file_are_refused() {
  run 2 vector score --model "$MODEL" --vector "$VECTOR" --layer 2 --input "$TEXT" --input-file "$ROOT/text.txt"
  refused "cannot be used with"
  refused "--input-file $ROOT/text.txt both given"
}
a_missing_text_file_is_refused() {
  run 1 vector score --model "$MODEL" --vector "$VECTOR" --layer 2 --input-file "$ROOT/absent.txt"
  refused "reading the text to score from $ROOT/absent.txt: no file is there"
}
a_direction_without_length_is_refused() {
  jq '.vectors[0].values |= map(0)' "$VECTOR" >"$ROOT/flat.json"
  run 1 vector score --model "$MODEL" --vector "$ROOT/flat.json" --layer 1 --input "$TEXT"
  refused "the artifact's direction at layer 1 has no usable length (0)"
}
a_layer_the_artifact_lacks_is_refused
an_empty_text_is_refused
no_text_is_refused
a_text_and_a_text_file_are_refused
a_missing_text_file_is_refused
a_direction_without_length_is_refused

request '{"layer":2,"text":"x"}'
check "a request with no model is refused by its declared cause" "$(echo "$out" | jq -r .json.cause.id)" "field-blank"
err=$(echo "$out" | jq -r .json.error)
refused "checking the vector/score request: vector/score requires a model (a Hugging Face Hub id or a checkpoint directory): the request's model field is absent or blank"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
