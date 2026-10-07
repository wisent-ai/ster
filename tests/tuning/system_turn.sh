#!/usr/bin/env bash
# Real test of an example's system turn through the built binary, on a toy
# checkpoint it writes under this run's directory.
#
# It writes the toy model, copies it with the documented toy chat template
# beside it, gives every example of docs/examples/tuning/chat-examples.json the
# same system turn, and scores both sets with `ster tune evaluate`: the set
# with a system turn is accepted, echoes its system turn, and scores exactly as
# many completion tokens as the set without one, because the system turn is
# prompt rather than target. Then the refusals: a blank system turn, and a
# system turn on the toy model itself, whose chat template is absent, and on
# the templated copy run with --chat-template off. Every command, whether it
# was accepted, and its answer (stdout of an accepted run, stdout and stderr of
# a refused one) go to the run's report.txt; a failed check stops the run
# (set -e) after naming itself there.
#
# Usage: tests/tuning/system_turn.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/tuning/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
# run accepted|refused ARGS...: the command must end the way named. An accepted
# run's answer is its stdout, the JSON report, with progress left on the
# console; a refused run's answer is everything it wrote, the refusal included.
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
CHAT="$ROOT/toy-chat-model"
BARE=docs/examples/tuning/chat-examples.json
SERVED="$ROOT/served.json"
BLANK="$ROOT/blank.json"
run accepted toy-model "$TOY"
cp -R "$TOY" "$CHAT"
cp docs/examples/chat-template/tokenizer_config.json "$CHAT/"
jq '.examples |= map(. + {system: "the sea is calm ."})' "$BARE" >"$SERVED"
jq '.examples |= map(. + {system: "  "})' "$BARE" >"$BLANK"
# The limits every scoring run shares, read rather than chosen: the toy
# model's own context length, so no example is skipped, and the whole set in
# one forward pass.
LIMITS=(--max-sequence "$(jq .max_position_embeddings "$TOY/config.json")"
  --batch-size "$(jq '.examples | length' "$BARE")")

run accepted tune evaluate --model "$CHAT" --examples "$BARE" "${LIMITS[@]}"
cp "$ROOT/output" "$ROOT/bare-report.json"
run accepted tune evaluate --model "$CHAT" --examples "$SERVED" "${LIMITS[@]}"
cp "$ROOT/output" "$ROOT/served-report.json"
check "each scored entry echoes its system turn" \
  "$(jq -r '[.entries[].system] | unique | join(",")' "$ROOT/served-report.json")" "the sea is calm ."
check "the system turn adds no completion token" \
  "$(jq .completion_tokens "$ROOT/served-report.json")" "$(jq .completion_tokens "$ROOT/bare-report.json")"
check "an entry without a system turn carries none" \
  "$(jq -c '[.entries[] | has("system")] | unique' "$ROOT/bare-report.json")" '[false]'

run refused tune evaluate --model "$CHAT" --examples "$BLANK" "${LIMITS[@]}"
refused "contains an empty system prompt"
run refused tune evaluate --model "$TOY" --examples "$SERVED" "${LIMITS[@]}"
refused "training example has a system prompt, which only a chat template can place, and this run's chat template is absent"
run refused tune evaluate --model "$CHAT" --chat-template off --examples "$SERVED" "${LIMITS[@]}"
refused "training example has a system prompt, which only a chat template can place, and this run's chat template is off"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
