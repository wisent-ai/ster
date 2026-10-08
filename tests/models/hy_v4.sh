#!/usr/bin/env bash
# Real test of the `hy_v4` family (Tencent's HY V4) through the built binary,
# on the tiny random checkpoint `hf-tiny-v2/tiny-random-HYV4ForCausalLM`
# downloaded from the Hub.
#
# That checkpoint is written by Transformers under the names its own writer
# produces (`hc_pre.hc_attn_layer.hc_fn`, `mlp.linear_gate`,
# `learnable_sink_param`), so a load reads every tensor through the `hy_v4`
# renames; its two layers hold one full indexer and one shared one, a dense
# feed-forward and a routed one, latent attention with sinks and the output
# gate, and four independent hyper-connected streams. Its `index_topk` is
# eight keys, so the indexer chooses keys on every step that sees more than
# eight of them and keeps them all otherwise.
#
# It checks that greedy decoding answers, that the batch path answers the
# same, that the model still answers with its hyper-connections turned off
# (`enable_ihc: false`, one plain residual stream), and that a `gating_type`
# Ster does not implement is refused by name. Every command, whether it was
# accepted, and its answer go to the run's report.txt; a failed check stops
# the run (set -e) after naming itself there.
#
# Usage: STER=target/debug/ster tests/models/hy_v4.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/hy_v4-$RUN"
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

MODEL=hf-tiny-v2/tiny-random-HYV4ForCausalLM
PROMPTS=docs/examples/prompts.json
FIRST=$(jq -r '.prompts | first' "$PROMPTS")
# Greedy decoding, so two paths can be compared; the token budget and seed are
# read from the checked-in prompt set, so none is a chosen number.
COUNT=$(jq '.prompts | length' "$PROMPTS")
SETTINGS=(--max-new-tokens "$COUNT" --temperature "$(jq '.prompts | length - length' "$PROMPTS")" --seed "$COUNT" --chat-template off)

run accepted generate --model "$MODEL" --prompt "$FIRST" "${SETTINGS[@]}"
SINGLE=$out
ANSWERS="$ROOT/answers.json"
run accepted generate --model "$MODEL" --prompts "$PROMPTS" --output "$ANSWERS" "${SETTINGS[@]}"
check "the batch answer is the single-prompt answer" "$(jq -r 'first | .model_output' "$ANSWERS")" "$SINGLE"

# A local copy of the downloaded checkpoint whose config is rewritten per case.
SNAPSHOT=
for dir in "$HOME"/.cache/huggingface/hub/models--hf-tiny-v2--tiny-random-HYV4ForCausalLM/snapshots/*/; do
  SNAPSHOT=$dir
done
[ -f "$SNAPSHOT/config.json" ] || fail "the Hub download left no config.json under $SNAPSHOT"
copy() {
  dir="$ROOT/$1"
  mkdir -p "$dir"
  for file in model.safetensors tokenizer.json tokenizer_config.json generation_config.json; do
    ln -s "$SNAPSHOT/$file" "$dir/$file"
  done
  jq "$2" "$SNAPSHOT/config.json" >"$dir/config.json"
  echo "$dir"
}

PLAIN=$(copy plain-residual '.enable_ihc = false')
run accepted generate --model "$PLAIN" --prompt "$FIRST" "${SETTINGS[@]}"

HEADWISE=$(copy headwise-gate '.gating_type = "headwise"')
run refused generate --model "$HEADWISE" --prompt "$FIRST" "${SETTINGS[@]}"
refused "declares gating_type"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
