#!/usr/bin/env bash
# Real test of the `minimax_m3_vl_text` family (MiniMax-M3's text decoder,
# inside `minimax_m3_vl`) through the built binary, on a local copy of a
# MiniMax-M3 checkpoint named by MINIMAX_M3_CHECKPOINT: the gated fixture
# `shagunsd/tiny-random-MiniMaxM3SparseForConditionalGeneration` (random
# F32 weights, access granted on request) or the released
# `MiniMaxAI/MiniMax-M3` where it fits.
#
# Every feed-forward is SwiGLU-OAI, every norm Gemma's, and the layers
# `sparse_attention_config.sparse_attention_freq` marks run MiniMax Sparse
# Attention: a block indexer keeps each key-value group's best key blocks.
#
# It checks that greedy decoding answers, that the batch path answers the
# same, that a prompt longer than the blocks a query keeps (so the indexer
# hides keys) still answers, and that a config Ster does not implement is
# refused by name: another `hidden_act`, another block score than `max`, and
# index heads that are not one per key-value head. Every command, whether it
# was accepted, and its answer go to the run's report.txt; a failed check
# stops the run (set -e) after naming itself there.
#
# Usage: STER=target/release/ster MINIMAX_M3_CHECKPOINT=/path/to/checkpoint tests/models/minimax_m3.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/release/ster}
CHECKPOINT=${MINIMAX_M3_CHECKPOINT:?set MINIMAX_M3_CHECKPOINT to a local copy of a MiniMax-M3 checkpoint}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/minimax_m3-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
echo "checkpoint: $CHECKPOINT" >>"$REPORT"

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

[ -f "$CHECKPOINT/config.json" ] || fail "$CHECKPOINT holds no config.json"
PROMPTS=docs/examples/prompts.json
FIRST=$(jq -r '.prompts | first' "$PROMPTS")
# Greedy decoding, so two paths can be compared; the token budget and seed are
# read from the checked-in prompt set, so none is a chosen number.
COUNT=$(jq '.prompts | length' "$PROMPTS")
SETTINGS=(--max-new-tokens "$COUNT" --temperature "$(jq '.prompts | length - length' "$PROMPTS")" --seed "$COUNT" --chat-template off)

run accepted generate --model "$CHECKPOINT" --prompt "$FIRST" "${SETTINGS[@]}"
SINGLE=$out
ANSWERS="$ROOT/answers.json"
run accepted generate --model "$CHECKPOINT" --prompts "$PROMPTS" --output "$ANSWERS" "${SETTINGS[@]}"
check "the batch answer is the single-prompt answer" "$(jq -r 'first | .model_output' "$ANSWERS")" "$SINGLE"

# The first prompt repeated once per key a query's kept blocks hold, so the
# prompt alone holds more blocks than a query keeps and the indexer hides
# some on every sparse layer.
LONG=$(jq -r --arg first "$FIRST" '.text_config.sparse_attention_config
  | [range(.sparse_topk_blocks * .sparse_block_size)] | map($first) | join(" ")' "$CHECKPOINT/config.json")
run accepted generate --model "$CHECKPOINT" --prompt "$LONG" "${SETTINGS[@]}"

# A copy of the checkpoint whose config is rewritten per case; every other
# file is a link to the original.
copy() {
  dir="$ROOT/$1"
  mkdir -p "$dir"
  for file in "$CHECKPOINT"/*; do
    [ "$(basename "$file")" = config.json ] || ln -s "$file" "$dir/$(basename "$file")"
  done
  jq "$2" "$CHECKPOINT/config.json" >"$dir/config.json"
  echo "$dir"
}

SILU=$(copy silu '.text_config.hidden_act = "silu"')
run refused generate --model "$SILU" --prompt "$FIRST" "${SETTINGS[@]}"
refused "feed-forwards are swigluoai"

SUMMED=$(copy summed-blocks '.text_config.sparse_attention_config.sparse_score_type = "sum"')
run refused generate --model "$SUMMED" --prompt "$FIRST" "${SETTINGS[@]}"
refused "declares sparse_score_type"

SHARED=$(copy shared-index-heads '.text_config.sparse_attention_config.sparse_num_index_heads = .text_config.num_key_value_heads + .text_config.num_key_value_heads')
run refused generate --model "$SHARED" --prompt "$FIRST" "${SETTINGS[@]}"
refused "one block selection per key-value head"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
