#!/usr/bin/env bash
# Real test of the `dots3_note` family (Dots3-Note) through the built binary,
# on a local copy of the released BF16 checkpoint `dots-studio/dots3-note-prev`
# named by DOTS3_NOTE_CHECKPOINT. The Hub holds no smaller Dots3-Note
# checkpoint, so the test runs where that one fits.
#
# Its layers alternate the two kinds Ster builds apart: `full_attention`
# layers run latent attention with the sparse-attention indexer, and
# `sliding_attention` layers their own narrower latent attention (64 heads,
# their own ranks and head parts, rotated by `swa_rope_theta`) through
# `sliding_window_size`; both norm the shared rotated key part, rescale their
# bottlenecks and gate every head by `g_proj`.
#
# It checks that greedy decoding answers, that the batch path answers the
# same, and that a config Ster does not implement is refused by name: an
# attention gate other than `headwise`, and sliding-window heads rotating
# another width than the full-attention ones. Every command, whether it was
# accepted, and its answer go to the run's report.txt; a failed check stops
# the run (set -e) after naming itself there.
#
# Usage: STER=target/release/ster DOTS3_NOTE_CHECKPOINT=/path/to/dots3-note-prev tests/models/dots3_note.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/release/ster}
CHECKPOINT=${DOTS3_NOTE_CHECKPOINT:?set DOTS3_NOTE_CHECKPOINT to a local copy of dots-studio/dots3-note-prev}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/dots3_note-$RUN"
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

ELEMENTWISE=$(copy elementwise-gate '.swa_attention_gate_type = "elementwise"')
run refused generate --model "$ELEMENTWISE" --prompt "$FIRST" "${SETTINGS[@]}"
refused "declares swa_attention_gate_type"

WIDER=$(copy wider-window-rotation '.swa_qk_rope_head_dim = .qk_rope_head_dim + .qk_rope_head_dim')
run refused generate --model "$WIDER" --prompt "$FIRST" "${SETTINGS[@]}"
refused "per sliding-window head"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
