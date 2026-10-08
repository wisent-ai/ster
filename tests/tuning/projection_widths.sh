#!/usr/bin/env bash
# Exercises the actual CLI with a caller-supplied published checkpoint, never toy weights.
# Use a local Qwen3 checkpoint whose attention width differs from hidden_size.
# Required inputs: STER, STER_TEST_MODEL, STER_TEST_EXAMPLES, STER_TEST_LEARNING_RATE,
# STER_TEST_MAX_SEQUENCE. Reports retain source, binary and checkpoint identities.
# Shell positional parameters and statuses follow https://www.gnu.org/software/bash/manual/bash.html#Special-Parameters
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the built Ster binary}
SOURCE=${STER_TEST_MODEL:?set STER_TEST_MODEL to an absolute published checkpoint directory}
EXAMPLES=${STER_TEST_EXAMPLES:?set STER_TEST_EXAMPLES to real supervised examples}
RATE=${STER_TEST_LEARNING_RATE:?set STER_TEST_LEARNING_RATE explicitly}
SEQUENCE=${STER_TEST_MAX_SEQUENCE:?set STER_TEST_MAX_SEQUENCE explicitly}
ROOT="$PWD/target/real-tests/tuning/projection-widths-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ROOT/model"
REPORT="$ROOT/report.txt"
MODEL="$ROOT/model"
ADAPTER="$ROOT/adapter.safetensors"
fail() { printf 'FAIL: %s\n' "$*" | tee -a "$REPORT"; false; }
printf 'source revision: %s\ncheckpoint: %s\n' "$(git rev-parse HEAD)" "$SOURCE" >"$REPORT"
git diff --binary >"$ROOT/source.patch"
shasum "$BIN" "$EXAMPLES" "$SOURCE/config.json" "$SOURCE"/*.safetensors >>"$REPORT"
for file in "$SOURCE"/*; do
  [ -f "$file" ] || continue
  case "$file" in
    */config.json) cp "$file" "$MODEL/config.json" ;;
    *) ln -s "$file" "$MODEL/$(basename "$file")" ;;
  esac
done
# This case requires real nonsquare query projections, not a config changed to pretend they exist.
jq -e '.head_dim * .num_attention_heads != .hidden_size' "$MODEL/config.json" >>"$REPORT" || fail 'checkpoint must have unequal attention and hidden widths'
cp "$MODEL/config.json" "$ROOT/original-config.json"
COUNT=$(jq '.examples | length' "$EXAMPLES")
EPOCHS=$(jq '.examples | length / length' "$EXAMPLES")
run() {
  expected=$1
  shift
  printf '$ %q ' "$BIN" >>"$REPORT"
  printf '%q ' "$@" >>"$REPORT"
  printf '\n' >>"$REPORT"
  # Bash specifies descriptor 1 as standard output and 2 as standard error.
  # https://www.gnu.org/software/bash/manual/bash.html#Redirections
  if "$BIN" "$@" >"$ROOT/output" 2>&1 </dev/null; then
    status=$?
    outcome=accepted
  else
    status=$?
    outcome=refused
  fi
  printf 'exit: %s\n' "$status" >>"$REPORT"
  cat "$ROOT/output" >>"$REPORT"
  [ "$outcome" = "$expected" ] || fail "$* was $outcome, expected $expected"
}
run accepted tune sft --model "$MODEL" --examples "$EXAMPLES" --output "$ADAPTER" \
  --targets query --rank "$COUNT" --alpha "$COUNT" --epochs "$EPOCHS" \
  --learning-rate "$RATE" --accumulation "$COUNT" --max-sequence "$SEQUENCE" \
  --batch-size "$COUNT" --seed "$COUNT" --chat-template off
run accepted tune inspect "$ADAPTER"
cp "$ROOT/output" "$ROOT/inspection.json"
WIDTH=$(jq '.head_dim * .num_attention_heads' "$MODEL/config.json")
jq -e --argjson width "$WIDTH" '.adapter.tensors | map(select(.name | endswith(".query.b"))) | all(.shape | first == $width)' "$ROOT/inspection.json" >>"$REPORT" || fail 'saved query factors do not match checkpoint attention width'
run accepted tune evaluate --model "$MODEL" --adapter "$ADAPTER" --examples "$EXAMPLES" \
  --max-sequence "$SEQUENCE" --batch-size "$COUNT" --chat-template off
run accepted tune export --model "$MODEL" --adapter "$ADAPTER" --format peft --output "$ROOT/peft"
jq -e --arg model "$MODEL" --argjson rank "$COUNT" '.base_model_name_or_path == $model and .r == $rank and .target_modules == ["q_proj"]' "$ROOT/peft/adapter_config.json" >>"$REPORT" || fail 'persisted PEFT metadata differs from trained query adapter'
# Only this run's config changes. Operator checkpoint files remain untouched.
# Increasing declared heads creates a mismatch with the already persisted adapter.
jq '.num_attention_heads += .num_attention_heads' "$ROOT/original-config.json" >"$MODEL/config.json"
for operation in export merge; do
  arguments=(tune "$operation" --model "$MODEL" --adapter "$ADAPTER" --output "$ROOT/refused-$operation")
  if [ "$operation" = export ]; then arguments+=(--format peft); fi
  run refused "${arguments[@]}"
  case "$(cat "$ROOT/output")" in
    *'adapter tensor '*'.query.b has shape '*'for the model projection'*) ;;
    *) fail "$operation did not identify the mismatched query factor" ;;
  esac
  [ ! -e "$ROOT/refused-$operation" ] || fail "$operation wrote output after refusing the factor shape"
done
run refused tune evaluate --model "$MODEL" --adapter "$ADAPTER" --examples "$EXAMPLES" \
  --max-sequence "$SEQUENCE" --batch-size "$COUNT" --chat-template off
case "$(cat "$ROOT/output")" in
  *'adapter tensor '*'.query.b has shape '*'for the model projection'*) ;;
  *) fail 'runtime attachment did not identify the mismatched query factor' ;;
esac
cp "$ROOT/original-config.json" "$MODEL/config.json"
printf 'PASS\n' | tee -a "$REPORT"
printf 'Report: %s\n' "$REPORT"
