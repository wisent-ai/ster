#!/usr/bin/env bash
# Real test of `ster tune export --format peft` through the built binary, on a
# toy checkpoint and an adapter it trains under this run's directory.
#
# It writes the toy model, trains a small adapter on the checked-in toy
# examples with every setting stated, exports it, and reads the directory
# back: the three files, the config's rank, alpha and target modules equal to
# the adapter's own sidecar, and one lora_A and one lora_B per adapted
# projection named after the toy checkpoint's projections. Then the refusals:
# an adapter exported against another model. Every command, whether it was
# accepted, and its answer go to the run's report.txt; a failed check stops
# the run (set -e) after naming itself there.
#
# Usage: STER=target/debug/ster tests/tuning/export.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/tuning/export-$RUN"
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
OTHER="$ROOT/other-model"
ADAPTER="$ROOT/calm.safetensors"
PEFT="$ROOT/calm-peft"
EXAMPLES=docs/examples/tuning/examples.json
run accepted toy-model "$TOY"
cp -R "$TOY" "$OTHER"
# The settings only need to train some adapter: the export is checked for
# names and shapes, which no setting changes. Each is read from the run's
# inputs — the set's size, one pass, the toy context — so none is a chosen number.
COUNT=$(jq '.examples | length' "$EXAMPLES")
run accepted tune sft --model "$TOY" --examples "$EXAMPLES" --output "$ADAPTER" \
  --rank "$COUNT" --alpha "$COUNT" --epochs "$(jq '.examples | length / length' "$EXAMPLES")" \
  --learning-rate "$(jq '1 / (.examples | length)' "$EXAMPLES")" --accumulation "$COUNT" \
  --max-sequence "$(jq .max_position_embeddings "$TOY/config.json")" --batch-size "$COUNT" --seed "$COUNT"
SIDECAR="$ROOT/calm.json"

run accepted tune export --model "$TOY" --adapter "$ADAPTER" --format peft --output "$PEFT"
for file in adapter_config.json adapter_model.safetensors tokenizer.json; do
  [ -s "$PEFT/$file" ] || fail "the export wrote no $file"
  echo "ok: $file written" >>"$REPORT"
done
check "the tokenizer is the base's own" "$(cmp -s "$PEFT/tokenizer.json" "$TOY/tokenizer.json" && echo same)" "same"
check "r is the adapter's rank" "$(jq .r "$PEFT/adapter_config.json")" "$(jq .rank "$SIDECAR")"
check "lora_alpha is the adapter's alpha" "$(jq .lora_alpha "$PEFT/adapter_config.json")" "$(jq .alpha "$SIDECAR")"
check "the config names the base model" "$(jq -r .base_model_name_or_path "$PEFT/adapter_config.json")" "$TOY"
check "the report counts two factors per adapted projection" \
  "$(jq .report.tensors "$ROOT/output")" \
  "$(jq '(.layers | length) * (.targets | length) * 2' "$SIDECAR")"
check "the target modules are the checkpoint's projection names" \
  "$(jq -c .target_modules "$PEFT/adapter_config.json")" '["q_proj","v_proj"]'

run refused tune export --model "$OTHER" --adapter "$ADAPTER" --format peft --output "$ROOT/other-peft"
refused "adapter was trained for model"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
