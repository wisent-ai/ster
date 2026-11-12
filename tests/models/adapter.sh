#!/bin/sh
# Real test of how attaching a trained artifact at load fails
# (`ster.model.load`), through the built binary.
#
# `ster model fixture create` writes a real Llama checkpoint; `ster tune sft`
# and `ster tune reward` train a generation adapter and a reward model on it
# for one epoch. `ster generate --adapter` is then refused a reward model, an
# adapter trained for another checkpoint, and an adapter whose checkpoint was
# rewritten at another width, each with its declared sentence and exit 1;
# the adapter on its own checkpoint loads and answers first.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: tests/models/adapter.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/adapter-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
TRAIN="--rank 2 --alpha 4 --epochs 1 --learning-rate 0.001 --accumulation 1 --warmup-steps 0 --max-sequence 32 --batch-size 1 --seed 1"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
step() {
  "$BIN" "$@" >>"$REPORT" 2>&1 </dev/null || fail "ster $* did not complete"
}
generate() {
  expected=$1
  shift
  set +e
  out=$("$BIN" generate "$@" --prompt "describe the sea ." --max-new-tokens 1 \
    --temperature 0 --seed 1 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  # grep exits 1 when every line was a wait line; any other failure stops the run.
  err=$(grep -v '^czekam\|^koniec czekania' "$ROOT/stderr" || [ $? -eq 1 ])
  printf '$ ster generate %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster generate $* exited $status, expected $expected: $err"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }
# fixture NAME JQ: a fixture checkpoint at $ROOT/NAME, its config the
# example's through JQ.
fixture() {
  jq "$2" docs/examples/model/config.json >"$ROOT/config.json"
  step model fixture create --from "$ROOT/fixture.json" --output "$ROOT/$1"
}

cp docs/examples/model/fixture.json docs/examples/model/vocabulary.json "$ROOT/"
fixture base .
fixture other .
# shellcheck disable=SC2086
step tune sft --model "$ROOT/base" --examples docs/examples/tuning/examples.json \
  --output "$ROOT/sft.lora.safetensors" $TRAIN
# shellcheck disable=SC2086
step tune reward --model "$ROOT/base" --pairs docs/examples/pairs.json \
  --output "$ROOT/reward.lora.safetensors" $TRAIN
generate 0 --model "$ROOT/base" --adapter "$ROOT/sft.lora.safetensors"
passed "the adapter loads on the checkpoint it was trained for"

a_reward_model_given_as_a_generation_adapter_is_refused() {
  generate 1 --model "$ROOT/base" --adapter "$ROOT/reward.lora.safetensors"
  said "Error: attaching artifact $ROOT/reward.lora.safetensors to $ROOT/base: adapter artifact is a reward model, not a generation adapter; observed: kind: reward"
  passed a_reward_model_given_as_a_generation_adapter_is_refused
}

an_adapter_for_another_model_is_refused() {
  generate 1 --model "$ROOT/other" --adapter "$ROOT/sft.lora.safetensors"
  said "adapter was trained for model \"$ROOT/base\", current model is \"$ROOT/other\""
  passed an_adapter_for_another_model_is_refused
}

an_adapter_of_another_width_is_refused() {
  rm -r "$ROOT/base"
  fixture base '.hidden_size = 32 | .intermediate_size = 64 | .head_dim = 8'
  generate 1 --model "$ROOT/base" --adapter "$ROOT/sft.lora.safetensors"
  said "Error: attaching artifact $ROOT/sft.lora.safetensors to $ROOT/base: adapter width 64 does not match model width 32"
  passed an_adapter_of_another_width_is_refused
}

a_reward_model_given_as_a_generation_adapter_is_refused
an_adapter_for_another_model_is_refused
an_adapter_of_another_width_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
