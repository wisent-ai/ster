#!/bin/sh
# Real test of the refusals of placing LoRA adapters on a model whose family
# cannot hold them (`ster.model.adapters`), through the built binary and
# tiny random checkpoints downloaded from the Hugging Face Hub into this
# run's cache: GPT-2, whose feed-forward has no gate; Mixtral, whose
# feed-forward is experts; DeepSeek-V3, whose attention is latent; Mamba,
# whose blocks have no attention or feed-forward projection at all; and
# Qwen3-Next and MiniMax, whose blocks include linear-attention mixers.
# `ster tune sft` is refused each before anything is trained, and no
# artifact is written.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: tests/models/loading/targets.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/targets-$RUN"
mkdir -p "$ROOT/home"
REPORT="$ROOT/report.txt"
OUT="$ROOT/adapter.lora.safetensors"
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
TRAIN="--rank 2 --alpha 4 --epochs 1 --learning-rate 0.001 --accumulation 1 --warmup-steps 0 --max-sequence 32 --batch-size 1 --seed 1"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
# sft EXPECTED MODEL TARGETS: train on MODEL with TARGETS, expecting EXPECTED.
sft() {
  set +e
  # shellcheck disable=SC2086
  out=$("$BIN" tune sft --model "$2" --targets "$3" --examples docs/examples/tuning/examples.json \
    --output "$OUT" $TRAIN 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  # grep exits 1 when every line was a wait line; any other failure stops the run.
  err=$(grep -v '^czekam\|^koniec czekania' "$ROOT/stderr" || [ $? -eq 1 ])
  printf '$ ster tune sft --model %s --targets %s\nexit: %s\nstderr: %s\n\n' "$2" "$3" "$status" "$err" >>"$REPORT"
  [ "$status" -eq "$1" ] || fail "tune sft on $2 exited $status, expected $1: $err"
  [ ! -e "$OUT" ] || fail "a refused run wrote $OUT"
}
said() {
  case "$err" in
    *"$1"*) echo "ok: $1" >>"$REPORT" ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}

a_target_the_model_lacks_is_refused() {
  sft 2 hf-internal-testing/tiny-random-gpt2 gate
  said "Error: placing LoRA adapters on hf-internal-testing/tiny-random-gpt2: this model has no single gate projection to adapt because its feed-forward has no gate; observed: targets this model has: query, key, value, output, up, down; targets asked for: gate; fix: choose adapter targets among query, key, value, output, up, down with --targets"
}

a_feed_forward_target_of_experts_is_refused() {
  sft 2 hf-internal-testing/tiny-random-MixtralForCausalLM gate
  said "Error: placing LoRA adapters on hf-internal-testing/tiny-random-MixtralForCausalLM: this model has no single gate projection to adapt because its feed-forward is a mixture of experts"
}

a_latent_attention_target_is_refused() {
  sft 2 hf-internal-testing/tiny-random-DeepseekV3ForCausalLM query
  said "Error: placing LoRA adapters on hf-internal-testing/tiny-random-DeepseekV3ForCausalLM: this model has no single query projection to adapt because its attention is latent"
}

a_model_that_trains_no_adapters_is_refused() {
  sft 1 hf-internal-testing/tiny-random-MambaForCausalLM query
  said "Error: placing LoRA adapters on hf-internal-testing/tiny-random-MambaForCausalLM: this model's blocks are state-space mixers (Mamba) with no attention or feed-forward projection to adapt; Ster steers it but trains no adapters on it"
  said "; targets asked for: query; fix: steer this model with a steering vector (\`ster vector train --model hf-internal-testing/tiny-random-MambaForCausalLM\`)"
}

a_delta_rule_model_is_refused() {
  sft 1 hf-internal-testing/tiny-random-Qwen3NextForCausalLM query
  said "this model's blocks include gated delta-rule mixers (Qwen3-Next's and Kimi-Linear's linear attention)"
}

a_lightning_attention_model_is_refused() {
  sft 1 hf-internal-testing/tiny-random-MiniMaxForCausalLM query
  said "this model's blocks include lightning-attention mixers (MiniMax-Text-01's and Ling 2.5's linear attention)"
}

a_target_the_model_lacks_is_refused
a_feed_forward_target_of_experts_is_refused
a_latent_attention_target_is_refused
a_model_that_trains_no_adapters_is_refused
a_delta_rule_model_is_refused
a_lightning_attention_model_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
