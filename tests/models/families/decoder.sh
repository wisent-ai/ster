#!/bin/sh
# Real test of how building a family's own blocks fails (`ster.model.decoder`)
# through the built binary.
#
# The checkpoint is the tiny random DeepSeek-V4 `yujiepan/deepseek-v4-bf16-tiny-random`,
# downloaded from the Hub into this run's own HF_HOME; its first three layers
# route by token id through `mlp.gate.tid2eid`, an I64 table of one row of
# `num_experts_per_tok` experts per token. The unbroken copy is checked to load
# and answer first; then the test writes a copy of the weights whose first
# routing table sends token 0 to expert 256, one past the 256 experts the
# layer stores, and asks `ster generate` to load it.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: tests/models/families/decoder.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/decoder-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
# The binary runs under this run's own home and Hub cache, never the operator's.
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
generate() {
  expected=$1
  model=$2
  set +e
  out=$("$BIN" generate --model "$model" --prompt "the sea" --max-new-tokens 1 \
    --temperature 0 --seed 1 --chat-template off 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  # grep exits 1 when every line was a wait line; any other failure stops the run.
  err=$(grep -v '^czekam\|^koniec czekania\|^blad czekania' "$ROOT/stderr" || [ $? -eq 1 ])
  printf '$ ster generate --model %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$model" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "loading $model exited $status, expected $expected: $err"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }

MODEL=yujiepan/deepseek-v4-bf16-tiny-random
generate 0 "$MODEL"
passed "the unbroken checkpoint loads and answers"
SNAPSHOT=
for dir in "$HF_HOME"/hub/models--yujiepan--deepseek-v4-bf16-tiny-random/snapshots/*/; do
  SNAPSHOT=$dir
done
[ -f "$SNAPSHOT/model.safetensors" ] || fail "the Hub download left no model.safetensors under $SNAPSHOT"

a_routing_table_naming_an_expert_past_the_layer_is_refused() {
  copy="$ROOT/route-outside"
  mkdir -p "$copy"
  for file in config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
    cp -L "$SNAPSHOT/$file" "$copy/$file"
  done
  cp -L "$SNAPSHOT/model.safetensors" "$copy/model.safetensors"
  # A safetensors file is an 8-byte little-endian header length, the JSON
  # header, then the tensors at the offsets the header names.
  header=$(od -An -t u8 -N8 "$copy/model.safetensors" | tr -d ' ')
  entry=$(dd if="$copy/model.safetensors" bs=1 skip=8 count="$header" 2>/dev/null \
    | jq -c '."model.layers.0.mlp.gate.tid2eid"')
  [ "$(printf '%s' "$entry" | jq -r .dtype)" = I64 ] \
    || fail "layer 0's routing table is not the I64 table this test rewrites: $entry"
  experts=$(jq -r .n_routed_experts "$copy/config.json")
  [ "$experts" = 256 ] || fail "the checkpoint stores $experts experts, and this test writes expert 256"
  start=$(printf '%s' "$entry" | jq -r '.data_offsets[0]')
  # Token 0's first expert becomes 256: 0x100 as eight little-endian bytes.
  printf '\000\001\000\000\000\000\000\000' \
    | dd of="$copy/model.safetensors" bs=1 seek=$((8 + header + start)) conv=notrunc 2>/dev/null
  generate 1 "$copy"
  said "reading the hash routing table of layer 0"
  said "layer 0's mlp.gate.tid2eid sends token 0 to expert 256, and the layer holds experts 0 to 256 exclusive"
  said "tid2eid row 0: expert 256; experts: 256"
  passed a_routing_table_naming_an_expert_past_the_layer_is_refused
}

a_routing_table_naming_an_expert_past_the_layer_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
