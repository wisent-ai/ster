#!/bin/sh
# Real test of how loading a model fails (`ster.model.checkpoint`,
# `ster.model.config`, `ster.model.weights`), through the built binary.
#
# `ster model fixture create` writes a real four-layer Llama checkpoint from
# docs/examples/model/fixture.json under this run's directory; each test
# copies it, breaks one thing (a file removed, a config key changed, a
# weight file replaced) and asks `ster generate` to load the copy. Each
# refusal is checked for its declared sentence and exit status 1, and the
# unbroken fixture is checked to load and answer first.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: tests/models/load.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/load-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
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
  out=$("$BIN" generate --model "$model" --prompt "describe the sea ." --max-new-tokens 1 \
    --temperature 0 --seed 1 2>"$ROOT/stderr" </dev/null)
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
# copy NAME: a fresh copy of the fixture at $ROOT/NAME, its path in $COPY.
copy() {
  COPY="$ROOT/$1"
  cp -R "$ROOT/base" "$COPY"
}
# config NAME JQ: a copy whose config.json is the fixture's through JQ.
config() {
  copy "$1"
  jq "$2" "$ROOT/base/config.json" >"$COPY/config.json"
}

cp docs/examples/model/fixture.json docs/examples/model/config.json docs/examples/model/vocabulary.json "$ROOT/"
"$BIN" model fixture create --from "$ROOT/fixture.json" --output "$ROOT/base" >>"$REPORT" 2>&1 \
  || fail "the fixture could not be written"
generate 0 "$ROOT/base"
passed "the unbroken fixture loads and answers"

a_local_path_that_is_no_directory_is_refused() {
  generate 1 "$ROOT/no-such-checkpoint"
  said "Error: finding model $ROOT/no-such-checkpoint: it names a local path (it starts with /, ./, ../ or ~/) and no checkpoint directory is there: nothing is there"
  case "$err" in *huggingface.co*) fail "a local path was asked of the Hub: $err" ;; esac
  generate 1 "$ROOT/fixture.json"
  said "no checkpoint directory is there: it is not a directory"
  passed a_local_path_that_is_no_directory_is_refused
}

an_unreadable_directory_is_refused() {
  copy closed
  chmod 000 "$COPY"
  generate 1 "$COPY"
  chmod 755 "$COPY"
  said "Error: reading the checkpoint directory $COPY: this account may not read it: the file's permission bits or owner keep this account out; observed: path: $COPY (a directory, "
  said ", mode 000, owner uid $(id -u)); the system answered: "
  passed an_unreadable_directory_is_refused
}

a_directory_without_a_config_is_refused() {
  copy no-config
  rm "$COPY/config.json"
  generate 1 "$COPY"
  said "Error: finding the config of checkpoint $COPY: no config.json is there; observed: path: $COPY/config.json"
  passed a_directory_without_a_config_is_refused
}

a_directory_without_a_tokenizer_is_refused() {
  copy no-tokenizer
  rm "$COPY/tokenizer.json"
  generate 1 "$COPY"
  said "Error: finding the tokenizer of checkpoint $COPY: $COPY publishes no tokenizer: none of tokenizer.json, tokenizer.jsonl, tokenizer.model, tiktoken.model is there"
  passed a_directory_without_a_tokenizer_is_refused
}

a_directory_without_weights_is_refused() {
  copy no-weights
  rm "$COPY/model.safetensors"
  generate 1 "$COPY"
  said "Error: finding the weights of checkpoint $COPY: $COPY holds no *.safetensors weights; observed: layout: Transformers (config.json)"
  passed a_directory_without_weights_is_refused
}

an_unreadable_config_is_refused() {
  copy closed-config
  chmod 000 "$COPY/config.json"
  generate 1 "$COPY"
  chmod 644 "$COPY/config.json"
  said "Error: reading the model config $COPY/config.json: this account may not read it: the file's permission bits or owner keep this account out; observed: path: $COPY/config.json (a file, "
  passed an_unreadable_config_is_refused
}

a_config_that_is_not_json_is_refused() {
  copy not-json
  echo '{' >"$COPY/config.json"
  generate 1 "$COPY"
  said "Error: reading the model config $COPY/config.json: it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway); observed: path: $COPY/config.json (a file, 2 bytes, "
  said ": EOF while parsing"
  echo '[1]' >"$COPY/config.json"
  generate 1 "$COPY"
  said "Error: reading model config $COPY/config.json: it is JSON, and not one object of settings: a config states each setting under its key; observed: path: $COPY/config.json; it holds [1]"
  passed a_config_that_is_not_json_is_refused
}

a_config_naming_no_model_type_is_refused() {
  config no-type 'del(.model_type)'
  generate 1 "$COPY"
  said "Error: reading the model type of $COPY/config.json: it states no model_type, and its architectures (absent) name no model class Ster reads in its place"
  said "fix: use a Hugging Face checkpoint whose model_type is one of llama, "
  passed a_config_naming_no_model_type_is_refused
}

a_model_type_that_is_not_text_is_refused() {
  config type-number '.model_type = 3'
  generate 1 "$COPY"
  said "Error: reading the model type of $COPY/config.json: it states model_type as 3, which is not a string"
  passed a_model_type_that_is_not_text_is_refused
}

# T5, an encoder-decoder, is one of the model types Ster does not run: its
# config is refused before any weight is read, as any unknown type is.
an_unknown_model_type_is_refused() {
  config unknown-type '.model_type = "nope"'
  generate 1 "$COPY"
  said "Error: choosing the decoder for $COPY/config.json: model architecture \"nope\" is unsupported by this Ster build"
  config t5 '.model_type = "t5" | .architectures = ["T5ForConditionalGeneration"]'
  generate 1 "$COPY"
  said "Error: choosing the decoder for $COPY/config.json: model architecture \"t5\" is unsupported by this Ster build"
  said "observed: model_type: \"t5\""
  passed an_unknown_model_type_is_refused
}

a_quantized_checkpoint_is_refused() {
  config quantized '.quantization_config = {"bits": 4}'
  generate 1 "$COPY"
  said "is a quantized checkpoint (it declares quantization_config); Ster maps unquantized safetensors only; observed: quantization_config: {\"bits\":4}; fix: use the checkpoint it was quantized from"
  passed a_quantized_checkpoint_is_refused
}

a_config_without_a_required_key_is_refused() {
  config no-width 'del(.hidden_size)'
  generate 1 "$COPY"
  said "Error: reading the llama decoder's hidden_size from $COPY/config.json: $COPY/config.json states no hidden_size (nor n_embd, n_embed, d_model); Ster needs the llama decoder's hidden_size (a whole number)"
  passed a_config_without_a_required_key_is_refused
}

# A key Candle needs and a key the family reads with a default are both
# refused when stated with the wrong type; only an absent key takes the
# default.
a_key_of_the_wrong_kind_is_refused() {
  config width-text '.hidden_size = "x"'
  generate 1 "$COPY"
  said "Error: reading hidden_size from $COPY/config.json: $COPY/config.json states hidden_size as \"x\", which is not a whole number; observed: hidden_size: \"x\"; fix: state hidden_size as a whole number, the type the publisher's config class reads it as"
  config bias-text '.attention_bias = "yes"'
  generate 1 "$COPY"
  said "Error: reading attention_bias from $COPY/config.json: $COPY/config.json states attention_bias as \"yes\", which is not true or false; observed: attention_bias: \"yes\"; fix: state attention_bias as true or false"
  config rope-type-number '.rope_scaling = {"rope_type": 3, "factor": 2.0}'
  generate 1 "$COPY"
  said "Error: reading rope_scaling.rope_type from $COPY/config.json: $COPY/config.json states rope_scaling.rope_type as 3, which is not a string"
  passed a_key_of_the_wrong_kind_is_refused
}

a_config_candle_cannot_read_is_refused() {
  config theta-text '.rope_theta = "x"'
  generate 1 "$COPY"
  said "Error: reading the llama decoder's settings from $COPY/config.json: Candle's Llama config does not read it: invalid type: string \"x\""
  passed a_config_candle_cannot_read_is_refused
}

a_setting_ster_does_not_implement_is_refused() {
  config activation '.hidden_act = "tanh"'
  generate 1 "$COPY"
  said "Error: reading the decoder's settings from the checkpoint's config: the config states a variant of a setting "
  said "that Ster's decoder for its family does not implement"
  said "; observed: $COPY/config.json declares hidden_act \"tanh\" for its llama feed-forward; Ster implements silu"
  said "; the config states hidden_act: \"tanh\", hidden_activation: absent, activation_function: absent, mlp_hidden_act: absent; fix: load a checkpoint of a variant Ster implements"
  passed a_setting_ster_does_not_implement_is_refused
}

two_gguf_files_are_refused() {
  copy two-gguf
  rm "$COPY/model.safetensors"
  : >"$COPY/a.gguf"
  : >"$COPY/b.gguf"
  generate 1 "$COPY"
  said "Error: mapping the GGUF weights of the checkpoint: a GGUF checkpoint is one .gguf file; this one lists 2"
  passed two_gguf_files_are_refused
}

a_file_that_is_not_gguf_is_refused() {
  copy not-gguf
  rm "$COPY/model.safetensors"
  printf 'not a gguf file\n' >"$COPY/a.gguf"
  generate 1 "$COPY"
  said "Error: reading GGUF weights $COPY/a.gguf: $COPY/a.gguf is not a GGUF file: its header does not open with the GGUF magic, or names a version or value type Candle does not read; observed: path: $COPY/a.gguf (a file, 16 bytes, "
  passed a_file_that_is_not_gguf_is_refused
}

# The GGUF magic and version 3, and nothing after them.
a_gguf_file_cut_short_is_refused() {
  copy cut-gguf
  rm "$COPY/model.safetensors"
  printf 'GGUF\003\000\000\000' >"$COPY/a.gguf"
  generate 1 "$COPY"
  said "Error: reading GGUF weights $COPY/a.gguf: $COPY/a.gguf ends before its header does: it was cut short, as a download that stopped leaves it; observed: path: $COPY/a.gguf (a file, 8 bytes, "
  passed a_gguf_file_cut_short_is_refused
}

a_weight_file_that_is_not_safetensors_is_refused() {
  copy not-safetensors
  echo junk >"$COPY/model.safetensors"
  generate 1 "$COPY"
  said "Error: reading a weight file $COPY/model.safetensors: $COPY/model.safetensors is not a safetensors file: its header or its table of tensors does not parse"
  passed a_weight_file_that_is_not_safetensors_is_refused
}

weights_that_do_not_fit_the_config_are_refused() {
  config narrow '.hidden_size = 16'
  generate 1 "$COPY"
  said "Error: building the decoder of $COPY from its weights: a tensor the checkpoint stores has another shape than the decoder its config describes needs"
  said "; observed: Candle: shape mismatch for model.embed_tokens.weight; the config sizes it [67, 16], the checkpoint stores [67, 64]; fix: "
  passed weights_that_do_not_fit_the_config_are_refused
}

weights_without_a_tensor_the_config_needs_are_refused() {
  config deeper '.num_hidden_layers = 5'
  generate 1 "$COPY"
  said "Error: building the decoder of $COPY from its weights: the decoder its config describes needs a tensor the checkpoint does not store"
  said "; observed: tensor model.layers.4."
  said ": the checkpoint holds no tensor of that name; fix: "
  passed weights_without_a_tensor_the_config_needs_are_refused
}

a_local_path_that_is_no_directory_is_refused
an_unreadable_directory_is_refused
a_directory_without_a_config_is_refused
a_directory_without_a_tokenizer_is_refused
a_directory_without_weights_is_refused
an_unreadable_config_is_refused
a_config_that_is_not_json_is_refused
a_config_naming_no_model_type_is_refused
a_model_type_that_is_not_text_is_refused
an_unknown_model_type_is_refused
a_quantized_checkpoint_is_refused
a_config_without_a_required_key_is_refused
a_key_of_the_wrong_kind_is_refused
a_config_candle_cannot_read_is_refused
a_setting_ster_does_not_implement_is_refused
two_gguf_files_are_refused
a_file_that_is_not_gguf_is_refused
a_gguf_file_cut_short_is_refused
a_weight_file_that_is_not_safetensors_is_refused
weights_that_do_not_fit_the_config_are_refused
weights_without_a_tensor_the_config_needs_are_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
