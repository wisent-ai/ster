#!/bin/sh
# Real test of how `ster model fixture create` fails (`ster.model.fixture`,
# docs/guide/models/fixtures.md): copies of the example declaration, config
# and tokenizer declaration, each changed in one place with jq, under the
# run's own HOME. Each refusal names its declared cause; nothing is written to
# an output a refusal names, and an output that existed is left as it was.
#
# Usage: tests/models/fixture.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/fixture-$RUN"
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
# create EXPECTED DECLARATION OUTPUT: run the command; $err is its stderr.
create() {
  set +e
  "$BIN" model fixture create --from "$2" --output "$3" >"$ROOT/stdout" 2>"$ROOT/stderr" </dev/null
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ ster model fixture create --from %s --output %s\nexit: %s\nstderr: %s\n\n' "$2" "$3" "$status" "$err" >>"$REPORT"
  [ "$status" -eq "$1" ] || fail "exited $status, expected $1: $err"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }
# inputs NAME [DECLARATION-JQ [CONFIG-JQ]]: a copy of the example inputs at
# $ROOT/NAME, changed by the jq programs; $FROM is its declaration.
inputs() {
  DIR="$ROOT/$1"
  mkdir -p "$DIR"
  cp docs/examples/model/vocabulary.json "$DIR/"
  jq "${2:-.}" docs/examples/model/fixture.json >"$DIR/fixture.json"
  jq "${3:-.}" docs/examples/model/config.json >"$DIR/config.json"
  FROM="$DIR/fixture.json"
}

inputs base
create 0 "$FROM" "$ROOT/base/out"
passed "the example declaration writes a checkpoint Ster loads"

an_input_that_does_not_exist_is_refused() {
  create 2 "$ROOT/nowhere.json" "$ROOT/missing-out"
  said "reading fixture declaration $ROOT/nowhere.json: no file is at $ROOT/nowhere.json; observed: path: $ROOT/nowhere.json (absent)"
  inputs no-config '.config = "absent.json"'
  create 2 "$FROM" "$DIR/out"
  said "reading fixture config $DIR/absent.json: no file is at $DIR/absent.json"
  [ ! -e "$DIR/out" ] || fail "a refused fixture left $DIR/out"
  passed an_input_that_does_not_exist_is_refused
}

an_unreadable_input_is_refused() {
  inputs closed
  chmod 000 "$DIR/config.json"
  create 1 "$FROM" "$DIR/out"
  chmod 644 "$DIR/config.json"
  said "reading fixture config $DIR/config.json: the file cannot be read: this account may not read it, or it is no regular file; observed: path: $DIR/config.json (a file, "
  said "mode 000"
  passed an_unreadable_input_is_refused
}

an_input_that_is_not_json_is_refused() {
  inputs not-json
  echo 'nope' >"$FROM"
  create 1 "$FROM" "$DIR/out"
  said "reading fixture declaration $FROM: it is not JSON; observed: the JSON reader stopped at line 1, column 2: "
  passed an_input_that_is_not_json_is_refused
}

an_input_of_another_shape_is_refused() {
  inputs shape '.' 'del(.hidden_size)'
  create 1 "$FROM" "$DIR/out"
  said "reading fixture config $DIR/config.json: it is JSON, and a field the fixture config requires is missing"
  said "missing field \`hidden_size\`"
  passed an_input_of_another_shape_is_refused
}

a_config_of_another_family_is_refused() {
  inputs family '.' '.model_type = "mistral"'
  create 1 "$FROM" "$DIR/out"
  said "checking fixture config $DIR/config.json: the fixture tensor layout requires model_type=llama, and the config declares \"mistral\""
  passed a_config_of_another_family_is_refused
}

a_feed_forward_bias_is_refused() {
  inputs mlp-bias '.' '.mlp_bias = true'
  create 1 "$FROM" "$DIR/out"
  said "the config declares mlp_bias=true, but Ster's Llama consumer does not apply feed-forward biases; no checkpoint was written"
  passed a_feed_forward_bias_is_refused
}

heads_that_do_not_group_are_refused() {
  inputs heads '.' '.num_key_value_heads = 3'
  create 1 "$FROM" "$DIR/out"
  said "the 4 attention heads do not split into whole groups over the 3 key/value heads"
  said "remainder 1"
  passed heads_that_do_not_group_are_refused
}

a_setting_that_is_not_finite_is_refused() {
  inputs infinite '.initialization.normalization = 1e39'
  create 1 "$FROM" "$DIR/out"
  said "checking fixture declaration $FROM: initialization.normalization must be a finite number, and it is inf"
  passed a_setting_that_is_not_finite_is_refused
}

a_negative_scale_is_refused() {
  inputs negative '.initialization.projection.scale = -1'
  create 1 "$FROM" "$DIR/out"
  said "drawing fixture initialization.projection from its normal distribution: its scale, a standard deviation, is negative"
  passed a_negative_scale_is_refused
}

a_scale_past_f32_is_refused() {
  inputs huge '.initialization.embedding.scale = 3e38'
  create 1 "$FROM" "$DIR/out"
  said "building fixture tensor model.embed_tokens.weight of shape [67, 64]: the normal distribution it is drawn from gave a value past what f32 holds"
  passed a_scale_past_f32_is_refused
}

an_attention_bias_without_its_initialization_is_refused() {
  inputs bias '.' '.attention_bias = true'
  create 1 "$FROM" "$DIR/out"
  said "initialization.bias is required because the config declares attention_bias=true, and the declaration gives none"
  passed an_attention_bias_without_its_initialization_is_refused
}

a_blank_verification_prompt_is_refused() {
  inputs prompt '.verification.prompt = "  "'
  create 1 "$FROM" "$DIR/out"
  said "verification.prompt is required: the checkpoint is verified by forwarding it, and it is blank; observed: verification.prompt: \"  \""
  passed a_blank_verification_prompt_is_refused
}

no_verification_layer_is_refused() {
  inputs no-layers '.verification.layers = []'
  create 1 "$FROM" "$DIR/out"
  said "verification.layers is required: the checkpoint is verified at those layers, and it lists none; observed: verification.layers: []; the config declares 4 layers"
  passed no_verification_layer_is_refused
}

a_verification_layer_past_the_model_is_refused() {
  inputs layers '.verification.layers = [0, 9]'
  create 1 "$FROM" "$DIR/out"
  said "verification layer 9 lies outside 4 declared layers; observed: verification.layers: [0, 9]; num_hidden_layers: 4"
  passed a_verification_layer_past_the_model_is_refused
}

a_tokenizer_the_library_does_not_read_is_refused() {
  inputs bad-tokenizer '.tokenizer = "tokenizer.json"'
  echo '{}' >"$DIR/tokenizer.json"
  create 1 "$FROM" "$DIR/out"
  said "reading fixture tokenizer $DIR/tokenizer.json: the tokenizers library does not read it as a tokenizer.json"
  passed a_tokenizer_the_library_does_not_read_is_refused
}

a_token_past_the_vocabulary_is_refused() {
  inputs narrow '.tokenizer = "tokenizer.json"' '.vocab_size = 8'
  cp "$ROOT/base/out/tokenizer.json" "$DIR/tokenizer.json"
  create 1 "$FROM" "$DIR/out"
  said "outside config vocab_size 8, so the embedding would hold no row for it"
  passed a_token_past_the_vocabulary_is_refused
}

a_declared_tokenizer_past_the_vocabulary_is_refused() {
  inputs narrow-declared '.' '.vocab_size = 8'
  create 1 "$FROM" "$DIR/out"
  said "token identities, exceeding config vocab_size 8"
  passed a_declared_tokenizer_past_the_vocabulary_is_refused
}

a_width_past_the_host_is_refused() {
  inputs wide '.' '.head_dim = 4611686018427387904'
  create 1 "$FROM" "$DIR/out"
  said "head_dim 4611686018427387904 times 4 query heads overflows this host's 64-bit sizes"
  passed a_width_past_the_host_is_refused
}

an_output_named_in_bytes_that_are_not_text_is_refused() {
  create 2 "$ROOT/base/fixture.json" "$(printf "$ROOT/out-\377")"
  said "cannot be passed to Ster's UTF-8 checkpoint loader: its name holds bytes that are not UTF-8"
  passed an_output_named_in_bytes_that_are_not_text_is_refused
}

an_output_that_exists_is_refused() {
  create 2 "$ROOT/base/fixture.json" "$ROOT/base/out"
  said "fixture output $ROOT/base/out already exists; choose a new directory; existing files were not changed"
  [ -f "$ROOT/base/out/model.safetensors" ] || fail "the existing output was changed"
  passed an_output_that_exists_is_refused
}

an_output_in_a_closed_directory_is_refused() {
  mkdir -p "$ROOT/closed"
  chmod 000 "$ROOT/closed"
  create 1 "$ROOT/base/fixture.json" "$ROOT/closed/out"
  chmod 755 "$ROOT/closed"
  said "Ster cannot tell whether $ROOT/closed/out exists: this account may not look inside $ROOT/closed"
  passed an_output_in_a_closed_directory_is_refused
}

an_output_in_a_read_only_directory_is_refused() {
  mkdir -p "$ROOT/read-only"
  chmod 555 "$ROOT/read-only"
  create 1 "$ROOT/base/fixture.json" "$ROOT/read-only/out"
  chmod 755 "$ROOT/read-only"
  said "creating fixture output directory $ROOT/read-only/out: this account may not create $ROOT/read-only/out"
  passed an_output_in_a_read_only_directory_is_refused
}

an_output_below_a_file_is_refused() {
  : >"$ROOT/a-file"
  create 1 "$ROOT/base/fixture.json" "$ROOT/a-file/out"
  said "creating fixture output parent $ROOT/a-file: $ROOT/a-file cannot be created although this account may write beside it"
  passed an_output_below_a_file_is_refused
}

activations_that_overflow_are_refused() {
  inputs overflow '.initialization.projection.scale = 1e30'
  create 1 "$FROM" "$DIR/out"
  said "returned non-finite activation"
  said "the declared initialization scales grow the activations past what f32 holds"
  [ ! -e "$DIR/out" ] || fail "the refused fixture left $DIR/out"
  passed activations_that_overflow_are_refused
}

an_input_that_does_not_exist_is_refused
an_unreadable_input_is_refused
an_input_that_is_not_json_is_refused
an_input_of_another_shape_is_refused
a_config_of_another_family_is_refused
a_feed_forward_bias_is_refused
heads_that_do_not_group_are_refused
a_setting_that_is_not_finite_is_refused
a_negative_scale_is_refused
a_scale_past_f32_is_refused
an_attention_bias_without_its_initialization_is_refused
a_blank_verification_prompt_is_refused
no_verification_layer_is_refused
a_verification_layer_past_the_model_is_refused
a_tokenizer_the_library_does_not_read_is_refused
a_token_past_the_vocabulary_is_refused
a_declared_tokenizer_past_the_vocabulary_is_refused
a_width_past_the_host_is_refused
an_output_named_in_bytes_that_are_not_text_is_refused
an_output_that_exists_is_refused
an_output_in_a_closed_directory_is_refused
an_output_in_a_read_only_directory_is_refused
an_output_below_a_file_is_refused
activations_that_overflow_are_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
