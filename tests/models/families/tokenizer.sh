#!/bin/sh
# Real test of how reading a checkpoint's tokenizer fails
# (`ster.model.tokenizer`), through `ster generate` on copies of the
# four-layer fixture whose tokenizer is replaced: a tokenizer.json the
# tokenizers library cannot read, a PLaMo tokenizer.jsonl written from the
# fixture's own vocabulary and then broken, and a tiktoken tokenizer.model
# of two byte tokens. Every refusal exits 1, before any weight is mapped.
#
# Usage: tests/models/families/tokenizer.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/tokenizer-$RUN"
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
# copy NAME: a copy of the fixture at $ROOT/NAME without its tokenizer.json.
copy() {
  COPY="$ROOT/$1"
  cp -R "$ROOT/base" "$COPY"
  rm "$COPY/tokenizer.json"
}
# plamo NAME [CONFIG]: a copy reading the fixture's vocabulary as PLaMo's
# tokenizer.jsonl, one [piece, score, kind] row per token, with the
# tokenizer_config.json CONFIG when given.
plamo() {
  copy "$1"
  VOCABULARY="$COPY/tokenizer.jsonl"
  jq -c '.modelTokens | to_entries[] | [.value.content, (0 - .key),
    (if .value.content == "[UNK]" then "UNKNOWN" elif (.value.content | startswith("<")) then "CONTROL" else "NORMAL" end)]' \
    "$ROOT/vocabulary.json" >"$VOCABULARY"
  [ $# -lt 2 ] || printf '%s' "$2" >"$COPY/tokenizer_config.json"
}
# glm NAME CONFIG: a copy reading a tiktoken tokenizer.model of the bytes a
# and b, with the tokenizer_config.json CONFIG.
glm() {
  copy "$1"
  VOCABULARY="$COPY/tokenizer.model"
  printf 'YQ== 0\nYg== 1\n' >"$VOCABULARY"
  printf '%s' "$2" >"$COPY/tokenizer_config.json"
}

cp docs/examples/model/fixture.json docs/examples/model/config.json docs/examples/model/vocabulary.json "$ROOT/"
"$BIN" model fixture create --from "$ROOT/fixture.json" --output "$ROOT/base" >>"$REPORT" 2>&1 \
  || fail "the fixture could not be written"
plamo plamo
generate 0 "$COPY"
passed "the fixture's vocabulary read as PLaMo's tokenizer.jsonl loads and answers"

a_tokenizer_json_that_is_not_json_is_refused() {
  cp -R "$ROOT/base" "$ROOT/not-json"
  echo 'nope' >"$ROOT/not-json/tokenizer.json"
  generate 1 "$ROOT/not-json"
  said "Error: reading tokenizer $ROOT/not-json/tokenizer.json: the tokenizers library cannot read it: it is not JSON, so it is no tokenizer.json a tokenizer was saved as; observed: path: $ROOT/not-json/tokenizer.json (a file, 5 bytes, "
  said "the JSON reader stopped at line 1, column 2: "
  passed a_tokenizer_json_that_is_not_json_is_refused
}

a_tokenizer_file_cut_short_is_refused() {
  cp -R "$ROOT/base" "$ROOT/cut-short"
  printf '{"version": "1.0",' >"$ROOT/cut-short/tokenizer.json"
  generate 1 "$ROOT/cut-short"
  said "Error: reading tokenizer $ROOT/cut-short/tokenizer.json: it ends before its JSON value closes: the file was cut short, as a download that stopped or a disk that filled up leaves it; observed: path: $ROOT/cut-short/tokenizer.json (a file, 18 bytes, "
  said "the JSON reader ran out at line 1, column "
  passed a_tokenizer_file_cut_short_is_refused
}

a_tokenizer_json_that_does_not_load_is_refused() {
  cp -R "$ROOT/base" "$ROOT/empty-json"
  echo '{"version": "1.0"}' >"$ROOT/empty-json/tokenizer.json"
  generate 1 "$ROOT/empty-json"
  said "Error: reading tokenizer $ROOT/empty-json/tokenizer.json: the tokenizers library cannot read it: it is JSON, and the model, normalizer, pre-tokenizer, post-processor or decoder it describes is missing or not one the library knows; observed: path: $ROOT/empty-json/tokenizer.json; top-level keys: version; the library answered: "
  passed a_tokenizer_json_that_does_not_load_is_refused
}

an_unreadable_tokenizer_config_is_refused() {
  plamo closed-config '{}'
  chmod 000 "$COPY/tokenizer_config.json"
  generate 1 "$COPY"
  chmod 644 "$COPY/tokenizer_config.json"
  said "Error: reading tokenizer config $COPY/tokenizer_config.json: the file cannot be read: this account may not read it; observed: path: $COPY/tokenizer_config.json (a file, 2 bytes, mode 000, owner uid "
  said "the system answered: Permission denied"
  passed an_unreadable_tokenizer_config_is_refused
}

a_tokenizer_config_that_is_not_json_is_refused() {
  plamo not-json nope
  generate 1 "$COPY"
  said "Error: reading tokenizer config $COPY/tokenizer_config.json: it is not JSON; observed: path: $COPY/tokenizer_config.json (a file, 4 bytes, "
  said "the JSON reader stopped at line 1, column 1: expected value at line 1 column 1"
  passed a_tokenizer_config_that_is_not_json_is_refused
}

an_unreadable_vocabulary_is_refused() {
  plamo closed-vocabulary
  chmod 000 "$VOCABULARY"
  generate 1 "$COPY"
  chmod 644 "$VOCABULARY"
  said "Error: reading vocabulary $VOCABULARY: the file cannot be read: this account may not read it; observed: path: $VOCABULARY (a file, "
  passed an_unreadable_vocabulary_is_refused
}

a_vocabulary_that_is_not_text_is_refused() {
  plamo not-text
  printf '\377\376["x", 0, "NORMAL"]\n' >>"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: the file cannot be read as text: it holds bytes that are not UTF-8"
  passed a_vocabulary_that_is_not_text_is_refused
}

an_empty_vocabulary_is_refused() {
  plamo empty
  printf '\n\n' >"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: it holds no [piece, score, kind] row, so the tokenizer would have no token to spell any text with; observed: path: $VOCABULARY (a file, 2 bytes, "
  said "rows: 0"
  glm empty-glm '{"tokenizer_class": "ChatGLM4Tokenizer"}'
  : >"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: it holds no \`base64 rank\` row"
  passed an_empty_vocabulary_is_refused
}

a_vocabulary_row_that_is_not_json_is_refused() {
  plamo not-json-row
  echo 'piece x' >>"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: line 68 is not JSON, and every line of a PLaMo tokenizer.jsonl is one JSON [piece, score, kind] row"
  said "; observed: line 68: \"piece x\"; the JSON reader answered: "
  passed a_vocabulary_row_that_is_not_json_is_refused
}

a_vocabulary_row_without_its_score_is_refused() {
  plamo bad-row
  echo '{"piece": "x"}' >>"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: line 68 is JSON and not a [piece, score, kind] row: it holds no piece string followed by a score number"
  said "; first two entries: absent and absent"
  passed a_vocabulary_row_without_its_score_is_refused
}

a_vocabulary_row_without_a_space_is_refused() {
  glm spaceless '{"tokenizer_class": "ChatGLM4Tokenizer"}'
  printf 'Yw==\n' >>"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: line 3 holds no space, and a tiktoken vocabulary's row is a \`base64 rank\`"
  said "; observed: line 3: \"Yw==\"; 4 characters"
  passed a_vocabulary_row_without_a_space_is_refused
}

a_vocabulary_token_that_is_not_base64_is_refused() {
  glm not-base64 '{"tokenizer_class": "ChatGLM4Tokenizer"}'
  printf '!!! 2\n' >>"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: the token \"!!!\" of line 3 is not base64"
  said "; observed: line 3: \"!!! 2\"; the base64 reader answered: "
  passed a_vocabulary_token_that_is_not_base64_is_refused
}

a_vocabulary_rank_that_is_not_an_id_is_refused() {
  glm bad-rank '{"tokenizer_class": "ChatGLM4Tokenizer"}'
  printf 'Yw== c\n' >>"$VOCABULARY"
  generate 1 "$COPY"
  said "Error: reading vocabulary $VOCABULARY: the rank \"c\" of line 3 is not a token id: a tiktoken vocabulary ranks each token by a whole number from 0 to 4294967295"
  said "; observed: line 3: \"Yw== c\"; the number reader answered: "
  passed a_vocabulary_rank_that_is_not_an_id_is_refused
}

a_tiktoken_vocabulary_without_its_class_is_refused() {
  glm no-class '{}'
  rm "$COPY/tokenizer_config.json"
  generate 1 "$COPY"
  said "Error: choosing how to read vocabulary $VOCABULARY: its tokenizer config names no auto_map.AutoTokenizer or tokenizer_class string, so Ster cannot tell GLM-4's (ChatGLM4Tokenizer) from Kimi's (TikTokenTokenizer); observed: tokenizer config: absent"
  passed a_tiktoken_vocabulary_without_its_class_is_refused
}

a_vocabulary_of_another_class_is_refused() {
  glm llama '{"tokenizer_class": "LlamaTokenizer"}'
  generate 1 "$COPY"
  said "Error: choosing how to read vocabulary $VOCABULARY: it is a tiktoken or SentencePiece vocabulary for \"LlamaTokenizer\""
  passed a_vocabulary_of_another_class_is_refused
}

a_bos_token_the_config_does_not_name_is_refused() {
  plamo bos-unnamed '{"add_bos_token": true}'
  generate 1 "$COPY"
  said "Error: starting every sequence of $VOCABULARY with its BOS token: the tokenizer config sets add_bos_token and names no bos_token string; observed: bos_token: absent"
  passed a_bos_token_the_config_does_not_name_is_refused
}

a_bos_token_the_vocabulary_lacks_is_refused() {
  plamo bos-absent '{"add_bos_token": true, "bos_token": "<bos>"}'
  generate 1 "$COPY"
  said "Error: starting every sequence of $VOCABULARY with \"<bos>\": the vocabulary holds no token \"<bos>\""
  passed a_bos_token_the_vocabulary_lacks_is_refused
}

a_split_threshold_past_the_engine_is_refused() {
  plamo split '{"break_around_repeated_chars_threshold": 200000}'
  generate 1 "$COPY"
  said "Error: isolating the stretches break_around_repeated_chars_threshold names in $VOCABULARY: the tokenizer config's break_around_repeated_chars_threshold 200000 asks for the split"
  said "a repetition count that large is past its limit; observed: break_around_repeated_chars_threshold: 200000; the engine answered: "
  said "too big number for repeat range"
  passed a_split_threshold_past_the_engine_is_refused
}

a_special_token_past_the_reserved_ids_is_refused() {
  glm special '{"tokenizer_class": "ChatGLM4Tokenizer", "added_tokens_decoder": {"999": {"content": "[gMASK]"}}}'
  generate 1 "$COPY"
  said "Error: adding the special tokens of $VOCABULARY: the tokenizer config numbers a special token 999, outside the 1 ids after the vocabulary's 2 ranks"
  # Kimi's class reserves 256 ids, and its split pattern is compiled before
  # the special tokens are added, so this also shows KIMI_PATTERN compiles.
  glm special-kimi '{"tokenizer_class": "TikTokenTokenizer", "added_tokens_decoder": {"999": {"content": "<|im_end|>"}}}'
  generate 1 "$COPY"
  said "Error: adding the special tokens of $VOCABULARY: the tokenizer config numbers a special token 999, outside the 256 ids after the vocabulary's 2 ranks"
  passed a_special_token_past_the_reserved_ids_is_refused
}

added_tokens_that_are_not_an_object_are_refused() {
  glm added-list '{"tokenizer_class": "ChatGLM4Tokenizer", "added_tokens_decoder": []}'
  generate 1 "$COPY"
  said "Error: adding the special tokens of $VOCABULARY: the tokenizer config's added_tokens_decoder is not an object of id to token"
  said "; observed: added_tokens_decoder: []; config: $COPY/tokenizer_config.json"
  passed added_tokens_that_are_not_an_object_are_refused
}

an_added_token_keyed_by_no_id_is_refused() {
  glm added-key '{"tokenizer_class": "ChatGLM4Tokenizer", "added_tokens_decoder": {"x": {"content": "[gMASK]"}}}'
  generate 1 "$COPY"
  said "the tokenizer config's added_tokens_decoder keys an entry \"x\", which is not a token id"
  said "; observed: entry \"x\": {\"content\":\"[gMASK]\"}; config: $COPY/tokenizer_config.json"
  passed an_added_token_keyed_by_no_id_is_refused
}

an_added_token_without_its_content_is_refused() {
  glm added-content '{"tokenizer_class": "ChatGLM4Tokenizer", "added_tokens_decoder": {"2": {"special": true}}}'
  generate 1 "$COPY"
  said "entry \"2\" of the tokenizer config's added_tokens_decoder has no \`content\` string"
  said "; observed: entry \"2\": {\"special\":true}; config: $COPY/tokenizer_config.json"
  passed an_added_token_without_its_content_is_refused
}

a_tokenizer_json_that_is_not_json_is_refused
a_tokenizer_file_cut_short_is_refused
a_tokenizer_json_that_does_not_load_is_refused
an_unreadable_tokenizer_config_is_refused
a_tokenizer_config_that_is_not_json_is_refused
an_unreadable_vocabulary_is_refused
a_vocabulary_that_is_not_text_is_refused
an_empty_vocabulary_is_refused
a_vocabulary_row_that_is_not_json_is_refused
a_vocabulary_row_without_its_score_is_refused
a_vocabulary_row_without_a_space_is_refused
a_vocabulary_token_that_is_not_base64_is_refused
a_vocabulary_rank_that_is_not_an_id_is_refused
a_tiktoken_vocabulary_without_its_class_is_refused
a_vocabulary_of_another_class_is_refused
a_bos_token_the_config_does_not_name_is_refused
a_bos_token_the_vocabulary_lacks_is_refused
a_split_threshold_past_the_engine_is_refused
a_special_token_past_the_reserved_ids_is_refused
added_tokens_that_are_not_an_object_are_refused
an_added_token_keyed_by_no_id_is_refused
an_added_token_without_its_content_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
