#!/bin/sh
# Real test of how a checkpoint's chat template fails (`ster.chat.template`),
# through `ster generate` and `ster tune sft` on copies of the four-layer
# fixture whose tokenizer_config.json is docs/examples/chat-template's
# through jq, or which carry a chat_template.jinja beside it. A template is
# read when the model loads and refused only by a run that applies it.
#
# Usage: tests/chat/template.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/chat/template-$RUN"
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
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  # grep exits 1 when every line was a wait line; any other failure stops the run.
  err=$(grep -v '^czekam\|^koniec czekania\|^blad czekania' "$ROOT/stderr" || [ $? -eq 1 ])
  printf '$ ster %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster $* exited $status, expected $expected: $err"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }
generate() {
  expected=$1
  model=$2
  shift 2
  run "$expected" generate --model "$model" --prompt "describe the sea ." --max-new-tokens 1 --temperature 0 --seed 1 "$@"
}
# templated NAME JQ: a copy of the fixture at $ROOT/NAME whose
# tokenizer_config.json is the example's through JQ.
templated() {
  COPY="$ROOT/$1"
  cp -R "$ROOT/base" "$COPY"
  jq "$2" docs/examples/chat-template/tokenizer_config.json >"$COPY/tokenizer_config.json"
}

cp docs/examples/model/fixture.json docs/examples/model/config.json docs/examples/model/vocabulary.json "$ROOT/"
"$BIN" model fixture create --from "$ROOT/fixture.json" --output "$ROOT/base" >>"$REPORT" 2>&1 \
  || fail "the fixture could not be written"
templated chat '.'
generate 0 "$COPY"
passed "the example template is applied and the model answers"

an_unknown_mode_is_refused() {
  generate 2 "$ROOT/chat" --chat-template sometimes
  said "unknown chat template mode \"sometimes\"; expected auto or off"
  passed an_unknown_mode_is_refused
}

an_unreadable_template_is_refused() {
  templated closed '.'
  printf '{{ messages[0].content }}' >"$COPY/chat_template.jinja"
  chmod 000 "$COPY/chat_template.jinja"
  generate 1 "$COPY"
  said "Error: reading chat template $COPY/chat_template.jinja: the file cannot be read: this account may not read it; observed: path: $COPY/chat_template.jinja (a file, "
  said "mode 000"
  generate 0 "$COPY" --chat-template off
  chmod 644 "$COPY/chat_template.jinja"
  passed an_unreadable_template_is_refused
}

a_template_of_no_known_form_is_refused() {
  templated number '.chat_template = 3'
  generate 1 "$COPY"
  said "Error: reading the chat_template of the tokenizer config: it is neither a template nor a list of them; observed: chat_template: 3; config: $COPY/tokenizer_config.json"
  passed a_template_of_no_known_form_is_refused
}

a_template_list_without_a_default_is_refused() {
  templated named '.chat_template = [{"name": "tool_use", "template": "x"}, {"name": "rag", "template": "y"}]'
  generate 1 "$COPY"
  said "it lists several named templates and none of them is the default"
  said "observed: names listed: \"tool_use\", \"rag\"; config: $COPY/tokenizer_config.json"
  passed a_template_list_without_a_default_is_refused
}

a_template_entry_without_text_is_refused() {
  templated entry '.chat_template = [{"name": "default"}]'
  generate 1 "$COPY"
  said "Error: reading the chat_template of the tokenizer config: its chosen entry \"default\" holds no template string; observed: entry: {\"name\":\"default\"}"
  generate 0 "$COPY" --chat-template off
  passed a_template_entry_without_text_is_refused
}

a_template_using_an_undeclared_token_is_refused() {
  templated no-bos 'del(.bos_token)'
  generate 1 "$COPY"
  said "Error: compiling the chat template: the template writes bos_token, but the tokenizer config declares none, so every rendering would lack it; observed: bos_token: absent from tokenizer config $COPY/tokenizer_config.json; the template read from the chat_template of $COPY/tokenizer_config.json"
  passed a_template_using_an_undeclared_token_is_refused
}

a_template_that_does_not_parse_is_refused() {
  templated unparsable '.chat_template = "{% for %}"'
  generate 1 "$COPY"
  said "Error: compiling the chat template: the template does not parse: it is not Jinja the template engine reads"
  said "observed: line 1 of the template; the engine answered: "
  passed a_template_that_does_not_parse_is_refused
}

a_template_that_raises_is_refused() {
  templated raising '.chat_template = "{{ raise_exception(\"only system turns\") }}"'
  generate 1 "$COPY"
  said "rendering 1 turn(s) through the chat template: the template failed to render them: the template itself refuses these turns"
  said "the template said: "
  said "only system turns"
  passed a_template_that_raises_is_refused
}

a_template_calling_an_unknown_method_is_refused() {
  templated method '.chat_template = "{{ messages[0].content.frobnicate() }}"'
  generate 1 "$COPY"
  said "rendering 1 turn(s) through the chat template: the template failed to render them: it calls a method or function, or reads a value, that Ster's renderer does not provide"
  said "error kind UnknownMethod"
  generate 0 "$COPY" --chat-template off
  passed a_template_calling_an_unknown_method_is_refused
}

# A template that upper-cases every turn renders a completion no longer
# spelled as the example wrote it, so where it starts cannot be found.
a_template_that_rewrites_the_prompt_is_refused() {
  templated upper '.chat_template = "{% for message in messages %}{{ message[\"content\"] | upper }} {% endfor %}{% if add_generation_prompt %}answer :{% endif %}"'
  cp docs/examples/tuning/examples.json "$ROOT/examples.json"
  run 1 tune sft --model "$COPY" --examples "$ROOT/examples.json" --output "$ROOT/upper/adapter.safetensors" \
    --rank 2 --alpha 4 --epochs 1 --learning-rate 0.001 --accumulation 1 --warmup-steps 0 \
    --max-sequence 64 --batch-size 1 --seed 1
  said "locating where the assistant's turn begins: the template does not render the assistant's turn after the prompt, so the completion's tokens cannot be told from the prompt's"
  passed a_template_that_rewrites_the_prompt_is_refused
}

an_unknown_mode_is_refused
an_unreadable_template_is_refused
a_template_of_no_known_form_is_refused
a_template_list_without_a_default_is_refused
a_template_entry_without_text_is_refused
a_template_using_an_undeclared_token_is_refused
a_template_that_does_not_parse_is_refused
a_template_that_raises_is_refused
a_template_calling_an_unknown_method_is_refused
a_template_that_rewrites_the_prompt_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
