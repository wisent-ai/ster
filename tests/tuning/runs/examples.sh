#!/bin/sh
# Real test of how reading an example set fails (`ster.tune.examples`), and
# of the one way encoding a valid example fails on a model without a chat
# template (`ster.runtime.encode`), through `ster tune sft` and
# `ster tune evaluate` on the four-layer fixture. Every refusal exits 1 and
# no run that refused writes an adapter.
#
# Usage: tests/tuning/runs/examples.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=examples
. "$(dirname "$0")/../harness.sh"

EVALUATE="--model $MODEL --examples $ROOT/examples.json --max-sequence 64 --batch-size 1"
# examples NAME JQ: an example set at $ROOT/NAME.json, the docs' set through JQ.
examples() {
  SET="$ROOT/$1.json"
  jq "$2" "$ROOT/examples.json" >"$SET"
}
no_adapter() {
  [ ! -e "$ROOT/sft/adapter.safetensors" ] || fail "a refused run wrote an adapter"
}

a_missing_example_set_is_refused() {
  tune 1 sft "$SFT" --examples "$ROOT/absent.json"
  said "Error: reading example set $ROOT/absent.json: no file is there: nothing at $ROOT/absent.json is the example set named; observed: path: $ROOT/absent.json (absent); its directory $ROOT; the system answered: "
  no_adapter
  tune 1 evaluate "$EVALUATE" --examples "$ROOT/absent.json"
  said "Error: reading example set $ROOT/absent.json: no file is there: "
  passed a_missing_example_set_is_refused
}

a_directory_named_as_an_example_set_is_refused() {
  mkdir -p "$ROOT/folder"
  tune 1 sft "$SFT" --examples "$ROOT/folder"
  said "Error: reading the example set $ROOT/folder: $ROOT/folder is a directory, and the example set is one file; observed: path: $ROOT/folder (a directory, "
  no_adapter
  passed a_directory_named_as_an_example_set_is_refused
}

a_file_that_is_not_an_example_set_is_refused() {
  echo nope >"$ROOT/nope.json"
  tune 1 sft "$SFT" --examples "$ROOT/nope.json"
  said "Error: reading the example set $ROOT/nope.json: it is not JSON, and the example set is {\"examples\": [{\"prompt\": \"<text>\", \"completion\": \"<text>\"}]}, as docs/examples/tuning/examples.json is: it is another kind of file, or one broken by hand; observed: path: $ROOT/nope.json (a file, 5 bytes, "
  examples no-completion 'del(.examples[0].completion)'
  tune 1 sft "$SFT" --examples "$SET"
  said "it is JSON of another form than the example set, which is {\"examples\": "
  said "the reader answered: missing field \`completion\`"
  no_adapter
  passed a_file_that_is_not_an_example_set_is_refused
}

an_example_set_without_examples_is_refused() {
  examples empty '.examples = []'
  tune 1 sft "$SFT" --examples "$SET"
  said "Error: reading example set $SET: example set $SET contains no examples; observed: examples: 0"
  no_adapter
  passed an_example_set_without_examples_is_refused
}

an_example_with_an_empty_side_is_refused() {
  examples blank '.examples[2].completion = "  "'
  tune 1 sft "$SFT" --examples "$SET"
  said "Error: reading example set $SET: example set $SET contains an empty prompt or completion; observed: example 2: prompt \"question : describe the harbor .\", completion \"  \"; fix: give example 2 text in both its prompt and its completion, or remove it"
  no_adapter
  passed an_example_with_an_empty_side_is_refused
}

an_empty_system_prompt_is_refused() {
  examples empty-system '.examples[1].system = ""'
  tune 1 sft "$SFT" --examples "$SET"
  said "Error: reading example set $SET: example set $SET contains an empty system prompt; observed: example 1: system \"\""
  no_adapter
  passed an_empty_system_prompt_is_refused
}

a_system_prompt_without_a_chat_template_is_refused() {
  examples system '.examples[0].system = "answer calmly ."'
  tune 1 sft "$SFT" --examples "$SET"
  said "encoding example 0"
  said "encoding a training example: the example has a system prompt, which only a chat template can place, and this run's chat template is absent; observed: system: \"answer calmly .\"; chat template: absent"
  tune 1 sft "$SFT" --examples "$SET" --chat-template off
  said "and this run's chat template is off; observed: system: \"answer calmly .\"; chat template: off"
  no_adapter
  passed a_system_prompt_without_a_chat_template_is_refused
}

# Each published form is read into the same examples: with example 2's completion
# blank, every form is refused naming example 2's prompt, so prompt and completion
# were read from where the form keeps them.
rows_of_other_trainers_are_read_as_examples() {
  blanked='.examples[2].completion = "  " | .examples[]'
  jq -c "$blanked"' | {messages: [{role: "user", content: .prompt}, {role: "assistant", content: .completion}]}' "$ROOT/examples.json" >"$ROOT/openai.jsonl"
  jq "[$blanked"' | {conversations: [{from: "human", value: .prompt}, {from: "gpt", value: .completion}]}]' "$ROOT/examples.json" >"$ROOT/sharegpt.json"
  jq -c "$blanked"' | {instruction: .prompt, input: "", output: .completion}' "$ROOT/examples.json" >"$ROOT/alpaca.jsonl"
  for SET in "$ROOT/openai.jsonl" "$ROOT/sharegpt.json" "$ROOT/alpaca.jsonl"; do
    tune 1 sft "$SFT" --examples "$SET"
    said "Error: reading example set $SET: example set $SET contains an empty prompt or completion; observed: example 2: prompt \"question : describe the harbor .\", completion \"  \""
  done
  no_adapter
  passed rows_of_other_trainers_are_read_as_examples
}

a_conversation_of_more_than_one_exchange_is_refused() {
  jq -c '.examples[] | {messages: [{role: "user", content: .prompt}, {role: "assistant", content: .completion}, {role: "user", content: .prompt}, {role: "assistant", content: .completion}]}' "$ROOT/examples.json" >"$ROOT/long.jsonl"
  tune 1 sft "$SFT" --examples "$ROOT/long.jsonl"
  said "row 0 of example set $ROOT/long.jsonl is a conversation of other turns than one example's"
  said "observed: row 0: turns user, assistant, user, assistant"
  no_adapter
  passed a_conversation_of_more_than_one_exchange_is_refused
}

a_row_of_no_known_form_is_refused() {
  echo '[{"question": "a", "answer": "b"}]' >"$ROOT/unknown.json"
  tune 1 sft "$SFT" --examples "$ROOT/unknown.json"
  said "row 0 of example set $ROOT/unknown.json is none of the forms an example row is read in"
  said "observed: row 0: fields answer, question"
  no_adapter
  passed a_row_of_no_known_form_is_refused
}

a_missing_example_set_is_refused
a_directory_named_as_an_example_set_is_refused
a_file_that_is_not_an_example_set_is_refused
an_example_set_without_examples_is_refused
an_example_with_an_empty_side_is_refused
an_empty_system_prompt_is_refused
a_system_prompt_without_a_chat_template_is_refused
rows_of_other_trainers_are_read_as_examples
a_conversation_of_more_than_one_exchange_is_refused
a_row_of_no_known_form_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
