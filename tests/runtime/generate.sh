#!/bin/sh
# Real test of how sampling a continuation fails (`ster.runtime.generate`)
# and what `ster generate` refuses before any weight is read
# (`ster.generate`), on the four-layer fixture, whose context holds 256
# positions: a token budget of zero is refused with exit status 2, a prompt
# of 300 words, each one token, before anything is sampled; a --vector
# without its --strength, a system turn that cannot be read or holds no
# text, and answers whose file cannot be written.
# Usage: tests/runtime/generate.sh   (STER selects the binary, default target/debug/ster)
AREA=generate
. "$(dirname "$0")/harness.sh"

no_new_token_is_refused() {
  run 2 generate --model "$MODEL" --prompt "describe the sea ." --max-new-tokens 0 --temperature 0 --seed 1
  said "Error: sampling a continuation: max_new_tokens must be greater than zero, or there is nothing to sample; observed: --max-new-tokens 0"
  passed no_new_token_is_refused
}

a_prompt_past_the_context_is_refused() {
  long=
  i=0
  while [ "$i" -lt 300 ]; do
    long="$long the"
    i=$((i + 1))
  done
  generate 1 "$MODEL" "$long"
  said "Error: sampling a continuation: the prompt is 300 tokens, and this model's context holds fewer than 256; observed: prompt: 300 tokens; max_position_embeddings: 256"
  passed a_prompt_past_the_context_is_refused
}

a_vector_without_its_strength_is_refused() {
  run 2 generate --model "$ROOT/no-such-model" --prompt "describe the sea ." --max-new-tokens 1 --temperature 0 --seed 1 \
    --vector "$ROOT/calm.json" --vector "$ROOT/calm.json" --strength 1
  said "Error: pairing each --vector with its --strength: every --vector needs its own --strength, and this run gave 2 vector(s) and 1 strength(s)"
  passed a_vector_without_its_strength_is_refused
}

a_system_turn_that_cannot_be_read_is_refused() {
  generate 1 "$ROOT/no-such-model" "describe the sea ." --system "$ROOT/absent.txt"
  said "Error: reading the system turn $ROOT/absent.txt: nothing is there: no file has the name $ROOT/absent.txt; observed: path: $ROOT/absent.txt (absent); its directory $ROOT; the system answered: "
  said "; fix: name the system turn where it is; \`ls -ln $ROOT\` shows what that directory holds"
  passed a_system_turn_that_cannot_be_read_is_refused
}

an_empty_system_turn_is_refused() {
  printf '  \n' >"$ROOT/blank.txt"
  generate 2 "$ROOT/no-such-model" "describe the sea ." --system "$ROOT/blank.txt"
  said "Error: reading the system turn $ROOT/blank.txt: the file holds no text, so there is no system turn to give the model; observed: --system $ROOT/blank.txt: 3 bytes of whitespace"
  passed an_empty_system_turn_is_refused
}

answers_under_a_file_are_refused() {
  cp docs/examples/prompts.json "$ROOT/prompts.json"
  run 1 generate --model "$MODEL" --prompts "$ROOT/prompts.json" --output "$ROOT/prompts.json/answers.json" \
    --max-new-tokens 1 --temperature 0 --seed 1
  said "Error: creating the directory of the answers $ROOT/prompts.json/answers.json: the path runs through a file: $ROOT/prompts.json is not a directory, so nothing can be created below it; observed: path: $ROOT/prompts.json/answers.json; $ROOT/prompts.json: a file, "
  said "; no prompt was answered and nothing was written; the system answered: "
  passed answers_under_a_file_are_refused
}

a_vector_without_its_strength_is_refused
a_system_turn_that_cannot_be_read_is_refused
an_empty_system_turn_is_refused
answers_under_a_file_are_refused
no_new_token_is_refused
a_prompt_past_the_context_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
