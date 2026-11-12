#!/bin/sh
# Real test of how turning text into a checkpoint's token ids fails
# (`ster.runtime.encode`), through the built binary on the four-layer
# fixture `ster model fixture create` writes.
#
#   * a prompt of only whitespace is refused with exit status 2;
#   * a prompt the tokenizer normalizes away is refused: the copy's
#     tokenizer.json gains a Replace normalizer that deletes "zz";
#   * a conversation is refused on a model without a chat template, while
#     the simulated user carries one;
#   * a decision is refused when the vocabulary has no token for the answer
#     letter, which the fixture's lower-case vocabulary lacks.
#
# Usage: tests/runtime/encode.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=encode
. "$(dirname "$0")/harness.sh"

a_blank_prompt_is_refused() {
  generate 2 "$MODEL" "   "
  said "Error: encoding the prompt: the prompt is empty or only whitespace, so there is nothing to encode; observed: prompt: \"   \"; fix: give the prompt at least one word"
  passed a_blank_prompt_is_refused
}

text_the_tokenizer_drops_is_refused() {
  cp -R "$MODEL" "$ROOT/normalized"
  jq '.normalizer = {"type": "Replace", "pattern": {"String": "zz"}, "content": ""}' \
    "$MODEL/tokenizer.json" >"$ROOT/normalized/tokenizer.json"
  generate 0 "$ROOT/normalized" "describe the sea ."
  generate 1 "$ROOT/normalized" "zz"
  said "Error: encoding the prompt: this model's tokenizer turns \"zz\" into no tokens, so the model has nothing to read; observed: prompt: \"zz\"; token ids: []"
  passed text_the_tokenizer_drops_is_refused
}

a_conversation_without_a_chat_template_is_refused() {
  cp -R "$MODEL" "$ROOT/user"
  cp docs/examples/chat-template/tokenizer_config.json "$ROOT/user/tokenizer_config.json"
  cp docs/examples/tuning/grpo-prompts.json "$ROOT/prompts.json"
  run 1 converse --model "$MODEL" --prompts "$ROOT/prompts.json" --user-model "$ROOT/user" \
    --turns 2 --max-new-tokens 2 --max-sequence 64 --temperature 1 --seed 1 --output "$ROOT/conversations.jsonl"
  said "encoding a conversation of several turns: only the model's chat template knows how turns are delimited, and this run's chat template is absent; observed: chat template: absent"
  passed a_conversation_without_a_chat_template_is_refused
}

# The fixture's vocabulary read as PLaMo's tokenizer.jsonl with no row of
# kind UNKNOWN and no byte rows: a word outside it has no piece to become.
a_text_the_tokenizer_refuses_is_refused() {
  cp -R "$MODEL" "$ROOT/no-unknown"
  rm "$ROOT/no-unknown/tokenizer.json"
  jq -c '.modelTokens | to_entries[] | [.value.content, (0 - .key), "NORMAL"]' \
    "$ROOT/vocabulary.json" >"$ROOT/no-unknown/tokenizer.jsonl"
  generate 1 "$ROOT/no-unknown" "zz"
  said "Error: encoding the prompt: the text holds a piece the tokenizer's vocabulary lacks, and its tokenizer.json names no unknown token (unk_id) to stand in for it; observed: prompt: \"zz\"; the tokenizer holds "
  said " tokens; the library answered: Encountered an unknown token but \`unk_id\` is missing"
  passed a_text_the_tokenizer_refuses_is_refused
}

a_label_the_vocabulary_lacks_is_refused() {
  printf '%s' '{"state": "the sea is calm .", "questions": {"mood": {"type": "choice", "instructions": "is the sea calm ?", "criteria": {"calm": "the sea is calm", "storm": "the sea is loud"}}}}' \
    >"$ROOT/decision.json"
  run 1 decide --model "$MODEL" --request "$ROOT/decision.json"
  said "finding the token of answer label \"A\": this tokenizer has no single token for \"A\", so the model's answer cannot be read from one position; observed: spellings looked up: \"A\", \"▁A\", \"ĠA\""
  passed a_label_the_vocabulary_lacks_is_refused
}

a_blank_prompt_is_refused
text_the_tokenizer_drops_is_refused
a_conversation_without_a_chat_template_is_refused
a_label_the_vocabulary_lacks_is_refused
a_text_the_tokenizer_refuses_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
