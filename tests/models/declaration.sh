#!/bin/sh
# Real test of how compiling a tokenizer declaration fails
# (`ster.tokenizer.declaration`) and how `ster model tokenizer create`
# writes its output (`ster.tokenizer.create`), through the built binary.
# Each test writes a copy of docs/examples/model/vocabulary.json through jq
# that is wrong in one way, and checks the refusal and that no output was
# left behind. The run's home is tests/runtime/harness.sh's.
#
# Usage: tests/models/declaration.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=models-declaration . "$(dirname "$0")/../runtime/harness.sh"
# declared NAME JQ: the example declaration through JQ, compiled to
# $ROOT/NAME.out, which must not exist afterwards.
declared() {
  jq "$2" docs/examples/model/vocabulary.json >"$ROOT/$1.json"
  run 1 model tokenizer create --from "$ROOT/$1.json" --output "$ROOT/$1.out"
  [ ! -e "$ROOT/$1.out" ] || fail "a refused declaration left $ROOT/$1.out"
}

run 0 model tokenizer create --from docs/examples/model/vocabulary.json --output "$ROOT/ok.out"
[ "$(echo "$out" | jq .vocabulary_size)" -gt 0 ] || fail "the example declaration compiled to no vocabulary: $out"
passed "the example declaration compiles"

a_missing_declaration_is_refused() {
  run 1 model tokenizer create --from "$ROOT/absent.json" --output "$ROOT/absent.out"
  said "reading the tokenizer declaration $ROOT/absent.json: no file is there"
  passed a_missing_declaration_is_refused
}

a_declaration_without_model_tokens_is_refused() {
  declared none '.modelTokens = []'
  said "tokenizer declaration $ROOT/none.json contains no modelTokens"
  said "observed: modelTokens: []; addedTokens: 3"
  passed a_declaration_without_model_tokens_is_refused
}

a_blank_unknown_token_is_refused() {
  declared unknown '.unknownToken = ""'
  said "tokenizer declaration requires unknownToken"
  passed a_blank_unknown_token_is_refused
}

a_blank_verification_text_is_refused() {
  declared blank '.verification.text = "   "'
  said "tokenizer declaration requires verification.text"
  said "verification.text: 3 bytes, all whitespace"
  passed a_blank_verification_text_is_refused
}

an_undeclared_unknown_token_is_refused() {
  declared undeclared '.unknownToken = "zzz"'
  said "tokenizer unknownToken \"zzz\" is absent from its declared modelTokens"
  passed an_undeclared_unknown_token_is_refused
}

a_blank_model_token_is_refused() {
  declared empty '.modelTokens[5].content = ""'
  said "tokenizer token at position 5 has empty content"
  passed a_blank_model_token_is_refused
}

a_repeated_model_token_is_refused() {
  declared twice '.modelTokens += [.modelTokens[3]]'
  said "duplicates earlier content; token identities must be unique"
  said "observed: modelTokens[3] and modelTokens[67] both hold"
  passed a_repeated_model_token_is_refused
}

a_blank_added_token_is_refused() {
  declared added '.addedTokens += [{"content": "", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}]'
  said "has empty addedTokens content at position 3"
  passed a_blank_added_token_is_refused
}

a_repeated_added_token_is_refused() {
  declared merged '.addedTokens += [.addedTokens[0]]'
  said "requested 4 added-token records, but the native operation applied 3 and stored 3 unique identities"
  said "at 0 and 3"
  passed a_repeated_added_token_is_refused
}

a_post_processor_emitting_an_unmapped_id_is_refused() {
  declared unmapped '.verification.addSpecialTokens = true | .pipeline.postProcessor = {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [999], "tokens": ["[CLS]"]}}}'
  said "encoded token \"[CLS]\" as id 999, but its vocabulary maps that token to no id"
  said "the output $ROOT/unmapped.out this run created was removed"
  passed a_post_processor_emitting_an_unmapped_id_is_refused
}

an_output_path_that_is_not_utf8_is_refused() {
  run 2 model tokenizer create --from docs/examples/model/vocabulary.json --output "$ROOT/$(printf 'bad\377').out"
  said "the output path is not UTF-8"
  passed an_output_path_that_is_not_utf8_is_refused
}

an_existing_output_is_not_replaced() {
  echo "kept" >"$ROOT/kept.out"
  run 1 model tokenizer create --from docs/examples/model/vocabulary.json --output "$ROOT/kept.out"
  said "something is already at $ROOT/kept.out, and existing output is not replaced"
  [ "$(cat "$ROOT/kept.out")" = "kept" ] || fail "an existing output was changed"
  passed an_existing_output_is_not_replaced
}

a_missing_declaration_is_refused
a_declaration_without_model_tokens_is_refused
a_blank_unknown_token_is_refused
a_blank_verification_text_is_refused
an_undeclared_unknown_token_is_refused
a_blank_model_token_is_refused
a_repeated_model_token_is_refused
a_blank_added_token_is_refused
a_repeated_added_token_is_refused
a_post_processor_emitting_an_unmapped_id_is_refused
an_output_path_that_is_not_utf8_is_refused
an_existing_output_is_not_replaced
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
