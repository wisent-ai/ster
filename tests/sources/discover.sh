#!/bin/sh
# Real test of `<program> sources discover <adapter> --path <file>` (`<program>.sources`)
# through the built binary: a ShareGPT file and a DPO file are discovered with what each
# gives under the adapter's name; a path that is not there and a file of no known form are
# answered with the exact reason; a missing adapter or --path is refused with clap's exact
# usage sentence; an adapter the program does not have is refused, and so is one named by
# the requirement number it was once listed under, naming its name, each with its exact
# declared sentence. Every case leaves the file it read byte for byte as it was, says
# nothing it should not on standard error, and keeps nothing on disk.
#
# Usage: tests/sources/discover.sh   (<environment prefix>BIN selects the binary, see tests/support/names.sh)
cd "$(dirname "$0")/../.."
. tests/sources/harness.sh discover
FILES="$SCRATCH/files"
mkdir -p "$FILES"
printf '%s\n' '{"conversations": [{"from": "human", "value": "Name a colour."}, {"from": "gpt", "value": "Blue."}]}' >"$FILES/sharegpt.jsonl"
printf '%s\n%s\n' '{"prompt": "Greet.", "chosen": "Hello.", "rejected": "Go away."}' '{"prompt": "Greet.", "chosen": "", "rejected": "No."}' >"$FILES/dpo.jsonl"
printf 'not json at all\n{"half": \n' >"$FILES/malformed.jsonl"
SHAREGPT=$(digest "$FILES/sharegpt.jsonl")
DPO=$(digest "$FILES/dpo.jsonl")
MALFORMED=$(digest "$FILES/malformed.jsonl")

# The files read are as they were, nothing else was kept, and nothing was said on stderr.
untouched() {
  [ "$(digest "$FILES/sharegpt.jsonl")" = "$SHAREGPT" ] || fail "$1 changed the ShareGPT file"
  [ "$(digest "$FILES/dpo.jsonl")" = "$DPO" ] || fail "$1 changed the DPO file"
  [ "$(digest "$FILES/malformed.jsonl")" = "$MALFORMED" ] || fail "$1 changed the malformed file"
  [ ! -e "$FILES/absent.jsonl" ] || fail "$1 made the absent file"
  [ -z "$(kept_files)" ] || fail "$1 kept files: $(kept_files)"
}

a_sharegpt_file_is_discovered_with_its_examples() {
  run 0 sources discover sharegpt --path "$FILES/sharegpt.jsonl"
  [ "$(printf '%s' "$out" | jq .examples)" = 1 ] || fail "the ShareGPT file gave no example: $out"
  [ "$(printf '%s' "$out" | jq -r .adapter)" = sharegpt ] || fail "the answer does not name sharegpt: $out"
  [ "$(printf '%s' "$out" | jq .found)" = true ] || fail "the ShareGPT file was not found to be ShareGPT: $out"
  [ "$(printf '%s' "$out" | jq -r .path)" = "$FILES/sharegpt.jsonl" ] || fail "the answer does not name the file read: $out"
  said_exactly ""
  untouched "discovering the ShareGPT file"
  passed a_sharegpt_file_is_discovered_with_its_examples
}

a_dpo_file_is_discovered_with_its_pairs() {
  run 0 sources discover dpo --path "$FILES/dpo.jsonl"
  [ "$(printf '%s' "$out" | jq .pairs)" = 1 ] || fail "the DPO file did not give one pair: $out"
  [ "$(printf '%s' "$out" | jq .left_out)" = 1 ] || fail "the DPO file's row without a chosen answer was not left out: $out"
  [ "$(printf '%s' "$out" | jq -r .adapter)" = dpo ] || fail "the answer does not name dpo: $out"
  said_exactly ""
  untouched "discovering the DPO file"
  passed a_dpo_file_is_discovered_with_its_pairs
}

a_path_that_is_not_there_is_answered_with_why() {
  run 0 sources discover dpo --path "$FILES/absent.jsonl"
  [ "$(printf '%s' "$out" | jq .found)" = false ] || fail "an absent file was found: $out"
  [ "$(printf '%s' "$out" | jq -r .why_not)" = "$FILES/absent.jsonl: No such file or directory (os error 2)" ] \
    || fail "an absent file's answer does not say exactly why: $out"
  said_exactly ""
  untouched "discovering an absent file"
  passed a_path_that_is_not_there_is_answered_with_why
}

a_file_of_no_known_form_is_answered_with_why() {
  run 0 sources discover dpo --path "$FILES/malformed.jsonl"
  [ "$(printf '%s' "$out" | jq .found)" = false ] || fail "a malformed file was found: $out"
  [ "$(printf '%s' "$out" | jq -r .why_not)" = "it is neither a JSON list nor JSON Lines: not readable as json-lines at line 1: line 1 is not JSON: expected ident at line 1 column 2" ] \
    || fail "a malformed file's answer does not say exactly why: $out"
  said_exactly ""
  untouched "discovering a malformed file"
  passed a_file_of_no_known_form_is_answered_with_why
}

a_missing_adapter_or_path_is_a_usage_error() {
  run 2 sources discover
  usage_refused "$(printf 'the following required arguments were not provided:\n  --path <PATH>\n  <ADAPTER>')" "sources discover --path <PATH> <ADAPTER>"
  run 2 sources discover dpo
  usage_refused "$(printf 'the following required arguments were not provided:\n  --path <PATH>')" "sources discover --path <PATH> <ADAPTER>"
  untouched "a discovery missing its words"
  passed a_missing_adapter_or_path_is_a_usage_error
}

an_unknown_adapter_is_refused() {
  run 2 sources discover no-such-source --path "$FILES/dpo.jsonl"
  said_exactly "Error: discovering $FILES/dpo.jsonl as no-such-source: no-such-source is not an adapter of $PROGRAM; observed: its adapters are hugging-face-preferences, openai-fine-tuning, sharegpt, alpaca, dpo, preferences-comparisons, transformers-checkpoint, gguf, hugging-face-cache, parquet-or-arrow, repeng, steering-vectors, transformerlens, nnsight, sae-lens, mlx-model, pytorch-checkpoint, mlx-lora-adapter, peft-lora-adapter, evaluator-verdicts; fix: name one of the adapters \`$PROGRAM sources\` lists"
  [ -z "$out" ] || fail "a refused discovery printed an answer: $out"
  untouched "an unknown adapter"
  passed an_unknown_adapter_is_refused
}

an_adapter_named_by_its_requirement_number_is_refused_naming_its_name() {
  # The import plan numbered the product's adapters IMP-<PROGRAM in capitals>-<NN>.
  dpo_number="IMP-$(printf '%s' "$PROGRAM" | tr '[:lower:]' '[:upper:]')-06"
  run 2 sources discover "$dpo_number" --path "$FILES/dpo.jsonl"
  said_exactly "Error: discovering $FILES/dpo.jsonl as $dpo_number: $dpo_number is the import plan's requirement number of an adapter, and $PROGRAM's adapters are named, not numbered; observed: $dpo_number stands for the adapter named dpo; fix: name it dpo: \`$PROGRAM sources discover dpo --path $FILES/dpo.jsonl\`"
  [ -z "$out" ] || fail "a refused number printed an answer: $out"
  # A number of a model checkpoint, written in lowercase, is refused naming its name too.
  cache_number="IMP-$(printf '%s' "$PROGRAM" | tr '[:lower:]' '[:upper:]')-16"
  lowercase_cache_number=$(printf '%s' "$cache_number" | tr '[:upper:]' '[:lower:]')
  run 2 sources discover "$lowercase_cache_number" --path "$FILES"
  said_exactly "Error: discovering $FILES as $lowercase_cache_number: $lowercase_cache_number is the import plan's requirement number of an adapter, and $PROGRAM's adapters are named, not numbered; observed: $cache_number stands for the adapter named hugging-face-cache; fix: name it hugging-face-cache: \`$PROGRAM sources discover hugging-face-cache --path $FILES\`"
  untouched "a requirement number"
  passed an_adapter_named_by_its_requirement_number_is_refused_naming_its_name
}

a_sharegpt_file_is_discovered_with_its_examples
a_dpo_file_is_discovered_with_its_pairs
a_path_that_is_not_there_is_answered_with_why
a_file_of_no_known_form_is_answered_with_why
a_missing_adapter_or_path_is_a_usage_error
an_unknown_adapter_is_refused
an_adapter_named_by_its_requirement_number_is_refused_naming_its_name
finished
