#!/bin/sh
# Real test of `<program> sources discover <adapter> --path <file>` (`<program>.sources`)
# through the built binary: a ShareGPT file and a DPO file are discovered with what each
# gives under the adapter's name; a path that is not there and a file of no known form are
# answered with why; a missing adapter or --path is a usage error; an adapter the program
# does not have is refused, and so is one named by the requirement number it was once listed
# under, naming its name; and no discovery keeps anything on disk.
#
# Usage: tests/sources/discover.sh   (<environment prefix>BIN selects the binary, see tests/support/names.sh)
cd "$(dirname "$0")/../.."
. tests/sources/harness.sh discover
FILES="$SCRATCH/files"
mkdir -p "$FILES"
printf '%s\n' '{"conversations": [{"from": "human", "value": "Name a colour."}, {"from": "gpt", "value": "Blue."}]}' >"$FILES/sharegpt.jsonl"
printf '%s\n%s\n' '{"prompt": "Greet.", "chosen": "Hello.", "rejected": "Go away."}' '{"prompt": "Greet.", "chosen": "", "rejected": "No."}' >"$FILES/dpo.jsonl"
printf 'not json at all\n{"half": \n' >"$FILES/malformed.jsonl"

a_sharegpt_file_is_discovered_with_its_examples() {
  run 0 sources discover sharegpt --path "$FILES/sharegpt.jsonl"
  [ "$(printf '%s' "$out" | jq .examples)" = 1 ] || fail "the ShareGPT file gave no example: $out"
  [ "$(printf '%s' "$out" | jq -r .adapter)" = sharegpt ] || fail "the answer does not name sharegpt: $out"
  [ "$(printf '%s' "$out" | jq .found)" = true ] || fail "the ShareGPT file was not found to be ShareGPT: $out"
  passed a_sharegpt_file_is_discovered_with_its_examples
}

a_dpo_file_is_discovered_with_its_pairs() {
  run 0 sources discover dpo --path "$FILES/dpo.jsonl"
  [ "$(printf '%s' "$out" | jq .pairs)" = 1 ] || fail "the DPO file did not give one pair: $out"
  [ "$(printf '%s' "$out" | jq .left_out)" = 1 ] || fail "the DPO file's row without a chosen answer was not left out: $out"
  [ "$(printf '%s' "$out" | jq -r .adapter)" = dpo ] || fail "the answer does not name dpo: $out"
  passed a_dpo_file_is_discovered_with_its_pairs
}

a_path_that_is_not_there_is_answered_with_why() {
  run 0 sources discover dpo --path "$FILES/absent.jsonl"
  [ "$(printf '%s' "$out" | jq .found)" = false ] || fail "an absent file was found: $out"
  case "$(printf '%s' "$out" | jq -r .why_not)" in
    "$FILES/absent.jsonl: "*) ;;
    *) fail "an absent file's answer does not say why, naming it: $out" ;;
  esac
  passed a_path_that_is_not_there_is_answered_with_why
}

a_file_of_no_known_form_is_answered_with_why() {
  run 0 sources discover dpo --path "$FILES/malformed.jsonl"
  [ "$(printf '%s' "$out" | jq .found)" = false ] || fail "a malformed file was found: $out"
  case "$(printf '%s' "$out" | jq -r .why_not)" in
    "it is neither a JSON list nor JSON Lines: "*) ;;
    *) fail "a malformed file's answer does not say why: $out" ;;
  esac
  passed a_file_of_no_known_form_is_answered_with_why
}

a_missing_adapter_or_path_is_a_usage_error() {
  run 2 sources discover
  said "<ADAPTER>"
  run 2 sources discover dpo
  said "--path <PATH>"
  passed a_missing_adapter_or_path_is_a_usage_error
}

an_unknown_adapter_is_refused() {
  run 2 sources discover no-such-source --path "$FILES/dpo.jsonl"
  said "Error: discovering $FILES/dpo.jsonl as no-such-source: no-such-source is not an adapter of $PROGRAM"
  said "its adapters are hugging-face-preferences, openai-fine-tuning, sharegpt, alpaca, dpo"
  passed an_unknown_adapter_is_refused
}

an_adapter_named_by_its_requirement_number_is_refused_naming_its_name() {
  run 2 sources discover IMP-STER-06 --path "$FILES/dpo.jsonl"
  said "Error: discovering $FILES/dpo.jsonl as IMP-STER-06: IMP-STER-06 is the import plan's requirement number of an adapter"
  said "observed: IMP-STER-06 stands for the adapter named dpo"
  said "fix: name it dpo: \`$PROGRAM sources discover dpo --path $FILES/dpo.jsonl\`"
  # A number of a model checkpoint, written in lowercase, is refused naming its name too.
  run 2 sources discover imp-ster-16 --path "$FILES"
  said "observed: IMP-STER-16 stands for the adapter named hugging-face-cache"
  passed an_adapter_named_by_its_requirement_number_is_refused_naming_its_name
}

no_discovery_keeps_anything() {
  [ -z "$(kept_files)" ] || fail "a discovery kept files: $(kept_files)"
  passed no_discovery_keeps_anything
}

a_sharegpt_file_is_discovered_with_its_examples
a_dpo_file_is_discovered_with_its_pairs
a_path_that_is_not_there_is_answered_with_why
a_file_of_no_known_form_is_answered_with_why
a_missing_adapter_or_path_is_a_usage_error
an_unknown_adapter_is_refused
an_adapter_named_by_its_requirement_number_is_refused_naming_its_name
no_discovery_keeps_anything
finished
