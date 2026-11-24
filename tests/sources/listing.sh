#!/bin/sh
# Real test of `<program> sources [--all]` (`<program>.sources`), the listing, through the
# built binary: it names every adapter by its name and none by a requirement number, asks
# the sharing question with an example event that names an adapter by its name, keeps
# nothing on disk, and refuses an unknown flag or word with clap's exact usage sentence.
#
# Usage: tests/sources/listing.sh   (<environment prefix>BIN selects the binary, see tests/support/names.sh)
cd "$(dirname "$0")/../.."
. tests/sources/harness.sh listing

NAMES="hugging-face-preferences parquet-or-arrow openai-fine-tuning sharegpt alpaca dpo
repeng steering-vectors transformerlens nnsight sae-lens transformers-checkpoint gguf mlx-model
pytorch-checkpoint hugging-face-cache peft-lora-adapter mlx-lora-adapter preferences-comparisons
evaluator-verdicts"

the_listing_names_every_adapter_by_its_name() {
  for flags in "" "--all"; do
    # shellcheck disable=SC2086 # the empty flag is no word at all
    run 0 sources $flags
    for adapter in $NAMES; do
      [ "$(printf '%s' "$out" | jq --arg id "$adapter" '[.adapters[] | select(.adapter == $id)] | length')" = 1 ] \
        || fail "sources $flags does not name $adapter: $out"
    done
    [ "$(printf '%s' "$out" | jq '.adapters | length')" = 20 ] || fail "sources $flags does not hold exactly the 20 adapters: $out"
    [ "$(printf '%s' "$out" | jq '.all')" = "$([ -n "$flags" ] && echo true || echo false)" ] || fail "sources $flags answers the wrong all: $out"
    case "$out" in
      *IMP-*) fail "sources $flags still shows a requirement number: $out" ;;
    esac
    [ -z "$err" ] || fail "sources $flags said something on standard error: $err"
  done
  passed the_listing_names_every_adapter_by_its_name
}

the_sharing_question_names_its_example_adapter_by_its_name() {
  run 0 sources
  [ "$(printf '%s' "$out" | jq -r '.sharing.sends.adapter')" = dpo ] || fail "the sharing question's example event does not name dpo: $out"
  [ "$(printf '%s' "$out" | jq -r '.sharing.answer_with')" = "$PROGRAM sources share yes | no" ] || fail "the sharing question names no answer: $out"
  passed the_sharing_question_names_its_example_adapter_by_its_name
}

the_listing_keeps_nothing() {
  run 0 sources --all
  [ -z "$(kept_files)" ] || fail "the listing kept files: $(kept_files)"
  passed the_listing_keeps_nothing
}

an_unknown_flag_or_word_is_refused_with_its_usage() {
  run 2 sources --bogus
  usage_refused "unexpected argument '--bogus' found" "sources [OPTIONS] [COMMAND]"
  run 2 sources extra
  usage_refused "unrecognized subcommand 'extra'" "sources [OPTIONS] [COMMAND]"
  [ -z "$out" ] || fail "a refused listing printed an answer: $out"
  [ -z "$(kept_files)" ] || fail "a refused listing kept files: $(kept_files)"
  passed an_unknown_flag_or_word_is_refused_with_its_usage
}

the_listing_names_every_adapter_by_its_name
the_sharing_question_names_its_example_adapter_by_its_name
the_listing_keeps_nothing
an_unknown_flag_or_word_is_refused_with_its_usage
finished
