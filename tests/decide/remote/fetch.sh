#!/bin/sh
# Real test of how `ster decisions fetch` fails before it asks the Hugging
# Face datasets server (`ster.decisions.fetch`,
# docs/guide/errors/decide-pipeline.md), for a classification set and for
# multiple-choice rows, whose row flags are checked as `ster pairs import
# --benchmark choices` checks them (`ster.pairs.import.fields`). What the
# server answers is listed in tests/errors/untested.json: only the real
# server produces it.
#
# Usage: tests/decide/remote/fetch.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=fetch . "$(dirname "$0")/../harness.sh"

unfetched() { [ ! -e "$ROOT/fetched.json" ] || fail "a refused fetch wrote $ROOT/fetched.json"; }

a_count_of_zero_is_refused() {
  run 2 decisions fetch --dataset stanfordnlp/imdb --config plain_text --split train --text-field text \
    --label-field label --question-id sentiment --instructions "is it kind ?" --count 0 --output "$ROOT/fetched.json"
  said "Error: checking the fetch settings: --count 0 asks for no row, so no labelled decision could be written; observed: --count 0; fix: give --count the number of rows to read, 1 or more"
  unfetched
  passed a_count_of_zero_is_refused
}

# ARC's rows, as the Hub serves them: the answer is one of the choices'
# labels. A pointer without its slash is refused before any row is asked for.
a_choice_pointer_without_its_slash_is_refused() {
  run 2 decisions fetch --dataset allenai/ai2_arc --config ARC-Challenge --split test --question question \
    --choices /choices/text --answer /answerKey --answer-form label --labels /choices/label \
    --question-id answer --instructions "which answer is right ?" --count 4 --output "$ROOT/fetched.json"
  said "--question \"question\" is not a JSON pointer, which starts with / (for example /question or /choices/text)"
  unfetched
  passed a_choice_pointer_without_its_slash_is_refused
}

a_missing_choice_flag_is_refused() {
  run 2 decisions fetch --dataset allenai/ai2_arc --config ARC-Challenge --split test --question /question \
    --choices /choices/text --answer /answerKey --question-id answer --instructions "which answer is right ?" \
    --count 4 --output "$ROOT/fetched.json"
  said "multiple-choice rows need --answer-form"
  unfetched
  passed a_missing_choice_flag_is_refused
}

class_and_choice_flags_together_are_refused() {
  run 2 decisions fetch --dataset allenai/ai2_arc --config ARC-Challenge --split test --text-field question \
    --label-field answerKey --choices /choices/text --question-id answer --instructions "which answer is right ?" \
    --count 4 --output "$ROOT/fetched.json"
  said "cannot be used with"
  unfetched
  passed class_and_choice_flags_together_are_refused
}

a_count_of_zero_is_refused
a_choice_pointer_without_its_slash_is_refused
a_missing_choice_flag_is_refused
class_and_choice_flags_together_are_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
