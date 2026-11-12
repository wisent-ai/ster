#!/bin/sh
# Real test of how `ster decisions fetch` fails before it asks the Hugging
# Face datasets server (`ster.decisions.fetch`,
# docs/guide/errors/decide-pipeline.md). What the server answers is listed in
# tests/errors/untested.json: only the real server produces it.
#
# Usage: tests/decide/fetch.sh   (STER selects the binary, default target/debug/ster)
AREA=fetch . "$(dirname "$0")/harness.sh"

a_count_of_zero_is_refused() {
  run 2 decisions fetch --dataset stanfordnlp/imdb --config plain_text --split train --text-field text \
    --label-field label --question-id sentiment --instructions "is it kind ?" --count 0 --output "$ROOT/fetched.json"
  said "Error: checking the fetch settings: --count 0 asks for no row, so no labelled decision could be written; observed: --count 0; fix: give --count the number of rows to read, 1 or more"
  [ ! -e "$ROOT/fetched.json" ] || fail "a refused fetch wrote $ROOT/fetched.json"
  passed a_count_of_zero_is_refused
}

a_count_of_zero_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
