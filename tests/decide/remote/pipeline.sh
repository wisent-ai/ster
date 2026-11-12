#!/bin/sh
# Real test of how building labelled decision sets fails
# (`ster.decisions.question`, `.import`, `.synthesize`, `.split` and the
# fetch count, docs/guide/errors/decide-pipeline.md), through `ster
# decisions` itself. Every setting is refused before a file is read or a
# model or server is asked, and no output is written.
#
# Usage: tests/decide/remote/pipeline.sh   (STER selects the binary, default target/debug/ster)
AREA=pipeline . "$(dirname "$0")/../harness.sh"

printf '{"context": "the sea is calm", "options": ["calm", "storm"], "label": 0}\n' >"$ROOT/rows.jsonl"
IMPORT="decisions import --input $ROOT/rows.jsonl --output $ROOT/imported.json"
jq -n '{domain: "a report of the sea", questions: {weather: {type: "choice", instructions: "how is the sea ?", criteria: {calm: null, storm: null}}}}' >"$ROOT/schema.json"
SYNTH="decisions synthesize --generator-model wisent-backend/chat/primary --temperature 0.7 --max-tokens 16 --output $ROOT/synthesized.json"

absent() {
  [ ! -e "$1" ] || fail "a refused run wrote $1"
}

a_blank_question_id_is_refused() {
  run 2 $IMPORT --question-id " " --instructions "how is the sea ?"
  said "Error: checking the question every imported example asks: --question-id is blank, and every example answers its question under that id; observed: --question-id \" \""
  run 2 decisions fetch --dataset stanfordnlp/imdb --config plain_text --split train --text-field text \
    --label-field label --question-id "" --instructions "is it kind ?" --count 4 --output "$ROOT/fetched.json"
  said "Error: checking the question every fetched example asks: --question-id is blank"
  absent "$ROOT/imported.json"
  absent "$ROOT/fetched.json"
  passed a_blank_question_id_is_refused
}

blank_instructions_are_refused() {
  run 2 $IMPORT --question-id weather --instructions ""
  said "Error: checking the question every imported example asks: --instructions is blank, and the model is asked the question in those words; observed: --instructions \"\""
  absent "$ROOT/imported.json"
  passed blank_instructions_are_refused
}

an_export_without_rows_is_refused() {
  printf '\n  \n' >"$ROOT/blank.jsonl"
  run 1 decisions import --input "$ROOT/blank.jsonl" --question-id weather --instructions "how is the sea ?" --output "$ROOT/blank.json"
  said "Error: importing $ROOT/blank.jsonl: the file holds no row: every line is blank; observed: path: $ROOT/blank.jsonl (a file, 4 bytes, "
  absent "$ROOT/blank.json"
  passed an_export_without_rows_is_refused
}

# The first refused row's own cause, with how every row was refused after it.
an_export_whose_every_row_is_refused_is_refused() {
  printf '{"context": "the sea", "options": ["calm"], "label": 0}\nnot json\n' >"$ROOT/refused.jsonl"
  run 1 decisions import --input "$ROOT/refused.jsonl" --question-id weather --instructions "how is the sea ?" --output "$ROOT/refused.json"
  said "Error: importing line 1 of $ROOT/refused.jsonl: line 1 holds too few options to choose between, 1, and a choice needs at least two; observed: the options: [\"calm\"]"
  said "; every row was refused, so no labelled decision could be written: rows: 2; refused by cause: row-not-json 1, row-options-few 1; fix: give line 1 at least two options"
  absent "$ROOT/refused.json"
  passed an_export_whose_every_row_is_refused_is_refused
}

no_state_per_option_is_refused() {
  run 2 $SYNTH --schema "$ROOT/absent-schema.json" --per-option 0 --retry-multiplier 2
  said "Error: checking the synthesis settings: --per-option 0 asks for no state, so no labelled decision could be written; observed: --per-option 0"
  absent "$ROOT/synthesized.json"
  passed no_state_per_option_is_refused
}

no_attempt_per_state_is_refused() {
  run 2 $SYNTH --schema "$ROOT/absent-schema.json" --per-option 1 --retry-multiplier 0
  said "Error: checking the synthesis settings: --retry-multiplier 0 allows no attempt, so no state could be written; observed: --retry-multiplier 0"
  absent "$ROOT/synthesized.json"
  passed no_attempt_per_state_is_refused
}

a_schema_without_a_domain_is_refused() {
  jq '.domain = " "' "$ROOT/schema.json" >"$ROOT/no-domain.json"
  run 1 $SYNTH --schema "$ROOT/no-domain.json" --per-option 1 --retry-multiplier 2
  said "Error: reading the decision schema $ROOT/no-domain.json: its domain is blank, and the writer is told what to write by it; observed: domain: \" \""
  absent "$ROOT/synthesized.json"
  passed a_schema_without_a_domain_is_refused
}

a_schema_without_questions_is_refused() {
  jq '.questions = {}' "$ROOT/schema.json" >"$ROOT/no-questions.json"
  run 1 $SYNTH --schema "$ROOT/no-questions.json" --per-option 1 --retry-multiplier 2
  said "Error: reading the decision schema $ROOT/no-questions.json: it asks no question, so no state would be labelled; observed: questions: 0"
  absent "$ROOT/synthesized.json"
  passed a_schema_without_questions_is_refused
}

a_holdout_outside_zero_and_one_is_refused() {
  run 2 decisions split --examples "$ROOT/absent.json" --holdout 1.5 --seed 1 \
    --train-output "$ROOT/train.json" --holdout-output "$ROOT/held.json"
  said "Error: checking the split settings: --holdout 1.5 is no fraction above zero and below one, so it parts no example from the rest; observed: --holdout 1.5"
  absent "$ROOT/train.json"
  absent "$ROOT/held.json"
  passed a_holdout_outside_zero_and_one_is_refused
}

a_blank_question_id_is_refused
blank_instructions_are_refused
an_export_without_rows_is_refused
an_export_whose_every_row_is_refused_is_refused
no_state_per_option_is_refused
no_attempt_per_state_is_refused
a_schema_without_a_domain_is_refused
a_schema_without_questions_is_refused
a_holdout_outside_zero_and_one_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
