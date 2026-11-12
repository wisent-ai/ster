#!/bin/sh
# Real test of why a row of a JSONL export cannot become a labelled decision
# (`ster.decisions.import.rows`, docs/guide/errors/decide-pipeline.md),
# through `ster decisions import` itself:
#
#   * an export with one good row and one refused row of every cause is
#     imported, and its report lists each refused row by its line under its
#     own cause;
#   * an export whose only row is refused for one cause is refused (exit 1)
#     with that row's cause, its values and its fix, and writes nothing.
#
# Usage: tests/decide/import-rows.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=import-rows . "$(dirname "$0")/harness.sh"

# refused NAME ROW: import the export holding ROW alone, which must be
# refused (exit 1) without writing the set; the refusal is in $err.
refused() {
  printf '%s\n' "$2" >"$ROOT/$1.jsonl"
  run 1 decisions import --input "$ROOT/$1.jsonl" --question-id weather --instructions "how is the sea ?" --output "$ROOT/$1.json"
  [ ! -e "$ROOT/$1.json" ] || fail "a refused import wrote $ROOT/$1.json"
  said "Error: importing line 1 of $ROOT/$1.jsonl: "
  said "; every row was refused, so no labelled decision could be written: rows: 1; refused by cause: "
}

every_refused_row_is_reported_under_its_cause() {
  printf '%s\n' \
    '{"context": "the sea is flat", "options": ["calm", "rough"], "label": 0}' \
    'not json' \
    '{"context": "the sea", "options": ["calm", "rough"], "label": 2}' >"$ROOT/mixed.jsonl"
  run 0 decisions import --input "$ROOT/mixed.jsonl" --question-id weather --instructions "how is the sea ?" --output "$ROOT/mixed.json"
  [ "$(echo "$out" | jq -c '[.report.rejected[] | [.line, .cause.id, .cause.exit, .cause.functionality]]')" = '[[2,"row-not-json",1,"ster.decisions.import.rows"],[3,"row-label-outside",1,"ster.decisions.import.rows"]]' ] \
    || fail "the report did not list lines 2 and 3 under their causes: $out"
  [ "$(echo "$out" | jq -r '.report.rejected[1].reason')" = "importing line 3 of $ROOT/mixed.jsonl: the label 2 of line 3 names no option: the row has 2 options, numbered from 0 to 1; observed: the options: [\"calm\",\"rough\"]; fix: set the label of line 3 to the index of the right option, 0 to 1" ] \
    || fail "line 3's reason is not its declared refusal: $out"
  [ "$(jq '.examples | length' "$ROOT/mixed.json")" = 1 ] || fail "the good row was not written"
  passed every_refused_row_is_reported_under_its_cause
}

a_line_that_is_not_json_is_refused() {
  refused not-json 'not json'
  said "line 1 is not JSON, and every line of the export is one {\"context\", \"options\", \"label\"} object; observed: line 1: not json; the JSON reader stopped at column 2: "
  said "fix: write line 1 as one JSON object on one line, such as {\"context\": \"the sea is flat\""
  passed a_line_that_is_not_json_is_refused
}

a_line_that_is_not_an_object_is_refused() {
  refused not-object '[1, 2]'
  said "line 1 is JSON, a list of 2, where a row is one object holding context, options and label; observed: line 1: [1, 2]"
  passed a_line_that_is_not_an_object_is_refused
}

a_row_without_a_field_is_refused() {
  refused field-missing '{"options": ["calm", "rough"]}'
  said "line 1 has no context, and a row states its context, its options and the index of the right one; observed: the fields it has: options; fix: add context to line 1"
  passed a_row_without_a_field_is_refused
}

a_context_that_is_not_text_is_refused() {
  refused context-not-text '{"context": 7, "options": ["calm", "rough"], "label": 0}'
  said "the context of line 1 is the number 7, where it is the text the question is asked about"
  passed a_context_that_is_not_text_is_refused
}

options_that_are_not_a_list_are_refused() {
  refused options-not-list '{"context": "the sea", "options": "calm", "label": 0}'
  said "the options of line 1 are text \"calm\", where they are a list of the texts to choose from"
  passed options_that_are_not_a_list_are_refused
}

an_option_that_is_not_text_is_refused() {
  refused option-not-text '{"context": "the sea", "options": ["calm", 3], "label": 0}'
  said "option 1 of line 1 is the number 3, where every option is text; observed: the options: [\"calm\",3]"
  passed an_option_that_is_not_text_is_refused
}

a_label_that_is_not_an_index_is_refused() {
  refused label-not-index '{"context": "the sea", "options": ["calm", "rough"], "label": -1}'
  said "the label of line 1 is the number -1, where it is the index of the right option, a whole number from 0"
  passed a_label_that_is_not_an_index_is_refused
}

a_blank_context_is_refused() {
  refused context-blank '{"context": "  ", "options": ["calm", "rough"], "label": 0}'
  said "the context of line 1 is blank, so there is nothing to ask the question about; observed: context: \"  \""
  passed a_blank_context_is_refused
}

a_row_with_one_option_is_refused() {
  refused options-few '{"context": "the sea", "options": ["calm"], "label": 0}'
  said "line 1 holds too few options to choose between, 1, and a choice needs at least two"
  passed a_row_with_one_option_is_refused
}

a_row_with_too_many_options_is_refused() {
  refused options-many "$(jq -nc '{context: "the sea", options: [range(27) | "option \(.)"], label: 0}')"
  said "line 1 has 27 options, and Ster labels at most 26, one letter each"
  passed a_row_with_too_many_options_is_refused
}

a_blank_option_is_refused() {
  refused option-blank '{"context": "the sea", "options": ["calm", " "], "label": 0}'
  said "option 1 of line 1 is blank, and a blank option cannot be told from no option"
  passed a_blank_option_is_refused
}

repeated_options_are_refused() {
  refused options-repeated '{"context": "the sea", "options": ["calm", "rough", "calm "], "label": 0}'
  said "options 0 and 2 of line 1 carry the same text once trimmed, \"calm\", so the label cannot say which is right"
  passed repeated_options_are_refused
}

a_label_past_the_options_is_refused() {
  refused label-outside '{"context": "the sea", "options": ["calm", "rough"], "label": 2}'
  said "the label 2 of line 1 names no option: the row has 2 options, numbered from 0 to 1; observed: the options: [\"calm\",\"rough\"]; every row was refused"
  said "fix: set the label of line 1 to the index of the right option, 0 to 1"
  passed a_label_past_the_options_is_refused
}

every_refused_row_is_reported_under_its_cause
a_line_that_is_not_json_is_refused
a_line_that_is_not_an_object_is_refused
a_row_without_a_field_is_refused
a_context_that_is_not_text_is_refused
options_that_are_not_a_list_are_refused
an_option_that_is_not_text_is_refused
a_label_that_is_not_an_index_is_refused
a_blank_context_is_refused
a_row_with_one_option_is_refused
a_row_with_too_many_options_is_refused
a_blank_option_is_refused
repeated_options_are_refused
a_label_past_the_options_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
