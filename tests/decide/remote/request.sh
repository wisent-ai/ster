#!/bin/sh
# Real test of how answering a decision request fails (`ster.decide.request`,
# `ster.decide`) through the built binary: `ster decide` and `ster request
# decide` on the lettered fixture tests/decide/harness.sh writes.
#
# The fixture is checked to answer ok.json first. Each test then names a
# missing, unreadable or broken document, or one whose state or questions
# break a rule, and checks the declared sentence and exit status 1 (2 from
# `ster request decide`, which refuses a request before anything runs);
# every refusal of the document is made before the checkpoint loads.
#
# Usage: tests/decide/remote/request.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=request . "$(dirname "$0")/../harness.sh"

run 0 $DECIDE --request "$ROOT/ok.json"
printf '%s' "$out" | jq -e '.answers.weather.choice and (.answers.loud.noul | type == "number")' >/dev/null \
  || fail "the fixture did not answer both questions: $out"
passed "the lettered fixture answers a choice and a noul"

a_missing_request_is_refused() {
  run 1 $DECIDE --request "$ROOT/absent.json"
  said "Error: reading decision request $ROOT/absent.json: no file is there: nothing at $ROOT/absent.json is the decision request named; observed: path: $ROOT/absent.json (absent); its directory $ROOT; the system answered: "
  said "; fix: name an existing decision document with --request (\`ls -ln $ROOT\` shows what that directory holds), or give \`--request -\` and write it on standard input"
  passed a_missing_request_is_refused
}

a_directory_named_as_the_request_is_refused() {
  run 1 $DECIDE --request "$ROOT/spec"
  said "Error: reading the decision request $ROOT/spec: $ROOT/spec is a directory, and the decision request is one file; observed: path: $ROOT/spec (a directory, "
  INPUT="$ROOT/spec" run 1 $DECIDE --request -
  said "Error: reading the decision request /dev/stdin: /dev/stdin is a directory, and the decision request is one file; observed: path: /dev/stdin (a directory, "
  passed a_directory_named_as_the_request_is_refused
}

a_request_that_is_not_json_is_refused() {
  printf '{"state": 1' >"$ROOT/broken.json"
  run 1 $DECIDE --request "$ROOT/broken.json"
  said "Error: reading the decision request $ROOT/broken.json: it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway); observed: path: $ROOT/broken.json (a file, 11 bytes, "
  said "; the JSON reader ran out at line 1, column "
  said ": EOF while parsing an object"
  INPUT="$ROOT/broken.json" run 1 $DECIDE --request -
  said "Error: reading the decision request /dev/stdin: it ends before its JSON value closes: "
  said "; fix: write the decision request again; it is {\"state\": "
  passed a_request_that_is_not_json_is_refused
}

an_empty_state_is_refused() {
  request empty-state '.state = "  "'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking the decision request: its state is empty, so every question would be judged against nothing; observed: state: \"  \""
  passed an_empty_state_is_refused
}

a_request_without_questions_is_refused() {
  request no-questions '.questions = {}'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking the decision request: it asks no question; observed: questions: {}"
  passed a_request_without_questions_is_refused
}

a_question_with_an_empty_id_is_refused() {
  request empty-id '.questions[" "] = .questions.loud'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking the decision request: a question's id is empty, and the answer comes back under that id; observed: question id: \" \""
  passed a_question_with_an_empty_id_is_refused
}

a_question_without_instructions_is_refused() {
  request no-instructions '.questions.loud.instructions = null'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking question 'loud': its instructions are empty, so the model is shown options with no question; observed: instructions: null"
  passed a_question_without_instructions_is_refused
}

a_question_with_one_option_is_refused() {
  request one-option '.questions.weather.criteria = {calm: null}'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking choice question 'weather': it offers 1 options, and a decision needs at least two to choose between; observed: options: 1"
  request one-level '.questions.weather = {type: "score", instructions: "how calm ?", criteria: ["calm"]}'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking score question 'weather': it offers 1 levels"
  passed a_question_with_one_option_is_refused
}

a_question_with_more_options_than_letters_is_refused() {
  request many '.questions.weather.criteria = ([range(27)] | map({key: "option \(.)", value: null}) | from_entries)'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking choice question 'weather': it offers 27 options, and each is answered under one of the 26 letters A to Z read at one token position; observed: options: 27; fix: split the question into questions of at most 26 options each"
  # The request operation refuses the same document before anything runs.
  jq -c --arg model "$MODEL" '{model: $model, chatTemplate: "off", request: .}' "$REQUEST" >"$ROOT/many-request.json"
  INPUT="$ROOT/many-request.json" run 2 request decide
  printf '%s' "$out" | jq -e 'select(.type == "result") | .status == 2
    and (.json.error | contains("it offers 27 options"))' >/dev/null \
    || fail "ster request decide did not refuse 27 options with status 2: $out"
  passed a_question_with_more_options_than_letters_is_refused
}

an_option_with_an_empty_name_is_refused() {
  request empty-option '.questions.weather.criteria[""] = null'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking choice question 'weather': one of its options has an empty name, and the answer names the option chosen; observed: option name: \"\""
  passed an_option_with_an_empty_name_is_refused
}

a_level_without_a_description_is_refused() {
  request empty-level '.questions.weather = {type: "score", instructions: "how calm ?", criteria: ["calm", ""]}'
  run 1 $DECIDE --request "$REQUEST"
  said "Error: checking score question 'weather': its level 1 has an empty description, so the model is shown a letter with nothing under it; observed: level 1: \"\""
  passed a_level_without_a_description_is_refused
}

a_question_longer_than_the_context_is_refused() {
  context=$(jq -r '.max_position_embeddings' "$MODEL/config.json")
  jq --arg words "$(yes 'the sea' | head -n "$context" | tr '\n' ' ')" '.state = $words' \
    "$ROOT/ok.json" >"$ROOT/long.json"
  run 1 $DECIDE --request "$ROOT/long.json"
  said "Error: rendering question 'loud': its prompt is "
  said "tokens, and this model reads at most $context at once"
  passed a_question_longer_than_the_context_is_refused
}

a_letter_the_model_does_not_score_is_refused() {
  # A copy whose tokenizer spells A a second way, as a token one past the
  # vocabulary the weights score.
  cp -R "$MODEL" "$ROOT/wide"
  size=$(jq -r '.vocab_size' "$MODEL/config.json")
  jq --argjson id "$size" '.added_tokens += [{id: $id, content: "ĠA", single_word: false,
    lstrip: false, rstrip: false, normalized: false, special: false}]' \
    "$MODEL/tokenizer.json" >"$ROOT/wide/tokenizer.json"
  run 1 decide --model "$ROOT/wide" --chat-template off --request "$ROOT/ok.json"
  said "Error: reading the answer under letter A: the tokenizer spells A as token $size, and the model scores only tokens 0 to $size exclusive"
  case "$err" in *panicked*) fail "the decision panicked: $err" ;; esac
  passed a_letter_the_model_does_not_score_is_refused
}

an_answer_that_cannot_be_written_is_refused() {
  run 1 $DECIDE --request "$ROOT/ok.json" --output "$ROOT/ok.json/answer.json"
  said "Error: writing the decision $ROOT/ok.json/answer.json: the path runs through a file: $ROOT/ok.json is not a directory, so nothing can be created below it; observed: path: $ROOT/ok.json/answer.json; $ROOT/ok.json: a file, "
  said "; the answer is printed on standard output above; the file may be emptied or cut short; the system answered: "
  printf '%s' "$out" | jq -e '.answers.weather.choice' >/dev/null || fail "the answer was not printed before the refused write: $out"
  passed an_answer_that_cannot_be_written_is_refused
}

a_missing_request_is_refused
a_directory_named_as_the_request_is_refused
a_request_that_is_not_json_is_refused
an_empty_state_is_refused
a_request_without_questions_is_refused
a_question_with_an_empty_id_is_refused
a_question_without_instructions_is_refused
a_question_with_one_option_is_refused
a_question_with_more_options_than_letters_is_refused
an_option_with_an_empty_name_is_refused
a_level_without_a_description_is_refused
a_question_longer_than_the_context_is_refused
a_letter_the_model_does_not_score_is_refused
an_answer_that_cannot_be_written_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
