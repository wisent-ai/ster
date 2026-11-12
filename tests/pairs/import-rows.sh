#!/bin/sh
# Real test of why a row of a benchmark export cannot become a pair
# (`ster.pairs.import.rows`, docs/guide/errors/pairs-import.md), through
# `ster pairs import` itself, on exports this run writes under its own
# directory. Each export holds one row that fails for one cause, so the
# import is refused (exit 1) with that row's cause, its values and its fix,
# and writes no set. Keeping a row and listing the skipped ones is
# tests/pairs/import.sh and tests/pairs/import-choices.mjs.
#
# Usage: tests/pairs/import-rows.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-import-rows/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
OUT="$ROOT/set.json"
# The binary runs under this run's own home, never the operator's.
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }

# refused NAME FILE TEXT ARGS...: write TEXT to FILE under the run's
# directory as $SRC, import it with ARGS, and require the refusal of its one
# row (exit 1, no set written) naming CAUSE as the count after it.
refused() {
  cause=$1
  SRC="$ROOT/$2"
  printf '%s\n' "$3" >"$SRC"
  shift 3
  set +e
  out=$("$BIN" pairs import --source "$SRC" "$@" --seed 7 --output "$OUT" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ ster pairs import --source %s %s\nexit: %s\nstderr: %s\n\n' "$SRC" "$*" "$status" "$err" >>"$REPORT"
  [ "$status" -eq 1 ] || fail "importing $SRC exited $status, expected 1: $err"
  [ ! -e "$OUT" ] || fail "a refused import wrote $OUT"
  said "no row of $SRC became a pair, and a set without pairs is refused: rows: 1; refused by cause: $cause 1; fix: "
}
CHOICES="--benchmark choices --question /q --choices /c --answer /a"
LABELS="$CHOICES --answer-form label --labels /l"

a_row_without_a_question_is_skipped() {
  refused question-blank dna.csv "$(printf 'question,id\n,1')" --benchmark dna
  said "Error: importing row 1 of $SRC: the question column of row 1 is blank, and every pair asks a question; observed: question: \"\""
  passed a_row_without_a_question_is_skipped
}

a_row_with_a_blank_answer_is_skipped() {
  refused answer-blank bipo.csv "$(printf 'question,matching,not_matching\nIs it?,Yes,')" --benchmark bipo
  said "the not_matching column of row 1 is blank, and it is the negative side of the pair; observed: question: \"Is it?\"; matching: \"Yes\"; not_matching: \"\""
  said "fix: write the not_matching of row 1 in $SRC, or remove the row"
  passed a_row_with_a_blank_answer_is_skipped
}

a_row_without_an_incorrect_answer_is_skipped() {
  refused incorrect-none truthfulqa.csv "$(printf 'Question,Best Answer,Incorrect Answers\nIs it?,Yes,; ;')" --benchmark truthfulqa
  said "the Incorrect Answers column of row 1 holds no answer once split at \`;\`, and the negative side is one of them"
  passed a_row_without_an_incorrect_answer_is_skipped
}

# LiveCodeBench: problems against the examples --examples names.
livecodebench() {
  printf '%s\n' "$3" >"$ROOT/examples-$2.json"
  refused "$1" "problems-$2.json" "$4" --benchmark livecodebench --examples "$ROOT/examples-$2.json"
}

a_problem_without_an_id_is_skipped() {
  livecodebench problem-id-missing id '{"questions": []}' '[{"question_id": 7, "question_content": "Add two numbers."}]'
  said "Error: importing problem 1 of $SRC: problem 1 has no question_id text, which is what pairs a problem with its examples; observed: question_id: the number 7"
  passed a_problem_without_an_id_is_skipped
}

a_problem_without_content_is_skipped() {
  livecodebench problem-content-blank content '{"questions": []}' '[{"question_id": "q1"}]'
  said "Error: importing problem 1 (q1) of $SRC: problem 1 (q1) has no question_content text, and it is the question of the pair; observed: question_content: absent"
  passed a_problem_without_content_is_skipped
}

a_problem_without_examples_is_skipped() {
  livecodebench example-missing examples '{"questions": [{"question_id": "q2"}]}' '[{"question_id": "q1", "question_content": "Add."}]'
  said "no question of $ROOT/examples-examples.json carries the question_id of problem 1 (q1), q1, and the pair's code comes from its examples; observed: questions in $ROOT/examples-examples.json: 1"
  passed a_problem_without_examples_is_skipped
}

an_example_without_code_is_skipped() {
  livecodebench example-code-missing code '{"questions": [{"question_id": "q1", "good_example": {"code": "a + b"}}]}' '[{"question_id": "q1", "question_content": "Add."}]'
  said "the example of q1 in $ROOT/examples-code.json has no bad_example code, and the pair is its good code against its bad; observed: good_example.code: present; bad_example.code: absent or blank"
  passed an_example_without_code_is_skipped
}

a_choice_row_without_a_question_is_skipped() {
  refused choice-question-missing question.jsonl '{"c": ["blue", "green"], "a": 0}' $CHOICES --answer-form index
  said "Error: importing row 1 of $SRC: row 1 has no question text at /q, where --question points; observed: /q: absent"
  passed a_choice_row_without_a_question_is_skipped
}

choices_that_are_not_a_list_are_skipped() {
  refused choices-not-list not-list.jsonl '{"q": "Sky?", "c": "blue", "a": 0}' $CHOICES --answer-form index
  said "row 1 holds no list at /c, where --choices points; observed: /c: text \"blue\""
  passed choices_that_are_not_a_list_are_skipped
}

a_choice_that_is_not_text_is_skipped() {
  refused choice-not-text not-text.jsonl '{"q": "Sky?", "c": ["blue", ""], "a": 0}' $CHOICES --answer-form index
  said "choice 1 of row 1 is text \"\", where every choice is non-empty text; observed: /c: [\"blue\",\"\"]"
  passed a_choice_that_is_not_text_is_skipped
}

a_row_with_one_choice_is_skipped() {
  refused choices-few one.jsonl '{"q": "Sky?", "c": ["blue"], "a": 0}' $CHOICES --answer-form index
  said "row 1 holds 1 choices at /c, and a pair is the correct choice against another; observed: /c: [\"blue\"]"
  passed a_row_with_one_choice_is_skipped
}

a_row_without_an_answer_is_skipped() {
  refused answer-missing no-answer.jsonl '{"q": "Sky?", "c": ["blue", "green"], "a": null}' $CHOICES --answer-form index
  said "row 1 has no answer at /a, where --answer points; observed: /a: null"
  passed a_row_without_an_answer_is_skipped
}

an_answer_that_is_not_an_index_is_skipped() {
  refused answer-not-index letter.jsonl '{"q": "Sky?", "c": ["blue", "green"], "a": "B"}' $CHOICES --answer-form index
  said "the answer of row 1 at /a is text \"B\", where --answer-form index reads a whole number from 0, or text holding one; observed: the text is not a whole number: "
  refused answer-not-index flag.jsonl '{"q": "Sky?", "c": ["blue", "green"], "a": true}' $CHOICES --answer-form index
  said "the answer of row 1 at /a is true, where --answer-form index reads a whole number from 0, or text holding one; observed: it is neither a whole number nor text"
  passed an_answer_that_is_not_an_index_is_skipped
}

an_answer_past_the_choices_is_skipped() {
  refused answer-index-outside past.jsonl '{"q": "Sky?", "c": ["blue", "green"], "a": 2}' $CHOICES --answer-form index
  said "the answer of row 1 names choice 2, and the row holds 2 choices, numbered from 0 to 1; observed: choices: [\"blue\", \"green\"]"
  passed an_answer_past_the_choices_is_skipped
}

labels_that_are_not_a_list_are_skipped() {
  refused labels-not-list labels-text.jsonl '{"q": "Sky?", "c": ["blue", "green"], "l": "AB", "a": "A"}' $LABELS
  said "row 1 holds no list at /l, where --labels points; observed: /l: text \"AB\""
  passed labels_that_are_not_a_list_are_skipped
}

a_label_that_is_not_text_is_skipped() {
  refused label-not-text label-number.jsonl '{"q": "Sky?", "c": ["blue", "green"], "l": ["A", 2], "a": "A"}' $LABELS
  said "label 1 of row 1 is the number 2, where every label is non-empty text; observed: /l: [\"A\",2]"
  passed a_label_that_is_not_text_is_skipped
}

labels_that_do_not_line_up_are_skipped() {
  refused labels-misaligned labels-short.jsonl '{"q": "Sky?", "c": ["blue", "green"], "l": ["A"], "a": "A"}' $LABELS
  said "row 1 holds 1 labels and 2 choices, and --answer-form label reads the label at each choice's position; observed: labels: [\"A\"]; choices: [\"blue\", \"green\"]"
  passed labels_that_do_not_line_up_are_skipped
}

an_answer_that_is_not_a_label_is_skipped() {
  refused answer-label-not-text label-index.jsonl '{"q": "Sky?", "c": ["blue", "green"], "l": ["A", "B"], "a": 0}' $LABELS
  said "the answer of row 1 at /a is the number 0, where --answer-form label reads one of the row's labels as text"
  passed an_answer_that_is_not_a_label_is_skipped
}

an_answer_that_names_no_label_is_skipped() {
  refused answer-label-unknown label-unknown.jsonl '{"q": "Sky?", "c": ["blue", "green"], "l": ["A", "B"], "a": "D"}' $LABELS
  said "the answer of row 1, \"D\", is none of its labels; observed: labels: [\"A\", \"B\"]"
  passed an_answer_that_names_no_label_is_skipped
}

an_answer_that_is_not_text_is_skipped() {
  refused answer-text-not-text text-index.jsonl '{"q": "Sky?", "c": ["blue", "green"], "a": 0}' $CHOICES --answer-form text
  said "the answer of row 1 at /a is the number 0, where --answer-form text reads the correct choice's text"
  passed an_answer_that_is_not_text_is_skipped
}

an_answer_that_names_no_choice_is_skipped() {
  refused answer-text-unknown text-unknown.jsonl '{"q": "Sky?", "c": ["blue", "green"], "a": "red"}' $CHOICES --answer-form text
  said "the answer of row 1, \"red\", is none of its choices; observed: choices: [\"blue\", \"green\"]"
  passed an_answer_that_names_no_choice_is_skipped
}

a_row_without_a_question_is_skipped
a_row_with_a_blank_answer_is_skipped
a_row_without_an_incorrect_answer_is_skipped
a_problem_without_an_id_is_skipped
a_problem_without_content_is_skipped
a_problem_without_examples_is_skipped
an_example_without_code_is_skipped
a_choice_row_without_a_question_is_skipped
choices_that_are_not_a_list_are_skipped
a_choice_that_is_not_text_is_skipped
a_row_with_one_choice_is_skipped
a_row_without_an_answer_is_skipped
an_answer_that_is_not_an_index_is_skipped
an_answer_past_the_choices_is_skipped
labels_that_are_not_a_list_are_skipped
a_label_that_is_not_text_is_skipped
labels_that_do_not_line_up_are_skipped
an_answer_that_is_not_a_label_is_skipped
an_answer_that_names_no_label_is_skipped
an_answer_that_is_not_text_is_skipped
an_answer_that_names_no_choice_is_skipped

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
