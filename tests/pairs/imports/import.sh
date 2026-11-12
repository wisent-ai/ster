#!/bin/sh
# Real test of `ster pairs import` and `ster request pairs/import`
# (`ster.pairs.import`, and `ster.pairs.export` reading the export), through
# the built binary, on exports this run writes under its own directory.
#
# It imports a BiPO CSV and reads the written set and the skipped row. Then
# every refusal with its declared sentence and exit status, and no set
# written: each flag refusal names an export that does not exist, which shows
# the flags are checked before the export is read; then a missing or
# unreadable export, CSV and JSON exports that do not parse as their
# benchmark's, a count of zero or above the pairs, and an export no row of
# which becomes a pair.
#
# Usage: tests/pairs/imports/import.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-import/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
OUT="$ROOT/set.json"
ABSENT="$ROOT/absent.csv"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
HEADER='"question", "matching", "not_matching"'
printf 'question,matching,not_matching\nIs the sky blue?,Yes,No\n,Yes,No\n' >"$ROOT/bipo.csv"
printf 'question,matching,not_matching\n,Yes,No\n' >"$ROOT/blank.csv"
printf 'question,matching,not_matching\n' >"$ROOT/header.csv"
: >"$ROOT/empty.csv"
printf 'prompt,good,bad\nIs it?,Yes,No\n' >"$ROOT/other.csv"
printf 'question,matching,not_matching\nIs it?,Yes,No\nTwo,fields\n' >"$ROOT/short.csv"
printf 'question,matching,not_matching\n"Is it?,Yes,No\n' >"$ROOT/quote.csv"
printf '[{"q": "Sky?", "c": ["blue", "green"], "a": 0},\n' >"$ROOT/list.json"
printf '{"q": "Sky?", "c": ["blue", "green"], "a": 0}\nnot json\n' >"$ROOT/rows.jsonl"
printf '{\n  "q": "Sky?"\n}\n' >"$ROOT/spread.jsonl"
printf '[{"question_id": "q1", "question_content": "Add two numbers."}]\n' >"$ROOT/problems.json"
printf '{"items": []}\n' >"$ROOT/examples.json"
printf 'not json\n' >"$ROOT/broken.json"
mkdir -p "$ROOT/directory.csv"
printf 'question,good,bad\n\377\376 caf\351,yes,no\n' >"$ROOT/latin.csv"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ ster %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster $* exited $status, expected $expected: $err"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
passed() { echo "ok: $1" >>"$REPORT"; }
# One refused import: status, then the rest of its arguments; no set may be written.
refused() {
  run "$@" --seed 7 --output "$OUT"
  [ ! -e "$OUT" ] || fail "a refused import wrote $OUT"
}
FLAGS="reading the row flags of --benchmark"
CHOICES="--question /q --choices /c --answer /a"

importing_a_bipo_export_writes_its_pairs() {
  run 0 pairs import --benchmark bipo --source "$ROOT/bipo.csv" --seed 7 --output "$ROOT/bipo.json"
  check "the set holds the one usable row" "$(jq '.pairs | length' "$ROOT/bipo.json")" "1"
  check "the positive is the matching answer" "$(jq -r '.pairs[0].positive' "$ROOT/bipo.json")" "Question: Is the sky blue?
Answer: Yes"
  check "the trait is the benchmark" "$(jq -r .trait_name "$ROOT/bipo.json")" "bipo"
  check "the blank row is skipped under its cause" "$(echo "$out" | jq -c '[.report.skipped[] | [.row, .cause.id, .cause.exit]]')" '[["2","question-blank",1]]'
  check "the row's cause names the functionality that declares it" "$(echo "$out" | jq -r '.report.skipped[0].cause.functionality')" "ster.pairs.import.rows"
  check "the row's cause links its line on the area page" "$(echo "$out" | jq -r '.report.skipped[0].cause.docs | split("#") | last')" "ster-pairs-import-rows-question-blank"
  check "its reason is the declared refusal" "$(echo "$out" | jq -r '.report.skipped[0].reason')" "importing row 2 of $ROOT/bipo.csv: the question column of row 2 is blank, and every pair asks a question; observed: question: \"\"; matching: \"Yes\"; not_matching: \"No\"; fix: write the question of row 2 in $ROOT/bipo.csv, or remove the row"
  passed importing_a_bipo_export_writes_its_pairs
}

an_unknown_benchmark_is_refused() {
  refused 2 pairs import --benchmark nope --source "$ABSENT"
  said "Error: reading the benchmark name: unknown benchmark nope: Ster imports no export by that name; observed: --benchmark \"nope\"; benchmarks: truthfulqa, dna, livecodebench, bipo, choices; fix: name one of them with --benchmark"
  set +e
  out=$(printf '{"benchmark": "nope", "source": "%s", "seed": 7, "output": "%s"}' "$ABSENT" "$OUT" | "$BIN" request pairs/import)
  status=$?
  set -e
  printf '$ ster request pairs/import\nexit: %s\nstdout: %s\n\n' "$status" "$out" >>"$REPORT"
  check "the request refuses before it runs" "$status" "2"
  check "the request says the same" "$(echo "$out" | jq -r 'select(.type == "result") | .json.error' | cut -d';' -f1)" \
    "reading the benchmark name: unknown benchmark nope: Ster imports no export by that name"
  passed an_unknown_benchmark_is_refused
}

row_flags_on_another_benchmark_are_refused() {
  refused 2 pairs import --benchmark bipo --source "$ABSENT" --question /question
  said "Error: $FLAGS bipo: --question, --choices, --answer, --answer-form and --labels apply only to --benchmark choices; observed: given: --question /question; fix: leave the row flags out for bipo, or import multiple-choice rows with --benchmark choices"
  passed row_flags_on_another_benchmark_are_refused
}

choices_without_row_flags_are_refused() {
  refused 2 pairs import --benchmark choices --source "$ABSENT"
  said "Error: $FLAGS choices: --benchmark choices needs --question, --choices, --answer and --answer-form: where each part of a row sits; observed: --benchmark choices; none of --question, --choices, --answer, --answer-form and --labels is given; fix: name each part's JSON pointer and the answer's form"
  passed choices_without_row_flags_are_refused
}

a_missing_row_flag_is_refused() {
  refused 2 pairs import --benchmark choices --source "$ABSENT" $CHOICES
  said "Error: $FLAGS choices: multiple-choice rows need --answer-form: --question, --choices, --answer and --answer-form name where each part of a row sits; observed: given: --question /q, --choices /c, --answer /a; fix: add --answer-form: "
  passed a_missing_row_flag_is_refused
}

an_unknown_answer_form_is_refused() {
  refused 2 pairs import --benchmark choices --source "$ABSENT" $CHOICES --answer-form maybe
  said "Error: $FLAGS choices: unknown answer form maybe: choose index, label or text; observed: --answer-form maybe; fix: give --answer-form index (a position counted from 0), label (one of --labels) or text (the choice itself)"
  passed an_unknown_answer_form_is_refused
}

a_pointer_without_its_slash_is_refused() {
  refused 2 pairs import --benchmark choices --source "$ABSENT" --question q --choices /c --answer /a --answer-form text
  said "Error: $FLAGS choices: --question \"q\" is not a JSON pointer, which starts with / (for example /question or /choices/text); observed: --question \"q\"; fix: write --question as a JSON pointer (RFC 6901) from the row's root, e.g. /question"
  passed a_pointer_without_its_slash_is_refused
}

label_form_without_labels_is_refused() {
  refused 2 pairs import --benchmark choices --source "$ABSENT" $CHOICES --answer-form label
  said "Error: $FLAGS choices: --answer-form label needs --labels, the list of labels the answer names; observed: --answer-form label without --labels; the answer is read at "
  passed label_form_without_labels_is_refused
}

labels_with_another_answer_form_are_refused() {
  refused 2 pairs import --benchmark choices --source "$ABSENT" $CHOICES --answer-form index --labels /l
  said "Error: $FLAGS choices: --labels applies only to --answer-form label; observed: --answer-form index with --labels /l; fix: leave --labels out, or read the answer as a label with --answer-form label"
  passed labels_with_another_answer_form_are_refused
}

livecodebench_without_examples_is_refused() {
  refused 2 pairs import --benchmark livecodebench --source "$ABSENT"
  said "Error: importing livecodebench from $ABSENT: --benchmark livecodebench needs --examples, the question_examples.json holding each question's good and bad code; observed: no --examples given; --source $ABSENT; fix: add --examples with the question_examples.json published beside problems.json"
  passed livecodebench_without_examples_is_refused
}

examples_on_another_benchmark_are_refused() {
  refused 2 pairs import --benchmark bipo --source "$ABSENT" --examples "$ROOT/examples.json"
  said "Error: importing bipo from $ABSENT: --examples applies only to --benchmark livecodebench; observed: --benchmark bipo with --examples $ROOT/examples.json; fix: leave --examples out for bipo"
  passed examples_on_another_benchmark_are_refused
}

a_count_of_zero_is_refused() {
  refused 2 pairs import --benchmark bipo --source "$ABSENT" --count 0
  said "Error: importing bipo from $ABSENT: --count 0 keeps no pair, and a set without pairs is refused; observed: --count 0; fix: give --count a number above zero, or leave it out to keep every pair"
  passed a_count_of_zero_is_refused
}

a_count_above_the_pairs_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/bipo.csv" --count 5
  said "Error: drawing 5 pairs from $ROOT/bipo.csv: $ROOT/bipo.csv yields 1 pairs, fewer than --count 5; observed: rows: 2, rows skipped: 1; fix: give --count 1 or fewer, or leave it out to keep every pair"
  passed a_count_above_the_pairs_is_refused
}

an_export_without_a_usable_row_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/blank.csv"
  said "Error: importing row 1 of $ROOT/blank.csv: the question column of row 1 is blank, and every pair asks a question; observed: question: \"\"; matching: \"Yes\"; not_matching: \"No\"; no row of $ROOT/blank.csv became a pair, and a set without pairs is refused: rows: 1; refused by cause: question-blank 1; fix: write the question of row 1 in $ROOT/blank.csv"
  refused 1 pairs import --benchmark bipo --source "$ROOT/header.csv"
  said "Error: importing bipo from $ROOT/header.csv: the export holds no row, so no pair could be made, and a set without pairs is refused; observed: rows: 0; $ROOT/header.csv (a file, "
  said "; fix: name the bipo export itself"
  passed an_export_without_a_usable_row_is_refused
}

a_missing_export_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ABSENT"
  said "Error: reading the export $ABSENT: no file is there: nothing at $ABSENT is the export --source names; observed: --source $ABSENT; its directory $ROOT; the system answered: "
  said "; fix: name the downloaded export with --source; \`ls -ln $ROOT\` shows what that directory holds"
  refused 1 pairs import --benchmark livecodebench --source "$ROOT/problems.json" --examples "$ROOT/absent.json"
  said "; observed: --examples $ROOT/absent.json; its directory $ROOT; the system answered: "
  said "; fix: name the downloaded export with --examples; "
  passed a_missing_export_is_refused
}

a_directory_named_as_an_export_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/directory.csv"
  said "Error: reading the export named by --source $ROOT/directory.csv: $ROOT/directory.csv is a directory, and the export named by --source is one file; observed: path: $ROOT/directory.csv (a directory, "
  said "; fix: name the file that holds the export named by --source, not a directory; \`ls -ln $ROOT/directory.csv\` shows what the directory holds"
  passed a_directory_named_as_an_export_is_refused
}

an_export_that_is_not_text_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/latin.csv"
  said "Error: reading the export named by --source $ROOT/latin.csv: it holds bytes that are not UTF-8, and the export named by --source is read as UTF-8 text; observed: path: $ROOT/latin.csv (a file, "
  said "; fix: convert $ROOT/latin.csv to UTF-8 (\`iconv -t UTF-8\`), or name the export named by --source as it was first written"
  passed an_export_that_is_not_text_is_refused
}

an_empty_csv_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/empty.csv"
  said "Error: reading the CSV export $ROOT/empty.csv: $ROOT/empty.csv holds no header row; observed: the file holds no record: "
  said "none of them a field; fix: name the bipo export, whose first record names its columns: $HEADER"
  passed an_empty_csv_is_refused
}

a_csv_of_another_export_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/other.csv"
  said "Error: reading the CSV export $ROOT/other.csv: $ROOT/other.csv has no column \"question\", so it is not the bipo export; observed: the header names \"prompt\", \"good\", \"bad\"; fix: name the bipo export, whose header names $HEADER, or the benchmark this export belongs to with --benchmark"
  passed a_csv_of_another_export_is_refused
}

a_csv_record_of_another_width_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/short.csv"
  said "Error: reading the CSV export $ROOT/short.csv: $ROOT/short.csv record 2 has 2 fields where the header names 3; observed: the header: $HEADER; fix: quote every field of record 2 that holds a comma, or name the bipo export as it was published"
  passed a_csv_record_of_another_width_is_refused
}

a_csv_ending_inside_quotes_is_refused() {
  refused 1 pairs import --benchmark bipo --source "$ROOT/quote.csv"
  said "Error: reading the CSV export $ROOT/quote.csv: $ROOT/quote.csv ends inside a quoted field; observed: a field opened with a quote is never closed before the end of the file: the quote opens on line "
  said "; fix: close the quoted field, doubling every quote inside it, or name the bipo export as it was published"
  passed a_csv_ending_inside_quotes_is_refused
}

a_broken_json_list_of_rows_is_refused() {
  refused 1 pairs import --benchmark choices --source "$ROOT/list.json" $CHOICES --answer-form index
  said "Error: reading the multiple-choice rows $ROOT/list.json: it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway); observed: path: $ROOT/list.json ("
  said "; the JSON reader ran out at line "
  passed a_broken_json_list_of_rows_is_refused
}

a_json_lines_row_that_is_not_json_is_refused() {
  refused 1 pairs import --benchmark choices --source "$ROOT/rows.jsonl" $CHOICES --answer-form index
  said "Error: reading the multiple-choice rows of $ROOT/rows.jsonl: $ROOT/rows.jsonl line 2 is not JSON, and an export that does not open with [ is JSON Lines, one JSON row object on each non-blank line; observed: line 2: "
  said "; fix: write line 2 as one JSON row object, as every non-blank line of JSON Lines is"
  refused 1 pairs import --benchmark choices --source "$ROOT/spread.jsonl" $CHOICES --answer-form index
  said "Error: reading the multiple-choice rows of $ROOT/spread.jsonl: the JSON value line 1 of $ROOT/spread.jsonl opens does not close on that line: "
  said "; fix: write each row on one line (\`jq -c . $ROOT/spread.jsonl\` prints every value of the file on a line of its own)"
  passed a_json_lines_row_that_is_not_json_is_refused
}

problems_that_are_not_a_json_list_are_refused() {
  refused 1 pairs import --benchmark livecodebench --source "$ROOT/list.json" --examples "$ROOT/examples.json"
  said "Error: reading the LiveCodeBench problems $ROOT/list.json: it ends before its JSON value closes: "
  said "; fix: write the LiveCodeBench problems again; it is LiveCodeBench's problems.json, one JSON list of problem objects"
  passed problems_that_are_not_a_json_list_are_refused
}

examples_that_are_not_json_are_refused() {
  refused 1 pairs import --benchmark livecodebench --source "$ROOT/problems.json" --examples "$ROOT/broken.json"
  said "Error: reading the LiveCodeBench examples $ROOT/broken.json: it is not JSON, and the LiveCodeBench examples is LiveCodeBench's question_examples.json, one JSON object whose questions list holds each question's examples: "
  passed examples_that_are_not_json_are_refused
}

examples_without_a_questions_list_are_refused() {
  refused 1 pairs import --benchmark livecodebench --source "$ROOT/problems.json" --examples "$ROOT/examples.json"
  said "Error: reading the LiveCodeBench examples $ROOT/examples.json: $ROOT/examples.json has no questions list; observed: its top level: an object with the keys items; fix: name LiveCodeBench's question_examples.json with --examples"
  passed examples_without_a_questions_list_are_refused
}

importing_a_bipo_export_writes_its_pairs
an_unknown_benchmark_is_refused
row_flags_on_another_benchmark_are_refused
choices_without_row_flags_are_refused
a_missing_row_flag_is_refused
an_unknown_answer_form_is_refused
a_pointer_without_its_slash_is_refused
label_form_without_labels_is_refused
labels_with_another_answer_form_are_refused
livecodebench_without_examples_is_refused
examples_on_another_benchmark_are_refused
a_count_of_zero_is_refused
a_count_above_the_pairs_is_refused
an_export_without_a_usable_row_is_refused
a_missing_export_is_refused
a_directory_named_as_an_export_is_refused
an_export_that_is_not_text_is_refused
an_empty_csv_is_refused
a_csv_of_another_export_is_refused
a_csv_record_of_another_width_is_refused
a_csv_ending_inside_quotes_is_refused
a_broken_json_list_of_rows_is_refused
a_json_lines_row_that_is_not_json_is_refused
problems_that_are_not_a_json_list_are_refused
examples_that_are_not_json_are_refused
examples_without_a_questions_list_are_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
