#!/bin/sh
# Real test of authoring a pair set one pair at a time (`ster.pairs.add`,
# `.remove` and `.edit`, and the layers `.index` and `.side` they use),
# through the built binary, on files this run creates under its own directory.
#
# It adds three pairs, edits one side of the second, then both sides of the
# first together with the set's trait, removes the third, and reads the file
# after each step. Then the refusals, each with its declared sentence and exit
# status: a side given as empty text, refused before any set is read (a
# missing file stays missing), an edit naming no change, and an index past the
# end for edit and remove; the file is read after each and is as it was.
#
# Usage: tests/pairs/author.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-author/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
SET="$ROOT/set.json"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

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
unchanged() { check "the set is as it was" "$(cat "$SET")" "$1"; }

adding_pairs_creates_and_extends_the_set() {
  [ ! -e "$SET" ] || fail "$SET exists before the first add"
  run 0 pairs add --pairs "$SET" --positive "I will help." --negative "I refuse." --trait helpfulness
  check "the first pair is index 0" "$(echo "$out" | jq -c .added)" '{"index":0}'
  run 0 pairs add --pairs "$SET" --positive "Here is how." --negative "Figure it out."
  run 0 pairs add --pairs "$SET" --positive "Ask me anything." --negative "Go away."
  check "the third pair is index 2" "$(echo "$out" | jq -c .added)" '{"index":2}'
  check "the set holds three pairs" "$(jq '.pairs | length' "$SET")" "3"
  check "the trait is kept" "$(jq -r .trait_name "$SET")" "helpfulness"
  check "the second pair is stored" "$(jq -r '.pairs[1].negative' "$SET")" "Figure it out."
  passed adding_pairs_creates_and_extends_the_set
}

editing_changes_only_the_sides_given() {
  run 0 pairs edit --pairs "$SET" --index 1 --negative "Work it out yourself."
  check "the answer names the edited pair" "$(echo "$out" | jq -c .edited)" \
    '{"index":1,"negative":"Work it out yourself.","positive":"Here is how."}'
  check "the second pair's negative is stored" "$(jq -r '.pairs[1].negative' "$SET")" "Work it out yourself."
  check "its positive is kept" "$(jq -r '.pairs[1].positive' "$SET")" "Here is how."
  run 0 pairs edit --pairs "$SET" --index 0 --positive "Glad to help." --negative "No." --trait warmth
  check "the first pair's positive is stored" "$(jq -r '.pairs[0].positive' "$SET")" "Glad to help."
  check "the first pair's negative is stored" "$(jq -r '.pairs[0].negative' "$SET")" "No."
  check "the trait is renamed" "$(jq -r .trait_name "$SET")" "warmth"
  passed editing_changes_only_the_sides_given
}

removing_a_pair_keeps_the_others() {
  run 0 pairs remove --pairs "$SET" --index 2
  check "the answer names the removed pair" "$(echo "$out" | jq -r .removed.positive)" "Ask me anything."
  check "the set holds two pairs" "$(jq '.pairs | length' "$SET")" "2"
  check "the first pair is kept" "$(jq -r '.pairs[0].positive' "$SET")" "Glad to help."
  passed removing_a_pair_keeps_the_others
}

an_index_past_the_end_is_refused() {
  before=$(cat "$SET")
  run 1 pairs edit --pairs "$SET" --index 2 --positive "Too far."
  said "Error: finding pair 2 of $SET: pair index 2 is outside the set: $SET holds 2 pair(s), indexed from 0; observed: pairs: 2, the last at index 1; fix: name an index from 0 to 1; the pairs of $SET are numbered in the order the file lists them"
  unchanged "$before"
  run 1 pairs remove --pairs "$SET" --index 5
  said "Error: finding pair 5 of $SET: pair index 5 is outside the set: $SET holds 2 pair(s), indexed from 0; observed: pairs: 2, the last at index 1"
  unchanged "$before"
  passed an_index_past_the_end_is_refused
}

an_edit_naming_no_change_is_refused() {
  before=$(cat "$SET")
  run 2 pairs edit --pairs "$SET" --index 1
  said "Error: editing pair 1 of $SET: pairs edit changes nothing without --positive, --negative or --trait; observed: none of --positive, --negative or --trait is given; $SET (a file"
  said "is left as it is; fix: name what changes: --positive <text>, --negative <text> or --trait <name>"
  unchanged "$before"
  passed an_edit_naming_no_change_is_refused
}

an_empty_side_is_refused_before_the_set_is_read() {
  run 2 pairs add --pairs "$ROOT/absent.json" --positive "Still." --negative " "
  said "Error: writing the negative side of a pair in $ROOT/absent.json: --negative is empty or only whitespace, and every pair needs text on both sides; observed: --negative \" \"; fix: give --negative the text of that side"
  [ ! -e "$ROOT/absent.json" ] || fail "a refused add wrote $ROOT/absent.json"
  run 2 pairs edit --pairs "$ROOT/absent.json" --index 0 --positive ""
  said "Error: writing the positive side of a pair in $ROOT/absent.json: --positive is empty or only whitespace"
  before=$(cat "$SET")
  run 2 pairs edit --pairs "$SET" --index 0 --negative "	"
  said "observed: --negative \"	\"; fix: give --negative the text of that side"
  unchanged "$before"
  passed an_empty_side_is_refused_before_the_set_is_read
}

adding_pairs_creates_and_extends_the_set
editing_changes_only_the_sides_given
removing_a_pair_keeps_the_others
an_index_past_the_end_is_refused
an_edit_naming_no_change_is_refused
an_empty_side_is_refused_before_the_set_is_read

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
