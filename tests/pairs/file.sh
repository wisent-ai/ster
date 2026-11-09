#!/bin/sh
# Real test of how reading, checking and writing a pair-set file fails
# (`ster.pairs.file`), through the built binary, on files this run creates
# under its own directory.
#
# A missing file, a directory, a file that is not a pair set, a set whose
# last pair is removed (the file is kept), a pair with an empty side, a set
# under a file and a set in a read-only directory are each refused with their
# declared sentence and exit status, and the file on disk is read after it. A
# missing file handed to `ster workspace import` is rejected with the same
# cause in its report.
#
# Usage: tests/pairs/file.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-file/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
trap 'chmod -R u+w "$ROOT" 2>/dev/null || true' EXIT

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
inspect() { run "$1" pairs inspect --pairs "$2" --dedupe-bits 3 --dedupe-bands 4 --unbalanced-ratio 3; }

[ "$(id -u)" != 0 ] || fail "run as an ordinary account: root ignores the read-only directory one refusal needs"

a_missing_pair_set_is_refused() {
  inspect 1 "$ROOT/absent.json"
  said "reading pair set $ROOT/absent.json: no file is there: "
  said "observed: path: $ROOT/absent.json; fix: name an existing pair-set file, or write one with \`ster pairs add --pairs $ROOT/absent.json --positive <text> --negative <text>\`"
  run 1 workspace import "$ROOT/absent.json"
  check "the import is rejected" "$(echo "$out" | jq -r .status)" "rejected"
  check "its report names the cause" "$(echo "$out" | jq -r .cause.id)" "pair-set-missing"
  passed a_missing_pair_set_is_refused
}

a_directory_named_as_a_pair_set_is_refused() {
  mkdir -p "$ROOT/directory.json"
  inspect 1 "$ROOT/directory.json"
  said "reading pair set $ROOT/directory.json: the file cannot be read: "
  said "observed: path: $ROOT/directory.json; fix: name a pair-set file this account can read"
  passed a_directory_named_as_a_pair_set_is_refused
}

a_file_that_is_not_a_pair_set_is_refused() {
  printf '{"not": "pairs"}' >"$ROOT/other.json"
  inspect 1 "$ROOT/other.json"
  said "reading pair set $ROOT/other.json: the file is not a pair-set document; observed: invalid pair set JSON in $ROOT/other.json: unknown field \`not\`, expected \`trait_name\` or \`pairs\` at line 1 column 6; fix: write it as {\"trait_name\": \"<trait>\", \"pairs\": [{\"positive\": \"<text>\", \"negative\": \"<text>\"}]}, as docs/examples/pairs.json is"
  passed a_file_that_is_not_a_pair_set_is_refused
}

removing_the_last_pair_is_refused_and_the_file_kept() {
  run 0 pairs add --pairs "$ROOT/one.json" --positive "Glad to help." --negative "No."
  run 1 pairs remove --pairs "$ROOT/one.json" --index 0
  said "checking pair set $ROOT/one.json: pair set $ROOT/one.json contains no pairs; observed: pairs: 0; fix: add a pair with \`ster pairs add --pairs <file> --positive <text> --negative <text>\`"
  check "the file keeps its pair" "$(jq '.pairs | length' "$ROOT/one.json")" "1"
  passed removing_the_last_pair_is_refused_and_the_file_kept
}

a_pair_with_an_empty_side_is_refused() {
  printf '{"trait_name": "calm", "pairs": [{"positive": "Still.", "negative": "Loud."}, {"positive": "Quiet.", "negative": " "}]}' >"$ROOT/blank.json"
  inspect 1 "$ROOT/blank.json"
  said "checking pair set $ROOT/blank.json: pair set $ROOT/blank.json contains an empty positive or negative prompt; observed: pair 1 has an empty negative; fix: write that side with \`ster pairs edit --pairs <file> --index 1 --negative <text>\`, or take the pair out with \`ster pairs remove --pairs <file> --index 1\`"
  passed a_pair_with_an_empty_side_is_refused
}

a_set_under_a_file_is_refused() {
  : >"$ROOT/afile"
  run 1 pairs add --pairs "$ROOT/afile/set.json" --positive "Still." --negative "Loud."
  said "writing pair set $ROOT/afile/set.json: its directory $ROOT/afile cannot be created: "
  said "observed: path: $ROOT/afile/set.json; fix: name a path whose directory this account can create, or create $ROOT/afile first"
  check "the file in the way is untouched" "$(wc -c <"$ROOT/afile" | tr -d ' ')" "0"
  passed a_set_under_a_file_is_refused
}

a_set_in_a_read_only_directory_is_refused() {
  mkdir -p "$ROOT/read-only"
  chmod 555 "$ROOT/read-only"
  run 1 pairs add --pairs "$ROOT/read-only/set.json" --positive "Still." --negative "Loud."
  chmod 755 "$ROOT/read-only"
  said "writing pair set $ROOT/read-only/set.json: the file cannot be written: "
  said "observed: path: $ROOT/read-only/set.json; fix: make its directory writable for this account, or name another path"
  [ ! -e "$ROOT/read-only/set.json" ] || fail "a refused write left $ROOT/read-only/set.json"
  passed a_set_in_a_read_only_directory_is_refused
}

a_missing_pair_set_is_refused
a_directory_named_as_a_pair_set_is_refused
a_file_that_is_not_a_pair_set_is_refused
removing_the_last_pair_is_refused_and_the_file_kept
a_pair_with_an_empty_side_is_refused
a_set_under_a_file_is_refused
a_set_in_a_read_only_directory_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
