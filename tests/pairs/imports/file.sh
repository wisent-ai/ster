#!/bin/sh
# Real test of how reading, checking and writing a pair-set file fails
# (`ster.pairs.file` and the `ster.file` layer it uses), through the built
# binary, on files this run creates under its own directory.
#
# A missing file, one this account may not read, a directory, a path through
# a file, bytes that are not JSON, JSON cut short, JSON that is not a pair
# set, a set whose last pair is removed (the file is kept), a pair with an
# empty side, a set under a file, a set in a read-only directory and a merge
# into a directory are each refused with their declared sentence and exit
# status, and the file on disk is read after it. A missing file handed to
# `ster workspace import` is rejected with the same cause in its report.
#
# Usage: tests/pairs/imports/file.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-file/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
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
  said "reading pair set $ROOT/absent.json: no file is there: nothing at $ROOT/absent.json is the pair set named; observed: path: $ROOT/absent.json (absent); its directory $ROOT; the system answered: "
  said "; fix: name an existing pair-set file (\`ls -ln $ROOT\` shows what that directory holds), or write one with \`ster pairs add --pairs $ROOT/absent.json --positive <text> --negative <text>\`"
  run 1 workspace import "$ROOT/absent.json"
  check "the import is rejected" "$(echo "$out" | jq -r .status)" "rejected"
  check "its report names the cause" "$(echo "$out" | jq -r .cause.id)" "pair-set-missing"
  passed a_missing_pair_set_is_refused
}

a_pair_set_this_account_may_not_read_is_refused() {
  printf '{"trait_name": "calm", "pairs": [{"positive": "Still.", "negative": "Loud."}]}' >"$ROOT/closed.json"
  chmod 000 "$ROOT/closed.json"
  inspect 1 "$ROOT/closed.json"
  chmod 644 "$ROOT/closed.json"
  said "reading the pair set $ROOT/closed.json: this account may not read it: the file's permission bits or owner keep this account out; observed: path: $ROOT/closed.json (a file, "
  said ", mode 000, owner uid $(id -u)); the system answered: "
  said "; fix: give this account read access to $ROOT/closed.json (\`chmod u+r $ROOT/closed.json\` as its owner), or name a copy it may read"
  passed a_pair_set_this_account_may_not_read_is_refused
}

a_directory_named_as_a_pair_set_is_refused() {
  mkdir -p "$ROOT/directory.json"
  inspect 1 "$ROOT/directory.json"
  said "reading the pair set $ROOT/directory.json: $ROOT/directory.json is a directory, and the pair set is one file; observed: path: $ROOT/directory.json (a directory, "
  said "; fix: name the file that holds the pair set, not a directory; \`ls -ln $ROOT/directory.json\` shows what the directory holds"
  passed a_directory_named_as_a_pair_set_is_refused
}

a_pair_set_below_a_file_is_refused() {
  : >"$ROOT/plain"
  inspect 1 "$ROOT/plain/set.json"
  said "reading the pair set $ROOT/plain/set.json: the path runs through a file: $ROOT/plain is not a directory, so nothing can be found below it; observed: path: $ROOT/plain/set.json; $ROOT/plain: a file, 0 bytes, "
  said "; fix: name the pair set by a path that runs through directories only; $ROOT/plain is a file"
  passed a_pair_set_below_a_file_is_refused
}

a_pair_set_that_is_not_json_is_refused() {
  printf 'nope' >"$ROOT/notjson.json"
  inspect 1 "$ROOT/notjson.json"
  said "reading the pair set $ROOT/notjson.json: it is not JSON, and the pair set is the JSON \`ster pairs add\`, \`ster pairs import\` or \`ster pairs synthesize\` writes, "
  said ": it is another kind of file, or one broken by hand; observed: path: $ROOT/notjson.json (a file, 4 bytes, "
  said "; the JSON reader stopped at line 1, column "
  passed a_pair_set_that_is_not_json_is_refused
}

a_pair_set_cut_short_is_refused() {
  printf '{"trait_name": "calm", "pairs": [' >"$ROOT/cut.json"
  inspect 1 "$ROOT/cut.json"
  said "reading the pair set $ROOT/cut.json: it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway); observed: path: $ROOT/cut.json (a file, 33 bytes, "
  said "; the JSON reader ran out at line 1, column "
  said "; fix: write the pair set again; it is the JSON \`ster pairs add\`"
  passed a_pair_set_cut_short_is_refused
}

a_file_that_is_not_a_pair_set_is_refused() {
  printf '{"not": "pairs"}' >"$ROOT/other.json"
  inspect 1 "$ROOT/other.json"
  said "reading the pair set $ROOT/other.json: it is JSON of another form than the pair set, which is the JSON \`ster pairs add\`, \`ster pairs import\` or \`ster pairs synthesize\` writes, "
  said ": a field it requires is missing, a field holds another type or value, or it is a document of another kind; observed: path: $ROOT/other.json (a file, 16 bytes, "
  said "; at line 1, column 6 the reader answered: unknown field \`not\`, expected \`trait_name\` or \`pairs\` at line 1 column 6"
  passed a_file_that_is_not_a_pair_set_is_refused
}

removing_the_last_pair_is_refused_and_the_file_kept() {
  run 0 pairs add --pairs "$ROOT/one.json" --positive "Glad to help." --negative "No."
  run 1 pairs remove --pairs "$ROOT/one.json" --index 0
  said "checking pair set $ROOT/one.json: pair set $ROOT/one.json contains no pairs; observed: pairs: 0; trait_name: "
  said "; fix: add a pair with \`ster pairs add --pairs <file> --positive <text> --negative <text>\`"
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
  said "creating the directory of the pair set $ROOT/afile/set.json: the path runs through a file: $ROOT/afile is not a directory, so nothing can be created below it; observed: path: $ROOT/afile/set.json; $ROOT/afile: a file, 0 bytes, "
  said "; nothing was written; the system answered: "
  said "; fix: name a path that runs through directories only, or move $ROOT/afile out of the way"
  check "the file in the way is untouched" "$(wc -c <"$ROOT/afile" | tr -d ' ')" "0"
  passed a_set_under_a_file_is_refused
}

a_set_in_a_read_only_directory_is_refused() {
  mkdir -p "$ROOT/read-only"
  chmod 555 "$ROOT/read-only"
  run 1 pairs add --pairs "$ROOT/read-only/set.json" --positive "Still." --negative "Loud."
  chmod 755 "$ROOT/read-only"
  said "opening for writing the pair set $ROOT/read-only/set.json: this account may not change $ROOT/read-only or $ROOT/read-only/set.json: their permission bits or owner keep this account out; observed: path: $ROOT/read-only/set.json (absent); directory $ROOT/read-only (a directory, "
  said ", mode 555, owner uid $(id -u)); nothing was written, and a set already there is unchanged; the system answered: "
  said "; fix: give this account write access to $ROOT/read-only (\`chmod u+w $ROOT/read-only\` as its owner), or name a path in a directory it may write"
  [ ! -e "$ROOT/read-only/set.json" ] || fail "a refused write left $ROOT/read-only/set.json"
  passed a_set_in_a_read_only_directory_is_refused
}

a_merge_into_a_directory_is_refused() {
  printf '{"trait_name": "calm", "pairs": [{"positive": "Still.", "negative": "Loud."}]}' >"$ROOT/calm.json"
  printf '{"trait_name": "kind", "pairs": [{"positive": "Thank you.", "negative": "Go away."}]}' >"$ROOT/kind.json"
  mkdir -p "$ROOT/merged.json"
  run 1 pairs merge --pairs "$ROOT/calm.json" --pairs "$ROOT/kind.json" --trait goodness --output "$ROOT/merged.json"
  said "opening for writing the pair set $ROOT/merged.json: $ROOT/merged.json is a directory, and the pair set is one file; observed: path: $ROOT/merged.json (a directory, "
  said "; fix: name a file path for the pair set, not the directory $ROOT/merged.json"
  [ -z "$(ls -A "$ROOT/merged.json")" ] || fail "a refused merge wrote into $ROOT/merged.json"
  passed a_merge_into_a_directory_is_refused
}

a_missing_pair_set_is_refused
a_pair_set_this_account_may_not_read_is_refused
a_directory_named_as_a_pair_set_is_refused
a_pair_set_below_a_file_is_refused
a_pair_set_that_is_not_json_is_refused
a_pair_set_cut_short_is_refused
a_file_that_is_not_a_pair_set_is_refused
removing_the_last_pair_is_refused_and_the_file_kept
a_pair_with_an_empty_side_is_refused
a_set_under_a_file_is_refused
a_set_in_a_read_only_directory_is_refused
a_merge_into_a_directory_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
