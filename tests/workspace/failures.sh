#!/bin/sh
# Real test of how Ster's workspace fails, through the built binary, in a
# workspace this run creates under its own directory.
#
# Every cause src/failures/workspace declares and a local workspace can
# provoke is driven through the command a person types: a missing home, a
# workspace document that cannot be read, is not its document or is of
# another schema, a workspace under a file, one that cannot be written (the
# import is undone), an unknown set, an invalid or underivable name, a taken
# name, a destination outside the index, a leftover staging file, selecting
# the active set, a copy that cannot be deleted (the set is kept, and removing
# it works once the directory allows it), and a run without --pairs with no
# active set, an active set the workspace does not hold, and one whose copy is
# gone. Each refusal is asserted with its declared sentence and exit status,
# and the workspace on disk is read after it. A refused import is also run
# through `ster request workspace/import`, whose result carries the cause.
#
# Usage: tests/workspace/failures.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/workspace/$RUN"
mkdir -p "$ROOT/source"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
SRC="$ROOT/source"
WS="$ROOT/data/ster"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
trap 'chmod -R u+w "$ROOT" 2>/dev/null || true' EXIT

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
# run STATUS ARGS...: run ster with ARGS (after $PREFIX, and reading $STDIN,
# when set), record it, and require exit STATUS.
run() {
  expected=$1
  shift
  set +e
  out=$(${PREFIX:-} "$BIN" "$@" 2>"$ROOT/stderr" <"${STDIN:-/dev/null}")
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
fresh() {
  chmod -R u+w "$ROOT/data" 2>/dev/null || true
  rm -rf "$ROOT/data" "$ROOT/state" "$ROOT/home"
  mkdir -p "$ROOT/home"
}
passed() { echo "ok: $1" >>"$REPORT"; }

[ "$(id -u)" != 0 ] || fail "run as an ordinary account: root ignores the read-only directories these refusals need"
cp docs/examples/pairs.json "$SRC/calm.json"
jq '.pairs |= .[0:2]' "$SRC/calm.json" >"$SRC/other.json"
cp "$SRC/calm.json" "$SRC/---.json"
a_workspace_without_a_home_is_refused() {
  fresh
  PREFIX="env -u HOME -u XDG_DATA_HOME"
  run 1 workspace show
  PREFIX=
  said "locating the Ster workspace: neither XDG_DATA_HOME nor HOME is set; observed: XDG_DATA_HOME is unset; HOME is unset; fix: set XDG_DATA_HOME, or HOME, to the directory the workspace lives under"
  [ ! -e "$WS" ] || fail "a refused run wrote $WS"
  passed a_workspace_without_a_home_is_refused
}

an_unreadable_workspace_is_refused() {
  fresh
  mkdir -p "$WS/workspace.json"
  run 1 workspace show
  said "reading the Ster workspace $WS/workspace.json: $WS/workspace.json is a directory, and the Ster workspace is one file; observed: path: $WS/workspace.json (a directory, "
  passed an_unreadable_workspace_is_refused
}

a_workspace_that_is_not_its_document_is_refused() {
  fresh
  mkdir -p "$WS"
  printf '{' >"$WS/workspace.json"
  run 1 workspace show
  said "reading the Ster workspace $WS/workspace.json: it ends before its JSON value closes: it was cut short while it was written (a full disk, a process stopped midway); observed: path: $WS/workspace.json (a file, 1 bytes, "
  said "; fix: write the Ster workspace again; it is the record Ster writes itself (restore it from a copy, or move it aside and import the pair sets again with \`ster workspace import\`)"
  check "the document is left as it was" "$(cat "$WS/workspace.json")" "{"
  passed a_workspace_that_is_not_its_document_is_refused
}

a_workspace_of_another_schema_is_refused() {
  fresh
  mkdir -p "$WS"
  printf '{"schema": "ster.workspace.v0"}' >"$WS/workspace.json"
  run 1 workspace select calm
  said "reading the Ster workspace $WS/workspace.json: unsupported Ster workspace schema ster.workspace.v0; this ster reads ster.workspace.v1; observed: schema: ster.workspace.v0; fix: read it with the ster that wrote it"
  passed a_workspace_of_another_schema_is_refused
}

a_workspace_under_a_file_is_refused() {
  fresh
  : >"$ROOT/afile"
  PREFIX="env XDG_DATA_HOME=$ROOT/afile"
  run 1 workspace import "$SRC/calm.json"
  PREFIX=
  said "creating the Ster workspace directory $ROOT/afile/ster/pairs: the path runs through a file: $ROOT/afile is not a directory, so nothing can be created below it; observed: path: $ROOT/afile/ster/pairs; $ROOT/afile: a file, 0 bytes, "
  said "; nothing was written under $ROOT/afile/ster; the system answered: "
  said "; fix: name a path that runs through directories only, or move $ROOT/afile out of the way"
  passed a_workspace_under_a_file_is_refused
}

a_read_only_workspace_is_refused_and_the_import_undone() {
  fresh
  run 0 workspace import "$SRC/calm.json"
  before=$(cat "$WS/workspace.json")
  chmod 555 "$WS"
  run 1 workspace import "$SRC/other.json" --name second
  chmod 755 "$WS"
  said "importing $SRC/other.json as second was undone: $WS/pairs/second.json was removed"
  said "creating the draft of the Ster workspace $WS/workspace.json."
  said ".tmp: this account may not change $WS or $WS/workspace.json."
  said "; nothing was written; $WS/workspace.json is unchanged; the system answered: "
  said "; fix: give this account write access to $WS (\`chmod u+w $WS\` as its owner), or name a path in a directory it may write"
  [ ! -e "$WS/pairs/second.json" ] || fail "the undone import left $WS/pairs/second.json"
  check "the workspace document is unchanged" "$(cat "$WS/workspace.json")" "$before"
  passed a_read_only_workspace_is_refused_and_the_import_undone
}

an_unknown_set_is_refused() {
  fresh
  run 1 workspace select ghost
  said "finding pair set ghost: the Ster workspace holds no pair set named ghost; it holds none — import one with \`ster workspace import\`; observed: workspace: $WS/workspace.json; fix: name a set the workspace holds"
  run 0 workspace import "$SRC/calm.json"
  run 1 workspace remove ghost
  said "the Ster workspace holds no pair set named ghost; it holds calm; observed"
  check "the set is still held" "$(jq -r '.pair_sets[].id' "$WS/workspace.json")" "calm"
  passed an_unknown_set_is_refused
}

an_invalid_name_is_refused() {
  fresh
  run 2 workspace import "$SRC/calm.json" --name=-bad
  said "naming the imported set -bad: pair-set name must start with an ASCII letter or digit and contain only letters, digits, dots, underscores, or hyphens; observed: --name -bad; fix: pass a --name such as my-pairs"
  check "the report says rejected" "$(echo "$out" | jq -r .status)" "rejected"
  check "the report names the cause" "$(echo "$out" | jq -r .cause.id)" "name-invalid"
  [ ! -e "$WS/workspace.json" ] || fail "a rejected import wrote $WS/workspace.json"
  passed an_invalid_name_is_refused
}

a_file_name_without_letters_needs_a_name() {
  fresh
  run 2 workspace import "$SRC/---.json"
  said "deriving a pair-set id from $SRC/---.json: $SRC/---.json has no ASCII letter or digit in its file name, so no pair-set id can be derived from it; observed: file name: ---.json; fix: name the set with --name"
  [ ! -e "$WS/workspace.json" ] || fail "a rejected import wrote $WS/workspace.json"
  passed a_file_name_without_letters_needs_a_name
}

a_taken_name_is_refused_and_the_first_set_kept() {
  fresh
  run 0 workspace import "$SRC/calm.json"
  before=$(cat "$WS/workspace.json")
  run 1 workspace import "$SRC/other.json" --name calm
  said "importing $SRC/other.json as calm: pair-set name conflicts with existing content; observed: calm holds $WS/pairs/calm.json, a different set; fix: choose another --name to preserve $WS/pairs/calm.json"
  check "the report says conflicting" "$(echo "$out" | jq -r .status)" "conflicting"
  printf '{"source": "%s", "name": "calm"}' "$SRC/other.json" >"$ROOT/body.json"
  STDIN="$ROOT/body.json"
  run 1 request workspace/import
  STDIN=
  check "the request's result carries the cause" "$(echo "$out" | jq -sr 'last | .json.cause.id')" "name-conflict"
  check "the cause names the CLI's exit status" "$(echo "$out" | jq -sr 'last | .json.cause.exit')" "1"
  check "the cause names the functionality that declares it" "$(echo "$out" | jq -sr 'last | .json.cause.functionality')" "ster.workspace.import"
  check "the cause links its line on the workspace page" "$(echo "$out" | jq -sr 'last | .json.cause.docs')" \
    "https://github.com/wisent-ai/ster/blob/main/docs/guide/errors/workspace.md#ster-workspace-import-name-conflict"
  check "the first set keeps its pairs" "$(jq '.pairs | length' "$WS/pairs/calm.json")" "8"
  check "the workspace document is unchanged" "$(cat "$WS/workspace.json")" "$before"
  passed a_taken_name_is_refused_and_the_first_set_kept
}

a_file_outside_the_index_is_not_replaced() {
  fresh
  mkdir -p "$WS/pairs"
  printf '{}' >"$WS/pairs/calm.json"
  run 1 workspace import "$SRC/calm.json"
  said "importing $SRC/calm.json as calm: the destination already exists outside the Ster workspace index; it was not replaced; observed: file: $WS/pairs/calm.json; fix: move $WS/pairs/calm.json out of the workspace, or import the set under another --name"
  check "the file outside the index is kept" "$(cat "$WS/pairs/calm.json")" "{}"
  passed a_file_outside_the_index_is_not_replaced
}

a_leftover_staging_file_is_refused() {
  fresh
  mkdir -p "$WS/pairs"
  # The staging file carries the importing process's id: the shell names it
  # with its own and hands that id to ster by exec.
  set +e
  sh -c 'touch "$1/.calm.$$.incoming"; exec "$2" workspace import "$3"' sh "$WS/pairs" "$BIN" "$SRC/calm.json" \
    >"$ROOT/stdout" 2>"$ROOT/stderr" </dev/null
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ ster workspace import %s (staging file in place)\nexit: %s\nstderr: %s\n\n' "$SRC/calm.json" "$status" "$err" >>"$REPORT"
  check "a leftover staging file refuses the import" "$status" "1"
  said "staging the import of $SRC/calm.json as calm: a staging file of an earlier import that did not finish is in the way; observed: staging file: $WS/pairs/.calm."
  said ".incoming: no workspace entry records it"
  [ ! -e "$WS/pairs/calm.json" ] || fail "the refused import wrote $WS/pairs/calm.json"
  passed a_leftover_staging_file_is_refused
}

selecting_the_active_set_is_refused() {
  fresh
  run 0 workspace import "$SRC/calm.json"
  run 1 workspace select calm
  said "selecting pair set calm: pair set calm is already the active one; observed: active pair set: calm; fix: nothing needs doing"
  check "calm stays active" "$(jq -r .active_pair_set "$WS/workspace.json")" "calm"
  passed selecting_the_active_set_is_refused
}

a_copy_that_cannot_be_deleted_keeps_the_set() {
  fresh
  run 0 workspace import "$SRC/calm.json"
  chmod 555 "$WS/pairs"
  run 1 workspace remove calm
  chmod 755 "$WS/pairs"
  said "removing the copy of pair set calm $WS/pairs/calm.json: this account may not change $WS/pairs or $WS/pairs/calm.json: their permission bits or owner keep this account out; observed: path: $WS/pairs/calm.json (a file, "
  said "; the workspace still holds calm; nothing was changed; the system answered: "
  said "; fix: give this account write access to $WS/pairs (\`chmod u+w $WS/pairs\` as its owner), or name a path in a directory it may write"
  check "the workspace still holds calm" "$(jq -r '.pair_sets[].id' "$WS/workspace.json")" "calm"
  check "calm is still active" "$(jq -r .active_pair_set "$WS/workspace.json")" "calm"
  [ -f "$WS/pairs/calm.json" ] || fail "the kept set lost its copy"
  run 0 workspace remove calm
  [ ! -e "$WS/pairs/calm.json" ] || fail "removing calm again left its copy"
  check "the workspace holds nothing" "$(jq -c .pair_sets "$WS/workspace.json")" "[]"
  passed a_copy_that_cannot_be_deleted_keeps_the_set
}

a_run_without_pairs_and_no_active_set_is_refused() {
  fresh
  run 1 $INSPECT
  said "choosing the pair set to read: no active Ster pair set; observed: the workspace holds none; fix: pass --pairs, select an imported set with \`ster workspace select <id>\`, or import one with \`ster workspace import\`"
  passed a_run_without_pairs_and_no_active_set_is_refused
}

an_active_set_the_workspace_does_not_hold_is_refused() {
  fresh
  run 0 workspace import "$SRC/calm.json"
  jq '.active_pair_set = "ghost"' "$WS/workspace.json" >"$ROOT/workspace.json"
  mv "$ROOT/workspace.json" "$WS/workspace.json"
  run 1 $INSPECT
  said "choosing the pair set to read: the Ster workspace names ghost active but holds no pair set by that name; observed: it holds calm; fix: select a set it holds with \`ster workspace select <id>\`, or pass --pairs"
  passed an_active_set_the_workspace_does_not_hold_is_refused
}

an_active_set_whose_copy_is_gone_is_refused() {
  fresh
  run 0 workspace import "$SRC/calm.json"
  mv "$WS/pairs/calm.json" "$ROOT/moved.json"
  run 1 $INSPECT
  said "reading the active pair set calm: active Ster pair set is missing: $WS/pairs/calm.json; observed: the workspace records calm at $WS/pairs/calm.json; fix: import it again with \`ster workspace import\`"
  passed an_active_set_whose_copy_is_gone_is_refused
}

a_workspace_without_a_home_is_refused
an_unreadable_workspace_is_refused
a_workspace_that_is_not_its_document_is_refused
a_workspace_of_another_schema_is_refused
a_workspace_under_a_file_is_refused
a_read_only_workspace_is_refused_and_the_import_undone
an_unknown_set_is_refused
an_invalid_name_is_refused
a_file_name_without_letters_needs_a_name
a_taken_name_is_refused_and_the_first_set_kept
a_file_outside_the_index_is_not_replaced
a_leftover_staging_file_is_refused
selecting_the_active_set_is_refused
a_copy_that_cannot_be_deleted_keeps_the_set
a_run_without_pairs_and_no_active_set_is_refused
an_active_set_the_workspace_does_not_hold_is_refused
an_active_set_whose_copy_is_gone_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
