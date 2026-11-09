#!/bin/sh
# Real test of `ster errors`, the failure catalogue every functionality
# declares in src/failures, through the built binary and its request.
#
#   * errors list answers every functionality, and every cause names its
#     code, exit status, test and documentation anchor;
#   * every area page under docs/guide/errors/ a functionality names is,
#     below its generated-section marker, byte for byte what
#     errors list --markdown --page prints for it;
#   * errors show answers one functionality with every cause it meets
#     through the functionalities it uses;
#   * errors check --repository . with tests/errors/uncatalogued.json and
#     tests/errors/untested.json passes on this checkout;
#   * the causes of ster.errors, each by the name its declaration gives as its
#     test, and the same catalogue through `ster request errors/list|show`.
#
# Usage: tests/errors/catalogue.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/errors/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
BASELINES="--repository . --uncatalogued tests/errors/uncatalogued.json --untested tests/errors/untested.json"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" <"${STDIN:-/dev/null}")
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
passed() { echo "ok: $1" >>"$REPORT"; }

run 0 errors list
LIST=$out
printf '%s' "$LIST" | jq -e '
  (.functionalities | length > 0)
  and all(.functionalities[]; (.operation | startswith("ster."))
    and all(.causes[]; .code != null and (.exit | type == "number") and (.test | contains("::"))
      and (.doc | startswith("https://github.com/wisent-ai/ster/blob/main/docs/guide/errors/"))))' >/dev/null \
  || fail "errors list does not answer every cause with its code, exit, test and doc: $LIST"
passed "errors list answers every functionality with its causes"

PAGES=$(printf '%s' "$LIST" | jq -r '.functionalities[].docs' | sort -u \
  | sed 's|^https://github.com/wisent-ai/ster/blob/main/||')
for page in $PAGES; do
  [ -f "$page" ] || fail "a functionality is listed on $page, which this checkout does not hold"
  "$BIN" errors list --markdown --page "$page" >"$ROOT/markdown.md"
  sed -n '/^<!-- generated: ster errors list --markdown -->$/,$p' "$page" | sed 1d >"$ROOT/page.md"
  diff -u "$ROOT/page.md" "$ROOT/markdown.md" >>"$REPORT" \
    || fail "$page differs from ster errors list --markdown --page $page; regenerate its section below the marker"
  passed "$page publishes exactly its functionalities' catalogue"
done

run 0 errors show ster.workspace.import
printf '%s' "$out" | jq -e '
  ([.met[] | select(.declared_by == "ster.workspace.import")] | length > 0)
  and ([.met[] | select(.declared_by == "ster.pairs.file")] | length > 0)
  and ([.met[] | select(.declared_by == "ster.workspace.store")] | length > 0)' >/dev/null \
  || fail "errors show does not list the causes the import meets through what it uses: $out"
passed "errors show lists a functionality's own causes and those it meets"

run 0 errors check $BASELINES
passed "errors check passes on this checkout"

show_refuses_an_undeclared_functionality() {
  run 2 errors show ster.nothing
  said "showing the failure catalogue of ster.nothing: no functionality of this ster declares a catalogue under ster.nothing; observed: declared functionalities: ster.errors, ster.pairs.file, ster.workspace.store"
  said "fix: name one of the declared functionalities; \`ster errors list\` prints them"
  passed show_refuses_an_undeclared_functionality
}

markdown_refuses_a_page_no_functionality_is_listed_on() {
  run 2 errors list --markdown --page docs/guide/errors/nothing.md
  said "printing the catalogue sections of docs/guide/errors/nothing.md: no functionality is listed on docs/guide/errors/nothing.md; observed: pages: docs/guide/errors/catalogue.md, docs/guide/errors/pairs.md, docs/guide/errors/workspace.md; fix: name one of those pages, or leave --page out for every section"
  passed markdown_refuses_a_page_no_functionality_is_listed_on
}

check_refuses_an_unreadable_baseline() {
  run 2 errors check --uncatalogued "$ROOT/absent.json"
  said "reading the uncatalogued-command baseline $ROOT/absent.json: the file could not be read: "
  passed check_refuses_an_unreadable_baseline
}

check_refuses_a_baseline_that_is_not_a_list() {
  printf '{"status": true}' >"$ROOT/object.json"
  run 2 errors check --untested "$ROOT/object.json"
  said "reading the untested-cause baseline $ROOT/object.json: it is not a JSON list of strings: "
  passed check_refuses_a_baseline_that_is_not_a_list
}

check_finds_a_baseline_entry_no_longer_needed() {
  jq '. + ["workspace import"]' tests/errors/uncatalogued.json >"$ROOT/stale.json"
  run 1 errors check --repository . --uncatalogued "$ROOT/stale.json" --untested tests/errors/untested.json
  said "checking the failure catalogue: 1 finding(s)"
  printf '%s' "$out" | jq -e '.findings == ["the uncatalogued baseline still allows ster workspace import, but a functionality names it now; remove it from the baseline"]' >/dev/null \
    || fail "errors check does not name the stale baseline entry: $out"
  passed check_finds_a_baseline_entry_no_longer_needed
}

the_request_answers_the_same_catalogue() {
  printf '{}' >"$ROOT/empty.json"
  STDIN="$ROOT/empty.json"
  run 0 request errors/list
  STDIN=
  [ "$(printf '%s' "$out" | jq -cS '.json.functionalities')" = "$(printf '%s' "$LIST" | jq -cS '.functionalities')" ] \
    || fail "request errors/list answers another catalogue than errors list"
  printf '{"id": "ster.nothing"}' >"$ROOT/unknown.json"
  STDIN="$ROOT/unknown.json"
  run 1 request errors/show
  STDIN=
  printf '%s' "$out" | jq -se 'last | .json.cause.id == "functionality-unknown" and .json.cause.exit == 2' >/dev/null \
    || fail "request errors/show does not carry the declared cause: $out"
  passed the_request_answers_the_same_catalogue
}

show_refuses_an_undeclared_functionality
markdown_refuses_a_page_no_functionality_is_listed_on
check_refuses_an_unreadable_baseline
check_refuses_a_baseline_that_is_not_a_list
check_finds_a_baseline_entry_no_longer_needed
the_request_answers_the_same_catalogue

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
