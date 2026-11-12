#!/bin/sh
# Real test of how a desktop request fails before its operation runs
# (`ster.request`, docs/guide/desktop-requests.md): through `ster request`
# itself, with no model, under the run's own HOME. Each refusal is the result
# event's status 2 with `json.cause` naming the declared cause, except a
# standard output nobody reads (status 2, the refusal on stderr) and a pipe
# that breaks under an event (the retryable status 69, on stderr). Every
# command, its exit status and output go to the run's report.txt.
#
# Usage: tests/request/refusals.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/request/refusals-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT/home"
# The binary runs under this run's own home, never the operator's.
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
# request OPERATION BODY: the body on stdin; $result is the last event.
request() {
  set +e
  out=$(printf '%s' "$2" | "$BIN" request "$1" 2>"$ROOT/stderr")
  status=$?
  set -e
  printf '$ ster request %s <<< %s\nexit: %s\nstdout: %s\n\n' "$1" "$2" "$status" "$out" >>"$REPORT"
  result=$(printf '%s\n' "$out" | tail -n 1)
}
# refused ID PHRASE: the request ended in status 2 with cause ID and PHRASE
# in its sentence.
refused() {
  [ "$status" -eq 2 ] || fail "expected status 2, got $status: $out"
  [ "$(echo "$result" | jq -r .json.cause.id)" = "$1" ] || fail "expected cause $1, got: $result"
  case "$(echo "$result" | jq -r .json.error)" in
    *"$2"*) echo "ok: $1: $2" >>"$REPORT" ;;
    *) fail "expected '$2' in the refusal, got: $result" ;;
  esac
}

an_unknown_operation_is_refused() {
  request vector/inspct '{}'
  refused operation-unknown "starting request vector/inspct: unknown operation: vector/inspct is not an operation \`ster request\` serves; observed: operation: vector/inspct; served: model/fixture/create, "
}

a_body_that_is_not_json_is_refused() {
  request errors/show 'nope'
  refused body-not-json "reading the body of request errors/show: request body does not match the operation schema: it is not JSON; observed: 4 bytes; the JSON reader stopped at line 1, column 2: "
}

a_body_cut_short_is_refused() {
  request errors/show '{"id": "ster.request"'
  refused body-truncated "request body does not match the operation schema: it ends before its JSON value closes"
}

a_body_of_another_shape_is_refused() {
  request errors/show '{"name": "ster.request"}'
  refused body-shape "request body does not match the operation schema: a field errors/show requires is missing, a field it does not take is present"
  case "$result" in *'unknown field `name`'*) ;; *) fail "the parser's reason is not observed: $result" ;; esac
}

a_request_without_a_required_field_is_refused() {
  request errors/show '{}'
  refused field-blank "checking the errors/show request: errors/show requires the functionality's id, as errors/list names it: the request's id field is absent or blank; observed: id: \"\""
}

a_list_shorter_than_its_operation_needs_is_refused() {
  request vector/compare '{"artifacts": ["one.ster.json"], "clusters": 1}'
  refused list-short "vector/compare requires at least 2 steering artifacts: the request's artifacts list holds 1; observed: artifacts: 1 entries"
}

a_field_without_its_partner_is_refused() {
  request vector/evaluate '{"model": "m", "pairs": "p.json", "vector": "v.ster.json", "batchSize": 4}'
  refused field-unpaired "batchSize requires strengths: the request gives batchSize and leaves strengths out; observed: batchSize: 4; strengths: absent"
}

# A directory as standard input: reading it fails with the system's answer,
# and the request is refused (status 2) with the directory's state.
a_body_from_a_directory_is_refused() {
  set +e
  out=$("$BIN" request errors/list <"$ROOT" 2>"$ROOT/stderr")
  status=$?
  set -e
  printf '$ ster request errors/list < %s\nexit: %s\nstdout: %s\n\n' "$ROOT" "$status" "$out" >>"$REPORT"
  result=$(printf '%s\n' "$out" | tail -n 1)
  refused body-directory "reading the body of request errors/list from standard input: standard input is a directory, not a stream of the body"
  case "$(echo "$result" | jq -r .json.error)" in
    *"observed: standard input: a directory, "*"error kind IsADirectory"*) ;;
    *) fail "the directory's state is not observed: $result" ;;
  esac
}

# A standard output nobody can read: closed (`>&-`, which Rust's runtime
# turns into the null device) and the null device itself. Nothing runs, and
# the refusal goes to stderr, the one stream left.
a_standard_output_nobody_reads_is_refused() {
  for redirect in closed null; do
    set +e
    case $redirect in
      closed) printf '{}' | "$BIN" request errors/list >&- 2>"$ROOT/stderr" ;;
      null) printf '{}' | "$BIN" request errors/list >/dev/null 2>"$ROOT/stderr" ;;
    esac
    status=$?
    set -e
    err=$(cat "$ROOT/stderr")
    printf '$ ster request errors/list <<< {} (stdout %s)\nexit: %s\nstderr: %s\n\n' "$redirect" "$status" "$err" >>"$REPORT"
    [ "$status" -eq 2 ] || fail "expected status 2 with stdout $redirect, got $status: $err"
    case "$err" in
      *"starting request errors/list: standard output reaches no reader"*"observed: standard output: "*"standard input: a pipe"*) ;;
      *) fail "unexpected refusal with stdout $redirect: $err" ;;
    esac
  done
  echo "ok: stdout-discarded" >>"$REPORT"
}

# A reader that has gone away: `true` exits at once, and the body arrives a
# second later, so the first event meets a pipe nobody reads.
an_event_nobody_reads_is_refused() {
  (sleep 1; printf '{}') | { "$BIN" request errors/list 2>"$ROOT/stderr"; echo $? >"$ROOT/status"; } | true
  status=$(cat "$ROOT/status")
  err=$(cat "$ROOT/stderr")
  printf '$ (sleep 1; printf {}) | ster request errors/list | true\nexit: %s\nstderr: %s\n\n' "$status" "$err" >>"$REPORT"
  [ "$status" -eq 69 ] || fail "expected status 69, got $status: $err"
  case "$err" in
    *"writing a result event to standard output: nobody reads the request's standard output any more"*) ;;
    *) fail "unexpected refusal: $err" ;;
  esac
  echo "ok: event-unwritten" >>"$REPORT"
}

an_unknown_operation_is_refused
a_body_that_is_not_json_is_refused
a_body_cut_short_is_refused
a_body_of_another_shape_is_refused
a_request_without_a_required_field_is_refused
a_list_shorter_than_its_operation_needs_is_refused
a_field_without_its_partner_is_refused
a_standard_output_nobody_reads_is_refused
a_body_from_a_directory_is_refused
an_event_nobody_reads_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
