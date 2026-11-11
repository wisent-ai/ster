#!/bin/sh
# Real test of `ster doctor` through the built binary, in a home of the run's
# own under target/real-tests/doctor.
#
#   * every functionality `ster errors list` declares with an entry point is
#     in the diagnosis exactly once, worked, failed or skipped with why;
#   * a fresh home has no active pair set: `ster.workspace.active` fails with
#     the workspace's own sentence and fix, and the diagnosis exits 1;
#   * after a pair set is imported everything that is tried works, and the
#     diagnosis exits 0;
#   * a telemetry directory that cannot be written fails `ster.telemetry`
#     naming the directory and how to fix it.
#
# Usage: tests/doctor/doctor.sh
#   STER   the binary, default target/debug/ster
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/doctor/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT/home" "$ROOT/data" "$ROOT/state"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" WISENT_TELEMETRY_HOME="$ROOT/state"
# Port 9, the IANA discard port: nothing is sent anywhere.
export WISENT_TELEMETRY_COLLECTOR_URL="http://127.0.0.1:9/api/analytics/mobile/collect"
export WISENT_SUPPORT_URL="http://127.0.0.1:9/v1/public/reports"
unset DO_NOT_TRACK

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
# run EXPECTED NAME ARGS...: the command's stdout in $ROOT/NAME.out.
run() {
  expected=$1 name=$2
  shift 2
  set +e
  "$BIN" "$@" >"$ROOT/$name.out" 2>"$ROOT/$name.err"
  status=$?
  set -e
  echo "ster $* -> $status" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster $* exited $status, expected $expected: $(cat "$ROOT/$name.err")"
}
state() {
  jq -r --arg id "$2" '.[] | select(.operation == $id) | .state' "$ROOT/$1.out"
}

run 0 catalogue errors list
jq -r '[.functionalities[] | select((.entries // []) | length > 0) | .operation] | sort | .[]' "$ROOT/catalogue.out" >"$ROOT/declared.txt"
[ -s "$ROOT/declared.txt" ] || fail "ster errors list declares no functionality with an entry point"

run 1 fresh doctor --json
jq -r '[.[].operation] | sort | .[]' "$ROOT/fresh.out" >"$ROOT/diagnosed.txt"
diff "$ROOT/declared.txt" "$ROOT/diagnosed.txt" >"$ROOT/coverage.diff" ||
  fail "the diagnosis and the catalogue's functionalities with an entry point differ: $(cat "$ROOT/coverage.diff")"
[ "$(state fresh ster.workspace.active)" = fails ] || fail "a home without a pair set does not fail ster.workspace.active"
jq -e '.[] | select(.operation == "ster.workspace.active") | .remedy | length > 0' "$ROOT/fresh.out" >/dev/null ||
  fail "ster.workspace.active fails without a fix"
for skipped in ster.workspace.import ster.workspace.select ster.workspace.remove; do
  [ "$(state fresh $skipped)" = skipped ] || fail "$skipped is tried, though it changes the user's data"
done
[ "$(state fresh ster.telemetry)" = works ] || fail "ster.telemetry does not work in a writable home: $(cat "$ROOT/fresh.out")"
run 1 fresh-lines doctor
grep -q '^failed  ster.workspace.active: .*; fix: ' "$ROOT/fresh-lines.out" || fail "the line for ster.workspace.active names no fix: $(cat "$ROOT/fresh-lines.out")"

cp docs/examples/pairs.json "$ROOT/calm.json"
run 0 import workspace import "$ROOT/calm.json"
run 0 ready doctor --json
jq -e 'all(.[]; .state != "fails")' "$ROOT/ready.out" >/dev/null || fail "a set-up home still fails: $(cat "$ROOT/ready.out")"
[ "$(state ready ster.workspace.active)" = works ] || fail "the imported set is not the active one"

chmod 0500 "$ROOT/state"
run 1 unwritable doctor --json
chmod 0700 "$ROOT/state"
[ "$(state unwritable ster.telemetry)" = fails ] || fail "an unwritable telemetry directory does not fail ster.telemetry"
jq -e --arg dir "$ROOT/state" '.[] | select(.operation == "ster.telemetry") | (.detail | contains($dir)) and (.remedy | contains("writable"))' \
  "$ROOT/unwritable.out" >/dev/null || fail "the telemetry failure does not name the directory and its fix: $(cat "$ROOT/unwritable.out")"

# The desktop's Diagnosis screen reads the same findings through the request.
set +e
echo '{}' | "$BIN" request doctor >"$ROOT/request.out" 2>"$ROOT/request.err"
status=$?
set -e
echo "ster request doctor -> $status" >>"$REPORT"
[ "$status" -eq 0 ] || fail "ster request doctor exited $status: $(cat "$ROOT/request.err")"
jq -s -e '[.[] | select(.type == "result")] | length == 1 and .[0].status == 0 and .[0].json.status == 0 and (.[0].json.findings | length) > 0' \
  "$ROOT/request.out" >/dev/null || fail "ster request doctor does not answer the findings: $(cat "$ROOT/request.out")"
jq -s -r '.[] | select(.type == "result") | .json.findings | sort_by(.operation) | map(.operation + " " + .state) | .[]' "$ROOT/request.out" >"$ROOT/request.txt"
jq -r 'sort_by(.operation) | map(.operation + " " + .state) | .[]' "$ROOT/ready.out" >"$ROOT/ready.txt"
diff "$ROOT/ready.txt" "$ROOT/request.txt" >/dev/null || fail "the request's findings differ from ster doctor --json's"

echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
