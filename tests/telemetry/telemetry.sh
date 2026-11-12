#!/bin/sh
# Real test of `ster telemetry` through the built binary, against the real
# Echo app collector and read back through Echo's own failure groups.
#
#   * nothing is queued before the user answers, and a script is never asked;
#   * with `errors`, a declared refusal is queued with its cause, and the
#     home directory is taken out of everything queued;
#   * `show` and `request telemetry/show` answer `{"events": [...]}`, the
#     object a request result must be;
#   * `flush` sends it to the collector, and Echo's failure groups hold it
#     under this installation;
#   * a command that ends with events waiting sends them from a detached
#     flush, and the command's own exit status is unchanged;
#   * `off` drops what waits; a wrong level, an unwritable directory and a
#     plain-http collector are refused by name.
#
# Usage: tests/telemetry/telemetry.sh
#   STER                            the binary, default target/debug/ster
#   STER_TEST_TELEMETRY_COLLECTOR   https://<echo web>/api/analytics/mobile/collect
#   STER_TEST_FLUSH_WAIT_SECONDS    how long the detached flush may take before the test fails
#   ECHO_CLI                        the echo-cli command that reads Echo back, e.g. "node ../echo/cli/main.mjs",
#                                   with ECHO_WEB_BASE_URL and ECHO_INTERNAL_BEARER set for it
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
: "${STER_TEST_TELEMETRY_COLLECTOR:?set it to the Echo app collector: this test sends to the real one}"
: "${STER_TEST_FLUSH_WAIT_SECONDS:?set it to how many seconds the detached flush may take}"
: "${ECHO_CLI:?set it to the echo-cli command that reads the failure groups back}"
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/telemetry/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
echo "collector: $STER_TEST_TELEMETRY_COLLECTOR" >>"$REPORT"
export WISENT_TELEMETRY_HOME="$ROOT/state"
# The binary runs under this run's own home, never the operator's; the pair
# set it misses lies inside that home, so the queued detail shows it as ~/.
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/xdg-state"
mkdir -p "$HOME"
unset WISENT_TELEMETRY_COLLECTOR_URL DO_NOT_TRACK

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
passed() { echo "ok: $1" >>"$REPORT"; }
inspect_missing() {
  run 1 pairs inspect --pairs "$HOME/absent.json" --dedupe-bits 3 --dedupe-bands 4 --unbalanced-ratio 3
}

run 0 telemetry status
printf '%s' "$out" | jq -e '.level == "unanswered" and .queued == 0 and .installation == null' >/dev/null \
  || fail "a new installation is not unanswered and empty: $out"
inspect_missing
said "reading pair set $HOME/absent.json: no file is there"
run 0 telemetry status
printf '%s' "$out" | jq -e '.queued == 0 and .installation == null' >/dev/null \
  || fail "a failure was queued before anyone answered: $out"
passed "nothing is queued before the user answers, and a script is never asked"

run 0 telemetry on --level errors
printf '%s' "$out" | jq -e '.level == "errors"' >/dev/null || fail "on --level errors did not keep errors: $out"
inspect_missing
run 0 telemetry show
EVENT=$out
printf '%s' "$EVENT" | jq -e '.events | length == 1 and .[0].eventName == "failure" and .[0].appId == "ster-cli"
  and .[0].properties.cause_id == "pair-set-missing" and .[0].properties.error_code == "not_found"' >/dev/null \
  || fail "the refusal is not queued with its declared cause: $EVENT"
case "$EVENT" in *"$HOME"*) fail "the queued event still holds the home directory: $EVENT" ;; esac
printf '%s' "$EVENT" | jq -e '.events[0].properties.detail | contains("~/")' >/dev/null \
  || fail "the home directory is not written as ~ in the detail: $EVENT"
EVENT_ID=$(printf '%s' "$EVENT" | jq -r '.events[0].eventId')
INSTALLATION=$(printf '%s' "$EVENT" | jq -r '.events[0].anonymousId')
POINT=$(printf '%s' "$EVENT" | jq -r '.events[0].properties.failure_point')
passed "a declared refusal is queued with its cause and without the home directory"

set +e
request=$(printf '{}' | "$BIN" request telemetry/show 2>"$ROOT/stderr")
request_status=$?
set -e
printf '$ ster request telemetry/show\nexit: %s\nstdout: %s\n\n' "$request_status" "$request" >>"$REPORT"
[ "$request_status" -eq 0 ] || fail "ster request telemetry/show exited $request_status: $(cat "$ROOT/stderr")"
printf '%s' "$request" | tail -n 1 | jq -e --arg event "$EVENT_ID" \
  '.status == 0 and (.json.events | length == 1) and .json.events[0].eventId == $event' >/dev/null \
  || fail "ster request telemetry/show does not answer the waiting event as {\"events\": [...]}: $request"
passed "request telemetry/show answers the object telemetry show prints"

export WISENT_TELEMETRY_COLLECTOR_URL="$STER_TEST_TELEMETRY_COLLECTOR"
run 0 telemetry flush
printf '%s' "$out" | jq -e '.sent == 1 and .remaining == 0 and .stopped == null' >/dev/null \
  || fail "flush did not send the event: $out"
set +e
groups=$($ECHO_CLI failures groups --app ster-cli --days 1 --installation "$INSTALLATION" 2>"$ROOT/echo-stderr")
echo_status=$?
set -e
printf '$ echo-cli failures groups --app ster-cli --days 1 --installation %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' \
  "$INSTALLATION" "$echo_status" "$groups" "$(cat "$ROOT/echo-stderr")" >>"$REPORT"
[ "$echo_status" -eq 0 ] || fail "Echo's failure groups could not be read: $(cat "$ROOT/echo-stderr")"
printf '%s' "$groups" | jq -e --arg event "$EVENT_ID" --arg point "$POINT" '
  any(.groups[]; .failure_point == $point and .cause_id == "pair-set-missing" and .installations == 1
    and .example.event_id == $event)' >/dev/null \
  || fail "Echo does not hold the sent failure under this installation: $groups"
passed "flush sends the failure and Echo groups it under this installation"

inspect_missing
waited=0
while :; do
  run 0 telemetry status
  queued=$(printf '%s' "$out" | jq -r '.queued')
  [ "$queued" = 0 ] && break
  [ "$waited" -ge "$STER_TEST_FLUSH_WAIT_SECONDS" ] && fail "the detached flush left $queued event(s) after $waited s: $out"
  sleep 1
  waited=$((waited + 1))
done
printf '%s' "$out" | jq -e '.last_delivery.sent == 1 and .last_delivery.stopped == null' >/dev/null \
  || fail "the detached flush did not record its delivery: $out"
passed "a failing command exits with its own status and its failure leaves from a detached flush"

run 0 telemetry off
printf '%s' "$out" | jq -e '.level == "off" and .queued == 0' >/dev/null || fail "off did not turn telemetry off: $out"
inspect_missing
run 0 telemetry status
printf '%s' "$out" | jq -e '.queued == 0' >/dev/null || fail "a failure was queued with telemetry off: $out"
passed "off sends nothing more"

run 2 telemetry on --level everything
said "invalid value 'everything' for '--level <LEVEL>'"
passed "a level that is not errors or usage is a usage error"

printf 'not a directory' >"$ROOT/a-file"
export WISENT_TELEMETRY_HOME="$ROOT/a-file/telemetry"
run 1 telemetry on --level errors
export WISENT_TELEMETRY_HOME="$ROOT/state"
said "keeping the telemetry consent in $ROOT/a-file/telemetry/consent.json: the telemetry directory could not be written"
said "fix: make $ROOT/a-file/telemetry writable by this user, or name a writable one with WISENT_TELEMETRY_HOME"
passed "an unwritable directory is refused by name"

run 0 telemetry on --level errors
inspect_missing
export WISENT_TELEMETRY_COLLECTOR_URL="http://collector.example/api/analytics/mobile/collect"
run 1 telemetry flush
export WISENT_TELEMETRY_COLLECTOR_URL="$STER_TEST_TELEMETRY_COLLECTOR"
said "reading the collector address from WISENT_TELEMETRY_COLLECTOR_URL: a collector address must be https://, or http:// on a loopback host; observed: WISENT_TELEMETRY_COLLECTOR_URL holds http://collector.example/api/analytics/mobile/collect"
passed "a plain-http collector is refused by name"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
