#!/bin/sh
# Real test of the frustration signals through the built binary, in a home of
# the run's own under target/real-tests/signals, the collector the discard
# port so every event stays queued and is read back with `ster telemetry show`.
#
#   * the same refusal twice queues frustration.repeat_failure with
#     consecutive 2 and its failure point;
#   * the same usage error (exit 2) twice queues frustration.usage_loop;
#   * `--help` after a failure queues frustration.help_after_failure naming
#     the command and the failure, and no argument;
#   * with consent `errors` neither usage signal is queued.
#
# Usage: tests/telemetry/signals.sh
#   STER   the binary, default .build/kompilator/debug/ster
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/signals/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT/home" "$ROOT/data"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data"
# Port 9, the IANA discard port: nothing is sent, every event stays queued.
export WISENT_TELEMETRY_COLLECTOR_URL="http://127.0.0.1:9/api/analytics/mobile/collect"
unset DO_NOT_TRACK

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
ster() {
  set +e
  "$BIN" "$@" >/dev/null 2>>"$ROOT/stderr.log"
  status=$?
  set -e
  echo "ster $* -> $status" >>"$REPORT"
}
# The properties of every queued event named $2, in the home $1's queue;
# `ster telemetry show` prints the queue as one array of events.
queued() {
  WISENT_TELEMETRY_HOME="$1" "$BIN" telemetry show 2>/dev/null |
    jq -c --arg name "$2" '[.[] | select(.eventName == $name) | .properties]'
}

export WISENT_TELEMETRY_HOME="$ROOT/usage"
ster telemetry on --level usage
ster workspace select no-such-set
ster workspace select no-such-set
[ "$status" -eq 1 ] || fail "selecting a set the workspace does not hold exited $status, expected 1"
ster workspace select
ster workspace select
[ "$status" -eq 2 ] || fail "select without its id exited $status, expected 2"
ster workspace select no-such-set
ster workspace --help

repeats=$(queued "$ROOT/usage" frustration.repeat_failure)
echo "repeat_failure: $repeats" >>"$REPORT"
echo "$repeats" | jq -e 'any(.[]; .consecutive == 2 and (.failure_point | startswith("ster.workspace")))' >/dev/null ||
  fail "the same refusal twice queued no repeat_failure: $repeats"
loops=$(queued "$ROOT/usage" frustration.usage_loop)
echo "usage_loop: $loops" >>"$REPORT"
echo "$loops" | jq -e 'any(.[]; .command == "workspace select" and .consecutive == 2)' >/dev/null ||
  fail "the same usage error twice queued no usage_loop: $loops"
helps=$(queued "$ROOT/usage" frustration.help_after_failure)
echo "help_after_failure: $helps" >>"$REPORT"
echo "$helps" | jq -e 'any(.[]; .command == "workspace" and (.failure_point | startswith("ster.workspace")) and (.seconds_since_failure | type == "number"))' >/dev/null ||
  fail "--help after a failure queued no help_after_failure naming it: $helps"
echo "$helps" | grep -q 'no-such-set' && fail "an argument reached the help signal: $helps"

export WISENT_TELEMETRY_HOME="$ROOT/errors"
ster telemetry on --level errors
ster workspace select
ster workspace select
ster workspace select no-such-set
ster workspace --help
[ "$(queued "$ROOT/errors" frustration.usage_loop)" = "[]" ] || fail "a usage loop was queued under errors"
[ "$(queued "$ROOT/errors" frustration.help_after_failure)" = "[]" ] || fail "a help signal was queued under errors"

echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
