#!/bin/sh
# Real test that Ster's privacy page describes exactly what the built binary
# can send: ster-landing's check reads `ster telemetry events` from this
# binary and the "What is sent" table of its telemetry page, and fails naming
# an event Ster sends that the page does not describe, one the page describes
# that Ster does not send, and a consent or property that differs. A change
# that makes Ster send something new fails here until the page says so.
#
# Usage: tests/telemetry/events.sh
#   STER                the binary, default target/debug/ster
#   STER_TEST_LANDING   the ster-landing checkout, default ../ster-landing
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
LANDING=${STER_TEST_LANDING:-$PWD/../ster-landing}
[ -f "$LANDING/src/cli/check-telemetry-events.mjs" ] ||
  { echo "FAIL: $LANDING holds no src/cli/check-telemetry-events.mjs: set STER_TEST_LANDING to the ster-landing checkout" >&2; exit 1; }
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/events/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
echo "landing: $LANDING at $(git -C "$LANDING" rev-parse HEAD)" >>"$REPORT"
"$BIN" telemetry events >"$ROOT/events.json"
jq -e 'length > 0 and all(.[]; (.name | length > 0) and (.consent == "errors" or .consent == "usage") and (.properties | length > 0))' \
  "$ROOT/events.json" >/dev/null || { echo "FAIL: ster telemetry events is not a list of named events with a consent and properties" | tee -a "$REPORT" >&2; exit 1; }
if node "$LANDING/src/cli/check-telemetry-events.mjs" --ster "$BIN" >>"$REPORT" 2>&1; then
  echo "PASS" >>"$REPORT"
  echo "PASS: $REPORT"
else
  echo "FAIL: the privacy page and ster telemetry events disagree: $(tail -n 8 "$REPORT")" >&2
  exit 1
fi
