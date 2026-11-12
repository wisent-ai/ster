#!/bin/sh
# Real test of how the GitHub star request fails (`ster.star`), through
# `ster star` and `ster request star/*` in the run's own home: a state
# directory XDG_STATE_HOME makes impossible, because a file stands where the
# directory belongs. The run's home is tests/runtime/harness.sh's.
#
# Usage: tests/onboarding/star.sh   (STER selects the binary, default target/debug/ster)
AREA=onboarding-star . "$(dirname "$0")/../runtime/harness.sh"

run 0 star status
[ "$(echo "$out" | jq -r .stage)" = "counting" ] || fail "a fresh home's star request is not counting: $out"
passed "a fresh home's star request is counting"

a_state_directory_that_cannot_be_made_is_refused() {
  echo "not a directory" >"$ROOT/blocker"
  set +e
  err=$(XDG_STATE_HOME="$ROOT/blocker" "$BIN" star dismiss 2>&1 >/dev/null </dev/null)
  status=$?
  set -e
  printf '$ XDG_STATE_HOME=%s ster star dismiss\nexit: %s\nstderr: %s\n\n' "$ROOT/blocker" "$status" "$err" >>"$REPORT"
  [ "$status" -eq 1 ] || fail "ster star dismiss exited $status, expected 1: $err"
  said "dismissing the GitHub star request kept in $ROOT/blocker/ster/onboarding.json"
  said "XDG_STATE_HOME $ROOT/blocker"
  said "the runtime answered: storage unavailable"
  set +e
  out=$(printf '{}' | XDG_STATE_HOME="$ROOT/blocker" "$BIN" request star/dismiss 2>/dev/null | tail -n 1)
  set -e
  printf '$ XDG_STATE_HOME=%s ster request star/dismiss\nresult: %s\n\n' "$ROOT/blocker" "$out" >>"$REPORT"
  [ "$(echo "$out" | jq -r .json.cause.id)" = "journey-storage" ] \
    || fail "the request was not refused as journey-storage: $out"
  [ "$(cat "$ROOT/blocker")" = "not a directory" ] || fail "the file standing in the way was changed"
  passed a_state_directory_that_cannot_be_made_is_refused
}

a_state_directory_that_cannot_be_made_is_refused
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
