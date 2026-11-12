#!/bin/sh
# Real test of how `ster onboarding` keeps an operator's progress
# (`ster.onboarding` and the `ster.file` layer it uses), through the built
# binary under this run's own HOME and XDG_STATE_HOME.
#
# A first run writes the state at the journey's entry and advances it; a
# state of another journey, a state at a screen the journey does not hold,
# a state that is not JSON, a state directory this account may not write
# and a run with neither HOME nor XDG_STATE_HOME are each refused with their
# declared sentence and exit status, and the state on disk is read after it.
#
# Usage: tests/onboarding/state.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/onboarding/state-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" XDG_STATE_HOME="$ROOT/state" XDG_DATA_HOME="$ROOT/data"
STATE="$ROOT/state/ster/onboarding.json"
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
passed() { echo "ok: $1" >>"$REPORT"; }

[ "$(id -u)" != 0 ] || fail "run as an ordinary account: root ignores the read-only directory one refusal needs"

a_first_run_keeps_its_progress() {
  run 0 onboarding
  [ -f "$STATE" ] || fail "a first run wrote no state at $STATE"
  [ "$(jq -r .status "$STATE")" = in_progress ] || fail "a first run's state is not in progress: $(cat "$STATE")"
  [ "$(jq -r .current_screen_id "$STATE")" = import_pairs ] \
    || fail "a first run did not advance to the import screen: $(cat "$STATE")"
  passed a_first_run_keeps_its_progress
}

a_state_of_another_journey_is_refused() {
  jq '.journey_version = "1999-01-01.1"' "$STATE" >"$ROOT/other.json"
  cp "$ROOT/other.json" "$STATE"
  run 1 onboarding
  said "Error: reading the onboarding state $STATE: the state records another journey than the one this Ster walks: its journey_version is \"1999-01-01.1\", and this Ster's is \"2026-09-05.1\"; observed: path: $STATE; journey_version: \"1999-01-01.1\""
  said "fix: run \`ster onboarding --reset\` to start this journey anew"
  run 0 onboarding --reset
  [ "$(jq -r .journey_version "$STATE")" = 2026-09-05.1 ] || fail "--reset kept the other journey: $(cat "$STATE")"
  passed a_state_of_another_journey_is_refused
}

a_state_at_an_unknown_screen_is_refused() {
  jq '.current_screen_id = "nowhere"' "$STATE" >"$ROOT/nowhere.json"
  cp "$ROOT/nowhere.json" "$STATE"
  run 1 onboarding
  said "Error: reading the onboarding state $STATE: the state stands at screen \"nowhere\", which this journey does not hold, so the walkthrough cannot go on from it; observed: path: $STATE; current_screen_id: \"nowhere\"; the journey's screens: promise, validation, import_pairs"
  [ "$(jq -r .current_screen_id "$STATE")" = nowhere ] || fail "a refused run changed the state"
  run 0 onboarding --reset
  passed a_state_at_an_unknown_screen_is_refused
}

a_state_that_is_not_json_is_refused() {
  printf 'nope' >"$STATE"
  run 1 onboarding
  said "Error: reading the onboarding state $STATE: it is not JSON, and the onboarding state is the JSON \`ster onboarding\` writes there (\`ster onboarding --reset\` writes it anew): it is another kind of file, or one broken by hand; observed: path: $STATE (a file, 4 bytes, "
  [ "$(cat "$STATE")" = nope ] || fail "a refused run rewrote the state"
  run 0 onboarding --reset
  passed a_state_that_is_not_json_is_refused
}

a_state_directory_this_account_may_not_write_is_refused() {
  mkdir -p "$ROOT/closed-state"
  chmod 555 "$ROOT/closed-state"
  set +e
  XDG_STATE_HOME="$ROOT/closed-state" "$BIN" onboarding >"$ROOT/stdout" 2>"$ROOT/stderr" </dev/null
  status=$?
  set -e
  chmod 755 "$ROOT/closed-state"
  err=$(cat "$ROOT/stderr")
  printf '$ XDG_STATE_HOME=%s ster onboarding\nexit: %s\nstderr: %s\n\n' "$ROOT/closed-state" "$status" "$err" >>"$REPORT"
  [ "$status" -eq 1 ] || fail "a read-only state directory exited $status, expected 1: $err"
  said "Error: creating the directory of the onboarding state $ROOT/closed-state/ster/onboarding.json: this account may not change $ROOT/closed-state/ster or "
  said "; the progress shown is not kept, and the state that was there is unchanged; the system answered: "
  [ ! -e "$ROOT/closed-state/ster" ] || fail "a refused run created $ROOT/closed-state/ster"
  passed a_state_directory_this_account_may_not_write_is_refused
}

a_walkthrough_without_a_home_is_refused() {
  set +e
  env -u HOME -u XDG_STATE_HOME "$BIN" onboarding >"$ROOT/stdout" 2>"$ROOT/stderr" </dev/null
  status=$?
  set -e
  err=$(cat "$ROOT/stderr")
  printf '$ env -u HOME -u XDG_STATE_HOME ster onboarding\nexit: %s\nstderr: %s\n\n' "$status" "$err" >>"$REPORT"
  [ "$status" -eq 1 ] || fail "a run without HOME exited $status, expected 1: $err"
  said "Error: locating the onboarding state: neither XDG_STATE_HOME nor HOME is set, so there is no directory of this account to keep the walkthrough's progress in; observed: XDG_STATE_HOME: unset or empty; HOME: unset or empty"
  passed a_walkthrough_without_a_home_is_refused
}

a_first_run_keeps_its_progress
a_state_of_another_journey_is_refused
a_state_at_an_unknown_screen_is_refused
a_state_that_is_not_json_is_refused
a_state_directory_this_account_may_not_write_is_refused
a_walkthrough_without_a_home_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
