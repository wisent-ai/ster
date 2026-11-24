#!/bin/sh
# What every real test of `<program> sources` shares: the product's names from
# tests/support/names.sh, a scratch directory of the run's own that HOME, every XDG
# directory and the report live under, so no run reads or writes the operator's
# configuration, and the steps that run the binary and check what it said.
#
# Usage: . tests/sources/harness.sh <area>   (from the checkout's top)
set -eu
. tests/support/names.sh
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SCRATCH="$PWD/target/real-tests/sources/$1-$RUN"
mkdir -p "$SCRATCH/home" "$SCRATCH/config"
SCRATCH=$(cd "$SCRATCH" && pwd -P)
REPORT="$SCRATCH/report.txt"
export HOME="$SCRATCH/home" XDG_CONFIG_HOME="$SCRATCH/config" XDG_STATE_HOME="$SCRATCH/state" XDG_DATA_HOME="$SCRATCH/data"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1  # https://pubs.opengroup.org/onlinepubs/9799919799/functions/exit.html
}
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$SCRATCH/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$SCRATCH/stderr")
  printf '$ %s %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$PROGRAM" "$*" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "$PROGRAM $* exited $status, expected $expected: $err"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
# The files under the run's home and XDG directories, one path a line: what a run kept.
kept_files() {
  find "$HOME" "$XDG_CONFIG_HOME" "$SCRATCH/state" "$SCRATCH/data" -type f 2>/dev/null | sort
}
passed() { echo "ok: $1" >>"$REPORT"; }
finished() {
  touch "$SCRATCH/passed"
  echo "PASS" >>"$REPORT"
  echo "PASS: $REPORT"
}
