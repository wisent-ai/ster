#!/bin/sh
# Real test of `ster support` through the built binary, against a real
# wisent-customer-support and read back through its agent API.
#
#   * a report carries Ster's version, the installation and the last queued
#     failure, and support holds it as a ticket with exactly those;
#   * the key is kept under the telemetry directory and never printed;
#   * an agent's reply is what `ster support status` prints, and
#     `ster request support/status` answers the same document;
#   * no description, a refused email, a plain-http address and no network
#     are refused by name, and nothing is kept for a report support did not take.
#
# Usage: tests/support/support.sh
#   STER                        the binary, default target/debug/ster
#   STER_TEST_SUPPORT           http(s)://<support>/v1/public/reports of a running wisent-customer-support
#   STER_TEST_SUPPORT_TOKEN     that service's support token, for the agent's side
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
: "${STER_TEST_SUPPORT:?set it to a running wisent-customer-support /v1/public/reports: this test files real reports}"
: "${STER_TEST_SUPPORT_TOKEN:?set it to that service support token, which the agent reply needs}"
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/support/$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
echo "support: $STER_TEST_SUPPORT" >>"$REPORT"
export WISENT_TELEMETRY_HOME="$ROOT/state"
# Port 9, the IANA discard port: the failure this test queues is sent nowhere.
export WISENT_TELEMETRY_COLLECTOR_URL="http://127.0.0.1:9/api/analytics/mobile/collect"
export WISENT_SUPPORT_URL="$STER_TEST_SUPPORT"
unset DO_NOT_TRACK
ORIGIN=${STER_TEST_SUPPORT%%/v1/*}

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
agent() {
  curl -sS -f -H "x-support-token: $STER_TEST_SUPPORT_TOKEN" "$@"
}
kept() { find "$ROOT/state/support" -name '*.json' 2>/dev/null | wc -l | tr -d ' '; }

run 0 telemetry on --level errors
run 1 pairs inspect --pairs "$ROOT/absent.json" --dedupe-bits 3 --dedupe-bands 4 --unbalanced-ratio 3
run 0 telemetry status
INSTALLATION=$(printf '%s' "$out" | jq -r '.installation')
[ -n "$INSTALLATION" ] && [ "$INSTALLATION" != null ] || fail "no installation id after a queued failure: $out"

run 0 support report --subject "pairs inspect fails" --description "ster pairs inspect stops on a pair set that is there"
FILED=$out
ID=$(printf '%s' "$FILED" | jq -r '.id')
printf '%s' "$FILED" | jq -e '.with_failure == true and (.id | test("^WCS-[0-9]{6}$")) and (has("access_key") | not)' >/dev/null \
  || fail "the report did not go with the failure, or printed its key: $FILED"
KEPT=$(printf '%s' "$FILED" | jq -r '.kept')
jq -e --arg id "$ID" '.id == $id and (.access_key | length) >= 40' "$KEPT" >/dev/null || fail "the key is not kept in $KEPT"
TICKETS=$(agent "$ORIGIN/v1/support/tickets")
TICKET=$(printf '%s' "$TICKETS" | jq -c --arg id "$ID" '.[] | select(.public_id == $id)')
printf '%s' "$TICKET" | jq -e --arg installation "$INSTALLATION" '
  .source == "installation" and .metadata.installation_report.product == "ster"
  and .metadata.installation_report.app == "ster-cli"
  and .metadata.installation_report.installation == $installation
  and .metadata.installation_report.failure.cause_id == "pair-set-missing"' >/dev/null \
  || fail "support does not hold the report with Ster's facts: $TICKET"
case "$TICKET" in *"$HOME"*) fail "the report support holds still has the home directory: $TICKET" ;; esac
passed "a report reaches support with the version, installation and last failure, and its key stays here"

TICKET_ID=$(printf '%s' "$TICKET" | jq -r '.id')
agent -X POST -H 'content-type: application/json' --data '{"body_text":"Fixed in the next release."}' \
  "$ORIGIN/v1/support/tickets/$TICKET_ID/reply" >/dev/null
run 0 support status
printf '%s' "$out" | jq -e --arg id "$ID" '
  any(.[]; .id == $id and .status == "waiting_on_customer" and .replies[0].text == "Fixed in the next release.")' >/dev/null \
  || fail "status does not show the reply: $out"
set +e
request=$(printf '{}' | "$BIN" request support/status 2>"$ROOT/stderr")
request_status=$?
set -e
printf '$ ster request support/status\nexit: %s\nstdout: %s\n\n' "$request_status" "$request" >>"$REPORT"
[ "$request_status" -eq 0 ] || fail "ster request support/status exited $request_status: $(cat "$ROOT/stderr")"
printf '%s' "$request" | tail -n 1 | jq -e --arg id "$ID" 'any(.result[]?; .id == $id) or any(.document[]?; .id == $id) or (tostring | contains($id))' >/dev/null \
  || fail "ster request support/status does not answer the report: $request"
passed "the agent's reply is what support status and request support/status show"

BEFORE=$(kept)
run 2 support report
said "--description <DESCRIPTION>"
run 1 support report --description "   "
said "the report has no description"
passed "a report without a description is refused before anything is sent"

run 1 support report --description "something failed" --email "not an address"
said "support refused HTTP 400: email \"not an address\" is not an email address"
passed "a report support refuses says support's reason"

export WISENT_SUPPORT_URL="http://support.example/v1/public/reports"
run 1 support report --description "something failed"
said "WISENT_SUPPORT_URL holds http://support.example/v1/public/reports"
passed "a plain-http support address is refused by name"

export WISENT_SUPPORT_URL="http://127.0.0.1:9/v1/public/reports"
run 1 support report --description "something failed"
said "support could not be reached"
said "nothing was filed"
passed "an unreachable support files nothing"
[ "$(kept)" -eq "$BEFORE" ] || fail "a key was kept for a report support did not take"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
