#!/bin/sh
# Real test that `ster pairs synthesize --generator brama` names no model
# service when BRAMA_URL is unset: each request is handed to `stado
# capabilities call inference --consumer ster`, through the built binary and
# the real Stado STER_TEST_STADO names (first on PATH as `stado`), in a home
# under this run's directory whose registry places Brama nowhere.
#
#   1. Nothing answers: the run is refused, naming brama with its cause and
#      the `stado capabilities attestation record` command that answers the
#      request Stado kept, and no pair set is written.
#   2. A statement is recorded for each request Stado kept, and the run is
#      repeated until every request it makes is answered: the run writes a
#      pair set, its report names attestation as the generator, and Stado
#      records that attestation served ster. Ster names no alternative.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: STER=target/debug/ster STER_TEST_STADO=<stado> tests/synthesis/resolved.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=$(cd "$(dirname "${STER:?set STER to the ster binary under test}")" && pwd)/$(basename "$STER")
STADO=${STER_TEST_STADO:?set STER_TEST_STADO to the real stado executable}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/synthesis/resolved-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT/home" "$ROOT/bin" "$ROOT/tmp" "$ROOT/storage"
ln -s "$STADO" "$ROOT/bin/stado"
echo "revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

export HOME="$ROOT/home" PATH="$ROOT/bin:$PATH" TMPDIR="$ROOT/tmp"
export WC_STORAGE_BACKEND=local WC_LOCAL_STORAGE_PATH="$ROOT/storage"
unset BRAMA_URL BRAMA_BEARER STADO_INFERENCE_ORDER STADO_ATTESTATIONS STADO_CAPABILITIES_RECORD STADO_CONFIG

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >&2
  exit 1
}
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(grep -v '^czekam\|^koniec czekania' "$ROOT/stderr" || true)
  printf '$ ster %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$*" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq "$expected" ] || fail "ster $* exited $status, expected $expected: $err"
}
contains() {
  case "$2" in
    *"$3"*) echo "ok: $1" >>"$REPORT" ;;
    *) fail "$1: expected '$3' in: $2" ;;
  esac
}

"$STADO" config init >>"$REPORT" 2>&1
SYNTH="pairs synthesize --generator brama --generator-model wisent-backend/chat/primary --trait formal --count 1 --retry-multiplier 10 --dedupe-bits 3 --dedupe-bands 8 --max-new-tokens 200 --temperature 0.7 --top-p 0.9 --seed 1 --opposite casual"

# 1. Nothing answers: every alternative is named, and the request is kept.
# shellcheck disable=SC2086
run 1 $SYNTH --output "$ROOT/refused.json"
contains "refusal names the call" "$err" "no inference alternative answered \`stado capabilities call inference --consumer ster"
contains "refusal names brama" "$err" "brama at "
contains "refusal names how to answer the kept request" "$err" "stado capabilities attestation record inference --consumer ster"
[ ! -e "$ROOT/refused.json" ] || fail "a refused run wrote a pair set"

# 2. Answer every request Stado keeps until the run asks nothing new, each
#    as its prompt asks: the question it asks for, the answer in the named
#    register, and the refusal judgement as the JSON object it requires.
answered=0
round=0
while :; do
  round=$((round + 1))
  [ "$round" -le 40 ] || fail "the run kept asking new requests after $answered statements: $(grep -v '^czekam\|^koniec' "$ROOT/stderr" | tail -2)"
  waiting=$("$STADO" capabilities attestation list inference --consumer ster --json | jq -r '.[] | select(.attested == false) | .request')
  for digest in $waiting; do
    prompt=$("$STADO" capabilities attestation show inference --consumer ster "$digest" | jq -r '.asked.messages[-1].content')
    case "$prompt" in
      "A model was asked"*) text='{"refuses": false}' ;;
      "Write one short question"*) text="Has the meeting been moved?" ;;
      *"personality: casual"*) text="yeah, it got bumped to friday, no biggie" ;;
      *) text="Indeed, the meeting has been rescheduled to Friday." ;;
    esac
    printf '%s' "$text" | "$STADO" capabilities attestation record inference --consumer ster "$digest" \
      --checked "wrote the answer in the register the request names" >>"$REPORT" 2>&1
    answered=$((answered + 1))
  done
  set +e
  out=$("$BIN" $SYNTH --output "$ROOT/served.json" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  [ "$status" -eq 0 ] && break
done
printf '$ ster %s (round %s, %s statements)\nexit: %s\nstdout: %s\n\n' "$SYNTH" "$round" "$answered" "$status" "$out" >>"$REPORT"
contains "the report names attestation as the generator" "$out" "attestation:wisent-backend/chat/primary"
[ -s "$ROOT/served.json" ] || fail "the served run wrote no pair set"
served=$("$STADO" capabilities inference --json | jq -r '.alternatives[0].served.ster.provider')
[ "$served" = attestation ] || fail "Stado records $served as serving ster, not attestation"
echo "ok: Stado records attestation serving ster" >>"$REPORT"

echo "PASS" >>"$REPORT"
echo "resolved evidence: $REPORT"
