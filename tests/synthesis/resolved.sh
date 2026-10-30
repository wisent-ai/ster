#!/bin/sh
# Real test that `ster pairs synthesize --generator brama` takes its model
# service from Stado's capability resolution, never from a gateway it dials
# itself, through the built binary and the real Stado STER_TEST_STADO names
# (first on PATH as `stado`), in a home under this run's directory whose
# registry places Brama nowhere.
#
#   1. BRAMA_URL unset and nothing declared in Brama's place: the run is
#      refused before any call, naming brama and model-endpoint with causes.
#   2. A real OpenAI-compatible endpoint declared in Brama's place
#      (capabilities.inference.endpoint.url, its model, and the role whose
#      item in an owner-only file store holds its key): the run writes pairs
#      with that endpoint's own model, the report names model-endpoint as the
#      generator, and Stado records that model-endpoint served ster.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: STER=target/debug/ster STER_TEST_STADO=<stado> \
#   STER_TEST_ENDPOINT=<base url> STER_TEST_ENDPOINT_MODEL=<model id> \
#   STER_TEST_ENDPOINT_KEY=<key> tests/synthesis/resolved.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=$(cd "$(dirname "${STER:?set STER to the ster binary under test}")" && pwd)/$(basename "$STER")
STADO=${STER_TEST_STADO:?set STER_TEST_STADO to the real stado executable}
ENDPOINT=${STER_TEST_ENDPOINT:?set STER_TEST_ENDPOINT to a real OpenAI-compatible base url}
MODEL=${STER_TEST_ENDPOINT_MODEL:?set STER_TEST_ENDPOINT_MODEL to the model that endpoint serves}
KEY=${STER_TEST_ENDPOINT_KEY:?set STER_TEST_ENDPOINT_KEY to that endpoint key}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/synthesis/resolved-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT/home" "$ROOT/bin" "$ROOT/tmp" "$ROOT/storage"
ln -s "$STADO" "$ROOT/bin/stado"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
echo "stado: $("$STADO" --version)" >>"$REPORT"

export HOME="$ROOT/home" PATH="$ROOT/bin:$PATH" TMPDIR="$ROOT/tmp"
export WC_STORAGE_BACKEND=local WC_LOCAL_STORAGE_PATH="$ROOT/storage"
unset BRAMA_URL BRAMA_BEARER STADO_INFERENCE_ENDPOINT_URL STADO_INFERENCE_ENDPOINT_MODEL STADO_INFERENCE_ENDPOINT_KEY_ROLE

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

# 1. Nothing answers: every alternative is named, before any call.
# shellcheck disable=SC2086
run 1 $SYNTH --output "$ROOT/refused.json"
contains "refusal names the resolution" "$err" "no inference alternative answers ster"
contains "refusal names brama" "$err" "brama at "
contains "refusal names the endpoint alternative" "$err" "model-endpoint at capabilities.inference.endpoint.url"
[ ! -e "$ROOT/refused.json" ] || fail "a refused run wrote a pair set"

# 2. A real endpoint answers in Brama's place.
"$STADO" config set credentials.store "file://$ROOT/credentials.json" >>"$REPORT" 2>&1
printf '{"token":"%s"}' "$KEY" | "$STADO" credentials put --role ster-journey-endpoint >>"$REPORT" 2>&1
"$STADO" config set capabilities.inference.endpoint.url "$ENDPOINT" >>"$REPORT" 2>&1
"$STADO" config set capabilities.inference.endpoint.model "$MODEL" >>"$REPORT" 2>&1
"$STADO" config set capabilities.inference.endpoint.key_role ster-journey-endpoint >>"$REPORT" 2>&1
# shellcheck disable=SC2086
run 0 $SYNTH --output "$ROOT/served.json"
contains "the report names the endpoint and its model" "$out" "model-endpoint:$MODEL"
[ -s "$ROOT/served.json" ] || fail "the served run wrote no pair set"
served=$("$STADO" capabilities inference --json | jq -r '.alternatives[0].served.ster.provider')
[ "$served" = model-endpoint ] || fail "Stado records $served as serving ster, not model-endpoint"
echo "ok: Stado records model-endpoint serving ster" >>"$REPORT"

echo "PASS" >>"$REPORT"
echo "resolved evidence: $REPORT"
