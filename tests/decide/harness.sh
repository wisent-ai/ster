# shellcheck shell=sh
# What every real test under tests/decide/ shares: the binary, the run's own
# directory and home under target/real-tests/decide/, the report each
# command, its exit status and its output are written to, and the checkpoint
# the decisions are read from.
#
# The checkpoint is the four-layer Llama `ster model fixture create` writes
# from docs/examples/model/fixture.json, with the answer letters A, B and C
# added to its vocabulary (and its vocab_size raised to match), so a choice
# of up to three options can be read from it. It is written under the run's
# directory as $MODEL; `ok.json` there is a request it answers.
#
# A test names its area and sources this file:
#   AREA=request . "$(dirname "$0")/harness.sh"
# STER selects the binary, default target/debug/ster.
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/decide/$AREA-$RUN"
mkdir -p "$ROOT/home" "$ROOT/spec"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
# The binary runs under this run's own home, never the operator's.
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
# run EXPECTED ARGS...: ster ARGS with standard input from $INPUT (default
# /dev/null), its output in $out and $err, its status required to be EXPECTED.
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" <"${INPUT:-/dev/null}")
  status=$?
  set -e
  # grep exits 1 when every line was a wait line; any other failure stops the run.
  err=$(grep -v '^czekam\|^koniec czekania\|^blad czekania' "$ROOT/stderr" || [ $? -eq 1 ])
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
# request NAME JQ: a copy of ok.json through JQ, its path in $REQUEST.
request() {
  REQUEST="$ROOT/$1.json"
  jq "$2" "$ROOT/ok.json" >"$REQUEST"
}

MODEL="$ROOT/model"
cp docs/examples/model/fixture.json "$ROOT/spec/"
jq '.modelTokens += [{"content": "A"}, {"content": "B"}, {"content": "C"}]' \
  docs/examples/model/vocabulary.json >"$ROOT/spec/vocabulary.json"
jq '.vocab_size += 3' docs/examples/model/config.json >"$ROOT/spec/config.json"
"$BIN" model fixture create --from "$ROOT/spec/fixture.json" --output "$MODEL" >>"$REPORT" 2>&1 \
  || fail "the lettered fixture could not be written"
jq -n '{state: "the sea is calm", questions: {
  weather: {type: "choice", instructions: "how is the sea ?", criteria: {calm: null, storm: null}},
  loud: {type: "noul", instructions: "is the sea loud ?"}}}' >"$ROOT/ok.json"
DECIDE="decide --model $MODEL --chat-template off"
