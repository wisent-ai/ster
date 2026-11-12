# What every real test under tests/runtime/ shares: the binary, the run's
# own directory and home under target/real-tests/runtime/, the report each
# command, its exit status and its output are written to, the four-layer
# Llama fixture `ster model fixture create` writes from
# docs/examples/model/fixture.json, and the checks a test makes of a
# refusal. HOME and HF_HOME point inside the run's directory.
#
# A test names its area and sources this file:
#   AREA=steer . "$(dirname "$0")/harness.sh"
# STER selects the binary, default target/debug/ster.
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/runtime/$AREA-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}
# run EXPECTED ARGS...: ster ARGS, its output in $out and $err, its status
# required to be EXPECTED.
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  err=$(grep -v '^czekam\|^koniec czekania\|^blad czekania' "$ROOT/stderr" || true)
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
# generate EXPECTED MODEL PROMPT [ARGS...]: one greedy token of MODEL's
# answer to PROMPT.
generate() {
  expected=$1
  model=$2
  prompt=$3
  shift 3
  run "$expected" generate --model "$model" --prompt "$prompt" --max-new-tokens 1 --temperature 0 --seed 1 "$@"
}

MODEL="$ROOT/base"
cp docs/examples/model/fixture.json docs/examples/model/config.json docs/examples/model/vocabulary.json "$ROOT/"
"$BIN" model fixture create --from "$ROOT/fixture.json" --output "$MODEL" >>"$REPORT" 2>&1 \
  || fail "the fixture could not be written"
generate 0 "$MODEL" "describe the sea ."
passed "the fixture encodes a prompt and answers"
