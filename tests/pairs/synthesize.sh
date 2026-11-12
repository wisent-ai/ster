#!/bin/sh
# Real test of the refusals `ster pairs synthesize` and
# `ster request pairs/synthesize` (`ster.pairs.synthesize`) give before any
# model loads or any Brama route is asked, through the built binary.
#
# Each run names a model or a route that does not exist and settings a run
# could not start with: a blank trait, a temperature at or below zero, a
# local generator without a model, a Brama generator without a route, and on
# the request an unknown generator and no count. Each is refused with its
# declared sentence and exit status 2, nothing is written, and no model is
# fetched. Writing pairs with a real model is tests/synthesis/judged.sh.
#
# Usage: tests/pairs/synthesize.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs-synthesize/$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
OUT="$ROOT/set.json"
MODEL="$ROOT/no-such-model"
export HOME="$ROOT/home" XDG_DATA_HOME="$ROOT/data" XDG_STATE_HOME="$ROOT/state"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

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
  [ ! -e "$OUT" ] || fail "a refused run wrote $OUT"
}
request() {
  set +e
  out=$(printf '%s' "$2" | "$BIN" request pairs/synthesize 2>"$ROOT/stderr")
  status=$?
  set -e
  printf '$ ster request pairs/synthesize <<< %s\nexit: %s\nstdout: %s\n\n' "$2" "$status" "$out" >>"$REPORT"
  [ "$status" -eq "$1" ] || fail "ster request pairs/synthesize exited $status, expected $1: $out"
  error=$(printf '%s\n' "$out" | jq -r 'select(.type == "result") | .json.error')
  [ ! -e "$OUT" ] || fail "a refused request wrote $OUT"
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
passed() { echo "ok: $1" >>"$REPORT"; }
SETTINGS="--count 1 --retry-multiplier 2 --dedupe-bits 3 --dedupe-bands 8 --max-new-tokens 16 --top-p 0.9 --seed 1 --output $OUT"
BODY="\"retryMultiplier\": 2, \"dedupeBits\": 3, \"dedupeBands\": 8, \"maxNewTokens\": 16, \"temperature\": 0.7, \"topP\": 0.9, \"seed\": 1, \"output\": \"$OUT\""

a_blank_trait_is_refused_before_any_model_loads() {
  run 2 pairs synthesize --generator local --model "$MODEL" --trait " " --temperature 0.7 $SETTINGS
  said "Error: writing pairs: pairs synthesize needs a description of the trait the positive side shows; observed: --trait \" \"; fix: describe the trait in one sentence with --trait"
  passed a_blank_trait_is_refused_before_any_model_loads
}

a_temperature_at_or_below_zero_is_refused_before_any_model_loads() {
  run 2 pairs synthesize --generator local --model "$MODEL" --trait calm --temperature 0 $SETTINGS
  said "Error: checking the sampling settings: synthesis requires a temperature above zero; argmax generation repeats a single prompt; observed: --temperature 0; fix: give --temperature a number above zero"
  run 2 pairs synthesize --generator brama --generator-model example/route --trait calm --temperature=-1 $SETTINGS
  said "observed: --temperature -1;"
  passed a_temperature_at_or_below_zero_is_refused_before_any_model_loads
}

a_local_generator_without_a_model_is_refused() {
  run 2 pairs synthesize --generator local --trait calm --temperature 0.7 $SETTINGS
  said "Error: choosing the model that writes the pairs: pairs synthesize with --generator local requires --model; observed: --generator local without --model; writing pairs for \"calm\"; fix: name the Hugging Face model id or local model directory that writes the pairs with --model"
  request 2 "{\"generator\": \"local\", \"trait\": \"calm\", \"count\": 1, $BODY}"
  check "the request says the same" "$(echo "$error" | cut -d';' -f1)" \
    "choosing the model that writes the pairs: pairs synthesize with --generator local requires --model"
  passed a_local_generator_without_a_model_is_refused
}

a_brama_generator_without_a_route_is_refused() {
  run 2 pairs synthesize --generator brama --trait calm --temperature 0.7 $SETTINGS
  said "Error: choosing the model that writes the pairs: pairs synthesize with --generator brama requires --generator-model; observed: --generator brama without --generator-model; writing pairs for \"calm\"; fix: name the route that writes the pairs with --generator-model"
  request 2 "{\"generator\": \"brama\", \"generatorModel\": \" \", \"trait\": \"calm\", \"count\": 1, $BODY}"
  check "the request refuses a blank route" "$(echo "$error" | cut -d';' -f1)" \
    "choosing the model that writes the pairs: pairs synthesize with --generator brama requires --generator-model"
  passed a_brama_generator_without_a_route_is_refused
}

an_unknown_generator_is_refused() {
  request 2 "{\"generator\": \"cloud\", \"trait\": \"calm\", \"count\": 1, $BODY}"
  check "the request names the generator" "$error" \
    "choosing the model that writes the pairs: pairs synthesize writes with local or brama, and \"cloud\" is neither; observed: generator: \"cloud\"; fix: give generator local, with a model, or brama, with generatorModel"
  request 2 "{\"trait\": \"calm\", \"count\": 1, $BODY}"
  check "the request without a generator names an empty one" "$(echo "$error" | cut -d';' -f2)" ' observed: generator: ""'
  passed an_unknown_generator_is_refused
}

a_request_without_a_count_is_refused() {
  request 2 "{\"generator\": \"brama\", \"generatorModel\": \"example/route\", \"trait\": \"calm\", $BODY}"
  check "the request asks for a count" "$error" \
    "reading the pairs/synthesize request: pairs synthesize requires count, a pair count above zero; observed: the request names no count; writing pairs for \"calm\"; fix: give count, the number of pairs to keep"
  passed a_request_without_a_count_is_refused
}

a_blank_trait_is_refused_before_any_model_loads
a_temperature_at_or_below_zero_is_refused_before_any_model_loads
a_local_generator_without_a_model_is_refused
a_brama_generator_without_a_route_is_refused
an_unknown_generator_is_refused
a_request_without_a_count_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
