# What every real test under tests/tuning/ shares: the binary, the run's own
# directory and home under target/real-tests/tuning/, the report each
# command, its exit status and its output are written to, the four-layer
# Llama fixture `ster model fixture create` writes from
# docs/examples/model/fixture.json, and the checks a test makes of a
# refusal. Nothing outside the run's directory is read or written: HOME and
# HF_HOME point inside it.
#
# A test names its area and sources this file from the repository root:
#   AREA=settings . "$(dirname "$0")/harness.sh"
# STER selects the binary, default target/debug/ster.
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/tuning/$AREA-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
unset STER_REWARD_BEARER || true
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
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
# unsaid TEXT: the refusal does not say TEXT, such as a sign the model was
# loaded before a setting was checked.
unsaid() {
  case "$err" in
    *"$1"*) fail "did not expect '$1' in the refusal, got: $err" ;;
    *) ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }

# merged DEFAULTS FLAG VALUE...: the flag-value pairs of DEFAULTS whose flag
# the arguments after it do not name. Values hold no spaces.
merged() {
  defaults=$1
  shift
  given=" $* "
  kept=
  # shellcheck disable=SC2086
  set -- $defaults
  while [ $# -gt 1 ]; do
    case "$given" in *" $1 "*) ;; *) kept="$kept $1 $2" ;; esac
    shift 2
  done
  echo "$kept"
}
# tune EXPECTED COMMAND DEFAULTS [FLAG VALUE]...: ster tune COMMAND with
# DEFAULTS, each flag given after them replacing its default.
tune() {
  expected=$1
  command=$2
  defaults=$3
  shift 3
  # shellcheck disable=SC2046
  run "$expected" tune "$command" $(merged "$defaults" "$@") "$@"
}

# The fixture every run trains on, and the inputs the runs read.
MODEL="$ROOT/base"
cp docs/examples/model/fixture.json docs/examples/model/config.json docs/examples/model/vocabulary.json "$ROOT/"
"$BIN" model fixture create --from "$ROOT/fixture.json" --output "$MODEL" >>"$REPORT" 2>&1 \
  || fail "the fixture could not be written"
cp docs/examples/tuning/examples.json "$ROOT/examples.json"
cp docs/examples/tuning/grpo-prompts.json "$ROOT/prompts.json"
cp docs/examples/pairs.json "$ROOT/pairs.json"
TRAIN="--rank 2 --alpha 4 --targets query,value --layers all --epochs 1 --learning-rate 0.001 --accumulation 1 --warmup-steps 0 --max-sequence 64 --batch-size 1 --seed 1"
SFT="--model $MODEL --examples $ROOT/examples.json --output $ROOT/sft/adapter.safetensors $TRAIN"
DPO="--model $MODEL --pairs $ROOT/pairs.json --output $ROOT/dpo/adapter.safetensors --beta 0.1 $TRAIN"
REWARD="--model $MODEL --pairs $ROOT/pairs.json --output $ROOT/reward/adapter.safetensors $TRAIN"
BIPO="--model $MODEL --pairs $ROOT/pairs.json --output $ROOT/bipo/vector.json --layer 1 --strength 1 --beta 0.1 --epochs 1 --learning-rate 0.001 --accumulation 1 --warmup-steps 0 --max-sequence 64 --batch-size 1 --seed 1"
GRPO="--model $MODEL --prompts $ROOT/prompts.json --output $ROOT/grpo/adapter.safetensors --reward length --group 2 --turns 1 --iterations 1 --beta 0 --rank 2 --alpha 4 --targets query,value --layers all --learning-rate 0.001 --accumulation 1 --warmup-steps 0 --max-new-tokens 2 --temperature 1 --top-p 1 --max-sequence 64 --seed 1"

# le64 N: N as eight little-endian bytes, the length that opens a
# safetensors file.
le64() {
  n=$1
  i=0
  while [ "$i" -lt 8 ]; do
    # shellcheck disable=SC2059
    printf "\\$(printf '%03o' $((n % 256)))"
    n=$((n / 256))
    i=$((i + 1))
  done
}
# safetensors OUT HEADER DATA: a safetensors file whose JSON header is
# HEADER and whose tensor bytes are the file DATA.
safetensors() {
  length=$(printf '%s' "$2" | wc -c | tr -d ' ')
  { le64 "$length"; printf '%s' "$2"; cat "$3"; } >"$1"
}
# floats OUT COUNT BYTES: COUNT f32 values, each the four bytes BYTES
# (printf escapes, little-endian), written to OUT.
floats() {
  : >"$1"
  i=0
  while [ "$i" -lt "$2" ]; do
    # shellcheck disable=SC2059
    printf "$3" >>"$1"
    i=$((i + 1))
  done
}
ZERO='\000\000\000\000'
NAN='\000\000\300\177'
HUGE='\377\377\177\177'

# sft_adapter NAME [FLAG VALUE]...: a real adapter ster tune sft trains on
# the fixture, at $ROOT/NAME/adapter.safetensors with its sidecar beside it.
sft_adapter() {
  name=$1
  shift
  tune 0 sft "$SFT" --output "$ROOT/$name/adapter.safetensors" "$@"
  ADAPTER="$ROOT/$name/adapter.safetensors"
  SIDECAR="$ROOT/$name/adapter.json"
}
