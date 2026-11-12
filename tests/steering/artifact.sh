#!/bin/sh
# Real test of how a steering artifact fails (`ster.steering.artifact`),
# through `ster vector inspect` and `ster vector train`. One real artifact
# is trained on the four-layer fixture; each test writes a copy of it
# through jq that is wrong in one way, or names a path that cannot be read
# or written.
#
# Usage: tests/steering/artifact.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/steering/artifact-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
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
# rewritten NAME JQ: the trained artifact through JQ, at $ROOT/NAME.json.
rewritten() {
  ARTIFACT="$ROOT/$1.json"
  jq "$2" "$TRAINED" >"$ARTIFACT"
}
inspect() {
  run "$1" vector inspect "$ARTIFACT"
}

cp docs/examples/model/fixture.json docs/examples/model/config.json docs/examples/model/vocabulary.json docs/examples/pairs.json "$ROOT/"
"$BIN" model fixture create --from "$ROOT/fixture.json" --output "$ROOT/base" >>"$REPORT" 2>&1 \
  || fail "the fixture could not be written"
TRAINED="$ROOT/calm.json"
run 0 vector train --model "$ROOT/base" --pairs "$ROOT/pairs.json" --output "$TRAINED" --layers 1
ARTIFACT="$TRAINED"
inspect 0
passed "a trained artifact is read back"

a_missing_artifact_is_refused() {
  ARTIFACT="$ROOT/absent.json"
  inspect 1
  said "Error: reading steering artifact $ARTIFACT: no file is there: nothing at $ARTIFACT is the steering artifact named; observed: path: $ARTIFACT (absent); its directory $ROOT; the system answered: "
  passed a_missing_artifact_is_refused
}

an_unreadable_artifact_is_refused() {
  rewritten closed '.'
  chmod 000 "$ARTIFACT"
  inspect 1
  chmod 644 "$ARTIFACT"
  said "Error: reading the steering artifact $ARTIFACT: this account may not read it: the file's permission bits or owner keep this account out; observed: path: $ARTIFACT (a file, "
  passed an_unreadable_artifact_is_refused
}

an_adapter_sidecar_given_as_a_vector_is_refused() {
  ARTIFACT="$ROOT/sidecar.json"
  printf '%s' '{"rank": 2, "targets": ["query"]}' >"$ARTIFACT"
  inspect 2
  said "Error: reading steering artifact $ARTIFACT: $ARTIFACT is a LoRA adapter sidecar, not a steering artifact"
  passed an_adapter_sidecar_given_as_a_vector_is_refused
}

a_file_that_is_not_an_artifact_is_refused() {
  ARTIFACT="$ROOT/nope.json"
  echo nope >"$ARTIFACT"
  inspect 1
  said "Error: reading the steering artifact $ARTIFACT: it is not JSON, and the steering artifact is the JSON \`ster vector train\` or \`ster tune bipo\` writes: it is another kind of file, or one broken by hand; observed: path: $ARTIFACT (a file, 5 bytes, "
  said "; the JSON reader stopped at line 1, column "
  passed a_file_that_is_not_an_artifact_is_refused
}

an_artifact_of_another_schema_is_refused() {
  rewritten schema '.schema_version = 2'
  inspect 1
  said "Error: checking the calm steering artifact: artifact schema 2 is unsupported; this Ster build reads schema 1"
  passed an_artifact_of_another_schema_is_refused
}

an_artifact_of_another_product_is_refused() {
  rewritten product '.product = "other"'
  inspect 1
  said "Error: checking the calm steering artifact: the artifact belongs to product \"other\", not Ster"
  passed an_artifact_of_another_product_is_refused
}

an_artifact_without_vectors_is_refused() {
  rewritten empty '.vectors = []'
  inspect 1
  said "Error: checking the calm steering artifact: the artifact holds no steering vector to add; observed: hidden_size: 64; vectors: 0"
  passed an_artifact_without_vectors_is_refused
}

a_vector_of_another_width_is_refused() {
  rewritten narrow '.vectors[0].values = .vectors[0].values[:32]'
  inspect 1
  said "Error: checking the calm steering artifact: the vector at layer 1 holds 32 values where the artifact's hidden_size is 64"
  passed a_vector_of_another_width_is_refused
}

# 1e39 is past the largest f32, so it is read as infinity.
a_vector_that_is_not_finite_is_refused() {
  rewritten infinite '.vectors[0].values[0] = 1e39'
  inspect 1
  said "Error: checking the calm steering artifact: the vector at layer 1 holds a value that is not finite"
  passed a_vector_that_is_not_finite_is_refused
}

an_output_under_a_file_is_refused() {
  run 1 vector train --model "$ROOT/base" --pairs "$ROOT/pairs.json" --output "$ROOT/pairs.json/calm.json" --layers 1
  said "Error: creating the directory of the steering artifact $ROOT/pairs.json/calm.json: the path runs through a file: $ROOT/pairs.json is not a directory, so nothing can be created below it; observed: path: $ROOT/pairs.json/calm.json; $ROOT/pairs.json: a file, "
  said "; no artifact was written there; the system answered: "
  passed an_output_under_a_file_is_refused
}

a_missing_artifact_is_refused
an_unreadable_artifact_is_refused
an_adapter_sidecar_given_as_a_vector_is_refused
a_file_that_is_not_an_artifact_is_refused
an_artifact_of_another_schema_is_refused
an_artifact_of_another_product_is_refused
an_artifact_without_vectors_is_refused
a_vector_of_another_width_is_refused
a_vector_that_is_not_finite_is_refused
an_output_under_a_file_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
