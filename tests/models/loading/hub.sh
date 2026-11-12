#!/bin/sh
# Real test of how reading a model from the Hugging Face Hub fails
# (`ster.model.checkpoint`), through the built binary and the real Hub.
#
# A repository id nobody publishes is asked for under an empty Hugging Face
# cache: the Hub answers it as invisible to this account, and the load is
# refused as an unknown repository (exit 1), not as the Hub being down.
#
# Every command, its exit status and output go to the run's report.txt.
#
# Usage: tests/models/loading/hub.sh   (STER selects the binary, default target/debug/ster)
set -eu
cd "$(dirname "$0")/../../.."
BIN=${STER:-target/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/models/hub-$RUN"
mkdir -p "$ROOT/home"
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" HF_HOME="$ROOT/home/huggingface"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"
# A repository name this run makes up, so nobody can have published it.
MISSING="wisent-ai/ster-no-such-model-$RUN"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1
}

an_unknown_repository_is_refused() {
  set +e
  out=$("$BIN" generate --model "$MISSING" --prompt "describe the sea ." --max-new-tokens 1 \
    --temperature 0 --seed 1 2>"$ROOT/stderr" </dev/null)
  status=$?
  set -e
  # grep exits 1 when every line was a wait line; any other failure stops the run.
  err=$(grep -v '^czekam\|^koniec czekania\|^blad czekania' "$ROOT/stderr" || [ $? -eq 1 ])
  printf '$ ster generate --model %s\nexit: %s\nstdout: %s\nstderr: %s\n\n' "$MISSING" "$status" "$out" "$err" >>"$REPORT"
  [ "$status" -eq 1 ] || fail "an unknown repository exited $status, expected 1: $err"
  case "$err" in
    *"Error: reading model repository $MISSING on the Hugging Face Hub: the Hub answered 40"*": no repository $MISSING at revision main is visible to this account"*) ;;
    *) fail "the refusal does not name the unknown repository: $err" ;;
  esac
  case "$err" in
    *"observed: GET https://huggingface.co/api/models/$MISSING/revision/main answered HTTP 40"*) ;;
    *) fail "the refusal does not name the request the Hub answered: $err" ;;
  esac
  echo "ok: an_unknown_repository_is_refused" >>"$REPORT"
}

an_unknown_repository_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
