#!/bin/sh
# Real test of `ster sources` (`ster.sources`), through the built binary: the listing
# names every adapter, a ShareGPT file and a DPO file are discovered with what each gives,
# and an adapter Ster does not have is refused with its declared sentence and exit status.
#
# Usage: tests/sources/sources.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/sources/sources-$RUN"
mkdir -p "$ROOT/home"
ROOT=$(cd "$ROOT" && pwd -P)
REPORT="$ROOT/report.txt"
export HOME="$ROOT/home" XDG_STATE_HOME="$ROOT/state" XDG_DATA_HOME="$ROOT/data"
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
}
said() {
  case "$err" in
    *"$1"*) ;;
    *) fail "expected '$1' in the refusal, got: $err" ;;
  esac
}
passed() { echo "ok: $1" >>"$REPORT"; }

the_listing_names_every_adapter() {
  run 0 sources
  for adapter in IMP-STER-01 IMP-STER-03 IMP-STER-04 IMP-STER-05 IMP-STER-06 IMP-STER-19; do
    [ "$(printf '%s' "$out" | jq --arg id "$adapter" '[.adapters[] | select(.adapter == $id)] | length')" = 1 ] \
      || fail "the listing does not name $adapter: $out"
  done
  passed the_listing_names_every_adapter
}

a_sharegpt_file_is_discovered_with_its_examples() {
  printf '%s\n' '{"conversations": [{"from": "human", "value": "Name a colour."}, {"from": "gpt", "value": "Blue."}]}' >"$ROOT/sharegpt.jsonl"
  run 0 sources discover IMP-STER-04 --path "$ROOT/sharegpt.jsonl"
  [ "$(printf '%s' "$out" | jq .examples)" = 1 ] || fail "the ShareGPT file gave no example: $out"
  passed a_sharegpt_file_is_discovered_with_its_examples
}

a_dpo_file_is_discovered_with_its_pairs() {
  printf '%s\n%s\n' '{"prompt": "Greet.", "chosen": "Hello.", "rejected": "Go away."}' '{"prompt": "Greet.", "chosen": "", "rejected": "No."}' >"$ROOT/dpo.jsonl"
  run 0 sources discover IMP-STER-06 --path "$ROOT/dpo.jsonl"
  [ "$(printf '%s' "$out" | jq .pairs)" = 1 ] || fail "the DPO file did not give one pair: $out"
  [ "$(printf '%s' "$out" | jq .left_out)" = 1 ] || fail "the DPO file's row without a chosen answer was not left out: $out"
  passed a_dpo_file_is_discovered_with_its_pairs
}

an_unknown_adapter_is_refused() {
  run 2 sources discover IMP-STER-99 --path "$ROOT/dpo.jsonl"
  said "Error: discovering $ROOT/dpo.jsonl as IMP-STER-99: IMP-STER-99 is not an adapter of ster"
  said "IMP-STER-06"
  passed an_unknown_adapter_is_refused
}

an_answer_to_sharing_that_cannot_be_kept_is_refused() {
  # XDG_CONFIG_HOME names a file, so the folder the answer goes in cannot be made.
  printf 'a file where the answer folder would go\n' >"$ROOT/config-is-a-file"
  export XDG_CONFIG_HOME="$ROOT/config-is-a-file"
  run 1 sources share no
  unset XDG_CONFIG_HOME
  said "keeping whether discoveries report their counts"
  said "could not be written"
  passed an_answer_to_sharing_that_cannot_be_kept_is_refused
}

the_listing_names_every_adapter
a_sharegpt_file_is_discovered_with_its_examples
a_dpo_file_is_discovered_with_its_pairs
an_unknown_adapter_is_refused
an_answer_to_sharing_that_cannot_be_kept_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
