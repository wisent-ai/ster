#!/bin/sh
# Real test of `ster sources` (`ster.sources`), through the built binary: the listing
# names every adapter by its name and none by a requirement number, a ShareGPT file and a
# DPO file are discovered with what each gives, an adapter Ster does not have is refused
# with its declared sentence and exit status, and an adapter named by the requirement
# number it was once listed under is refused naming its name.
#
# Usage: tests/sources/sources.sh   (STER selects the binary, default .build/kompilator/debug/ster)
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:-.build/kompilator/debug/ster}
case "$BIN" in /*) ;; *) BIN="$PWD/$BIN" ;; esac
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SCRATCH="$PWD/target/real-tests/sources/sources-$RUN"
mkdir -p "$SCRATCH/home"
SCRATCH=$(cd "$SCRATCH" && pwd -P)
REPORT="$SCRATCH/report.txt"
export HOME="$SCRATCH/home" XDG_STATE_HOME="$SCRATCH/state" XDG_DATA_HOME="$SCRATCH/data"
echo "revision: $(git --no-optional-locks rev-parse HEAD)$(git --no-optional-locks diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $*" | tee -a "$REPORT" >&2
  exit 1  # https://pubs.opengroup.org/onlinepubs/9799919799/functions/exit.html
}
run() {
  expected=$1
  shift
  set +e
  out=$("$BIN" "$@" 2>"$SCRATCH/stderr" </dev/null)
  status=$?
  set -e
  err=$(cat "$SCRATCH/stderr")
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
  for adapter in hugging-face-preferences parquet-or-arrow openai-fine-tuning sharegpt alpaca dpo \
    repeng steering-vectors transformerlens nnsight sae-lens transformers-checkpoint gguf mlx-model \
    pytorch-checkpoint hugging-face-cache peft-lora-adapter mlx-lora-adapter preferences-comparisons \
    wisent-evaluators; do
    [ "$(printf '%s' "$out" | jq --arg id "$adapter" '[.adapters[] | select(.adapter == $id)] | length')" = 1 ] \
      || fail "the listing does not name $adapter: $out"
  done
  [ "$(printf '%s' "$out" | jq '.adapters | length')" = 20 ] || fail "the listing does not hold exactly the 20 adapters: $out"
  case "$out" in
    *IMP-*) fail "the listing still shows a requirement number: $out" ;;
  esac
  [ "$(printf '%s' "$out" | jq -r '.sharing.sends.adapter')" = dpo ] || fail "the sharing question's example event does not name dpo: $out"
  passed the_listing_names_every_adapter
}

a_sharegpt_file_is_discovered_with_its_examples() {
  printf '%s\n' '{"conversations": [{"from": "human", "value": "Name a colour."}, {"from": "gpt", "value": "Blue."}]}' >"$SCRATCH/sharegpt.jsonl"
  run 0 sources discover sharegpt --path "$SCRATCH/sharegpt.jsonl"
  [ "$(printf '%s' "$out" | jq .examples)" = 1 ] || fail "the ShareGPT file gave no example: $out"
  [ "$(printf '%s' "$out" | jq -r .adapter)" = sharegpt ] || fail "the answer does not name sharegpt: $out"
  passed a_sharegpt_file_is_discovered_with_its_examples
}

a_dpo_file_is_discovered_with_its_pairs() {
  printf '%s\n%s\n' '{"prompt": "Greet.", "chosen": "Hello.", "rejected": "Go away."}' '{"prompt": "Greet.", "chosen": "", "rejected": "No."}' >"$SCRATCH/dpo.jsonl"
  run 0 sources discover dpo --path "$SCRATCH/dpo.jsonl"
  [ "$(printf '%s' "$out" | jq .pairs)" = 1 ] || fail "the DPO file did not give one pair: $out"
  [ "$(printf '%s' "$out" | jq .left_out)" = 1 ] || fail "the DPO file's row without a chosen answer was not left out: $out"
  passed a_dpo_file_is_discovered_with_its_pairs
}

an_unknown_adapter_is_refused() {
  run 2 sources discover no-such-source --path "$SCRATCH/dpo.jsonl"
  said "Error: discovering $SCRATCH/dpo.jsonl as no-such-source: no-such-source is not an adapter of ster"
  said "hugging-face-preferences, openai-fine-tuning"
  passed an_unknown_adapter_is_refused
}

an_adapter_named_by_its_requirement_number_is_refused_naming_its_name() {
  run 2 sources discover IMP-STER-06 --path "$SCRATCH/dpo.jsonl"
  said "Error: discovering $SCRATCH/dpo.jsonl as IMP-STER-06: IMP-STER-06 is the import plan's requirement number of an adapter"
  said "IMP-STER-06 stands for the adapter named dpo"
  said "name it dpo: \`ster sources discover dpo --path $SCRATCH/dpo.jsonl\`"
  # A number of a model checkpoint, written in lowercase, is refused naming its name too.
  run 2 sources discover imp-ster-16 --path "$SCRATCH"
  said "IMP-STER-16 stands for the adapter named hugging-face-cache"
  passed an_adapter_named_by_its_requirement_number_is_refused_naming_its_name
}

an_answer_to_sharing_that_cannot_be_kept_is_refused() {
  # XDG_CONFIG_HOME names a file, so the folder the answer goes in cannot be made.
  printf 'a file where the answer folder would go\n' >"$SCRATCH/config-is-a-file"
  export XDG_CONFIG_HOME="$SCRATCH/config-is-a-file"
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
an_adapter_named_by_its_requirement_number_is_refused_naming_its_name
an_answer_to_sharing_that_cannot_be_kept_is_refused

touch "$SCRATCH/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
