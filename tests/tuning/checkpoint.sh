#!/usr/bin/env bash
# Real test of `--checkpoint`, `--checkpoint-every` and `--resume` on every
# `ster tune` objective, through the built binary, on a toy checkpoint it
# writes under this run's directory.
#
# For each of sft, dpo, reward, grpo, bipo and decide it trains uninterrupted,
# then starts the same real training again and sends SIGTERM when its progress
# reports a completed checkpoint. It resumes from the actual persisted step
# and compares the final trained tensors with the uninterrupted run. Missing
# checkpoint progress or training that finished before interruption fails the
# journey rather than being reported as interruption proof. Commands, exits,
# progress and the actual saved step remain under the isolated run directory.
#
# Usage: STER=target/debug/ster tests/tuning/checkpoint.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/tuning/checkpoint-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
git diff --binary HEAD >"$ROOT/source.patch"
git ls-files --cached --others --exclude-standard -z | tar --null -T - -cf "$ROOT/source.tar"
openssl dgst -sha256 "$BIN" >"$ROOT/binary.sha256"
echo "binary: $BIN" >>"$REPORT"
ACTIVE=""
cleanup() {
  if [ -n "$ACTIVE" ]; then
    if kill -TERM "$ACTIVE"; then
      if wait "$ACTIVE"; then
        echo "cleanup: child $ACTIVE exited cleanly" >>"$REPORT"
      else
        echo "cleanup: child $ACTIVE exited $?" >>"$REPORT"
      fi
    else
      echo "cleanup: child $ACTIVE was no longer running" >>"$REPORT"
    fi
  fi
}
trap cleanup EXIT

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
# run accepted|refused ARGS...: the command must end the way named. An accepted
# run's answer is its stdout; a refused run's answer is everything it wrote.
run() {
  expected=$1
  shift
  if [ "$expected" = accepted ]; then
    if "$BIN" "$@" >"$ROOT/output" 2>"$ROOT/stderr" </dev/null; then
      status=$?; outcome=accepted
    else
      status=$?; outcome=refused
    fi
  else
    if "$BIN" "$@" &>"$ROOT/output" </dev/null; then
      status=$?; outcome=accepted
    else
      status=$?; outcome=refused
    fi
  fi
  out=$(cat "$ROOT/output")
  printf '$' >>"$REPORT"
  printf ' %q' "$BIN" "$@" >>"$REPORT"
  printf '\noutcome: %s\nexit: %s\noutput: %s\n' "$outcome" "$status" "$out" >>"$REPORT"
  if [ "$expected" = accepted ]; then cat "$ROOT/stderr" >>"$REPORT"; fi
  printf '\n' >>"$REPORT"
  [ "$outcome" = "$expected" ] || fail "$* was $outcome, expected $expected: $out"
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
refused() {
  case "$out" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) fail "expected a refusal containing '$1', got: $out" ;;
  esac
}

TOY="$ROOT/toy-model"
EXAMPLES=docs/examples/tuning/examples.json
PAIRS=docs/examples/pairs.json
PROMPTS=docs/examples/tuning/grpo-prompts.json
DECISIONS=docs/guide/decisions/examples.json
run accepted toy-model "$TOY"
# Test inputs deliberately use a single pass and a single item per update.
# The fixture supplies the model context, example count used for rank/seed,
# and final layer used for BiPO; its reciprocal count is the learning rate.
ONE=$(jq '.examples | length / length' "$EXAMPLES")
TWO=$(jq -n "$ONE + $ONE")
COUNT=$(jq '.examples | length' "$EXAMPLES")
CONTEXT=$(jq .max_position_embeddings "$TOY/config.json")
LAYERS=$(jq .num_hidden_layers "$TOY/config.json")
LAST=$(jq '.num_hidden_layers - (.num_hidden_layers / .num_hidden_layers)' "$TOY/config.json")
RATE=$(jq '1 / (.examples | length)' "$EXAMPLES")
ADAPTER=(--rank "$COUNT" --alpha "$COUNT")
STEPS=(--epochs "$ONE" --learning-rate "$RATE" --accumulation "$ONE" --batch-size "$ONE" --max-sequence "$CONTEXT")

# durable NAME TENSORS ARGS...: compare uninterrupted training with a real
# SIGTERM interruption after a completed checkpoint and its resumed result.
# Tensor files are compared byte for byte; BiPO compares its stored vectors.
durable() {
  name=$1
  shift
  tensors=$1
  shift
  dir="$ROOT/$name"
  mkdir -p "$dir"
  run accepted tune "$@" --output "$dir/whole.out"
  steps=$(jq .report.steps "$ROOT/output")
  final_loss=$(jq -e .report.final_loss "$ROOT/output")
  [ "$steps" -gt "$ONE" ] || fail "$name: this run has no intermediate step to interrupt"
  progress="$dir/progress.fifo"
  mkfifo "$progress"
  printf '$' >>"$REPORT"
  printf ' %q' "$BIN" tune "$@" --output "$dir/stopped.out" --checkpoint "$dir/checkpoints" --checkpoint-every "$ONE" >>"$REPORT"
  printf '\n' >>"$REPORT"
  "$BIN" tune "$@" --output "$dir/stopped.out" --checkpoint "$dir/checkpoints" --checkpoint-every "$ONE" \
    >"$dir/interrupted.stdout" 2>"$progress" </dev/null &
  ACTIVE=$!
  interrupted=false
  while IFS= read -r line; do
    printf '%s\n' "$line" >>"$dir/interrupted.stderr"
    case "$line" in
      checkpoint*" written at step "*)
        if kill -TERM "$ACTIVE"; then interrupted=true; fi
        break
        ;;
    esac
  done <"$progress"
  if wait "$ACTIVE"; then
    ACTIVE=""
    fail "$name: training finished before it could be interrupted"
  else
    status=$?
    printf 'interrupted child exit: %s\n' "$status" >>"$REPORT"
    ACTIVE=""
  fi
  [ "$interrupted" = true ] || fail "$name: no completed checkpoint was observed before the process ended"
  checkpoint=$(cat "$dir/checkpoints/latest")
  saved="$dir/checkpoints/$checkpoint"
  [ -d "$saved" ] || fail "$name: latest does not name a completed directory"
  every=$(jq .position.step "$saved/state.json")
  [ "$every" -lt "$steps" ] || fail "$name: the process completed every step before interruption"
  # A previous writer can have completed a step directory before it was
  # interrupted publishing latest. That unreferenced older-format attempt
  # must not prevent a resumed writer from saving the same numbered step.
  orphan="$dir/checkpoints/step-$steps"
  cp -R "$saved" "$orphan"
  cp "$orphan/state.json" "$dir/orphan-state.before.json"
  run accepted tune "$@" --output "$dir/resumed.out" --resume "$dir/checkpoints" \
    --checkpoint "$dir/checkpoints" --checkpoint-every "$ONE"
  cmp "$orphan/state.json" "$dir/orphan-state.before.json" || fail "$name: resume overwrote an unreferenced checkpoint attempt"
  check "$name: the resumed run continued from that checkpoint" "$(jq .report.resumed_from "$ROOT/output")" "$every"
  check "$name: the resumed run took every step" "$(jq .report.steps "$ROOT/output")" "$steps"
  check "$name: the final loss equals the uninterrupted run" "$(jq -e .report.final_loss "$ROOT/output")" "$final_loss"
  if [ "$tensors" = file ]; then
    cmp "$dir/whole.out" "$dir/resumed.out" || fail "$name: the resumed adapter differs from the uninterrupted one"
    echo "ok: $name: the resumed adapter is byte for byte the uninterrupted one" >>"$REPORT"
  else
    check "$name: the resumed tensors are the uninterrupted ones" \
      "$(jq -c "$tensors" "$dir/resumed.out")" "$(jq -c "$tensors" "$dir/whole.out")"
  fi
}

durable sft file sft --model "$TOY" --examples "$EXAMPLES" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$COUNT"
durable dpo file dpo --model "$TOY" --pairs "$PAIRS" "${ADAPTER[@]}" "${STEPS[@]}" --beta "$ONE" --seed "$COUNT"
durable reward file reward --model "$TOY" --pairs "$PAIRS" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$COUNT"
durable grpo file grpo --model "$TOY" --prompts "$PROMPTS" "${ADAPTER[@]}" --group "$TWO" \
  --iterations "$ONE" --beta "$ONE" --learning-rate "$RATE" --accumulation "$ONE" --max-new-tokens "$COUNT" \
  --temperature "$ONE" --top-p "$ONE" --max-sequence "$CONTEXT" --seed "$COUNT"
durable bipo '.vectors' bipo --model "$TOY" --pairs "$PAIRS" --layer "$LAST" --strength "$ONE" \
  --beta "$ONE" --warmup-steps "$(jq -n "$ONE - $ONE")" "${STEPS[@]}" --seed "$COUNT"
durable decide file decide --model "$TOY" --examples "$DECISIONS" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$COUNT"

run refused tune sft --model "$TOY" --examples "$EXAMPLES" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$LAYERS" \
  --output "$ROOT/other.out" --resume "$ROOT/sft/checkpoints"
refused "belongs to another run"
refused "seed was $COUNT and is $LAYERS"

run refused tune sft --model "$TOY" --examples "$EXAMPLES" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$COUNT" \
  --output "$ROOT/half.out" --checkpoint "$ROOT/half"
refused "--checkpoint-every"

run refused tune sft --model "$TOY" --examples "$EXAMPLES" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$COUNT" \
  --output "$ROOT/empty.out" --resume "$ROOT/no-checkpoints"
refused "holds no checkpoint"

# A malformed latest pointer must neither resume from nor retire an artifact
# outside the checkpoint output, even if it is otherwise a valid checkpoint.
cp "$ROOT/sft/checkpoints/latest" "$ROOT/sft-latest.before"
latest=$(cat "$ROOT/sft-latest.before")
cp -R "$ROOT/sft/checkpoints/$latest" "$ROOT/step-outside"
cp "$ROOT/step-outside/state.json" "$ROOT/outside-state.before"
cp "$ROOT/step-outside/tensors.safetensors" "$ROOT/outside-tensors.before"
printf '../step-outside\n' >"$ROOT/sft/checkpoints/latest"
run refused tune sft --model "$TOY" --examples "$EXAMPLES" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$COUNT" \
  --output "$ROOT/escaping-resume.out" --resume "$ROOT/sft/checkpoints"
refused "names invalid checkpoint"
run refused tune sft --model "$TOY" --examples "$EXAMPLES" "${ADAPTER[@]}" "${STEPS[@]}" --seed "$COUNT" \
  --output "$ROOT/escaping-save.out" --checkpoint "$ROOT/sft/checkpoints" --checkpoint-every "$ONE"
refused "names invalid checkpoint"
cmp "$ROOT/step-outside/state.json" "$ROOT/outside-state.before" || fail "a checkpoint write removed or changed the outside state"
cmp "$ROOT/step-outside/tensors.safetensors" "$ROOT/outside-tensors.before" || fail "a checkpoint write removed or changed the outside tensors"
cmp "$ROOT/sft/checkpoints/latest" <(printf '../step-outside\n') || fail "a refused write changed latest"
cp "$ROOT/sft-latest.before" "$ROOT/sft/checkpoints/latest"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
