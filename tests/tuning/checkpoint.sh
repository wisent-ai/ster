#!/bin/sh
# Real test of how a training run's checkpoints fail: writing one
# (`ster.tune.save`), the `latest` pointer (`ster.tune.latest`) and
# continuing from one (`ster.tune.resume`), through `ster tune sft` on the
# four-layer fixture.
#
# One real run writes a checkpoint every 3 of its 8 steps, so its newest
# checkpoint stands 6 steps into the first pass; each resume test copies
# that directory, breaks one thing in the copy (the state removed or
# rewritten, the tensors replaced by another run's) and resumes from it.
# Every refusal exits 1.
#
# Usage: tests/tuning/checkpoint.sh   (STER selects the binary, default target/debug/ster)
AREA=checkpoint
. "$(dirname "$0")/harness.sh"

DURABLE="--checkpoint-every 3"
# checkpointed DIRECTORY [FLAG VALUE]...: a real run checkpointing into DIRECTORY.
checkpointed() {
  directory=$1
  shift
  # shellcheck disable=SC2086
  tune 0 sft "$SFT" --checkpoint "$directory" $DURABLE "$@"
}
checkpointed "$ROOT/good"
GOOD_STEP=$(cat "$ROOT/good/latest")
case "$GOOD_STEP" in step-6-*) ;; *) fail "the run's newest checkpoint is $GOOD_STEP, not one at step 6" ;; esac
passed "a run writes its newest checkpoint and names it in latest"
# broken NAME: a copy of the good checkpoint directory at $ROOT/NAME; $STEP
# its newest step directory.
broken() {
  CHECKPOINT="$ROOT/$1"
  cp -R "$ROOT/good" "$CHECKPOINT"
  STEP="$CHECKPOINT/$GOOD_STEP"
}
# resume EXPECTED DIRECTORY [FLAG VALUE]...: the good run, continued from DIRECTORY.
resume() {
  expected=$1
  directory=$2
  shift 2
  tune "$expected" sft "$SFT" --resume "$directory" "$@"
}

a_checkpoint_directory_under_a_file_is_refused() {
  # shellcheck disable=SC2086
  tune 1 sft "$SFT" --checkpoint "$ROOT/examples.json/checkpoints" $DURABLE
  said "Error: creating the checkpoint directory $ROOT/examples.json/checkpoints: the path runs through a file: $ROOT/examples.json is not a directory, so nothing can be created below it; observed: path: $ROOT/examples.json/checkpoints; $ROOT/examples.json: a file, "
  said "; no checkpoint of this step was written, and latest names the checkpoint it named before; the system answered: "
  passed a_checkpoint_directory_under_a_file_is_refused
}

a_read_only_checkpoint_directory_is_refused() {
  mkdir -p "$ROOT/closed"
  chmod 555 "$ROOT/closed"
  # shellcheck disable=SC2086
  tune 1 sft "$SFT" --checkpoint "$ROOT/closed" $DURABLE
  chmod 755 "$ROOT/closed"
  said "Error: creating the checkpoint attempt $ROOT/closed/.step-3-"
  said ".partial: this account may not change $ROOT/closed or $ROOT/closed/.step-3-"
  said "observed: path: $ROOT/closed/.step-3-"
  said "directory $ROOT/closed (a directory, "
  said ", mode 555, owner uid $(id -u)); no checkpoint of this step was written"
  passed a_read_only_checkpoint_directory_is_refused
}

a_predecessor_that_cannot_be_removed_is_refused() {
  mkdir -p "$ROOT/kept/step-0-old/locked"
  : >"$ROOT/kept/step-0-old/locked/file"
  chmod 555 "$ROOT/kept/step-0-old/locked"
  printf 'step-0-old' >"$ROOT/kept/latest"
  # shellcheck disable=SC2086
  tune 1 sft "$SFT" --checkpoint "$ROOT/kept" $DURABLE
  chmod 755 "$ROOT/kept/step-0-old/locked"
  said "Error: removing the checkpoint latest named before $ROOT/kept/step-0-old: this account may not change $ROOT/kept or $ROOT/kept/step-0-old: their permission bits or owner keep this account out"
  said "; checkpoint $ROOT/kept/step-3-"
  said " is written and named in latest, so --resume continues from it; the one it replaced is still there; the system answered: "
  case "$(cat "$ROOT/kept/latest")" in step-3-*) ;; *) fail "latest does not name the checkpoint written before the refusal" ;; esac
  passed a_predecessor_that_cannot_be_removed_is_refused
}

a_latest_pointer_that_is_a_directory_is_refused() {
  mkdir -p "$ROOT/pointer-folder/latest"
  # shellcheck disable=SC2086
  tune 1 sft "$SFT" --checkpoint "$ROOT/pointer-folder" $DURABLE
  said "Error: reading the latest pointer $ROOT/pointer-folder/latest: $ROOT/pointer-folder/latest is a directory, and the latest pointer is one file; observed: path: $ROOT/pointer-folder/latest (a directory, "
  resume 1 "$ROOT/pointer-folder"
  said "Error: reading the latest pointer $ROOT/pointer-folder/latest: $ROOT/pointer-folder/latest is a directory, and the latest pointer is one file"
  passed a_latest_pointer_that_is_a_directory_is_refused
}

a_latest_pointer_naming_another_place_is_refused() {
  broken elsewhere
  printf '../good/%s' "$GOOD_STEP" >"$CHECKPOINT/latest"
  resume 1 "$CHECKPOINT"
  said "Error: reading $CHECKPOINT/latest, which names the newest checkpoint: it names \"../good/$GOOD_STEP\", which is not one step directory inside $CHECKPOINT"
  passed a_latest_pointer_naming_another_place_is_refused
}

resuming_a_directory_without_a_checkpoint_is_refused() {
  mkdir -p "$ROOT/empty"
  resume 1 "$ROOT/empty"
  said "Error: finding the newest checkpoint in --resume $ROOT/empty: --resume $ROOT/empty holds no checkpoint: no latest pointer in it names one, so no run checkpointed there; observed: --resume $ROOT/empty (a directory, "
  said "); $ROOT/empty/latest: absent"
  passed resuming_a_directory_without_a_checkpoint_is_refused
}

a_checkpoint_without_its_state_is_refused() {
  broken stateless
  rm "$STEP/state.json"
  resume 1 "$CHECKPOINT"
  said "Error: reading checkpoint state $STEP/state.json: the step directory latest names holds no state.json: Ster did not finish writing it there, or it was edited by hand; observed: path: $STEP/state.json (absent); its directory $STEP; the system answered: "
  passed a_checkpoint_without_its_state_is_refused
}

a_state_that_is_not_json_is_refused() {
  broken not-json
  echo nope >"$STEP/state.json"
  resume 1 "$CHECKPOINT"
  said "Error: reading the checkpoint state $STEP/state.json: it is not JSON, and the checkpoint state is the state.json a ster tune run writes into each step directory of its --checkpoint: it is another kind of file, or one broken by hand; observed: path: $STEP/state.json (a file, 5 bytes, "
  passed a_state_that_is_not_json_is_refused
}

a_state_of_another_kind_is_refused() {
  broken other-kind
  jq '.kind = "other"' "$ROOT/good/$GOOD_STEP/state.json" >"$STEP/state.json"
  resume 1 "$CHECKPOINT"
  said "Error: reading checkpoint state $STEP/state.json: $STEP/state.json says it is \"other\", not a ster tune checkpoint; observed: kind: \"other\""
  passed a_state_of_another_kind_is_refused
}

a_checkpoint_of_another_run_is_refused() {
  resume 1 "$ROOT/good" --learning-rate 0.002
  said "Error: continuing from checkpoint $ROOT/good/$GOOD_STEP: checkpoint $ROOT/good/$GOOD_STEP belongs to another run, so continuing it would train an adapter no single run produced; observed: learning_rate was 0.001 and is 0.002"
  passed a_checkpoint_of_another_run_is_refused
}

checkpoint_tensors_that_are_not_safetensors_are_refused() {
  broken junk-tensors
  echo junk >"$STEP/tensors.safetensors"
  resume 1 "$CHECKPOINT"
  said "Error: reading the checkpoint's tensors $STEP/tensors.safetensors: $STEP/tensors.safetensors is not a safetensors file: its header or its table of tensors does not parse"
  passed checkpoint_tensors_that_are_not_safetensors_are_refused
}

# other NAME [FLAG VALUE]...: the good run's checkpoint with the tensors a
# run of other flags checkpointed in its place.
other() {
  name=$1
  shift
  checkpointed "$ROOT/run-$name" "$@"
  broken "$name"
  cp "$ROOT/run-$name/$(cat "$ROOT/run-$name/latest")/tensors.safetensors" "$STEP/tensors.safetensors"
}

checkpoint_tensors_of_other_targets_are_refused() {
  other key --targets key
  resume 1 "$CHECKPOINT"
  said "Error: reading the tensors of checkpoint $STEP: $STEP/tensors.safetensors holds no "
  said ", which this run trains; observed: missing: "
  passed checkpoint_tensors_of_other_targets_are_refused
}

checkpoint_tensors_of_another_rank_are_refused() {
  other rank --rank 4
  resume 1 "$CHECKPOINT"
  said "Error: reading the tensors of checkpoint $STEP: "
  said " of $STEP/tensors.safetensors has another shape than this run trains: the checkpoint was written with another rank, target width or model; observed: "
  said ": stored ["
  passed checkpoint_tensors_of_another_rank_are_refused
}

checkpoint_tensors_of_more_targets_are_refused() {
  other more --targets query,value,key
  resume 1 "$CHECKPOINT"
  said "Error: reading the tensors of checkpoint $STEP: $STEP/tensors.safetensors holds "
  said ", which this run does not train; observed: extra: "
  passed checkpoint_tensors_of_more_targets_are_refused
}

# positioned NAME JQ: a copy of the good checkpoint whose newest state is
# the good one through JQ.
positioned() {
  broken "$1"
  jq "$2" "$ROOT/good/$GOOD_STEP/state.json" >"$STEP/state.json"
}

a_position_past_its_pass_is_refused() {
  positioned past '.position.within = 99'
  resume 1 "$CHECKPOINT"
  said "continuing the supervised fine-tuning run from its checkpoint: the checkpoint's position stands 99 steps into pass 0, and this run plans 8 steps for that pass"
  said "; observed: pass 0, step 6, 99 steps into the pass; this run's pass 0: 8 steps"
  passed a_position_past_its_pass_is_refused
}

a_running_report_of_another_form_is_refused() {
  positioned foreign '.position.running = "a report"'
  resume 1 "$CHECKPOINT"
  said "continuing the supervised fine-tuning run from its checkpoint: the running report the checkpoint's position carries is not the one a supervised fine-tuning run keeps"
  said "; observed: pass 0, step 6, 6 steps into the pass; the report reader answered: "
  passed a_running_report_of_another_form_is_refused
}

# At the end of the only pass, with no step of it recorded.
an_epoch_the_report_records_no_step_of_is_refused() {
  positioned no-step '.position.within = 8 | .position.running.epoch_units = [] | .position.running.epoch_losses = []'
  resume 1 "$CHECKPOINT"
  said "closing epoch 0 of the resumed supervised fine-tuning run: the checkpoint's position puts the run at the end of epoch 0, and its running report records no step of that epoch"
  said "; observed: pass 0, step 6, 8 steps into the pass; steps recorded for the epoch: 0"
  passed an_epoch_the_report_records_no_step_of_is_refused
}

# At the end of the only pass, its steps recorded and its first loss not.
a_report_without_its_losses_is_refused() {
  positioned no-loss '.position.within = 8 | .position.running.first_loss = null'
  resume 1 "$CHECKPOINT"
  said "reporting the resumed supervised fine-tuning run: the resumed run took its last step, and the checkpoint's running report records no first or final loss"
  passed a_report_without_its_losses_is_refused
}

a_checkpoint_directory_under_a_file_is_refused
a_read_only_checkpoint_directory_is_refused
a_predecessor_that_cannot_be_removed_is_refused
a_latest_pointer_that_is_a_directory_is_refused
a_latest_pointer_naming_another_place_is_refused
resuming_a_directory_without_a_checkpoint_is_refused
a_checkpoint_without_its_state_is_refused
a_state_that_is_not_json_is_refused
a_state_of_another_kind_is_refused
a_checkpoint_of_another_run_is_refused
checkpoint_tensors_that_are_not_safetensors_are_refused
checkpoint_tensors_of_other_targets_are_refused
checkpoint_tensors_of_another_rank_are_refused
checkpoint_tensors_of_more_targets_are_refused
a_position_past_its_pass_is_refused
a_running_report_of_another_form_is_refused
an_epoch_the_report_records_no_step_of_is_refused
a_report_without_its_losses_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
