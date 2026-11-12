#!/bin/sh
# Real test of how the numbers every trainer takes fail (`ster.tune.settings`)
# and what every run shares once its model is loaded (`ster.tune.run`),
# through the built binary on the four-layer fixture.
#
# Each setting is refused with exit status 2 before anything is read: the
# runs name a model that does not exist, so a refusal that came after the
# model was looked for would say so instead. The run-wide causes are
# provoked by a sequence limit no example fits and a pair side the
# fixture's tokenizer turns into one token.
#
# Usage: tests/tuning/scales/settings.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=settings
. "$(dirname "$0")/../harness.sh"

ABSENT="$ROOT/no-such-model"

no_pass_is_refused_before_the_model_loads() {
  tune 2 sft "$SFT" --model "$ABSENT" --epochs 0
  said "Error: starting supervised fine-tuning: supervised fine-tuning requires at least one epoch; observed: --epochs 0; fix: give --epochs a count of at least 1"
  tune 2 grpo "$GRPO" --model "$ABSENT" --iterations 0
  said "Error: starting group-relative policy optimization: group-relative policy optimization requires at least one iteration; observed: --iterations 0"
  unsaid "finding model"
  passed no_pass_is_refused_before_the_model_loads
}

no_accumulation_is_refused_before_the_model_loads() {
  tune 2 dpo "$DPO" --model "$ABSENT" --accumulation 0
  said "Error: starting direct preference optimization: direct preference optimization requires an accumulation of at least one pair; observed: --accumulation 0"
  unsaid "finding model"
  passed no_accumulation_is_refused_before_the_model_loads
}

no_batch_is_refused_before_the_model_loads() {
  tune 2 reward "$REWARD" --model "$ABSENT" --batch-size 0
  said "Error: starting reward modeling: reward modeling requires a batch of at least one pair; observed: --batch-size 0"
  unsaid "finding model"
  passed no_batch_is_refused_before_the_model_loads
}

an_unusable_learning_rate_is_refused_before_the_model_loads() {
  tune 2 sft "$SFT" --model "$ABSENT" --learning-rate 0
  said "Error: starting supervised fine-tuning: supervised fine-tuning requires a finite learning rate above zero; observed: --learning-rate 0"
  tune 2 sft "$SFT" --model "$ABSENT" --learning-rate=-0.5
  said "observed: --learning-rate -0.5"
  tune 2 sft "$SFT" --model "$ABSENT" --learning-rate inf
  said "observed: --learning-rate inf"
  passed an_unusable_learning_rate_is_refused_before_the_model_loads
}

a_sequence_limit_below_two_is_refused_before_the_model_loads() {
  tune 2 sft "$SFT" --model "$ABSENT" --max-sequence 1
  said "Error: starting supervised fine-tuning: max_sequence must be at least two tokens, so that one token can predict another; observed: --max-sequence 1; fix: give --max-sequence at least 2, as many tokens as the longest example the run should train on"
  passed a_sequence_limit_below_two_is_refused_before_the_model_loads
}

a_limit_shorter_than_every_example_is_refused() {
  tune 1 sft "$SFT" --max-sequence 3
  said "Error: encoding the examples to train on: every example is longer than the sequence limit, so there is nothing to train on; observed: 8 examples, the shortest "
  said " tokens; the limit is 3 tokens, the lower of --max-sequence and the model's context"
  [ ! -e "$ROOT/sft/adapter.safetensors" ] || fail "a run that trained on nothing wrote an adapter"
  tune 1 dpo "$DPO" --max-sequence 3
  said "Error: encoding the pairs to train on: every pair is longer than the sequence limit"
  passed a_limit_shorter_than_every_example_is_refused
}

a_side_the_tokenizer_drops_is_refused() {
  jq '.pairs[0].positive = "calm"' "$ROOT/pairs.json" >"$ROOT/one-token.json"
  tune 1 dpo "$DPO" --pairs "$ROOT/one-token.json"
  said "Error: encoding pair 0: pair 0 chosen side encodes to one token, so there is nothing to predict; observed: chosen side \"calm\" encodes to the token ids ["
  passed a_side_the_tokenizer_drops_is_refused
}

no_pass_is_refused_before_the_model_loads
no_accumulation_is_refused_before_the_model_loads
no_batch_is_refused_before_the_model_loads
an_unusable_learning_rate_is_refused_before_the_model_loads
a_sequence_limit_below_two_is_refused_before_the_model_loads
a_limit_shorter_than_every_example_is_refused
a_side_the_tokenizer_drops_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
