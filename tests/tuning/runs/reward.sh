#!/bin/sh
# Real test of how policy optimization's reward fails (`ster.tune.reward`),
# through `ster tune grpo` on the four-layer fixture.
#
# What `--reward` says is refused before the policy loads: the runs that
# only read it name a model that does not exist, so a refusal that came
# after the model was looked for would say so instead. A scorer address
# nothing listens at (port 1 of this machine) is refused once the first
# completion is scored, with the retry status 69. A reward artifact whose
# head is the largest f32 in every place scores the fixture's normalized
# residual stream past the f32 range, and the group is refused before a
# policy update.
#
# Usage: tests/tuning/runs/reward.sh   (STER selects the binary, default .build/kompilator/debug/ster)
AREA=reward
. "$(dirname "$0")/../harness.sh"

ABSENT="$ROOT/no-such-model"
SCORER="http://127.0.0.1:1/score"
no_adapter() {
  [ ! -e "$ROOT/grpo/adapter.safetensors" ] || fail "a refused run wrote an adapter"
}

a_blank_reward_source_is_refused_before_the_model_loads() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --reward " "
  said "Error: choosing where the reward comes from: group-relative policy optimization requires a reward source; pass length or a reward artifact; observed: --reward \" \""
  unsaid "finding model"
  passed a_blank_reward_source_is_refused_before_the_model_loads
}

a_reward_source_that_names_nothing_is_refused() {
  tune 1 grpo "$GRPO" --model "$ABSENT" --reward "$ROOT/no-such-reward.safetensors"
  said "Error: choosing where the reward comes from: reward source \"$ROOT/no-such-reward.safetensors\" is neither the keyword length nor a file that exists"
  unsaid "finding model"
  passed a_reward_source_that_names_nothing_is_refused
}

a_scorer_without_a_field_is_refused() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --reward "$SCORER"
  said "Error: reading the reward scorer --reward names: reward scorer \"$SCORER\" names no field: end it with #/<json pointer> to the reward, for example #/score"
  passed a_scorer_without_a_field_is_refused
}

a_scorer_field_that_is_no_pointer_is_refused() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --reward "$SCORER#score"
  said "Error: reading the reward scorer --reward names: reward scorer field \"score\" is not a JSON pointer; it starts with /"
  passed a_scorer_field_that_is_no_pointer_is_refused
}

an_empty_scorer_bearer_is_refused() {
  export STER_REWARD_BEARER=" "
  tune 1 grpo "$GRPO" --model "$ABSENT" --reward "$SCORER#/score"
  unset STER_REWARD_BEARER
  said "Error: reading STER_REWARD_BEARER for the reward scorer: STER_REWARD_BEARER is present but empty"
  passed an_empty_scorer_bearer_is_refused
}

a_scorer_bearer_that_is_not_text_is_refused() {
  STER_REWARD_BEARER=$(printf 'token\377')
  export STER_REWARD_BEARER
  tune 1 grpo "$GRPO" --model "$ABSENT" --reward "$SCORER#/score"
  unset STER_REWARD_BEARER
  said "Error: reading STER_REWARD_BEARER for the reward scorer: STER_REWARD_BEARER is not valid Unicode, so it cannot be sent as a header"
  passed a_scorer_bearer_that_is_not_text_is_refused
}

a_scorer_nothing_listens_at_is_refused() {
  tune 69 grpo "$GRPO" --reward "$SCORER#/score"
  said "asking the reward scorer $SCORER for the score of one completion: no answer came from the reward scorer $SCORER: nothing listens there, its name does not resolve, or the connection broke; observed: url: $SCORER; the client's error kind "
  said "fix: start the scorer at $SCORER or name one that answers in --reward, then run again"
  no_adapter
  passed a_scorer_nothing_listens_at_is_refused
}

rewards_that_are_not_finite_are_refused() {
  jq -n --arg model "$MODEL" '{schema_version: 1, product: "ster", kind: "reward", model: $model,
    model_revision: null, rank: 1, alpha: 1, targets: ["query"], layers: [0], hidden_size: 64, train: {}}' \
    >"$ROOT/huge.json"
  floats "$ROOT/huge.zeros" 128 "$ZERO"
  floats "$ROOT/huge.head" 64 "$HUGE"
  cat "$ROOT/huge.zeros" "$ROOT/huge.head" >"$ROOT/huge.data"
  safetensors "$ROOT/huge.safetensors" \
    '{"layers.0.query.a":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]},"layers.0.query.b":{"dtype":"F32","shape":[64,1],"data_offsets":[256,512]},"reward.head":{"dtype":"F32","shape":[1,64],"data_offsets":[512,768]}}' \
    "$ROOT/huge.data"
  run 0 tune inspect "$ROOT/huge.safetensors"
  tune 1 grpo "$GRPO" --reward "$ROOT/huge.safetensors"
  said "summing the rewards of conversation 1 of the 2 sampled for one prompt: the conversation's reward is not a finite number"
  said "; observed: after turn 1: this turn's reward "
  said "; reward source: a reward artifact's head"
  no_adapter
  passed rewards_that_are_not_finite_are_refused
}

a_blank_reward_source_is_refused_before_the_model_loads
a_reward_source_that_names_nothing_is_refused
a_scorer_without_a_field_is_refused
a_scorer_field_that_is_no_pointer_is_refused
an_empty_scorer_bearer_is_refused
a_scorer_bearer_that_is_not_text_is_refused
a_scorer_nothing_listens_at_is_refused
rewards_that_are_not_finite_are_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
