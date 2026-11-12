#!/bin/sh
# Real test of how the settings only one objective takes fail
# (`ster.tune.objective`) and how sampling a policy-optimization
# conversation fails (`ster.tune.rollout`), through the built binary.
#
# Every setting is refused with exit status 2 before anything is read: the
# runs name a model that does not exist, so a refusal that came after the
# model was looked for would say so instead. The layer BiPO trains is
# refused once the fixture is loaded, the only point its layer count is
# known. The rollout causes run policy optimization on the fixture and on a
# copy of it carrying docs/examples/chat-template/tokenizer_config.json.
#
# Usage: tests/tuning/objectives.sh   (STER selects the binary, default target/debug/ster)
AREA=objectives
. "$(dirname "$0")/harness.sh"

ABSENT="$ROOT/no-such-model"
CHAT="$ROOT/chat"
cp -R "$MODEL" "$CHAT"
cp docs/examples/chat-template/tokenizer_config.json "$CHAT/tokenizer_config.json"

an_unknown_preference_loss_is_refused() {
  tune 2 dpo "$DPO" --model "$ABSENT" --loss nope
  said "Error: reading the preference loss: unknown preference loss \"nope\"; expected dpo or ipo; observed: --loss \"nope\""
  unsaid "finding model"
  passed an_unknown_preference_loss_is_refused
}

a_beta_of_zero_is_refused_before_the_model_loads() {
  tune 2 dpo "$DPO" --model "$ABSENT" --beta 0
  said "Error: starting direct preference optimization: direct preference optimization requires a finite beta above zero; at zero every pair's loss is the constant log 2 and no gradient reaches the model; observed: --beta 0"
  tune 2 bipo "$BIPO" --model "$ABSENT" --beta 0
  said "Error: starting bi-directional preference optimization: bi-directional preference optimization requires a finite beta above zero"
  unsaid "finding model"
  passed a_beta_of_zero_is_refused_before_the_model_loads
}

a_strength_of_zero_is_refused_before_the_model_loads() {
  tune 2 bipo "$BIPO" --model "$ABSENT" --strength 0
  said "Error: starting bi-directional preference optimization: bi-directional preference optimization requires a finite strength above zero, the scale +v and -v are added at; observed: --strength 0"
  unsaid "finding model"
  passed a_strength_of_zero_is_refused_before_the_model_loads
}

a_layer_past_the_model_is_refused() {
  tune 2 bipo "$BIPO" --layer 9
  said "Error: placing the steering vector bi-directional preference optimization learns: layer 9 is outside the model's 4 layers; observed: --layer 9; the model has layers 0 to 3; fix: give --layer a layer from 0 to 3"
  [ ! -e "$ROOT/bipo/vector.json" ] || fail "a refused run wrote a vector"
  passed a_layer_past_the_model_is_refused
}

a_group_of_one_is_refused_before_the_model_loads() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --group 1
  said "Error: starting group-relative policy optimization: group-relative policy optimization requires a group of at least two completions, because the group is the baseline; observed: --group 1"
  unsaid "finding model"
  passed a_group_of_one_is_refused_before_the_model_loads
}

no_assistant_turn_is_refused_before_the_model_loads() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --turns 0
  said "Error: starting group-relative policy optimization: group-relative policy optimization requires at least one assistant turn; observed: --turns 0"
  passed no_assistant_turn_is_refused_before_the_model_loads
}

a_simulated_user_for_one_turn_is_refused() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --turns 1 --user-model "$CHAT"
  said "Error: starting group-relative policy optimization: a simulated user writes turns between assistant turns, so it needs turns above one; this run asked for one; observed: --turns 1 --user-model $CHAT"
  passed a_simulated_user_for_one_turn_is_refused
}

a_conversation_without_a_simulated_user_is_refused() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --turns 2
  said "Error: starting group-relative policy optimization: a conversation of 2 assistant turns needs a simulated user model to write the turns in between; observed: --turns 2, no --user-model"
  passed a_conversation_without_a_simulated_user_is_refused
}

a_negative_policy_beta_is_refused_before_the_model_loads() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --beta=-1
  said "Error: starting group-relative policy optimization: group-relative policy optimization requires a finite beta of zero or more; observed: --beta -1"
  passed a_negative_policy_beta_is_refused_before_the_model_loads
}

a_temperature_of_zero_is_refused_before_the_model_loads() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --temperature 0
  said "Error: starting group-relative policy optimization: group-relative policy optimization requires a temperature above zero; argmax sampling would draw one identical completion per group; observed: --temperature 0"
  passed a_temperature_of_zero_is_refused_before_the_model_loads
}

no_new_token_is_refused_before_the_model_loads() {
  tune 2 grpo "$GRPO" --model "$ABSENT" --max-new-tokens 0
  said "Error: starting group-relative policy optimization: max_new_tokens must be greater than zero, or no completion has a token to reward; observed: --max-new-tokens 0"
  passed no_new_token_is_refused_before_the_model_loads
}

a_simulated_user_without_a_chat_template_is_refused() {
  tune 1 grpo "$GRPO" --turns 2 --user-model "$MODEL"
  said "Error: loading the simulated user $MODEL: the simulated user $MODEL publishes no chat template, so it cannot write a user's turn in a conversation; observed: chat template of $MODEL: Absent"
  [ ! -e "$ROOT/grpo/adapter.safetensors" ] || fail "a refused run wrote an adapter"
  passed a_simulated_user_without_a_chat_template_is_refused
}

# Every prompt's raw text fits 8 tokens with 2 sampled, but the policy's
# chat template wraps the first turn in markers that do not.
a_conversation_past_the_limit_is_refused() {
  tune 1 grpo "$GRPO" --model "$CHAT" --turns 2 --user-model "$CHAT" --max-sequence 8
  said "sampling assistant turn 1 of a conversation: turn 1 needs 11 conversation tokens plus 2 sampled tokens, past the 8 token limit; observed: turn 1: 11 + 2 tokens; limit 8"
  passed a_conversation_past_the_limit_is_refused
}

a_conversation_of_one_turn_is_refused() {
  run 2 converse --model "$ABSENT" --prompts docs/examples/prompts.json --user-model "$ABSENT" \
    --turns 1 --max-new-tokens 2 --max-sequence 64 --temperature 1 --seed 1 --output "$ROOT/conversations.jsonl"
  said "a conversation with a simulated user needs --turns of at least 2; one turn is ster generate"
  said "observed: --turns 1"
  [ ! -e "$ROOT/conversations.jsonl" ] || fail "a refused converse wrote its output"
  passed a_conversation_of_one_turn_is_refused
}

an_unknown_preference_loss_is_refused
a_beta_of_zero_is_refused_before_the_model_loads
a_strength_of_zero_is_refused_before_the_model_loads
a_layer_past_the_model_is_refused
a_group_of_one_is_refused_before_the_model_loads
no_assistant_turn_is_refused_before_the_model_loads
a_simulated_user_for_one_turn_is_refused
a_conversation_without_a_simulated_user_is_refused
a_negative_policy_beta_is_refused_before_the_model_loads
a_temperature_of_zero_is_refused_before_the_model_loads
no_new_token_is_refused_before_the_model_loads
a_simulated_user_without_a_chat_template_is_refused
a_conversation_past_the_limit_is_refused
a_conversation_of_one_turn_is_refused

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
