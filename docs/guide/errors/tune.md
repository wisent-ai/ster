# How training, scoring and rewriting adapters fail

Every declared way a `ster tune` command, `ster vector evaluate --strengths`
and `ster vector ablate` fail, with their `ster request` twins: the settings
every trainer and each objective takes, the example sets, what every run
shares once its model is loaded, its checkpoints and resuming from them, the
reward policy optimization scores with and the conversations it samples,
choosing a steering strength, and fitting a trained artifact to the
checkpoint a merge, an export or an ablation writes. The adapter artifact
itself is on [lora.md](lora.md), loading the model on
[model.md](model.md), and encoding text on [runtime.md](runtime.md). How a
refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/tune.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.tune.settings`

check the passes, accumulation, batch, learning rate and sequence limit a training run is given, before any model, data or checkpoint is read

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.tune.settings`: 5 declared cause(s) of its own.

<a id="ster-tune-settings-passes-none"></a>

### `ster.tune.settings.passes-none`

- **What failed:** starting {subject}
- **Cause:** {subject} requires at least one {pass}
- **Observed state shown:** --{pass}s 0
- **Fix:** give --{pass}s a count of at least 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::no_pass_is_refused_before_the_model_loads`

<a id="ster-tune-settings-accumulation-none"></a>

### `ster.tune.settings.accumulation-none`

- **What failed:** starting {subject}
- **Cause:** {subject} requires an accumulation of at least one {unit}
- **Observed state shown:** --accumulation {value}
- **Fix:** give --accumulation a count of at least 1; 1 steps the optimizer after every forward pass
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::no_accumulation_is_refused_before_the_model_loads`

<a id="ster-tune-settings-batch-none"></a>

### `ster.tune.settings.batch-none`

- **What failed:** starting {subject}
- **Cause:** {subject} requires a batch of at least one {unit}
- **Observed state shown:** --batch-size {value}
- **Fix:** give --batch-size a count of at least 1; 1 reproduces an unbatched run byte for byte
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::no_batch_is_refused_before_the_model_loads`

<a id="ster-tune-settings-learning-rate-unusable"></a>

### `ster.tune.settings.learning-rate-unusable`

- **What failed:** starting {subject}
- **Cause:** {subject} requires a finite learning rate above zero
- **Observed state shown:** --learning-rate {value}
- **Fix:** give --learning-rate a finite number above zero, such as 0.0001
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::an_unusable_learning_rate_is_refused_before_the_model_loads`

<a id="ster-tune-settings-sequence-short"></a>

### `ster.tune.settings.sequence-short`

- **What failed:** starting {subject}
- **Cause:** max_sequence must be at least two tokens, so that one token can predict another
- **Observed state shown:** --max-sequence {value}
- **Fix:** give --max-sequence at least 2, as many tokens as the longest {unit} the run should train on
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::a_sequence_limit_below_two_is_refused_before_the_model_loads`

## `ster.tune.objective`

check the settings only one objective takes: DPO's loss and beta, the strength and layer of BiPO's vector, and the group, turns, simulated user, beta, temperature and token budget of policy optimization

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.tune.objective`: 11 declared cause(s) of its own.

<a id="ster-tune-objective-loss-unknown"></a>

### `ster.tune.objective.loss-unknown`

- **What failed:** reading the preference loss
- **Cause:** unknown preference loss {value}; expected dpo or ipo
- **Observed state shown:** --loss {value}
- **Fix:** give --loss dpo for the sigmoid loss of DPO or ipo for IPO's squared loss
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::an_unknown_preference_loss_is_refused`

<a id="ster-tune-objective-beta-unusable"></a>

### `ster.tune.objective.beta-unusable`

- **What failed:** starting {subject}
- **Cause:** {subject} requires a finite beta above zero; at zero every pair's loss is the constant log 2 and no gradient reaches the model
- **Observed state shown:** --beta {value}
- **Fix:** give --beta a finite number above zero; 0.1 is the value the DPO paper reports across its settings
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_beta_of_zero_is_refused_before_the_model_loads`

<a id="ster-tune-objective-strength-unusable"></a>

### `ster.tune.objective.strength-unusable`

- **What failed:** starting bi-directional preference optimization
- **Cause:** bi-directional preference optimization requires a finite strength above zero, the scale +v and -v are added at
- **Observed state shown:** --strength {value}
- **Fix:** give --strength a finite number above zero, such as 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_strength_of_zero_is_refused_before_the_model_loads`

<a id="ster-tune-objective-layer-outside"></a>

### `ster.tune.objective.layer-outside`

- **What failed:** placing the steering vector bi-directional preference optimization learns
- **Cause:** layer {layer} is outside the model's {layers} layers
- **Observed state shown:** --layer {layer}; the model has layers 0 to {last}
- **Fix:** give --layer a layer from 0 to {last}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_layer_past_the_model_is_refused`

<a id="ster-tune-objective-group-too-small"></a>

### `ster.tune.objective.group-too-small`

- **What failed:** starting group-relative policy optimization
- **Cause:** group-relative policy optimization requires a group of at least two completions, because the group is the baseline
- **Observed state shown:** --group {group}
- **Fix:** give --group at least 2
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_group_of_one_is_refused_before_the_model_loads`

<a id="ster-tune-objective-turns-none"></a>

### `ster.tune.objective.turns-none`

- **What failed:** starting group-relative policy optimization
- **Cause:** group-relative policy optimization requires at least one assistant turn
- **Observed state shown:** --turns {turns}; --group {group}
- **Fix:** give --turns at least 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::no_assistant_turn_is_refused_before_the_model_loads`

<a id="ster-tune-objective-user-unneeded"></a>

### `ster.tune.objective.user-unneeded`

- **What failed:** starting group-relative policy optimization
- **Cause:** a simulated user writes turns between assistant turns, so it needs turns above one; this run asked for one
- **Observed state shown:** --turns 1 --user-model {user_model}
- **Fix:** give --turns 2 or more for a conversation, or leave out --user-model
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_simulated_user_for_one_turn_is_refused`

<a id="ster-tune-objective-user-missing"></a>

### `ster.tune.objective.user-missing`

- **What failed:** starting group-relative policy optimization
- **Cause:** a conversation of {turns} assistant turns needs a simulated user model to write the turns in between
- **Observed state shown:** --turns {turns}, no --user-model
- **Fix:** name the model that writes the user's turns with --user-model, or give --turns 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_conversation_without_a_simulated_user_is_refused`

<a id="ster-tune-objective-policy-beta-negative"></a>

### `ster.tune.objective.policy-beta-negative`

- **What failed:** starting group-relative policy optimization
- **Cause:** group-relative policy optimization requires a finite beta of zero or more
- **Observed state shown:** --beta {value}
- **Fix:** give --beta 0 to switch the KL term off, or a finite number above zero
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_negative_policy_beta_is_refused_before_the_model_loads`

<a id="ster-tune-objective-temperature-unusable"></a>

### `ster.tune.objective.temperature-unusable`

- **What failed:** starting group-relative policy optimization
- **Cause:** group-relative policy optimization requires a temperature above zero; argmax sampling would draw one identical completion per group
- **Observed state shown:** --temperature {value}
- **Fix:** give --temperature a number above zero, such as 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_temperature_of_zero_is_refused_before_the_model_loads`

<a id="ster-tune-objective-new-tokens-none"></a>

### `ster.tune.objective.new-tokens-none`

- **What failed:** starting group-relative policy optimization
- **Cause:** max_new_tokens must be greater than zero, or no completion has a token to reward
- **Observed state shown:** --max-new-tokens {tokens}; --max-sequence {max_sequence}
- **Fix:** give --max-new-tokens at least 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::no_new_token_is_refused_before_the_model_loads`

## `ster.tune.examples`

read an example-set file of prompts and completions and check every example carries text

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.tune.examples`: 6 declared cause(s) of its own.

<a id="ster-tune-examples-example-set-missing"></a>

### `ster.tune.examples.example-set-missing`

- **What failed:** reading example set {path}
- **Cause:** no file is there: nothing at {path} is the example set named
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name an existing example-set file, shaped as docs/examples/tuning/examples.json is; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/examples.sh::a_missing_example_set_is_refused`

<a id="ster-tune-examples-example-set-empty"></a>

### `ster.tune.examples.example-set-empty`

- **What failed:** reading example set {label}
- **Cause:** example set {label} contains no examples
- **Observed state shown:** examples: 0; name: "{name}"
- **Fix:** add at least one object with a `prompt` and a `completion` text to its examples
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/examples.sh::an_example_set_without_examples_is_refused`

<a id="ster-tune-examples-example-blank"></a>

### `ster.tune.examples.example-blank`

- **What failed:** reading example set {label}
- **Cause:** example set {label} contains an empty prompt or completion
- **Observed state shown:** example {index}: prompt {prompt}, completion {completion}
- **Fix:** give example {index} text in both its prompt and its completion, or remove it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/examples.sh::an_example_with_an_empty_side_is_refused`

<a id="ster-tune-examples-system-blank"></a>

### `ster.tune.examples.system-blank`

- **What failed:** reading example set {label}
- **Cause:** example set {label} contains an empty system prompt
- **Observed state shown:** example {index}: system {system}
- **Fix:** give example {index} the text of its system turn, or remove its "system" key
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/examples.sh::an_empty_system_prompt_is_refused`

<a id="ster-tune-examples-example-conversation-unfit"></a>

### `ster.tune.examples.example-conversation-unfit`

- **What failed:** reading example set {label}
- **Cause:** row {index} of example set {label} is a conversation of other turns than one example's: an optional system turn, one of the user, one of the assistant
- **Observed state shown:** row {index}: turns {turns}
- **Fix:** split row {index} into rows of one exchange each, carrying the turns before it in its user turn, or remove it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/examples.sh::a_conversation_of_more_than_one_exchange_is_refused`

<a id="ster-tune-examples-example-row-unknown"></a>

### `ster.tune.examples.example-row-unknown`

- **What failed:** reading example set {label}
- **Cause:** row {index} of example set {label} is none of the forms an example row is read in: OpenAI's messages, ShareGPT's conversations, Alpaca's instruction and output, or prompt and completion
- **Observed state shown:** row {index}: fields {fields}
- **Fix:** write row {index} in one of those forms, or the set as one object whose `examples` list holds objects with a `prompt` and a `completion` text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/examples.sh::a_row_of_no_known_form_is_refused`

## `ster.tune.run`

open the adapters a loaded model trains, fingerprint the data for its checkpoints, encode every unit that fits the sequence limit, score what the run measures without training, and take each optimizer step on the device

Started through: run only by the functionalities that use it. Uses: `ster.runtime.encode`, `ster.tensor`. Failure catalogue `ster.tune.run`: 4 declared cause(s) of its own.

<a id="ster-tune-run-data-unreadable"></a>

### `ster.tune.run.data-unreadable`

- **What failed:** fingerprinting {path} for the run's checkpoints
- **Cause:** the file the run trains on was read at the start and cannot be read again: it was moved, removed or made unreadable while the run started
- **Observed state shown:** path: {path} ({state}); reading it again: error kind {kind}; the system answered: {error}
- **Fix:** leave {path} in place and readable while the run starts, then run again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::data_removed_while_the_run_starts_is_refused`

<a id="ster-tune-run-nothing-fits"></a>

### `ster.tune.run.nothing-fits`

- **What failed:** encoding the {unit}s to {purpose}
- **Cause:** every {unit} is longer than the sequence limit, so there is nothing to {purpose}
- **Observed state shown:** {count} {unit}s, the shortest {shortest} tokens; the limit is {limit} tokens, the lower of --max-sequence and the model's context
- **Fix:** give --max-sequence at least {shortest}, or shorten the {unit}s
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::a_limit_shorter_than_every_example_is_refused`

<a id="ster-tune-run-side-unpredictable"></a>

### `ster.tune.run.side-unpredictable`

- **What failed:** encoding pair {index}
- **Cause:** pair {index} {side} side encodes to one token, so there is nothing to predict
- **Observed state shown:** {side} side {text} encodes to the token ids {tokens}
- **Fix:** give pair {index}'s {side} side text this model's tokenizer turns into at least one token, or remove the pair
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::a_side_the_tokenizer_drops_is_refused`

<a id="ster-tune-run-step-empty"></a>

### `ster.tune.run.step-empty`

- **What failed:** taking optimizer step {step}
- **Cause:** the step was planned with {forwards} forwards holding {units} {unit}s, so no loss was summed for its backward pass: the planner cuts every step from a non-empty slice of the epoch's order, so this is a defect of Ster, not of the data
- **Observed state shown:** step {step}: {forwards} forwards, {units} {unit}s, {tensors} trained tensors; nothing was written for this step
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command and its --batch-size and --accumulation; the run's earlier checkpoints are left as they were
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/settings.sh::a_step_planned_without_units_is_refused`

## `ster.tune.latest`

read the latest file of a checkpoint directory and resolve the one step directory it names

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.tune.latest`: 1 declared cause(s) of its own.

<a id="ster-tune-latest-latest-invalid"></a>

### `ster.tune.latest.latest-invalid`

- **What failed:** reading {pointer}, which names the newest checkpoint
- **Cause:** it names {name}, which is not one step directory inside {directory}
- **Observed state shown:** latest: {name}
- **Fix:** write the name of one step-* directory of {directory} into {pointer}, or start the run with an empty --checkpoint directory
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_latest_pointer_naming_another_place_is_refused`

## `ster.tune.save`

write a run's trained tensors, optimizer moments and position under a new step directory of --checkpoint, name it in latest, and remove the checkpoint it replaces

Started through: run only by the functionalities that use it. Uses: `ster.tune.latest`, `ster.file`. Failure catalogue `ster.tune.save`: 1 declared cause(s) of its own.

<a id="ster-tune-save-clock-unreadable"></a>

### `ster.tune.save.clock-unreadable`

- **What failed:** naming the checkpoint written at step {step}
- **Cause:** the system clock stands before 1970, so the attempt cannot be named by the time it was written
- **Observed state shown:** step: {step}; the clock answered: {error}
- **Fix:** set the system clock to the current time and run again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_clock_before_1970_is_refused`

## `ster.tune.resume`

read the newest checkpoint of --resume, refuse one another run wrote, and put its tensors, optimizer moments, learning rates and position back into this run

Started through: run only by the functionalities that use it. Uses: `ster.tune.latest`, `ster.file`. Failure catalogue `ster.tune.resume`: 12 declared cause(s) of its own.

<a id="ster-tune-resume-no-checkpoint"></a>

### `ster.tune.resume.no-checkpoint`

- **What failed:** finding the newest checkpoint in --resume {directory}
- **Cause:** --resume {directory} holds no checkpoint: no latest pointer in it names one, so no run checkpointed there
- **Observed state shown:** --resume {directory} ({directory_state}); {pointer}: {state}
- **Fix:** give --resume the --checkpoint directory of the run to continue (`ls -ln {directory}` shows what it holds), or start the run without --resume
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::resuming_a_directory_without_a_checkpoint_is_refused`

<a id="ster-tune-resume-state-missing"></a>

### `ster.tune.resume.state-missing`

- **What failed:** reading checkpoint state {path}
- **Cause:** the step directory latest names holds no state.json: Ster did not finish writing it there, or it was edited by hand
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** resume from a --checkpoint directory whose latest names a step directory Ster wrote (`ls -ln {parent}` shows what this one holds), or start the run without --resume
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_checkpoint_without_its_state_is_refused`

<a id="ster-tune-resume-kind-foreign"></a>

### `ster.tune.resume.kind-foreign`

- **What failed:** reading checkpoint state {path}
- **Cause:** {path} says it is {kind}, not a ster tune checkpoint
- **Observed state shown:** kind: {kind}
- **Fix:** resume from a checkpoint Ster wrote, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_state_of_another_kind_is_refused`

<a id="ster-tune-resume-foreign-run"></a>

### `ster.tune.resume.foreign-run`

- **What failed:** continuing from checkpoint {checkpoint}
- **Cause:** checkpoint {checkpoint} belongs to another run, so continuing it would train an adapter no single run produced
- **Observed state shown:** {differing}
- **Fix:** run with the model, data and options the checkpoint records, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_checkpoint_of_another_run_is_refused`

<a id="ster-tune-resume-tensor-absent"></a>

### `ster.tune.resume.tensor-absent`

- **What failed:** reading the tensors of checkpoint {checkpoint}
- **Cause:** {path} holds no {key}, which this run trains
- **Observed state shown:** missing: {key}
- **Fix:** resume from the checkpoint this run wrote, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_other_targets_are_refused`

<a id="ster-tune-resume-tensor-unfit"></a>

### `ster.tune.resume.tensor-unfit`

- **What failed:** reading the tensors of checkpoint {checkpoint}
- **Cause:** {key} of {path} has another shape than this run trains: the checkpoint was written with another rank, target width or model
- **Observed state shown:** {key}: stored {stored}, trained {trained}
- **Fix:** resume from the checkpoint this run wrote, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_another_rank_are_refused`

<a id="ster-tune-resume-tensor-dtype"></a>

### `ster.tune.resume.tensor-dtype`

- **What failed:** reading the tensors of checkpoint {checkpoint}
- **Cause:** {key} of {path} is stored in another dtype than this run trains it in: the checkpoint was written at another --precision
- **Observed state shown:** {key}: stored {stored}, trained {trained}
- **Fix:** resume with the --precision the checkpoint was written at, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_another_dtype_are_refused`

<a id="ster-tune-resume-tensor-extra"></a>

### `ster.tune.resume.tensor-extra`

- **What failed:** reading the tensors of checkpoint {checkpoint}
- **Cause:** {path} holds {key}, which this run does not train
- **Observed state shown:** extra: {key}
- **Fix:** resume from the checkpoint this run wrote, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_more_targets_are_refused`

<a id="ster-tune-resume-position-past-pass"></a>

### `ster.tune.resume.position-past-pass`

- **What failed:** continuing the {objective} run from its checkpoint
- **Cause:** the checkpoint's position stands {within} steps into pass {pass}, and this run plans {steps} steps for that pass: the state was edited, or the data, batch or accumulation it was planned with changed
- **Observed state shown:** {position}; this run's pass {pass}: {steps} steps
- **Fix:** resume with the data and settings the checkpoint was written with, from an earlier --checkpoint directory, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_position_past_its_pass_is_refused`

<a id="ster-tune-resume-running-report-foreign"></a>

### `ster.tune.resume.running-report-foreign`

- **What failed:** continuing the {objective} run from its checkpoint
- **Cause:** the running report the checkpoint's position carries is not the one a {objective} run keeps: the state was edited, or another objective wrote it
- **Observed state shown:** {position}; the report reader answered: {error}
- **Fix:** resume from a checkpoint a {objective} run wrote, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_running_report_of_another_form_is_refused`

<a id="ster-tune-resume-epoch-lossless"></a>

### `ster.tune.resume.epoch-lossless`

- **What failed:** closing epoch {epoch} of the resumed {objective} run
- **Cause:** the checkpoint's position puts the run at the end of epoch {epoch}, and its running report records no step of that epoch, so the epoch has no mean loss: the state was edited
- **Observed state shown:** {position}; steps recorded for the epoch: 0
- **Fix:** resume from an earlier --checkpoint directory, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::an_epoch_the_report_records_no_step_of_is_refused`

<a id="ster-tune-resume-run-lossless"></a>

### `ster.tune.resume.run-lossless`

- **What failed:** reporting the resumed {objective} run
- **Cause:** the resumed run took its last step, and the checkpoint's running report records no first or final loss for the steps its position says were taken: the state was edited
- **Observed state shown:** {position}; first loss, final loss or epoch means absent from the report
- **Fix:** resume from an earlier --checkpoint directory, or start the run without --resume
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/checkpoint.sh::a_report_without_its_losses_is_refused`

## `ster.tune.reward`

resolve --reward to the length reward, an outside scorer posted the prompt and text over HTTP, or a reward artifact, and score each sampled completion with it

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.tune.reward`: 17 declared cause(s) of its own.

<a id="ster-tune-reward-source-blank"></a>

### `ster.tune.reward.source-blank`

- **What failed:** choosing where the reward comes from
- **Cause:** group-relative policy optimization requires a reward source; pass length or a reward artifact
- **Observed state shown:** --reward {value}
- **Fix:** give --reward length, the path of an artifact ster tune reward wrote, or https://<host>/<path>#/<json pointer> of a scorer
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_blank_reward_source_is_refused_before_the_model_loads`

<a id="ster-tune-reward-source-unknown"></a>

### `ster.tune.reward.source-unknown`

- **What failed:** choosing where the reward comes from
- **Cause:** reward source {value} is neither the keyword length nor a file that exists
- **Observed state shown:** --reward {value}
- **Fix:** give --reward length, the path of an artifact ster tune reward wrote, or https://<host>/<path>#/<json pointer> of a scorer
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_reward_source_that_names_nothing_is_refused`

<a id="ster-tune-reward-scorer-field-absent"></a>

### `ster.tune.reward.scorer-field-absent`

- **What failed:** reading the reward scorer --reward names
- **Cause:** reward scorer {value} names no field: end it with #/<json pointer> to the reward, for example #/score
- **Observed state shown:** --reward {value}
- **Fix:** end --reward with #/ and the JSON pointer of the number the scorer answers, such as #/score
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_without_a_field_is_refused`

<a id="ster-tune-reward-scorer-pointer-invalid"></a>

### `ster.tune.reward.scorer-pointer-invalid`

- **What failed:** reading the reward scorer --reward names
- **Cause:** reward scorer field {pointer} is not a JSON pointer; it starts with /
- **Observed state shown:** --reward {value}
- **Fix:** write the field after # as a JSON pointer, such as #/score or #/result/0
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_field_that_is_no_pointer_is_refused`

<a id="ster-tune-reward-bearer-empty"></a>

### `ster.tune.reward.bearer-empty`

- **What failed:** reading STER_REWARD_BEARER for the reward scorer
- **Cause:** STER_REWARD_BEARER is present but empty
- **Observed state shown:** STER_REWARD_BEARER: {bytes} bytes, all whitespace; the scorer: {url}
- **Fix:** set STER_REWARD_BEARER to the scorer's bearer token, or unset it for a scorer that needs none
- **Code:** `auth`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::an_empty_scorer_bearer_is_refused`

<a id="ster-tune-reward-bearer-not-unicode"></a>

### `ster.tune.reward.bearer-not-unicode`

- **What failed:** reading STER_REWARD_BEARER for the reward scorer
- **Cause:** STER_REWARD_BEARER is not valid Unicode, so it cannot be sent as a header
- **Observed state shown:** STER_REWARD_BEARER holds {bytes} bytes that are not UTF-8; the scorer: {url}
- **Fix:** set STER_REWARD_BEARER to the scorer's bearer token as text
- **Code:** `auth`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_bearer_that_is_not_text_is_refused`

<a id="ster-tune-reward-scorer-unreachable"></a>

### `ster.tune.reward.scorer-unreachable`

- **What failed:** asking the reward scorer {url} for the score of one completion
- **Cause:** no answer came from the reward scorer {url}: nothing listens there, its name does not resolve, or the connection broke
- **Observed state shown:** url: {url}; the client's error kind {kind}; it answered: {error}
- **Fix:** start the scorer at {url} or name one that answers in --reward, then run again
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_nothing_listens_at_is_refused`

<a id="ster-tune-reward-scorer-refused"></a>

### `ster.tune.reward.scorer-refused`

- **What failed:** asking the reward scorer {url} for the score of one completion
- **Cause:** the reward scorer {url} answered HTTP {status}, which says the request itself is wrong: it takes another body than an object of `prompt` and `text` or another address
- **Observed state shown:** HTTP {status}; STER_REWARD_BEARER {bearer}; the scorer's body: {body}
- **Fix:** name in --reward the scorer address that takes an object of `prompt` and `text`; the scorer's body says what it expected
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_answering_an_error_status_is_refused`

<a id="ster-tune-reward-scorer-unauthorized"></a>

### `ster.tune.reward.scorer-unauthorized`

- **What failed:** asking the reward scorer {url} for the score of one completion
- **Cause:** the reward scorer {url} answered HTTP {status}: it does not accept the bearer token sent, or wants one and none was sent
- **Observed state shown:** HTTP {status}; STER_REWARD_BEARER {bearer}; the scorer's body: {body}
- **Fix:** set STER_REWARD_BEARER to a token the scorer at {url} accepts, then run again
- **Code:** `auth`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_refusing_the_bearer_is_refused`

<a id="ster-tune-reward-scorer-limited"></a>

### `ster.tune.reward.scorer-limited`

- **What failed:** asking the reward scorer {url} for the score of one completion
- **Cause:** the reward scorer {url} answered HTTP 429: it limits how often it is asked, and this run passed the limit
- **Observed state shown:** HTTP {status}; the scorer's body: {body}
- **Fix:** run again after a while, or score fewer completions per prompt (--group)
- **Code:** `rate_limit`, exit 69; retrying unchanged can help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_limiting_requests_is_refused`

<a id="ster-tune-reward-scorer-failing"></a>

### `ster.tune.reward.scorer-failing`

- **What failed:** asking the reward scorer {url} for the score of one completion
- **Cause:** the reward scorer {url} answered HTTP {status}: it failed on its side, so the request was right and the scorer could not score it
- **Observed state shown:** HTTP {status}; the scorer's body: {body}
- **Fix:** run again once the scorer at {url} serves; its own log says why it failed
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_failing_on_its_side_is_refused`

<a id="ster-tune-reward-scorer-answer-unreadable"></a>

### `ster.tune.reward.scorer-answer-unreadable`

- **What failed:** reading the reward scorer's answer from {url}
- **Cause:** the answer broke off before it was read whole: the scorer closed the connection midway
- **Observed state shown:** url: {url}; error kind {kind}; the client answered: {error}
- **Fix:** run again; when it repeats, check that the scorer at {url} finishes its answers
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_answer_that_breaks_off_is_refused`

<a id="ster-tune-reward-scorer-not-text"></a>

### `ster.tune.reward.scorer-not-text`

- **What failed:** reading the reward scorer's answer from {url}
- **Cause:** the scorer {url} answered bytes that are not UTF-8 text, and a score is read from a JSON document
- **Observed state shown:** url: {url}; the client answered: {error}
- **Fix:** name a scorer address in --reward that answers a JSON document
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_answer_that_is_not_text_is_refused`

<a id="ster-tune-reward-scorer-not-json"></a>

### `ster.tune.reward.scorer-not-json`

- **What failed:** reading the reward scorer's answer from {url}
- **Cause:** the reward scorer {url} answered text that is not JSON, and a score is read from a JSON document
- **Observed state shown:** answer ({size} bytes): {body}; the JSON reader stopped at line {line}, column {column}: {error}
- **Fix:** name a scorer address in --reward that answers a JSON document
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_answer_that_is_not_json_is_refused`

<a id="ster-tune-reward-scorer-no-number"></a>

### `ster.tune.reward.scorer-no-number`

- **What failed:** reading the reward scorer's answer from {url}
- **Cause:** the reward scorer {url} answered without a number at {pointer}
- **Observed state shown:** answer: {body}
- **Fix:** point the #/ fragment of --reward at the number the scorer answers
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::a_scorer_answer_without_the_number_is_refused`

<a id="ster-tune-reward-reward-not-finite"></a>

### `ster.tune.reward.reward-not-finite`

- **What failed:** summing the rewards of conversation {conversation} of the {group} sampled for one prompt
- **Cause:** the conversation's reward is not a finite number: the reward source answered an infinite or NaN reward for a turn, or the turns' rewards summed past the largest finite number, so no advantage can be taken from it; the group is refused before a policy update
- **Observed state shown:** after turn {turn}: this turn's reward {score}, the sum {total}; reward source: {source}
- **Fix:** use a reward source that answers finite numbers: a scorer whose number stays below 1e300, or a reward artifact trained without diverging
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::rewards_that_are_not_finite_are_refused`

<a id="ster-tune-reward-group-spread-not-finite"></a>

### `ster.tune.reward.group-spread-not-finite`

- **What failed:** normalizing the rewards of the group sampled for one prompt
- **Cause:** every conversation's reward is finite, and the group's mean or variance is not: the rewards lie so far apart that their squared spread overflows, so no advantage can be taken from them; the group is refused before a policy update
- **Observed state shown:** {count} conversations, rewards {rewards}: mean {mean}, variance {variance}; reward source: {source}
- **Fix:** use a reward source whose numbers stay below 1e150, so their squares stay finite: rescale the scorer's answer
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/reward.sh::rewards_too_far_apart_are_refused`

## `ster.tune.rollout`

sample the conversations of one policy-optimization group: the policy's assistant turns and, above one turn, the simulated user's turns between them

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.tune.rollout`: 3 declared cause(s) of its own.

<a id="ster-tune-rollout-user-template-absent"></a>

### `ster.tune.rollout.user-template-absent`

- **What failed:** loading the simulated user {model}
- **Cause:** the simulated user {model} publishes no chat template, so it cannot write a user's turn in a conversation
- **Observed state shown:** chat template of {model}: {status}
- **Fix:** give --user-model a chat model whose tokenizer_config.json carries a chat_template
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_simulated_user_without_a_chat_template_is_refused`

<a id="ster-tune-rollout-conversation-too-long"></a>

### `ster.tune.rollout.conversation-too-long`

- **What failed:** sampling assistant turn {turn} of a conversation
- **Cause:** turn {turn} needs {context} conversation tokens plus {sampled} sampled tokens, past the {limit} token limit
- **Observed state shown:** turn {turn}: {context} + {sampled} tokens; limit {limit}
- **Fix:** lower --turns or --max-new-tokens, or raise --max-sequence
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_conversation_past_the_limit_is_refused`

<a id="ster-tune-rollout-turn-empty"></a>

### `ster.tune.rollout.turn-empty`

- **What failed:** sampling assistant turn {turn} of a conversation
- **Cause:** assistant turn {turn} of a sampled conversation was empty: its first token ended the sequence, so it has nothing to take a gradient through, and dropping it would bias the group's baseline
- **Observed state shown:** turn {turn}: 0 tokens
- **Fix:** raise --temperature so the policy draws other first tokens, or train the policy with ster tune sft first so it answers these prompts
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::an_empty_assistant_turn_is_refused`

## `ster.converse`

read the prompt set, load the model (with its adapter) and the simulated user, and write each conversation they play as one JSON line

Started through: `ster converse`. Uses: `ster.prompts`, `ster.model.load`, `ster.tune.rollout`, `ster.file`. Failure catalogue `ster.converse`: 1 declared cause(s) of its own.

<a id="ster-converse-turns-few"></a>

### `ster.converse.turns-few`

- **What failed:** playing conversations of --turns {turns} with the simulated user {user}
- **Cause:** a conversation with a simulated user needs --turns of at least 2; one turn is ster generate, since the user answers only after the model's first turn
- **Observed state shown:** --turns {turns}
- **Fix:** give --turns 2 or more, or answer each prompt once with ster generate --prompts
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/objectives.sh::a_conversation_of_one_turn_is_refused`

## `ster.tune.strength`

measure how far each named strength moves the model's log-probability of every pair's positive side over its negative side, and choose the one that orders the most pairs

Started through: `ster vector evaluate --strengths`, `ster request vector/evaluate`. Uses: `ster.tune.run`. Failure catalogue `ster.tune.strength`: 4 declared cause(s) of its own.

<a id="ster-tune-strength-strengths-none"></a>

### `ster.tune.strength.strengths-none`

- **What failed:** choosing the strength to add the steering artifact at
- **Cause:** strength selection needs at least one strength to measure
- **Observed state shown:** strengths: []; --batch-size {batch}; --max-sequence {max_sequence}
- **Fix:** name the strengths to measure, comma-separated, such as --strengths 0.5,1,2
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/strength.sh::no_strength_is_refused`

<a id="ster-tune-strength-strength-not-finite"></a>

### `ster.tune.strength.strength-not-finite`

- **What failed:** choosing the strength to add the steering artifact at
- **Cause:** strength selection needs finite strengths, not {strength}
- **Observed state shown:** --strengths {strengths}
- **Fix:** give --strengths finite numbers only, such as 0.5,1,2
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/strength.sh::a_strength_that_is_not_finite_is_refused`

<a id="ster-tune-strength-setting-missing"></a>

### `ster.tune.strength.setting-missing`

- **What failed:** choosing the strength to add the steering artifact at
- **Cause:** --strengths needs --{flag}; Ster assumes none
- **Observed state shown:** --strengths given without --{flag}
- **Fix:** give --{flag} beside --strengths
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/strength.sh::strengths_without_a_batch_are_refused`

<a id="ster-tune-strength-setting-zero"></a>

### `ster.tune.strength.setting-zero`

- **What failed:** choosing the strength to add the steering artifact at
- **Cause:** strength selection requires {name} of at least one
- **Observed state shown:** --{flag} 0
- **Fix:** give --{flag} at least 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/strength.sh::a_batch_of_zero_is_refused`

## `ster.tune.fit`

fit a trained artifact to the safetensors checkpoint it names: its kind, model, width, layers, and the tensor of every projection it rewrites

Started through: run only by the functionalities that use it. Uses: `ster.model.load`. Failure catalogue `ster.tune.fit`: 15 declared cause(s) of its own.

<a id="ster-tune-fit-reward-unmergeable"></a>

### `ster.tune.fit.reward-unmergeable`

- **What failed:** merging adapter {path}
- **Cause:** the artifact is a {kind} model, not a generation adapter: a merge folds an adapter's low-rank updates into the base and writes no head, so the merged model would drop the head the reward model was trained to score with
- **Observed state shown:** kind: {kind}; model: {model}
- **Fix:** name an artifact ster tune sft, dpo, grpo or decide wrote; `ster tune inspect {path}` prints an artifact's kind
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_reward_model_is_refused_by_merge`

<a id="ster-tune-fit-reward-unexportable"></a>

### `ster.tune.fit.reward-unexportable`

- **What failed:** exporting adapter {path}
- **Cause:** the artifact is a {kind} model, not a generation adapter: a PEFT adapter directory holds low-rank updates only and has no place for the reward model's head
- **Observed state shown:** kind: {kind}; model: {model}
- **Fix:** export an artifact ster tune sft, dpo, grpo or decide wrote; `ster tune inspect {path}` prints an artifact's kind
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_reward_model_is_refused_by_export`

<a id="ster-tune-fit-model-other"></a>

### `ster.tune.fit.model-other`

- **What failed:** {step} artifact {path}
- **Cause:** the artifact was trained for model {trained}, current model is {model}
- **Observed state shown:** trained for: {trained}
- **Fix:** name the model the artifact was trained for with --model {trained}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::an_artifact_for_another_model_is_refused`

<a id="ster-tune-fit-gguf-unmergeable"></a>

### `ster.tune.fit.gguf-unmergeable`

- **What failed:** merging artifact {path}
- **Cause:** {model} is a GGUF checkpoint, and an adapter is merged into safetensors weights only: a GGUF base holds quantized tensors, and writing them back dequantized would publish a model of another size and precision under the base's name
- **Observed state shown:** {model}: a .gguf file
- **Fix:** run the adapter on it unmerged (`ster generate --model {model} --adapter {path} --prompt …`), or merge into the base model's safetensors release
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_gguf_checkpoint_is_refused_by_merge`

<a id="ster-tune-fit-gguf-unexportable"></a>

### `ster.tune.fit.gguf-unexportable`

- **What failed:** exporting artifact {path}
- **Cause:** {model} is a GGUF checkpoint, and a PEFT adapter names the modules of the base model's safetensors release, which a GGUF file's tensor names are not
- **Observed state shown:** {model}: a .gguf file
- **Fix:** export against the base model's safetensors release (`ster tune export --model <release> --adapter {path} --format peft --output <directory>`)
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_gguf_checkpoint_is_refused_by_export`

<a id="ster-tune-fit-gguf-unablatable"></a>

### `ster.tune.fit.gguf-unablatable`

- **What failed:** ablating artifact {path}
- **Cause:** {model} is a GGUF checkpoint, and a direction is projected out of safetensors weights only: a GGUF file holds quantized tensors Ster does not write back
- **Observed state shown:** {model}: a .gguf file
- **Fix:** ablate the base model's safetensors release (`ster vector ablate --model <release> --vector {path} --strength <share> --output <directory>`)
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_gguf_checkpoint_is_refused_by_ablate`

<a id="ster-tune-fit-width-other"></a>

### `ster.tune.fit.width-other`

- **What failed:** {step} artifact {path}
- **Cause:** artifact width {trained} does not match model width {width}
- **Observed state shown:** hidden_size: {width}
- **Fix:** name the model the artifact was trained for; an artifact fits only the width it was trained at
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::an_artifact_of_another_width_is_refused`

<a id="ster-tune-fit-layer-past"></a>

### `ster.tune.fit.layer-past`

- **What failed:** {step} artifact {path}
- **Cause:** layer {layer} is outside the model's {layers} layers
- **Observed state shown:** artifact layers: {artifact_layers}
- **Fix:** name the model the artifact was trained for; its layers run from 0 to {last}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::an_artifact_past_the_model_layers_is_refused`

<a id="ster-tune-fit-projection-absent"></a>

### `ster.tune.fit.projection-absent`

- **What failed:** {step} the {target} projection of layer {layer}
- **Cause:** this model has no {target} projection at layer {layer}
- **Observed state shown:** artifact targets: {targets}
- **Fix:** name the checkpoint the artifact was trained on
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_layer_without_the_projection_is_refused`

<a id="ster-tune-fit-fused-unexportable"></a>

### `ster.tune.fit.fused-unexportable`

- **What failed:** exporting the {target} projection of layer {layer}
- **Cause:** this model stores the {target} projection fused with others in one tensor, and PEFT has no module of its own for one part of a fused tensor
- **Observed state shown:** tensor: {tensor}
- **Fix:** merge the adapter with `ster tune merge --model {model} --adapter {path}` instead, which writes the update into the rows the projection owns
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_fused_projection_is_refused_for_export`

<a id="ster-tune-fit-transposed-unexportable"></a>

### `ster.tune.fit.transposed-unexportable`

- **What failed:** exporting the {target} projection of layer {layer}
- **Cause:** this model stores the {target} projection transposed (GPT-2's Conv1D), and a PEFT LoRA module's factors are laid out for a Linear weight, so PEFT would apply them to the wrong axes
- **Observed state shown:** tensor: {tensor}
- **Fix:** merge the adapter with `ster tune merge --model {model} --adapter {path}` instead
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_transposed_projection_is_refused_for_export`

<a id="ster-tune-fit-fused-unablatable"></a>

### `ster.tune.fit.fused-unablatable`

- **What failed:** ablating the {target} projection of layer {layer}
- **Cause:** this model stores the {target} projection fused with others in one tensor, and an ablation rewrites each projection's whole tensor: Ster does not rewrite one part of a fused tensor
- **Observed state shown:** tensor: {tensor}
- **Fix:** ablate a checkpoint of a family that stores each projection as its own tensor, or steer at run time with `ster generate --model {model} --vector {path}`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_fused_projection_is_refused_for_ablation`

<a id="ster-tune-fit-tensor-missing"></a>

### `ster.tune.fit.tensor-missing`

- **What failed:** {step} the {target} projection of layer {layer}
- **Cause:** {holder} holds no tensor {name}
- **Observed state shown:** {holder}: {count} tensors
- **Fix:** name the checkpoint and the artifact that belong together: `ster tune inspect` prints the artifact's tensors
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_checkpoint_without_the_projection_is_refused`

<a id="ster-tune-fit-update-unfit"></a>

### `ster.tune.fit.update-unfit`

- **What failed:** {step} the {target} projection of layer {layer}
- **Cause:** the update the artifact carries has another shape than {name} as the checkpoint stores it: the artifact was trained on another model or width
- **Observed state shown:** {name}: {shape} in {path}; Candle answered: {error}
- **Fix:** name the checkpoint the artifact was trained on; `ster tune inspect` prints the model and the shapes the artifact holds
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::an_update_that_does_not_fit_is_refused`

<a id="ster-tune-fit-update-uncomputed"></a>

### `ster.tune.fit.update-uncomputed`

- **What failed:** {step} the {target} projection of layer {layer}
- **Cause:** the shapes fit, and the device did not compute the update into {name}: it has no memory left for the dense update, or does not compute in the weight's dtype
- **Observed state shown:** {name}: {shape} in {path}; the device answered: {error}
- **Fix:** run again with `--device cpu`, where the update is computed in host memory
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::an_update_the_device_does_not_compute_is_refused`

## `ster.tune.write`

read every tensor of a safetensors checkpoint and write the rewritten tensors beside copies of its config, tokenizer and chat template

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.tune.write`: 1 declared cause(s) of its own.

<a id="ster-tune-write-tensor-twice"></a>

### `ster.tune.write.tensor-twice`

- **What failed:** reading the weights of {model}
- **Cause:** model weight {name} appears in more than one weight file, so which one the checkpoint means is unknown
- **Observed state shown:** {name} in {path} and an earlier shard
- **Fix:** use a checkpoint whose shards hold each tensor once, as model.safetensors.index.json lists them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_tensor_in_two_shards_is_refused`

## `ster.tune.merge`

fold a LoRA adapter into its base's safetensors weights and write a standalone checkpoint

Started through: `ster tune merge`, `ster request tune/merge`. Uses: `ster.tune.fit`, `ster.tune.write`. Failure catalogue `ster.tune.merge`: 0 declared cause(s) of its own.

## `ster.tune.export`

write a LoRA adapter as a PEFT adapter directory for vLLM and peft, beside its base's tokenizer

Started through: `ster tune export`, `ster request tune/export`. Uses: `ster.tune.fit`, `ster.tune.write`. Failure catalogue `ster.tune.export`: 0 declared cause(s) of its own.

## `ster.tune.ablate`

rewrite a checkpoint's attention-output and MLP-down projections so they no longer write a steering direction, and save it as a standalone checkpoint

Started through: `ster vector ablate`, `ster request vector/ablate`. Uses: `ster.tune.fit`, `ster.tune.write`. Failure catalogue `ster.tune.ablate`: 3 declared cause(s) of its own.

<a id="ster-tune-ablate-ablation-strength"></a>

### `ster.tune.ablate.ablation-strength`

- **What failed:** removing the direction of {path} from {model}
- **Cause:** ablation strength {strength} changes nothing or nothing sensible; it must be a finite, non-zero number
- **Observed state shown:** --strength {strength}
- **Fix:** give --strength a finite, non-zero number: 1 removes the direction's whole component
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::an_ablation_strength_of_zero_is_refused`

<a id="ster-tune-ablate-direction-empty"></a>

### `ster.tune.ablate.direction-empty`

- **What failed:** removing the direction at layer {layer} of {path}
- **Cause:** the direction at layer {layer} has length {length}, so there is nothing to remove
- **Observed state shown:** length of layer {layer}: {length}
- **Fix:** train the direction again with ster vector train; a direction of length zero says the pairs did not separate
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_direction_of_length_zero_is_refused`

<a id="ster-tune-ablate-write-width"></a>

### `ster.tune.ablate.write-width`

- **What failed:** removing the direction at layer {layer} of {path}
- **Cause:** {name} writes {outputs} outputs where the residual stream is {width} wide
- **Observed state shown:** {name}: {outputs} outputs
- **Fix:** name the checkpoint the direction was trained on; its residual-stream writes are as wide as its hidden_size
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/rewrite.sh::a_write_of_another_width_is_refused`

## `ster.tune.sft`

train LoRA adapters on prompt and completion examples, the loss taken over each completion only

Started through: `ster tune sft`, `ster request tune/sft`. Uses: `ster.lora.spec`, `ster.tune.settings`, `ster.tune.examples`, `ster.model.load`, `ster.tune.run`, `ster.tune.resume`, `ster.tune.save`, `ster.lora.artifact`. Failure catalogue `ster.tune.sft`: 0 declared cause(s) of its own.

## `ster.tune.preference`

train on contrastive pairs: LoRA adapters that prefer each positive side (DPO, IPO), one steering vector both ways (BiPO), or a scalar reward head that ranks the two sides

Started through: `ster tune dpo`, `ster tune bipo`, `ster tune reward`, `ster request tune/dpo`, `ster request tune/bipo`, `ster request tune/reward`. Uses: `ster.lora.spec`, `ster.tune.settings`, `ster.tune.objective`, `ster.workspace.active`, `ster.pairs.file`, `ster.model.load`, `ster.tune.run`, `ster.tune.resume`, `ster.tune.save`, `ster.lora.artifact`. Failure catalogue `ster.tune.preference`: 0 declared cause(s) of its own.

## `ster.tune.grpo`

optimize LoRA adapters against a reward, each prompt's sampled group its own baseline

Started through: `ster tune grpo`, `ster request tune/grpo`. Uses: `ster.lora.spec`, `ster.tune.settings`, `ster.tune.objective`, `ster.tune.reward`, `ster.prompts`, `ster.model.load`, `ster.tune.rollout`, `ster.tune.run`, `ster.tune.resume`, `ster.tune.save`, `ster.lora.artifact`. Failure catalogue `ster.tune.grpo`: 0 declared cause(s) of its own.

## `ster.tune.decide`

train LoRA adapters to answer typed decisions with calibrated probabilities, every option order of each labelled question one row

Started through: `ster tune decide`, `ster request tune/decide`. Uses: `ster.lora.spec`, `ster.tune.settings`, `ster.decide.examples`, `ster.model.load`, `ster.tune.run`, `ster.tune.resume`, `ster.tune.save`, `ster.lora.artifact`. Failure catalogue `ster.tune.decide`: 0 declared cause(s) of its own.

## `ster.tune.evaluate`

score a checkpoint, with or without an adapter, on held-out examples: loss and perplexity over each completion, nothing trained or written

Started through: `ster tune evaluate`, `ster request tune/evaluate`. Uses: `ster.tune.settings`, `ster.tune.examples`, `ster.model.load`, `ster.tune.run`. Failure catalogue `ster.tune.evaluate`: 0 declared cause(s) of its own.
