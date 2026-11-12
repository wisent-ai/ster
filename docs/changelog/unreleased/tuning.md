# Unreleased: tuning

Part of the [changelog](../../../CHANGELOG.md).

- Device failures in a training step say which phase failed (forward and
  loss, or gradients and update) and the learning rate; the raw device
  answer moves to the observed state. A step that summed no loss, a GRPO
  group of fewer than two conversations or a conversation without turns, and
  logits read from a pass that projected none are refused by declared causes
  where `expect`s leaned on other functions' invariants.
- A tensor the device cannot allocate, a score it cannot compute and an
  optimizer step it cannot take are refused with the cause `ster.tensor`
  declares for Candle's error kind (`device-refused`, exit 69, for an
  accelerator out of memory or a failed kernel; `dtype-unsupported`;
  `tensors-misbuilt` and the other defects of Ster's own tensors), naming the
  tensor and its shape, what was scored, or the step, its phase, its learning
  rate and its trained tensors. `step-failed`, `score-failed` and
  `tensor-unallocatable`, which said "most often out of memory" whatever
  Candle's kind, are removed. A refusal declared at a command's end names
  the command (`running \`ster tune dpo\``, `ster request tune/dpo`) as its
  step and shows the context lines above Candle's error as where it was
  raised, where it used to take those lines as its step.
- A reward artifact without its head, met while scoring, and an adapter
  merging no projection are refused through their declared causes
  (`ster.lora.artifact.head-missing`, `targets-none` or `layers-none`)
  where the run used to stop on an assertion.
- A sidecar no Ster trainer writes is refused through the cause of what it
  declares, where one cause (`declaration-invalid`) carried the site's
  sentence as its reason: `rank-zero`, `alpha-invalid`, `hidden-size-zero`,
  `targets-none`, `layers-none`, `target-repeated` and `layer-repeated`,
  each naming the sidecar and the weights it read.
  `tests/tuning/sidecar.sh` provokes each.
- Merging, exporting and ablating refuse through causes of their own, where
  three causes took their reason and fix from the step as text: a reward
  model (`reward-unmergeable`, `reward-unexportable`), a GGUF checkpoint
  (`gguf-unmergeable`, `gguf-unexportable`, `gguf-unablatable`) and a
  projection stored fused or transposed (`fused-unexportable`,
  `transposed-unexportable`, `fused-unablatable`). `tests/tuning/rewrite.sh`
  provokes the reward and GGUF ones for each step.
- A checkpoint position a resumed run cannot stand at is refused through
  the cause of its reason, where one cause (`position-inconsistent`) carried
  the site's sentence as its reason: a position past its pass
  (`position-past-pass`, naming the steps this run plans), a running report
  of another form (`running-report-foreign`), an epoch the report records no
  step of (`epoch-lossless`) and a run whose report holds no loss
  (`run-lossless`). `tests/tuning/checkpoint.sh` provokes each.
- A checkpoint whose config states `num_hidden_layers` 0 is refused as
  `ster.lora.spec.layers-none` when adapters are placed on it, where training
  used to stop on an internal assertion.
- The tune catalogue's 105 declared tests exist: `tests/tuning/settings.sh`,
  `examples.sh`, `objectives.sh`, `strength.sh`, `checkpoint.sh`,
  `reward.sh`, `adapters.sh` and `rewrite.sh` drive every provokable cause
  through the binary on the fixture, sharing `tests/tuning/harness.sh`;
  `tests/errors/untested.json` lists, and `docs/guide/errors.md` explains,
  the device-memory, clock, race and outside-scorer causes no local state
  provokes. Two declared causes nothing raised are gone (a runtime without
  adapters, a reward head unfit after the artifact was already checked).
- `ster vector evaluate --strengths` and `ster tune bipo` clamp the sequence
  limit to the model's context as every trainer does, where a pair past it
  reached the batched forward and failed there.
- `ster tune merge` names the model it was given once in its
  another-model refusal.
