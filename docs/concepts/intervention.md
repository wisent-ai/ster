# Intervention

An intervention is the act of adding a trained [direction](direction.md) back
into the model's residual stream during generation. It exists only for the
lifetime of one `Runtime::generate` call; Ster does not modify the checkpoint
or save a steered model.

## Exact insertion point

For each decoder block `i`, Ster performs:

1. attention with its residual addition;
2. MLP with its residual addition;
3. optional final-token [activation capture](activation.md);
4. if the artifact has a vector for `i`, broadcast-add
   `strength × vector` to every sequence position in the block output;
5. pass the resulting residual stream to block `i + 1`.

During autoregressive generation the prompt pass is steered, and every later
single-token cached pass is also steered. The addition therefore occurs at
every forward step, not only once on the prompt. Capture precedes steering at
the same layer; normal CLI generation does not request captures.

## Steering plan

Before the first token is generated, Ster validates the artifact and builds a
`SteeringPlan`: a layer-sorted map of vectors copied onto the selected Candle
device in F32, with one shared `f64` strength. The artifact must:

- be schema-valid and belong to Ster;
- name the exact same model string as `--model`;
- have the same hidden width as the loaded model;
- contain only model-valid layer indices;
- contain at least one finite, correctly sized vector.

No partial application is allowed. A failure prevents generation entirely.

## Strength

`strength` is a signed scalar. Because all trained directions are unit length:

- positive values push toward the pair set's positive column;
- negative values push toward its negative column;
- `0` constructs the plan but makes the numeric addition zero;
- larger absolute values dominate the residual stream more strongly.

The CLI default and serve default are `1.0`. Negative CLI values must be
attached to the flag, for example `--strength=-1.0`, so clap does not interpret
the minus sign as another option.

There is no automatic calibration, clipping, or per-layer strength. Compare the
same prompt and seed without `--vector`, then sweep modest signed values and
judge model output. In the executed random toy checkpoint, strength `1.0`
changed

```text
loud sea hard hard , white , white , white loud loud
```

to

```text
white white white howls howls howls drifts , white white white white
```

This proves the insertion path changed logits, not that the random model
learned semantic calmness. At strength `2.0` the toy collapsed to one repeated
token; still larger positive strengths reached EOS immediately. Real models
also need empirical calibration.

## Generation interaction

The model uses a KV cache. `temperature <= 0` selects deterministic argmax;
positive temperature selects full-distribution sampling or top-p sampling when
`top_p` is present. Generation stops on the token limit, any configured EOS,
or the context limit. An immediate EOS is a successful empty result, not an
intervention failure.

Source-authored plan and application refusals are:

- `layer <L> steering vector width <W> does not match model width <H>`
- `steering plan contains no vectors`
- `artifact was trained for model "<A>", current model is "<B>"`
- `artifact width <A> does not match model width <M>`
- `failed to apply steering at layer <L>`

See [steering artifact](steering-artifact.md) for the earlier artifact gates
and the [runbook](../runbook.md) for fixes.
