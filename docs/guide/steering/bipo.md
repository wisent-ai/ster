# Steering-vector preference optimization (BiPO)

`ster tune bipo` learns one steering vector instead of an adapter, by the
bi-directional preference optimization of Cao et al. (arXiv 2406.00045). The
policy is the frozen model with `d · strength · v` added to the residual stream
at `--layer`; the reference is the same model with nothing added, scored once up
front exactly as `ster tune dpo` scores it. Each forward draws `d` from {+, −}:
with `+v` the loss asks the model to prefer each pair's positive side, with `−v`
its negative side, which is the same `--loss` (`dpo` or `ipo`) with `--beta`
negated. One vector therefore learns to move the behaviour both ways, and
`ster generate --vector <artifact> --strength s` turns it up, down or past zero
afterwards. `v` starts at zero, so the first policy is the reference itself, and
it is the only variable: no adapter is registered and no base weight moves.

Every setting is the operator's: `--layer`, `--strength` (the multiple added while
training), `--beta`, `--epochs`, `--learning-rate`, `--accumulation`,
`--warmup-steps`, `--max-sequence`, `--batch-size` and `--seed`. A layer outside
the model is refused with `layer <n> is outside the model's <layers> layers`; a
beta, strength or learning rate that is not a finite number above zero with
`bi-directional preference optimization requires a finite <name> above zero, not
<value>`; epochs, accumulation, batch size or sequence limit of zero with
`bi-directional preference optimization requires <name> of at least one`. Pairs
are encoded and skipped as in `ster tune dpo`. The output is a steering artifact
(`method: "bipo"`, one layer vector whose `train_margin` and `train_accuracy`
are the final epoch's mean reward margin and accuracy), and the report records
`layer`, `strength`, `loss`, `beta`, `pairs`, `trained_pairs`, `skipped_long`,
`epochs`, `steps`, `first_loss`, `final_loss`, `mean_final_epoch_loss`,
`accuracy`, `mean_reward_margin`, `vector_norm`, `learning_rate`,
`accumulation`, `batch`, `resumed_from`, `checkpoints`, `chat_template` and
`precision`. A run that takes no step is refused with `the run took no step, so
it has no loss to report`.

```bash
ster model fixture create --from docs/examples/model/fixture.json --output .build/bipo-fixture
ster tune bipo --model .build/bipo-fixture --pairs docs/examples/pairs.json --output calm.bipo.json \
  --layer 2 --strength 1 --beta 0.1 --epochs 2 --learning-rate 0.01 --accumulation 1 \
  --warmup-steps 0 --max-sequence 256 --batch-size 4 --seed 7
ster generate --model .build/bipo-fixture --vector calm.bipo.json --strength 2 --prompt "describe the sea ." \
  --max-new-tokens 16 --temperature 0 --seed 7
```

## Checkpoints and resuming a run

`--checkpoint <DIR> --checkpoint-every <STEPS>` and `--resume <DIR>` work as for
`ster tune sft` ([fine-tuning](../tuning/fine-tuning.md#checkpoints-and-resuming-a-run)):
the checkpoint holds the vector as `var/steering` with its two AdamW moments,
and its state the epoch's loss and accuracy totals. A resumed run replays the
shuffle and the direction drawn for every forward of the steps already taken,
so each later forward draws the direction the stopped run would have drawn, and
the run ends with the vector an uninterrupted run writes. The fingerprint is the
model and revision, a digest of the pair file (the active imported set and the
same file named with `--pairs` match), the chat-template choice, the precision,
every option above and the seed; a checkpoint of another run is refused naming
every field that differs.

BiPO has no request API operation and no Ster Desktop screen; the CLI is its
only surface.
