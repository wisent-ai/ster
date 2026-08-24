# Runnable offline examples

These files are the exact inputs used for the recorded CPU walkthroughs. They
exercise the real Llama/Safetensors path without a download or GPU. The model
has seeded random weights, so generated text is deterministic but meaningless.

## Files

- [`make-toy-model.py`](make-toy-model.py) — stdlib-only generator for a tiny
  four-layer, hidden-width-64 Llama-shaped checkpoint with tokenizer and F32
  Safetensors weights.
- [`pairs.json`](pairs.json) — the eight calm/stormy contrastive pairs used by
  train, optimize, evaluate, and serve.
- [`prompts.json`](prompts.json) — the two prompts used by extract.

## Run the CLI sequence

From a checkout with `target/release/ster` already built:

```bash
WORK=$(mktemp -d)
cd "$WORK"
python3 <repo>/docs/examples/make-toy-model.py toy-llama
cp <repo>/docs/examples/pairs.json .
cp <repo>/docs/examples/prompts.json .
STER=<repo>/target/release/ster

$STER train --model toy-llama --pairs pairs.json --layers 1..3 \
  --method caa --output calm.ster.json
$STER optimize --model toy-llama --pairs pairs.json --layers all \
  --output calm.best.ster.json
$STER evaluate --model toy-llama --pairs pairs.json --vector calm.ster.json
$STER generate --model toy-llama --prompt "describe the evening lake ." \
  --max-new-tokens 12
$STER generate --model toy-llama --vector calm.ster.json --strength 1.0 \
  --prompt "describe the evening lake ." --max-new-tokens 12
$STER extract --model toy-llama --input prompts.json --layers 0,3 \
  --output activations.json
$STER inspect calm.best.ster.json
```

Recorded output and interpretation:
[train/select/steer walkthrough](../walkthrough-steering.md).

## Run the serve sequence

Start the server from the work directory so relative input/output paths resolve
there:

```bash
cd "$WORK"
$STER serve --port 0
```

Copy the printed port into `P`, then:

```bash
curl -s "http://127.0.0.1:$P/v1/health"
curl -s -X POST "http://127.0.0.1:$P/v1/train" \
  -H 'content-type: application/json' \
  -d '{"model":"toy-llama","pairs":"pairs.json","output":"served.ster.json","layers":"2","method":"caa"}'
curl -s -X POST "http://127.0.0.1:$P/v1/generate" \
  -H 'content-type: application/json' \
  -d '{"model":"toy-llama","prompt":"describe the evening lake .","vector":"served.ster.json","strength":1.0,"maxNewTokens":12}'
```

Recorded NDJSON: [serve walkthrough](../walkthrough-serve.md).

## Replacing the toy model

For useful representation work, replace every `--model toy-llama` with one
existing local Llama-family Safetensors directory or a Hub model id. Use an
immutable `--revision` for Hub reproducibility, design pair columns to differ
only in the trait, keep a separate evaluation set, and calibrate strength on
baseline-versus-steered outputs. The toy artifacts are bound to `toy-llama`
and hidden width 64; they must not be applied to a real checkpoint.
