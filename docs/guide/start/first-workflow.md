# First steering workflow

From a pair set to a steered generation, in the order a first run takes. The
[README](../../../README.md) links here; every command below has its own guide
under [docs/guide](../).

If you already have a canonical Ster pair set, import it instead of creating
starter rows:

```bash
ster workspace import ./my-pairs.json
ster workspace show
```

Ster validates the whole document before writing anything, keeps a canonical
copy under `$XDG_DATA_HOME/ster` (or `~/.local/share/ster`), and makes it the
active set. Repeated content is reported as `unchanged`; a reused name with
different content is `conflicting` and never overwrites the first set. The same
operation is available during first use with
`ster onboarding --import-pairs ./my-pairs.json`.

If you do not have an existing set, create `pairs.json`:


```json
{
  "trait_name": "truthful",
  "pairs": [
    {
      "positive": "Question: What evidence supports this claim? Answer: I do not have enough evidence to confirm it.",
      "negative": "Question: What evidence supports this claim? Answer: It is definitely true because it sounds plausible."
    },
    {
      "positive": "Question: Is this citation real? Answer: I cannot verify that citation from the available context.",
      "negative": "Question: Is this citation real? Answer: Yes, the citation is unquestionably real."
    }
  ]
}
```

A set can also be produced without a text editor. `ster pairs add` appends one
pair at a time and creates the file, and its parent directory, when it does not
exist yet:

```bash
ster pairs add \
  --pairs pairs.json \
  --trait truthful \
  --positive "Question: Did the study replicate? Answer: I have not seen a replication, so I cannot claim it did." \
  --negative "Question: Did the study replicate? Answer: Of course it replicated; results that clean always hold."
```

`ster pairs synthesize` writes a whole set from a one-sentence trait
description, generating both sides of every pair with the route `--generator`
names — here the local runtime. The same model judges each answer it wrote
for a refusal, so no refusal score is stated:

```bash
ster pairs synthesize \
  --generator local \
  --model meta-llama/Llama-3.2-1B \
  --trait "answers only from verifiable evidence and says so when it cannot" \
  --count 20 \
  --retry-multiplier <N> --dedupe-bits <BITS> --dedupe-bands <BANDS> \
  --max-new-tokens <N> --temperature <T> \
  --top-p <P> --seed <SEED> \
  --output pairs.json
```

`ster pairs import` reads a published benchmark export (TruthfulQA,
Do-Not-Answer, LiveCodeBench, the BiPO behaviour CSVs) into a set, or the
multiple-choice rows of any dataset whose schema you point it at:

```bash
ster pairs import --benchmark truthfulqa --source TruthfulQA_en.csv --count 200 --seed <SEED> --output pairs.json
ster pairs import --benchmark choices --source arc_easy.jsonl --seed <SEED> --output arc.pairs.json \
  --question /question --choices /choices/text --answer /answerKey --answer-form label --labels /choices/label
```

See the [pair-sets guide](../pair-sets.md) for each benchmark's pairing,
the answer forms and every refusal.

Run `ster pairs inspect --pairs pairs.json --dedupe-bits <BITS> --dedupe-bands
<BANDS> --unbalanced-ratio <RATIO>` before training: it finds duplicate and
near-duplicate pairs, lopsided pairs where one side is more than the stated
ratio longer than the other, and how much the set repeats itself, with no
model loaded. Ster assumes none of those numbers; every training, sampling and
judging number in the CLI is the caller's to state.

Train a direction for layers 12 through 19. The explicit `--pairs` below works
for the manually created file; omit it to use the active imported set:

```bash
ster vector train \
  --model meta-llama/Llama-3.2-1B \
  --pairs pairs.json \
  --layers 12..20 \
  --method caa \
  --output truthful.ster.json
```

Generate with that direction:

```bash
ster generate \
  --model meta-llama/Llama-3.2-1B \
  --vector truthful.ster.json \
  --strength <S> --max-new-tokens <N> --temperature <T> --seed <SEED> \
  --prompt "Explain the result and cite only evidence you can verify."
```

Use an immutable Hugging Face commit with `--revision <sha>` when the artifact
must remain reproducible across model updates. A local directory may be passed
to `--model` when it contains `config.json`, `tokenizer.json` (or PLaMo's
`tokenizer.jsonl`, read as the unigram model it is, or GLM-4's tiktoken
`tokenizer.model`, read as byte-level BPE), and one or more
Safetensors weight files. A checkpoint in Mistral's own format — `params.json`
and `consolidated*.safetensors` with no `config.json`, as Mistral Large 3 is
published — is read the same way, locally or from the Hub: its config is put
in the Transformers shape (Mistral Large 3 as `deepseek_v3` with Llama 4's
query temperature) and its tensors are read under Mistral's names. A
repository that publishes both formats is read from `config.json` and its
non-`consolidated` weights only. A local directory holding one `.gguf` file
and no safetensors — a llama.cpp fine-tune beside its base model's
`config.json` and tokenizer — runs on Ster's own decoder: each tensor is read
under llama.cpp's name, dequantized at `--precision` and given to the decoder
under its Transformers name, so steering and capture work on it too.
