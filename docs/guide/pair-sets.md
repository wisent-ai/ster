# Pair sets

The file every training and evaluation command reads, and the commands that
author it. The [README](../../README.md) links here from its command list.
How a finished set is taken into the workspace, named and recognised again is
on [pair sets in the workspace](pair-sets/workspace.md).



`ster pairs` owns the file the training and evaluation commands read. It has
five subcommands:

```text
ster pairs inspect --pairs <FILE> --dedupe-bits <BITS> --dedupe-bands <BANDS>
                   --unbalanced-ratio <RATIO>
ster pairs add --pairs <FILE> --positive <TEXT> --negative <TEXT> [--trait <NAME>]
ster pairs remove --pairs <FILE> --index <N>
ster pairs edit --pairs <FILE> --index <N> [--positive <TEXT>] [--negative <TEXT>] [--trait <NAME>]
ster pairs synthesize --trait <TRAIT_DESCRIPTION> --count <COUNT> --output <OUTPUT>
                      --generator local|brama [--generator-model <ROUTE>]
                      [--model <MODEL>] [--revision <REVISION>] [--device <DEVICE>]
                      [--chat-template auto|off] [--precision f32|f16|bf16]
                      [--trait-name <NAME>] [--opposite <TEXT>]
                      --retry-multiplier <N> --dedupe-bits <BITS>
                      --dedupe-bands <BANDS>
                      --max-new-tokens <N> --temperature <T>
                      --top-p <P> --seed <SEED>
ster pairs import --benchmark truthfulqa|dna|livecodebench|bipo|choices --source <FILE> --output <FILE>
                  [--examples <FILE>] [--count <N>] --seed <SEED> [--trait <NAME>]
                  [--question <POINTER> --choices <POINTER> --answer <POINTER>
                   --answer-form index|label|text [--labels <POINTER>]]
```

Every number is the caller's: Ster assumes no deduplication distance, band
count, length ratio, attempt budget, token budget, temperature, nucleus mass
or seed, so leaving one out is a usage error that names the missing flag
(`error: the following required arguments were not provided:
--unbalanced-ratio <UNBALANCED_RATIO>`, exit 2). The same fields are
required in the `pairs/inspect`, `pairs/synthesize` and `pairs/import` request
bodies, where a missing one is refused as `missing field \`unbalancedRatio\``.
No refusal score is asked for: whether a side refused is the writing model's
judgement during synthesis, and `--refusal-threshold` is refused as an
unexpected argument.

Each subcommand prints a pretty JSON document on stdout, as the other commands
do, and each write leaves a pretty JSON pair set with a trailing newline. `add`
creates the file, and its parent directory, when it does not exist, and
`--trait` sets or replaces the trait name on the file. `remove` takes a
zero-based index and refuses one outside the set (exit 1) with `pair index {i} is outside the set: {path} holds {n} pair(s), indexed from 0`
and the indexes that do exist.
It also refuses to remove the last pair, because a set is validated before it is
saved: the file is left untouched and the refusal is `pair set {path} contains no pairs`.
`edit` takes the same index and changes only the sides given, or with
`--trait` the trait name; an edit with none of them is refused (exit 2) with
`pairs edit changes nothing without --positive, --negative or --trait`, and an
index outside the set with the same refusal `remove` gives. A side given to
`add` or `edit` as empty text is refused (exit 2) before the set is read.
Every refusal, its exit status and the test that provokes it are listed on
[How a pair set fails](errors/pairs.md).

`ster pairs merge` writes one set out of several, for a direction fitted
across all of them — the unified direction wisent's `train-unified-goodness`
fitted across benchmarks:

```text
ster pairs merge --pairs <SET> --pairs <SET> [--pairs <SET>...] --trait <NAME> --output <FILE>
```

Every pair of every set is written in the order the sets were named, under
the trait name given, and the answer lists each source with its own trait
name and pair count. Each source passes the validation every pair set
passes. Fewer than two sets are refused with
`pairs merge needs at least two pair sets and got <N>`, and an empty trait
name with `pairs merge needs the trait name the merged set is fitted for`, both
with exit 2 and before any set is read.
Through `ster request pairs/merge` the body is
`{"sources": [...], "traitName": "…", "output": "…"}`.

A pair set can also arrive from another product, and exactly one produces them:
Preferences, Wisent's pairwise voting tool. Every choice in one of its text
categories is already a contrastive pair, so its exporter writes a document
`--pairs` accepts unchanged, and `ster-pairs` is the only format it will emit:

```bash
preferences export --arena wisent --category tagline --format ster-pairs > pairs.json
ster vector train --model meta-llama/Llama-3.2-1B --pairs pairs.json --output taste.ster.json
```

`ster pairs import` reads a published benchmark export into a set:

```text
ster pairs import --benchmark truthfulqa|dna|livecodebench|bipo|choices --source <FILE> --output <FILE>
                  [--examples <FILE>] [--count <N>] --seed <SEED> [--trait <NAME>]
                  [--question <POINTER> --choices <POINTER> --answer <POINTER>
                   --answer-form index|label|text [--labels <POINTER>]]
```

Every row becomes one pair written as `pairs synthesize` writes one,
`Question: …` then `Answer: …` on both sides. TruthfulQA (`TruthfulQA_en.csv`)
pairs the row's Best Answer with one of its Incorrect Answers, picked with
`--seed`. Do-Not-Answer (`DnA_en.csv`) carries questions only; each is paired
as "I am happy to help!" (positive) against "I am sorry, I cannot comply with
this request" (negative). LiveCodeBench pairs each problem's
`question_content` from `problems.json` with the good and bad example code in
the `question_examples.json` that `--examples` names; without `--examples` the
import refuses. The BiPO
paper's behaviour CSVs (`question,matching,not_matching`, e.g. its `power-seeking/train.csv`)
pair each question's matching answer (positive) with the other (negative), the set
`ster tune bipo` learns a vector from. A row
that cannot become a pair is listed under `report.skipped` with its `row`
(its record number, or `problem <n>` for a LiveCodeBench problem without a
`question_id`), its `reason` (the declared refusal, naming the column or
field and what it held) and its `cause` (`functionality`, `id`, `docs`,
`code`, `exit`, `usage`, `retryable`, `remedy`) of `ster.pairs.import.rows`: `question-blank`,
`answer-blank` (naming the column and the side), `incorrect-none`,
`problem-id-missing`, `problem-content-blank`, `example-missing` and
`example-code-missing`. An export no row of which becomes a pair is refused
with its first skipped row's cause and fix, followed by how many rows were
skipped under each cause; an unreadable file, a CSV that is not the benchmark's
(a missing column, a record whose field count differs from its header, a
quoted field never closed), an export that holds no row (`no-pairs`), or
`--count` above the pairs the export yields refuses the import and writes
nothing; `--count 0`, and `--examples` with any benchmark but
`livecodebench`, are refused before the export is read. Every refusal is on
[How importing a benchmark export fails](errors/pairs-import.md) and
[How reading a benchmark export fails](errors/pairs-export.md). The exports
themselves live in the `wisent-benchmark` repository under `benchmarks/`.

`--benchmark choices` reads the multiple-choice rows of any dataset — ARC,
HellaSwag, MMLU and the other tasks the retired `wisent-extractors` Python
package had one extractor each for — exported as JSON Lines (one object per
line) or as one JSON list. The dataset's own schema says where a row keeps its
parts, so the caller names them as JSON pointers: `--question` the question
text, `--choices` the list of choices, `--answer` the correct answer, and
`--answer-form` how that answer names its choice: `index` (its position from
zero, as a number or as text holding one), `label` (one of the row's labels,
found through `--labels`, which must line up with the choices) or `text` (the
choice itself). The pair is the correct choice (positive) against one of the
other choices drawn with `--seed`. Ster guesses none of it: a row is skipped
under the cause that names what it lacks, with the pointer and what it held:
`choice-question-missing`, `choices-not-list`, `choice-not-text` (naming the
choice), `choices-few` (fewer than two), `answer-missing`, `answer-not-index`
(with the parse error when the answer is text), `answer-index-outside`,
`labels-not-list`, `label-not-text`, `labels-misaligned`,
`answer-label-not-text`, `answer-label-unknown`, `answer-text-not-text` or
`answer-text-unknown`. Its row is its line (JSON Lines) or its position from
one (a list); every cause is on [How importing a benchmark export
fails](errors/pairs-import.md). Every row is read unless `--count` keeps fewer.

```bash
# ARC-Easy: {"question": …, "choices": {"text": [...], "label": ["A", …]}, "answerKey": "B"}
ster pairs import --benchmark choices --source arc_easy.jsonl --seed <SEED> --output arc.pairs.json \
  --question /question --choices /choices/text --answer /answerKey --answer-form label --labels /choices/label
# HellaSwag: {"ctx": …, "endings": [...], "label": "2"}
ster pairs import --benchmark choices --source hellaswag.jsonl --seed <SEED> --output hellaswag.pairs.json \
  --question /ctx --choices /endings --answer /label --answer-form index
```

The row flags are refused on any other benchmark (`--question, --choices,
--answer, --answer-form and --labels apply only to --benchmark choices`), and
`choices` without them is refused (`--benchmark choices needs --question,
--choices, --answer and --answer-form: where each part of a row sits`), as are
a missing one (`multiple-choice rows need --answer: …`), a pointer that does not
start with `/`, `--answer-form label` without `--labels`, and `--labels` with
any other form; each exits 2 before the export is read. A file that is neither
a JSON list nor JSON Lines refuses the import and names the line that is not
JSON. `tests/pairs/import-choices.mjs` runs the built binary over ARC- and
HellaSwag-shaped rows, and `tests/pairs/import.sh` provokes every refusal
above; both keep each command and its result in their report.

Nothing else in the suite produces pair sets. Every other set is one you wrote,
one `ster pairs add` built, one `ster pairs synthesize` generated, or one
`ster pairs import` read from a benchmark.

`ster pairs inspect` loads no model; every judgement it makes is textual. For
the set it reports `trait_name`, `pair_count`, `duplicate_count`,
`unbalanced_count`, `unbalanced_ratio`, and `diversity`. For
each pair it reports `index`, both texts, `positive_chars` and `negative_chars`,
`positive_words` and `negative_words`, `duplicate`, and `length_ratio`. It
makes no refusal finding: whether a side declines its role is a judgement a
model makes, and inspection loads none. A `--dedupe-bands` that does not
divide 64, a `--dedupe-bits` above 64, and an `--unbalanced-ratio` below 1 or
not a number (which would count every pair, or none, as unbalanced) are
refused with exit 2 before the workspace or the set is read; the request
refuses the same fields with status 2.

Duplicates are found by SimHash over the normalized positive and negative text,
with 64-bit fingerprints built from BLAKE2b feature hashes and bucketed by
banded LSH over `--dedupe-bands` bands. Two pairs are near-duplicates when their
fingerprints differ in at most `--dedupe-bits` bits. The `duplicate` field is
`{"kind":"exact","of":N}` or `{"kind":"near","of":N,"distance":B}`, naming the
earlier pair. The first occurrence wins, so pair order decides which of two
near-identical pairs is flagged: the later one.

`length_ratio` is the longer side over the shorter side in characters, and
`unbalanced_count` counts the pairs above `--unbalanced-ratio`, which the report
echoes as `unbalanced_ratio`. A pair whose sides differ that much in length
teaches length instead of the trait, which is the confound to remove before
training rather than to discover afterwards in a flattering margin.

`diversity` reports `unique_unigrams`, `unique_bigrams`, `avg_jaccard`,
`mean_simhash_hamming`, and `min_simhash_hamming`. Inspection measures them over
every positive side; the pairwise passes compare every pair of texts, so a larger
set takes longer to report on rather than being sampled.

`ster pairs synthesize` builds a set from a trait description, in this order:

1. The opposite trait is derived with one generation, unless `--opposite` states
   it. An empty answer refuses the run (exit 1) with
   `<generator> returned no opposite trait for "<trait>"` and the fix
   `state the opposite trait with --opposite`,
   because every negative would otherwise answer as a trait nobody stated.
2. Each attempt generates a question, then an answer in the trait's voice, then
   an answer in the opposite's voice. Each side is stored as
   `Question: {q}\nAnswer: {a}`, the shape the example above already uses, which
   keeps the two sides matched on everything but the trait.
3. Each answer is judged by the same generator that wrote it: it is asked
   whether the answer declined to answer as its personality, and must reply
   with exactly `{"refuses": true}` or `{"refuses": false}`. Any other reply
   stops the run: `verdict-malformed` for a reply that is not JSON,
   `verdict-shape` for JSON of another form (another key, a value not `true`/`false`),
   because a verdict nobody can read is not a verdict. No list of refusal
   phrasings is consulted, so a refusal worded any way, in any language, is
   judged the same. A refusal is a useless example because it differs from the
   other side along the refusal axis rather than along the trait axis. A
   negative judged a refusal is asked again exactly once with a repair
   instruction, and the replacement is judged in turn; if it still refuses,
   the pair is dropped. A refusing positive is dropped immediately, because
   the trait itself is what the model declined and re-asking would refuse
   again.
4. A pair within `--dedupe-bits` of a pair already kept is dropped.
5. The attempt budget is `--count` times `--retry-multiplier`, and every attempt
   prints one `synthesizing pair 3/20 (attempt 7)` progress line on stderr. A
   run whose every attempt was rejected writes nothing and is refused (exit 1)
   with `none of <n> attempts kept a pair` and how many were rejected as
   empty, as refusals and as duplicates.

On the local route the seed advances by one on every model call. Ster builds a
fresh sampler per call, so a fixed seed would return one identical continuation
for the whole run and the set would collapse to a single pair; advancing from
`--seed` keeps the run reproducible from the one seed the caller supplied. A
temperature of zero, a blank `--trait` and near-duplicate settings that do not
fit the fingerprint are refused (exit 2) before any model loads or any route is
asked: `synthesis requires a temperature above zero; argmax generation repeats a single prompt`.

`--generator` chooses who writes the text, and is required: `local` uses the same
local open-weight runtime every other Ster command uses, and `brama` hands the
generation to Stado's inference alternatives, Brama among them. `--model`,
`--revision` and `--device` belong to the local route and are not read by the
hosted one, which loads no weights, resolves no device, and downloads nothing.
The local route refuses a missing model with `pairs synthesize with --generator local requires --model`,
the hosted route refuses a missing route with `pairs synthesize with --generator brama requires --generator-model`,
and anything else is refused by the argument parser as
`error: invalid value 'cloud' for '--generator <GENERATOR>'` with the two
possible values, exit 2.

`--chat-template` belongs to the local route for the same reason `--model`
does. `auto`, the default, asks the local generator through the model's own chat
template when the checkpoint publishes one, and `off` asks it as raw text.
Synthesis is the first step of the funnel and everything downstream inherits
what it writes, which makes this the one place the mistake is expensive twice:
an instruct checkpoint addressed without its markers reads a request for pair
text as a document to continue, and answers with meta-instructional debris —
`Step 3: Make sure your emojis are visually appealing` — instead of the answer
that was asked for. `--generator brama` ignores it, because the inference
request carries messages with roles rather than a rendered string: a chat API
is already a chat API.

The hosted route — handed to Stado's inference alternatives, the routes
`--generator-model` may name, and how a refusal reads — is described in
[synthesis through Stado's inference alternatives](pair-sets/hosted-generator.md).

The run reports `generator`, which records which of the two wrote the set as
`local:<model id>` or `<alternative>:<route>` (the alternative Stado says
answered, e.g. `brama:<route>`), then `trait_name`,
`trait_description`, `opposite`, `requested`, `attempts`, `kept`,
`rejected_empty`, `rejected_refusals`, `rejected_duplicates`,
`refusal_retries`, and `diversity`, so a short set is
explained by the counts rather than guessed at.

The same three operations are operations of `ster request`, which reads the
request body as JSON on stdin and prints NDJSON events like every other
operation ([desktop requests](desktop-requests.md)).
`ster request pairs/inspect` returns the inspection document for a path,
`ster request pairs/save` writes a set from `traitName` and `entries` and
returns the path and pair count, and `ster request pairs/synthesize` runs the
loop and returns the written path and the report. The synthesize operation requires `generator`,
`local` or `brama`, takes `generatorModel`, takes `chatTemplate` and
`precision` on the local route, and requires `model` only on that route; it
refuses (status 2) a request without a generator or with any other value with
`pairs synthesize writes with local or brama, and "<value>" is neither`, a
hosted run without a route with
`pairs synthesize with --generator brama requires --generator-model`, a local
run without a model with
`pairs synthesize with --generator local requires --model`, and a request
without `count` with `pairs synthesize requires count, a pair count above zero`.
Every refusal is on [How writing a pair set with a model fails](errors/pairs-synthesize.md).
This is how Ster Desktop offers pair authoring, inspection, and synthesis
beside its six workflows.

