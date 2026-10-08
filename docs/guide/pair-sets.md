# Pair sets

The file every training and evaluation command reads, and the commands that
author it. The [README](../../README.md) links here from its command list.



`ster pairs` owns the file the training and evaluation commands read. It has
five subcommands:

```text
ster pairs inspect --pairs <FILE> --dedupe-bits <BITS> --dedupe-bands <BANDS>
                   --refusal-threshold <SCORE> --unbalanced-ratio <RATIO>
ster pairs add --pairs <FILE> --positive <TEXT> --negative <TEXT> [--trait <NAME>]
ster pairs remove --pairs <FILE> --index <N>
ster pairs edit --pairs <FILE> --index <N> [--positive <TEXT>] [--negative <TEXT>] [--trait <NAME>]
ster pairs synthesize --trait <TRAIT_DESCRIPTION> --count <COUNT> --output <OUTPUT>
                      --generator local|brama [--generator-model <ROUTE>]
                      [--model <MODEL>] [--revision <REVISION>] [--device <DEVICE>]
                      [--chat-template auto|off] [--precision f32|f16|bf16]
                      [--trait-name <NAME>] [--opposite <TEXT>]
                      --retry-multiplier <N> --dedupe-bits <BITS>
                      --dedupe-bands <BANDS> --refusal-threshold <SCORE>
                      --max-new-tokens <N> --temperature <T>
                      --top-p <P> --seed <SEED>
ster pairs import --benchmark truthfulqa|dna|livecodebench|bipo|choices --source <FILE> --output <FILE>
                  [--examples <FILE>] [--count <N>] --seed <SEED> [--trait <NAME>]
                  [--question <POINTER> --choices <POINTER> --answer <POINTER>
                   --answer-form index|label|text [--labels <POINTER>]]
```

Every number is the caller's: Ster assumes no deduplication distance, band
count, refusal threshold, length ratio, attempt budget, token budget,
temperature, nucleus mass or seed, so leaving one out is a usage error that
names the missing flag (`error: the following required arguments were not
provided: --refusal-threshold <REFUSAL_THRESHOLD>`, exit 2). The same fields are
required in the `pairs/inspect`, `pairs/synthesize` and `pairs/import` request
bodies, where a missing one is refused as `missing field \`refusalThreshold\``.

Each subcommand prints a pretty JSON document on stdout, as the other commands
do, and each write leaves a pretty JSON pair set with a trailing newline. `add`
creates the file, and its parent directory, when it does not exist, and
`--trait` sets or replaces the trait name on the file. `remove` takes a
zero-based index and refuses one outside the set with `pair index {i} is outside the set's 0..{n-1} range`.
It also refuses to remove the last pair, because a set is validated before it is
saved: the file is left untouched and the refusal is `pair set {path} contains no pairs`.
`edit` takes the same index and changes only the sides given, or with
`--trait` the trait name; an edit with none of them is refused with `pairs
edit changes nothing without --positive, --negative or --trait`, and an index
outside the set with the same refusal `remove` gives.

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
name with `pairs merge needs the trait name the merged set is fitted for`.
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
`question_examples.json` (beside the source unless `--examples` names it). The BiPO
paper's behaviour CSVs (`question,matching,not_matching`, e.g. its `power-seeking/train.csv`)
pair each question's matching answer (positive) with the other (negative), the set
`ster tune bipo` learns a vector from. A row
that cannot become a pair is listed under `report.skipped` with its row and the
reason (`no question`, `no best answer`, `no incorrect answer`, `no
question_content`, `no good and bad example code`, `no matching answer`, `no
not_matching answer`); an unreadable file, a CSV
record whose field count differs from its header, or `--count` above the pairs
the export yields refuses the import and writes nothing. The exports themselves
live in the `wisent-benchmark` repository under `benchmarks/`.

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
other choices drawn with `--seed`. Ster guesses none of it: a row with no
question, choices that are not all non-empty text, no answer, or an answer
that does not resolve is skipped with `no question`, `choices are not a list
of non-empty text`, `no answer`, `answer is not a whole-number index`,
`answer is not a label`, `answer is not one of the labels`, `labels are not a
list of non-empty text`, `labels and choices differ in length`, `answer is not
text`, `answer is not one of the choices`, `answer index is outside the
choices` or `no incorrect choice`, and its row is its line (JSON Lines) or its
position from one (a list). Every row is read unless `--count` keeps fewer.

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
any other form. A file that is neither a JSON list nor JSON Lines refuses the
import and names the line that is not JSON. `tests/pairs/import-choices.mjs`
runs the built binary over ARC- and HellaSwag-shaped rows and every refusal
above, and keeps each command and its result in `report.json`.

Nothing else in the suite produces pair sets. Every other set is one you wrote,
one `ster pairs add` built, one `ster pairs synthesize` generated, or one
`ster pairs import` read from a benchmark.

`ster pairs inspect` loads no model; every judgement it makes is textual. For
the set it reports `trait_name`, `pair_count`, `duplicate_count`,
`refusal_count`, `unbalanced_count`, `unbalanced_ratio`, and `diversity`. For
each pair it reports `index`, both texts, `positive_chars` and `negative_chars`,
`positive_words` and `negative_words`, `duplicate`, `positive_refusal` and
`negative_refusal`, and `length_ratio`.

Duplicates are found by SimHash over the normalized positive and negative text,
with 64-bit fingerprints built from BLAKE2b feature hashes and bucketed by
banded LSH over `--dedupe-bands` bands. Two pairs are near-duplicates when their
fingerprints differ in at most `--dedupe-bits` bits. The `duplicate` field is
`{"kind":"exact","of":N}` or `{"kind":"near","of":N,"distance":B}`, naming the
earlier pair. The first occurrence wins, so pair order decides which of two
near-identical pairs is flagged: the later one.

Refusals are scored across ten weighted families — `ai_disclaimer`, `policy`,
`apology_hedge`, `unable`, `cannot_action`, `prefer_rather`, `decline_refuse`,
`no_support`, `no_ability`, and `refusal_word` — and a side is flagged at or
above `--refusal-threshold`. A flag carries the score, the family, and the text
that matched, for example
`{"score":0.9,"family":"ai_disclaimer","snippet":"As an AI language model"}`. A
refusal is a useless example because it differs from the other side along the
refusal axis rather than along the trait axis.

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
   it. An empty answer falls back to `neutral and plain`.
2. Each attempt generates a question, then an answer in the trait's voice, then
   an answer in the opposite's voice. Each side is stored as
   `Question: {q}\nAnswer: {a}`, the shape the example above already uses, which
   keeps the two sides matched on everything but the trait.
3. A negative that reads as a refusal is asked again exactly once with a repair
   instruction; if it still refuses, the pair is dropped. A refusing positive is
   dropped immediately, because the trait itself is what the model declined and
   re-asking would refuse again.
4. A pair within `--dedupe-bits` of a pair already kept is dropped.
5. The attempt budget is `--count` times `--retry-multiplier`, and every attempt
   prints one `synthesizing pair 3/20 (attempt 7)` progress line on stderr.

On the local route the seed advances by one on every model call. Ster builds a
fresh sampler per call, so a fixed seed would return one identical continuation
for the whole run and the set would collapse to a single pair; advancing from
`--seed` keeps the run reproducible from the one seed the caller supplied. A
temperature of zero is refused before the first generation on either route:
`synthesis requires a temperature above zero; argmax generation repeats a single prompt`.

`--generator` chooses who writes the text, and is required: `local` uses the same
local open-weight runtime every other Ster command uses, and `brama` sends the
generation to a hosted model through the Brama gateway. `--model`,
`--revision` and `--device` belong to the local route and are not read by the
hosted one, which loads no weights, resolves no device, and downloads nothing.
The local route refuses a missing model with `pairs synthesize with --generator local requires --model`,
the hosted route refuses a missing route with `pairs synthesize with --generator brama requires --generator-model`,
and anything else is `unknown generator "cloud"; expected local or brama`.

`--chat-template` belongs to the local route for the same reason `--model`
does. `auto`, the default, asks the local generator through the model's own chat
template when the checkpoint publishes one, and `off` asks it as raw text.
Synthesis is the first step of the funnel and everything downstream inherits
what it writes, which makes this the one place the mistake is expensive twice:
an instruct checkpoint addressed without its markers reads a request for pair
text as a document to continue, and answers with meta-instructional debris —
`Step 3: Make sure your emojis are visually appealing` — instead of the answer
that was asked for. `--generator brama` ignores it, because the gateway request
carries messages with roles rather than a rendered string: a chat API is
already a chat API.

The hosted route reads Brama's own documented client variables: `BRAMA_URL`,
the gateway base, and `BRAMA_BEARER`, the caller's bearer. Both are required;
there is no built-in gateway. Ster reads both from its own environment
and never reads a vault itself. Nothing puts them there for you: there is no
launcher script in this repository, no wrapper on the fleet, and no service
unit that exports them, and Ster Desktop runs `ster request` with the
environment it inherited without setting a variable of its own. You export
both before the run. An unset base is refused with
`BRAMA_URL is unset or empty; export the Brama gateway base Ster should call`,
an empty bearer with
`BRAMA_BEARER is unset or empty; export Ster's own Brama bearer before running this command`,
and a base that is neither https nor explicit loopback with
`BRAMA_URL must be an https:// base or an explicit http:// loopback address, because Brama answers plain http elsewhere with 426 secure_transport_required`.

The bearer to export is Ster's own client identity, never another product's.
Ster is a declared consumer of `brama` in Stado's service directory —
`stado resolver resolve brama --consumer ster` answers
`stado://service/brama capabilities=model-routing` — and the bearer itself is a
Skarbiec grant for the consumer `ster` whose capability is `call:brama#<route>`.
Brama resolves a bearer its own start did not preload by introspecting Skarbiec
for exactly that grant, so the route the grant names is the only route
`--generator-model` may then use.

Which Skarbiec matters, and this is the part that is easy to get wrong: the
gateway introspects the vault on its **own** host, not yours. A grant minted in
a workstation's vault authenticates against that workstation and is answered
`401 unauthenticated` by the gateway, which never consults it. Mint it where the
gateway will look, through Stado's managed channel rather than over ssh:

```bash
export BRAMA_BEARER="$(stado host vault-token-mint <gateway-host> ster \
  --capabilities 'call:brama#openai/gpt-5-mini' --audience ster --raw-token)"
```

`--raw-token` prints the bearer and nothing else, for a direct pipe into a
variable or a secret store; Stado never writes it to disk or to a command line,
and neither should you. Drop `--raw-token` for non-secret metadata only. Read
the value at the moment of the call, keep it in a process-only variable, and let
it die with the process: a bearer belongs in neither a file nor an argv.

`--generator-model` may name exactly four things: a declared alias, such as
Wisent's own chat alias `wisent-backend/chat/primary` or its sibling
`wisent-backend/chat/fallback`; the delegation alias `best`; a canonical
`provider/model` route, such as `anthropic/claude-3-5-haiku-latest`; or a
selector, `any` or `task:<name>`. `best` and the selectors additionally require
an agent-signed request, which Ster does not construct — it sends a bearer and
nothing more — so a Ster run uses a declared alias or a canonical route.
Anything outside that vocabulary is refused with
`the generator model must be a Brama alias, a canonical provider/model route, or a selector`.

The request to the gateway carries exactly `model`, `messages` with one user
message, `max_tokens` and `temperature`, because Brama refuses unknown fields
by name. Neither `--seed` nor `--top-p` therefore travels, and a hosted run is
not seed-reproducible: the provider owns its sampler, and the running
deduplicator is what suppresses repeated draws. Ster duplicates Brama's two
documented bounds locally to save a round trip per pair, in the gateway's own
words: `max_tokens must be between one and 32768` and
`temperature must be finite and between zero and 2`. A gateway refusal is
surfaced in Brama's own words, as `brama refused the completion: 401 unauthorized`
— the status followed by the `message` from Brama's `{"error":{...}}` envelope.
A body without that envelope becomes
`brama refused the completion with {status} and a body that is not its error envelope: {excerpt}`.

The run reports `generator`, which records which of the two wrote the set as
`local:<model id>` or `brama:<route>`, then `trait_name`,
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
loop and returns the written path and the report. The synthesize operation takes `generator`,
defaulting to `"local"`, and `generatorModel`, takes `chatTemplate` and
`precision` on the local route, and requires `model` only on that route; it
refuses a hosted run without a route with
`pairs synthesize with the brama generator requires generatorModel`, an
unrecognized value with `unknown generator; expected local or brama`, and a
local run without a model with `pairs synthesize requires a model`. This is how
Ster Desktop offers pair authoring, inspection, and synthesis beside its six
workflows.

