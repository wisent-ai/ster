# Unreleased: pair sets, the workspace and synthesis

Part of the [changelog](../../../CHANGELOG.md).

- `ster sources [--all]` lists the data other tools keep that Ster reads, one
  adapter each: example sets for `tune sft --examples` (IMP-STER-03 OpenAI,
  04 ShareGPT, 05 Alpaca) and preference rows for `pairs import --benchmark
  preference` (01 Hugging Face, 06 DPO, 19 Preferences). `ster sources
  discover <adapter> --path <file>` reads a file as that command would and says
  how many examples or pairs it gives, and how many rows give none, writing
  nothing; an adapter Ster does not have is refused
  (`ster.sources.sources-adapter-unknown`). `tests/sources/sources.sh`
  provokes it. The model checkpoints `--model` loads are listed too: a
  Transformers directory (12), a GGUF file beside its base model's config and
  tokenizer (13) and the Hugging Face hub cache (16). `discover` resolves a
  directory as `--model` does, saying how many weight files it has or why it
  is not that adapter's, and lists the Hub ids a hub cache directory keeps.
  Parquet or Arrow files, other steering libraries' vectors (repeng,
  steering-vectors, TransformerLens, nnsight, SAE Lens), MLX and PyTorch
  weights, PEFT and MLX LoRA adapters and wisent-evaluators' judgments (02,
  07–11, 14, 15, 17, 18, 20) are listed as sources no command reads, each with
  why and what to do; `discover` of one answers that row.

- Every refusal of the workspace and of a pair-set file is a declared cause
  (wisent-errors): it says what failed, why, the state observed and the fix,
  and exits `2` for a usage refusal, `69` when retrying unchanged can help and
  `1` otherwise. An invalid `--name`, or a file name that gives no id, now
  exits `2`.
- `ster workspace remove` deletes the copy before it changes the workspace: a
  copy that cannot be deleted leaves the set in place, where it used to leave
  a file no entry recorded that the next import of that id met as a conflict.
- A workspace write that fails says whether its temporary file was removed,
  and an import whose record cannot be written says it was undone.
- `ster pairs add`, `remove`, `edit`, `merge`, `inspect` and `import`, and
  `ster request pairs/save|merge|inspect|import`, refuse through declared
  causes listed on `docs/guide/errors/pairs.md`, `pairs-import.md` and
  `pairs-export.md`. Every flag refusal now exits `2` and comes before any
  set, workspace or export is read: an edit naming no change, a side given as
  empty text, fewer than two sets or a blank trait for `merge`, near-duplicate
  settings that do not fit the 64-bit fingerprint, and every row flag of
  `import`.
- `ster pairs inspect` refuses an `--unbalanced-ratio` below 1 or not a
  number, which used to count every pair, or none, as unbalanced.
- `ster pairs import` refuses row flags given to any benchmark but `choices`
  before it checks them, where it used to ask for the next missing row flag;
  it refuses `--examples` with any benchmark but `livecodebench`, which it
  used to ignore, and `--count 0`. A CSV without a column its benchmark needs
  is refused once, naming the columns it does have in their order, and an
  export no row of which becomes a pair says how many rows it read and why
  the first was skipped.
- A multiple-choice JSON list, LiveCodeBench's problems.json and
  question_examples.json whose JSON does not read are refused through the
  `ster.file` causes of their reason (not JSON, cut short, JSON of another
  form), naming the file's state and the line and column the reader stopped
  at, where `rows-not-a-list`, `problems-not-a-list` and `examples-not-json`
  joined those reasons. A JSON Lines row that is not JSON quotes the line and
  the column; one whose value runs past its line (a file printed with
  indentation) is `row-unclosed`, whose fix is `jq -c . <file>`.
- `tests/pairs/author.sh` replaces `tests/pairs/edit.sh` and covers `add`,
  `remove` and `edit`; `tests/pairs/merge.sh`, `inspect.sh` and `import.sh`
  provoke every declared refusal of those commands.
- `ster pairs synthesize` and `ster request pairs/synthesize` refuse through
  the causes on `docs/guide/errors/pairs-synthesize.md`. A blank `--trait`, a
  temperature at or below zero and near-duplicate settings that do not fit
  the fingerprint are refused with exit `2` before a local model loads, where
  the temperature and the settings used to be checked only after it had; a
  run that keeps no pair is refused with how each attempt was rejected
  instead of failing to write an empty set.
- Every hosted completion — `ster pairs synthesize --generator brama`,
  `ster request pairs/synthesize` and `ster decisions synthesize` — goes
  through `stado capabilities call inference --consumer ster --json`. The
  `BRAMA_URL`/`BRAMA_BEARER` route that sent completions straight to Brama is
  gone: Ster names no gateway and reads no bearer. The module is
  `ster::inference` (`Inference`, `pairs::Generator::Inference`), and its
  refusals are `ster.inference` on `docs/guide/errors/inference.md`: a blank
  route (exit 2, and `decisions synthesize` refuses it before it reads the
  schema), `stado` not on PATH, Stado breaking the call, no alternative
  answering (now exit 69, naming Stado's sentence), Stado refusing the
  request, an answer not in Stado's `--json` form, and a completion without an
  assistant message. Stado's own refusal is read from its `--json` envelope.
- Ster finds Stado in the order Stado states for every program that runs it
  (`stado --help`, "Where another program finds Stado"): the path in
  `STADO_BIN` when it is set, else `stado` on `PATH`, else
  `${STADO_BIN_DIR:-$HOME/.stado/bin}/stado`, an empty variable counting as
  unset, where it used to look on `PATH` only. `stado-absent` names every
  place it looked; a `STADO_BIN` naming no file is refused
  (`stado-bin-missing`), and a `stado` whose interpreter is missing is
  `stado-unlaunchable` where it was reported as no Stado at all; every start
  refusal names the place the `stado` was found through. Stado's refusals (`unanswered`,
  `stado-refused`) now carry the command line Ster ran, the directory, the
  exit status, Stado's cause id (`stado.capabilities.alternatives.none-answers`)
  and Stado's fix beside its whole sentence, where they carried the sentence
  alone; without an envelope, Stado's standard error is quoted as it is
  instead of a line picked by its `Error: ` prefix.
- `ster pairs import` skips a row under its own cause of
  `ster.pairs.import.rows` (21, from `question-blank` to
  `answer-text-unknown`), naming the row, the export, the column or JSON
  pointer and what it held, where it skipped it with a fixed phrase (`no
  question`, `answer is not one of the labels`). Each `report.skipped`
  entry carries `reason`, now the declared refusal, and `cause`. An export
  no row of which becomes a pair is refused with its first skipped row's
  cause and fix and the count per cause; `no-pairs` is left for an export
  that holds no row. A LiveCodeBench problem without a `question_id` is
  reported as `problem <n>`, where it was `(no question_id)`.
  `tests/pairs/import-rows.sh` provokes each cause.
- `ster pairs import --benchmark preference` writes a pair set from preference
  rows (IMP-STER-06, 01, 19): `prompt`, `chosen` and `rejected` as TRL's DPO
  trainer and the Hugging Face preference datasets publish them, as strings
  or in the conversational message form; the chosen answer is the positive
  side. A row without text on a side is skipped as
  `preference-side-missing`.
- `ster doctor --check import-sources` answers alone: nothing is outside Ster,
  since `--model` reads a checkpoint where it is; it lists the models the
  Hugging Face hub cache keeps (`HF_HUB_CACHE`, else `HF_HOME/hub`, else
  `~/.cache/huggingface/hub`) under `read_in_place`.
- `ster sources share yes | no` keeps, once, whether each `ster sources
  discover` reports `ster.sources.discovered` (adapter, whether the file is
  that source, its examples or pairs in buckets, a random subject) to Wisent
  Integrations with `STER_STADO_INTEGRATION_TOKEN`; `ster sources` asks with
  exactly that event until then, each discovery answers `reported`, and an
  answer that cannot be written is `ster.sources.sources-share-unkept`
  (`tests/sources/sources.sh`).
- JSON Lines is read by wisent-formats' one JSON Lines reader, which Zrodlo
  shares, in place of Ster's four copies: example sets for `tune sft
  --examples`, preference rows `ster sources discover` reads, multiple-choice
  rows (`ster.pairs.import.export`: `row-unclosed` when a value runs past its
  line, else `row-not-json`, with the line and column as before) and decision
  exports (`ster decisions import`, each line still read on its own so a line
  that does not read is one rejected row). The refusals and their values are
  unchanged.
