# Unreleased: typed decisions and calibration

Part of the [changelog](../../../CHANGELOG.md).

- `ster decisions fetch`, `import`, `synthesize`, `split` and `benchmark`
  refuse through declared causes (`ster.decisions.*`,
  `docs/guide/errors/decide-pipeline.md`), where 30 free-text errors stood.
  Every setting is refused (exit 2) before a file is read or a model or
  server is asked: a zero `--count`, a blank `--question-id` or
  `--instructions`, `--per-option 0`, `--retry-multiplier 0` (now before the
  schema is read), `--holdout` outside zero and one (now before the set is
  read) and `--ece-bins 0` for `benchmark` (now before the model loads). The
  datasets server's answer is split by its status (refused, limited with exit
  69, failing) apart from no answer, a page that breaks off, is not JSON or
  lacks rows or features; a dataset's columns name what they hold (a label
  column the dataset lacks lists its columns, one that is no class label
  shows its type, too few or too many classes count them, a row without the
  text column lists the row's columns). An export with no row is told from
  one whose every row was refused, naming the first refusal; a writer that
  wrote nothing usable counts its empty and repeated answers. Every set and
  report is written through `ster.file`, and `benchmark` prints its report
  before writing it. `tests/decide/pipeline.sh` and `fetch.sh` provoke the
  local causes; `decisions` leaves `tests/errors/uncatalogued.json`.
- `ster decide`, `ster calibrate` and their `ster request` twins refuse
  through declared causes (`ster.decide.request`, `.calibration`,
  `.examples`, `ster.decide`, `ster.decide.calibrate`, published on
  `docs/guide/errors/decide.md` and `decide-calibration.md`): a missing,
  unreadable or broken request, calibration or example set, an empty state,
  question id, instructions, option name or level, too few options, more
  options than the 26 answer letters, a label naming a question not asked or
  an option not offered, a calibration of another schema, temperature or
  model, a prompt longer than the model's context and an answer or a
  calibration that cannot be written. Rendering a question checks the
  request first, so no path into it indexes past the letters any more.
- A tokenizer that spells an answer letter with a token the model's weights
  do not score is refused as `ster.decide.letter-outside`; the decision used
  to panic on an index past the vocabulary.
- `ster calibrate --ece-bins 0` is refused (exit 2) before the examples or
  the model are read; it used to be refused after the checkpoint loaded.
- `ster request decide` without a request document is refused as a body
  that does not match the operation's schema.
- `ster decisions fetch` refusals of the datasets server name the request
  it answered (`GET https://datasets-server.huggingface.co/rows?dataset=…&config=…&split=…&offset=…&length=…`)
  and its HTTP status beside the server's own words, where they carried the
  status and the words only. A label the dataset's classes do not name is
  skipped by comparing the index, not through a conversion that could
  swallow it.
- `ster decisions import` refuses a line under its own cause of
  `ster.decisions.import.rows` (13: not JSON, not an object, a missing
  field, a context, options or label of another type, a blank context, too
  few or too many options, a blank or repeated option, a label past the
  options), each naming the line, the file and what the line held. The
  report's `rejected` entries are `{line, reason, cause}` objects where they
  were `line N: <sentence>` strings, and an export whose every line is
  refused is refused with its first line's cause and fix and the count per
  cause, where it was `rows-rejected` quoting the first sentence (removed).
  `tests/decide/import-rows.sh` provokes each cause.
