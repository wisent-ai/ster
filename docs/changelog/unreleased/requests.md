# Unreleased: desktop requests, the failure catalogue and the real tests

Part of the [changelog](../../../CHANGELOG.md).

- The desktop first-use journey (`.probierz/ster/journeys/desktop-first-use.mjs`)
  sets no clock of its own. The walkthrough's screens are awaited with no
  deadline (the 15000 ms nobody decided is gone, and the Probierz toolkit no
  longer takes one). After the malformed pair set is chosen, the journey no
  longer sleeps 5 s before looking: it waits for the app's answer, the
  walkthrough's refusal `while importing the first pair set: …` or an
  `Imported: …` line, and an import is reported as `a malformed pair set was
  imported`. The reopening check, which slept 5 s and then looked for the
  walkthrough, is removed with its promise in `.probierz/ster.yaml`: a
  finished first use shows nothing, so no answer of the app marks when that
  look could be made. The journey's `timeoutMs: 240000` is removed from
  `.probierz/ster.yaml`: Probierz requires none and reads none
  (`probierz-rs/src/manifest/validate/journeys.rs`). The open panel's steps
  wait on the panel's own state in the Probierz toolkit instead of pauses.
  Written and read, not run here.
- Ster compiles again: every report handed to `ster::json::value`, which
  takes a borrow, is borrowed (`ster vector evaluate`, `curve`, `project`,
  `score` and the `vector/*`, `decide`, `calibrate`, `model/*/create` and
  `workspace/*` requests); `workflow::check_counts` is exported for the
  `vector compare` command and request that refuse their counts before any
  artifact is read; and the request jobs see `JobOutcome`, which
  `workspace/import` answers with. The release manifest's quality gate on
  every platform, `error-pages`, runs `kompilator test --test error-pages`
  with the `private-sources` input, which compiles the library and the
  `ster` binary (darwin-arm64 with `metal`, as its release builds) before
  checking every error page against the catalogue, so a tree that does not
  compile is refused before it lands.
- `ster doctor`'s `ster.errors` remedy and the install guide name the ster
  release installer at `stado://releases/ster/<version>/<platform>/install.sh`,
  which places `ster` in `~/.local/bin`, instead of the retired `stado product
  install` and the nonexistent `stado product update`; updating is running the
  installer of the newer version.
- No cause of the failure catalogue holds a brace its renderer would refuse:
  wisent-errors past the revision Ster pins refuses a template whose `{`
  opens no placeholder name, which literal JSON and a `${STADO_BIN_DIR:-…}`
  default are. The JSONL row causes take the row's shape and example as
  values (`row_fields`, `row_example`) and `questions-none` its empty object
  as `questions`, so what a reader sees and `tests/decide/*.sh` assert is
  unchanged; the refusal judgement, example set, reward scorer, `rows-none`
  and Stado location causes say the shape in words.
  `sources-share-unkept` fills `{why}` instead of `{problem}`, a raw-message
  name the same wisent-errors refuses outside `observed`, and says in words
  why the answer was not kept (it, or the random id kept beside it, could
  not be made in the configuration directory), with what the system said
  moved into `observed` as `the answer was not kept: {why}`: the same
  wisent-errors refuses a cause that is only the text it wraps.
- `ster request` with its standard output closed (`>&-`) or sent to the null
  device is refused before standard input is read, as
  `ster.request.stdout-discarded` on stderr with exit `2`, naming what the
  stream is; it used to run, write its result into nothing and end silently.
  A directory redirected into standard input is `body-directory`, a status-2
  result naming the directory's state, where it used to exit `1` with the
  system's "Is a directory" only; standard input breaking mid-read stays
  `body-unread` and now observes the stream's state. A path's state names a
  pipe, a character or block device and a socket instead of "neither a file
  nor a directory". `tests/request/refusals.sh` provokes both refusals.
- Reading and writing Ster's files fails through one layer, `ster.file`
  (`docs/guide/errors/file.md`), split by the reason the system or the
  reader gives, each cause naming the path, its state (kind, size, mode,
  owner) and what the refused step left: a file this account may not read, a
  directory where a file belongs, bytes that are not UTF-8, a path running
  through a file, a read the system broke off; JSON that is not JSON, cut
  short, or of another form than the document (naming the form and who
  writes it); safetensors that do not parse, a dtype Candle does not load,
  tensors the device does not take; a directory this account may not write,
  a read-only file system, a full disk, a directory where the file belongs,
  and a change the system broke off. The pair set, steering artifact,
  adapter, calibration, labelled examples, decision request, example set,
  checkpoint state and tensors, latest pointer, merge and export output,
  workspace record, model config, system turn, answers file and the error
  baselines use it; each keeps its own cause for a file that is not there.
  71 declared causes used to carry the library's own message as their
  reason; that message is now only in the observed state, and causes that
  covered several reasons are split: the Hugging Face Hub limiting
  (`hub-limited`, exit 69), failing (`hub-failing`) or refusing a request
  (`hub-refused`) apart from no answer at all, and a file not written into
  its cache (`download-uncached`); a reward scorer refusing the bearer, rate
  limiting or failing on its side; `stado` absent from PATH apart from one
  this account may not execute; a GGUF file cut short apart from one that is
  not GGUF; weights of another shape apart from a tensor the checkpoint does
  not store; an update of another shape apart from one the device did not
  compute; a text holding a piece the vocabulary lacks with no unknown token.
  `ster decide --output` prints the answer before writing the file, so a
  refused write keeps the decision, and `ster generate --prompts --output`
  creates the answers' directory before the first prompt is answered.
- `ster onboarding` refuses through declared causes of `ster.onboarding`
  (`docs/guide/errors/onboarding.md`): neither HOME nor XDG_STATE_HOME set, a
  progress file of another journey (naming the field and both values), and
  one standing at a screen the journey does not hold (listing the screens);
  reading and writing the progress file go through `ster.file`. The embedded
  journey is read into types and checked once, where 21 free-text errors
  stood that no input could reach. `tests/onboarding/state.sh` provokes each.
- The changelog keeps its unreleased changes per area under
  `docs/changelog/unreleased/`, so no part passes 300 lines.
- `ster request` refuses through declared causes of `ster.request`
  (`docs/guide/errors/request.md`, `docs/guide/desktop-requests.md`): an
  unknown operation (now naming the operations it serves), a body that is
  not JSON, ends before its value closes, or does not fit the operation's
  fields (each observing the byte count, line, column and the parser's
  reason), a blank field (naming what it holds and the CLI flag that takes
  it), a list shorter than the operation needs, a field without the one it
  works with (`batchSize` without `strengths`, `checkpoint` without
  `checkpointEvery`), standard input that cannot be read and standard output
  nobody reads (exit 69, where they used to exit 1 or vanish). Every status-2
  result now carries `json.cause` beside `json.error`, and an operation's own
  declared refusal (a decide request's options, a dedupe setting) reaches the
  desktop with its cause instead of as flattened text. `tests/request/refusals.sh`
  provokes each.
- Writing Ster's own reports, artifacts and states as JSON goes through one
  writer (`ster::json`), which states once why it cannot fail.
- `ster errors list|show|check` and `ster request errors/list|show` print and
  check the failure catalogue; `docs/guide/errors.md` and one page per area
  under `docs/guide/errors/` publish it, and `tests/errors/catalogue.sh`
  refuses a page that differs from the catalogue.
- A failed `ster request` result carries the declared cause in `json.cause`,
  and a rejected or conflicting workspace import carries it in its report.
  Every such cause, and each refused row of `ster pairs import` and
  `ster decisions import`, names the `functionality` that declares it (the id
  `ster errors show` takes) and `docs`, its line on the area page, so a
  consumer can show the whole declaration and link it.
- `tests/errors/pages.rs` (`kompilator test --manifest-path Cargo.toml --test error-pages`) compares every
  page under `docs/guide/errors/` with `Operation::markdown` of the
  functionalities that name it, so a page edited by hand cannot drift from
  the catalogue.
- The real tests that ran the binary under the operator's home now give it
  one of the run's own: `tests/errors/catalogue.sh`,
  `tests/steering/score.sh`, `tests/models/dots3_note.sh`,
  `minimax_m3.sh`, `tests/synthesis/judged.sh`,
  `tests/tuning/projection_widths.sh`, `provenance.mjs` and
  `tests/pairs/import-choices.mjs`.
- Every real test records the source revision with `git
  --no-optional-locks`, so recording whether the checkout is dirty never
  takes its `index.lock` or rewrites its index while the operator works in
  it.

- `ster --version` names the commit it was built from, `ster <version>
  (revision <commit>)`, read from WISENT_SOURCE_COMMIT, which every Stado
  build sets (`build.rs`); a build without it says `revision unknown`.
  Before, it named only the package version, so whether a repair was
  installed could not be read from the binary. Written and read, not built
  or run here.

- Ster compiles against wisent-onboarding-client 0.2.0, whose journey
  storage failure is a typed `StorageError`: the star request no longer
  wraps a first-use state that cannot be moved out of its file as the
  runtime's storage failure, and says Ster's own declared refusal for it
  instead (`src/cli/admin/star.rs`). Written and read, not built or run
  here.

- `ster sources` reports what it saw through wisent-onboarding-client's
  `SourcesSharing`, declared once in `src/cli/sharing.rs`; Ster's own copy of
  the consent file, the event, the Stado route and the curl transport is
  gone. The client is declared by version, 0.3.1, the first release that
  carries `SourcesSharing` and finds Stado in the order stado-program
  states, with its `integration` feature off. A non-UTF-8
  WISENT_SOURCE_COMMIT stops the build by name instead of reading as an
  unknown revision. Written and read, not built or run here.
