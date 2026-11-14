# How a completion asked of Stado's inference alternatives fails

Every declared way the hosted writer of `ster pairs synthesize --generator
brama` and `ster decisions synthesize` fails: the route it names, starting
Stado, and what Stado answers. Every completion is handed to `stado
capabilities call inference --consumer ster --json`; Ster names no gateway,
key or alternative ([synthesis through Stado's inference
alternatives](../pair-sets/hosted-generator.md)). Ster finds `stado` in the
order Stado states for every program that runs it (`stado --help`, "Where
another program finds Stado"): the path in `STADO_BIN` when it is set, else
`stado` on `PATH`, else `${STADO_BIN_DIR:-$HOME/.stado/bin}/stado`, an empty
variable counting as unset. A `STADO_BIN` naming no file is refused, not
passed over, and every start refusal names the place the `stado` was found
through. A refusal Stado prints as
its `--json` envelope is relayed with the command line Ster ran, the
directory it ran in, Stado's exit status, Stado's cause id and its fix. How
a refusal reads and which status it exits with is on [How Ster
fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/inference.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.inference`

hand one chat completion to `stado capabilities call inference --consumer ster --json`, which asks the first alternative in Stado's order that answers, and read the assistant text and the alternative that answered

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.inference`: 11 declared cause(s) of its own.

<a id="ster-inference-route-blank"></a>

### `ster.inference.route-blank`

- **What failed:** choosing the route Ster asks Stado's inference alternatives for
- **Cause:** the route is blank
- **Observed state shown:** --generator-model "{route}"
- **Fix:** name a Brama alias, a canonical provider/model route or a selector with --generator-model, e.g. wisent-backend/chat/primary
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_blank_route_is_refused`

<a id="ster-inference-stado-bin-missing"></a>

### `ster.inference.stado-bin-missing`

- **What failed:** starting `{call}`
- **Cause:** STADO_BIN names {stado}, where no file is, and STADO_BIN is the first place Stado says another program finds it (`stado --help`), so PATH and $HOME/.stado/bin are not looked at
- **Observed state shown:** STADO_BIN: {stado} ({state})
- **Fix:** set STADO_BIN to the path of the `stado` program (`command -v stado` prints it when one is on PATH), or unset it so PATH and then $STADO_BIN_DIR/stado (or $HOME/.stado/bin/stado when STADO_BIN_DIR is unset or empty) are looked at
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_stado_bin_naming_nothing_is_refused`

<a id="ster-inference-stado-absent"></a>

### `ster.inference.stado-absent`

- **What failed:** starting `{call}`
- **Cause:** no `stado` is at any place Stado says another program finds it (`stado --help`): STADO_BIN is unset, no `stado` is on PATH and none is at $STADO_BIN_DIR/stado (or $HOME/.stado/bin/stado when STADO_BIN_DIR is unset or empty), and Stado hands every completion Ster asks to its inference alternatives
- **Observed state shown:** STADO_BIN: {program}; PATH: {path}; {place}: {state}
- **Fix:** install Stado with its install-stado.sh, which puts it in $STADO_BIN_DIR when that names a directory, else in $HOME/.stado/bin, or set STADO_BIN to the path of `stado`; `stado capabilities call --help` answers once it is found
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_missing_stado_is_refused`

<a id="ster-inference-stado-not-executable"></a>

### `ster.inference.stado-not-executable`

- **What failed:** starting `{call}`
- **Cause:** the `stado` Ster found through {through} may not be executed by this account: its permission bits or a directory before it keep this account out
- **Observed state shown:** {stado} ({state}); the system's error kind: {kind}; it answered: {error}
- **Fix:** make {stado} executable for this account (`chmod u+x {stado}` as its owner), or set STADO_BIN to the path of an executable `stado`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_stado_this_account_may_not_execute_is_refused`

<a id="ster-inference-stado-unlaunchable"></a>

### `ster.inference.stado-unlaunchable`

- **What failed:** starting `{call}`
- **Cause:** the system answered that the `stado` Ster found through {through} does not exist when Ster started it: it is a script whose interpreter is not installed, or it was removed after it was found
- **Observed state shown:** {stado} ({state}); its interpreter: {interpreter}; the system's error kind: {kind}; it answered: {error}
- **Fix:** reinstall Stado with its install-stado.sh, which writes the program itself rather than a script, or set STADO_BIN to the path of a `stado` that starts (`{stado} --help` shows whether it does)
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_stado_whose_interpreter_is_missing_is_refused`

<a id="ster-inference-call-unfinished"></a>

### `ster.inference.call-unfinished`

- **What failed:** handing the request to `{call}`
- **Cause:** the pipe between Ster and Stado broke: Stado exited before it read the whole request, or its answer stopped coming
- **Observed state shown:** the system's error kind: {kind}; it answered: {error}
- **Fix:** run the command again; when it repeats, `stado capabilities resolve inference --consumer ster` says whether Stado itself answers
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/synthesis/inference.sh::a_call_whose_pipe_breaks_is_refused`

<a id="ster-inference-unanswered"></a>

### `ster.inference.unanswered`

- **What failed:** asking Stado's inference alternatives for a completion from {route}
- **Cause:** no inference alternative answered `{call}`: every alternative in Stado's order was unreachable or refused the request, so asking again later can help
- **Observed state shown:** `{call}` in {directory} exited {status}; Stado's cause {named}: {said}; its fix: {fix}
- **Fix:** follow Stado's fix: `stado capabilities attestation record inference --consumer ster <request> --checked <what was checked>` answers a request Stado kept, `stado capabilities resolve inference --consumer ster` names each alternative's cause, and `stado capabilities select inference --order <alternatives>` changes the order
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/synthesis/resolved.sh::nothing_answering_is_refused_and_the_request_kept`

<a id="ster-inference-stado-refused"></a>

### `ster.inference.stado-refused`

- **What failed:** asking Stado's inference alternatives for a completion from {route}
- **Cause:** Stado refused `{call}`, and asking again unchanged will not help: it refuses the request's form, or does not serve ster as a consumer
- **Observed state shown:** `{call}` in {directory} exited {status}; Stado's cause {named}: {said}; its fix: {fix}
- **Fix:** follow Stado's fix; `stado capabilities call --help` says what a request must be, and `stado capabilities resolve inference --consumer ster` whether ster is a consumer it serves
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_request_stado_refuses_is_refused`

<a id="ster-inference-answer-unreadable"></a>

### `ster.inference.answer-unreadable`

- **What failed:** reading what `{call}` printed
- **Cause:** it succeeded and printed no JSON, where `--json` promises one JSON object: the `stado` Ster ran is one without --json, or printed something else beside it
- **Observed state shown:** printed: {printed}; the JSON reader answered: {error}
- **Fix:** install a Stado whose `stado capabilities call --help` offers --json, and run the command again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::an_answer_that_is_not_stados_form_is_refused`

<a id="ster-inference-answer-fieldless"></a>

### `ster.inference.answer-fieldless`

- **What failed:** reading what `{call}` printed
- **Cause:** it succeeded and printed a JSON document without the answer field `--json` promises: the `stado` Ster ran writes another envelope
- **Observed state shown:** printed: {printed}
- **Fix:** install a Stado whose `stado capabilities call --help` offers --json, and run the command again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::an_answer_without_its_answer_field_is_refused`

<a id="ster-inference-completion-malformed"></a>

### `ster.inference.completion-malformed`

- **What failed:** reading the completion {provider} answered for {route}
- **Cause:** the answer carries no assistant message at choices[0].message.content, which every chat completion holds
- **Observed state shown:** answer: {answer}
- **Fix:** ask another route with --generator-model, or put another alternative first with `stado capabilities select inference --order <alternatives>`
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_completion_without_a_message_is_refused`
