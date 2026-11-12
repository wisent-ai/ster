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

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.inference`: 7 declared cause(s) of its own.

<a id="ster-inference-route-blank"></a>

### `ster.inference.route-blank`

- **What failed:** choosing the route Ster asks Stado's inference alternatives for
- **Cause:** the route is blank
- **Observed state shown:** --generator-model "{route}"
- **Fix:** name a Brama alias, a canonical provider/model route or a selector with --generator-model, e.g. wisent-backend/chat/primary
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_blank_route_is_refused`

<a id="ster-inference-stado-unstartable"></a>

### `ster.inference.stado-unstartable`

- **What failed:** starting `{call}`
- **Cause:** Stado, which hands every completion Ster asks to its inference alternatives, could not be started: {error}
- **Observed state shown:** PATH: {path}
- **Fix:** install Stado so that `stado capabilities call --help` answers, or put the directory holding `stado` on PATH
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_missing_stado_is_refused`

<a id="ster-inference-call-unfinished"></a>

### `ster.inference.call-unfinished`

- **What failed:** handing the request to `{call}`
- **Cause:** the request could not be written to Stado, or Stado's answer could not be read: {error}
- **Observed state shown:** the system's error kind: {kind}
- **Fix:** run the command again; when it repeats, `stado capabilities resolve inference --consumer ster` says whether Stado itself answers
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/synthesis/inference.sh::a_call_whose_pipe_breaks_is_refused`

<a id="ster-inference-unanswered"></a>

### `ster.inference.unanswered`

- **What failed:** asking Stado's inference alternatives for a completion from {route}
- **Cause:** no inference alternative answered `{call}` ({status}): {said}
- **Observed state shown:** Stado exited {status} and named each alternative with its cause: {said}
- **Fix:** answer it as Stado's sentence says: `stado capabilities attestation record inference --consumer ster <request> --checked <what was checked>` answers a request Stado kept, `stado capabilities resolve inference --consumer ster` names each alternative's cause, and `stado capabilities select inference --order <alternatives>` changes the order
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/synthesis/resolved.sh::nothing_answering_is_refused_and_the_request_kept`

<a id="ster-inference-stado-refused"></a>

### `ster.inference.stado-refused`

- **What failed:** asking Stado's inference alternatives for a completion from {route}
- **Cause:** Stado refused `{call}` ({status}), and asking again unchanged will not help: {said}
- **Observed state shown:** Stado exited {status} and said: {said}
- **Fix:** act on Stado's sentence; `stado capabilities call --help` says what a request must be, and `stado capabilities resolve inference --consumer ster` whether ster is a consumer it serves
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_request_stado_refuses_is_refused`

<a id="ster-inference-answer-unreadable"></a>

### `ster.inference.answer-unreadable`

- **What failed:** reading what `{call}` printed
- **Cause:** it succeeded but printed no JSON object with an answer, the form `--json` promises
- **Observed state shown:** printed: {printed}
- **Fix:** install a Stado whose `stado capabilities call --help` offers --json, and run the command again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::an_answer_that_is_not_stados_form_is_refused`

<a id="ster-inference-completion-malformed"></a>

### `ster.inference.completion-malformed`

- **What failed:** reading the completion {provider} answered for {route}
- **Cause:** the answer carries no assistant message at choices[0].message.content, which every chat completion holds
- **Observed state shown:** answer: {answer}
- **Fix:** ask another route with --generator-model, or put another alternative first with `stado capabilities select inference --order <alternatives>`
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/synthesis/inference.sh::a_completion_without_a_message_is_refused`
