# How a decision fails

Every declared way `ster decide` and its `ster request decide` twin fail:
reading and checking a decision request (the layer `ster calibrate` and
`ster tune decide` read every example's request through) and answering it.
Reading a calibration and labelled examples, and fitting a temperature, are
on [decide-calibration.md](decide-calibration.md); loading the model is on
[model.md](model.md) and [model-load.md](model-load.md), and finding the
token of an answer letter on [runtime.md](runtime.md). How a refusal reads
and which status it exits with is on [How Ster fails](../errors.md).

A request's state and questions are checked before a weight is mapped, so a
refusal of them comes in milliseconds; `ster request decide` refuses the same
document with status 2 before anything runs.

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/decide.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.decide.request`

read a decision request from a file or standard input and check its state and every question before a weight is mapped

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.decide.request`: 11 declared cause(s) of its own.

<a id="ster-decide-request-request-missing"></a>

### `ster.decide.request.request-missing`

- **What failed:** reading decision request {path}
- **Cause:** no file is there: {error}
- **Observed state shown:** path: {path}
- **Fix:** name an existing decision document with --request, or give `--request -` and write it on standard input
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_missing_request_is_refused`

<a id="ster-decide-request-request-unreadable"></a>

### `ster.decide.request.request-unreadable`

- **What failed:** reading decision request {path}
- **Cause:** it cannot be read as a file: {error}
- **Observed state shown:** path: {path}
- **Fix:** name a decision document this account can read; a directory is not one
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_directory_named_as_the_request_is_refused`

<a id="ster-decide-request-request-malformed"></a>

### `ster.decide.request.request-malformed`

- **What failed:** reading decision request {path}
- **Cause:** it is not a decision document: {error}
- **Observed state shown:** source: {path}
- **Fix:** write it as {"state": <text or JSON>, "questions": {"<id>": {"type": "choice", "instructions": "<text>", "criteria": {"<option>": null, "<option>": null}}}}; a question's type is choice, score or noul
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_request_that_is_not_json_is_refused`

<a id="ster-decide-request-state-empty"></a>

### `ster.decide.request.state-empty`

- **What failed:** checking the decision request
- **Cause:** its state is empty, so every question would be judged against nothing
- **Observed state shown:** state: {state}
- **Fix:** give the request a non-empty state: the text or JSON the questions are about
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::an_empty_state_is_refused`

<a id="ster-decide-request-questions-none"></a>

### `ster.decide.request.questions-none`

- **What failed:** checking the decision request
- **Cause:** it asks no question
- **Observed state shown:** questions: {}
- **Fix:** add a question under questions, such as {"<id>": {"type": "noul", "instructions": "<text>"}}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_request_without_questions_is_refused`

<a id="ster-decide-request-question-id-empty"></a>

### `ster.decide.request.question-id-empty`

- **What failed:** checking the decision request
- **Cause:** a question's id is empty, and the answer comes back under that id
- **Observed state shown:** question id: "{id}"
- **Fix:** give the question an id with at least one character other than white space
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_question_with_an_empty_id_is_refused`

<a id="ster-decide-request-instructions-empty"></a>

### `ster.decide.request.instructions-empty`

- **What failed:** checking question '{id}'
- **Cause:** its instructions are empty, so the model is shown options with no question
- **Observed state shown:** instructions: {instructions}
- **Fix:** write the question in its instructions field
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_question_without_instructions_is_refused`

<a id="ster-decide-request-options-too-few"></a>

### `ster.decide.request.options-too-few`

- **What failed:** checking {kind} question '{id}'
- **Cause:** it offers {count} {unit}, and a decision needs at least two to choose between
- **Observed state shown:** {unit}: {count}
- **Fix:** give its criteria at least two {unit}; a yes or no question is a noul question
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_question_with_one_option_is_refused`

<a id="ster-decide-request-options-too-many"></a>

### `ster.decide.request.options-too-many`

- **What failed:** checking {kind} question '{id}'
- **Cause:** it offers {count} {unit}, and each is answered under one of the {max} letters A to Z read at one token position
- **Observed state shown:** {unit}: {count}
- **Fix:** split the question into questions of at most {max} {unit} each
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_question_with_more_options_than_letters_is_refused`

<a id="ster-decide-request-option-name-empty"></a>

### `ster.decide.request.option-name-empty`

- **What failed:** checking choice question '{id}'
- **Cause:** one of its options has an empty name, and the answer names the option chosen
- **Observed state shown:** option name: "{name}"
- **Fix:** give every key of its criteria at least one character other than white space
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::an_option_with_an_empty_name_is_refused`

<a id="ster-decide-request-level-empty"></a>

### `ster.decide.request.level-empty`

- **What failed:** checking score question '{id}'
- **Cause:** its level {index} has an empty description, so the model is shown a letter with nothing under it
- **Observed state shown:** level {index}: {level}
- **Fix:** describe every level of its criteria, in order from the first
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_level_without_a_description_is_refused`

## `ster.decide`

answer every typed question of a decision request with the probability the model puts on each option

Started through: `ster decide`, `ster request decide`. Uses: `ster.decide.request`, `ster.decide.calibration`, `ster.model.load`. Failure catalogue `ster.decide`: 3 declared cause(s) of its own.

<a id="ster-decide-question-too-long"></a>

### `ster.decide.question-too-long`

- **What failed:** rendering question '{id}'
- **Cause:** its prompt is {tokens} tokens, and this model reads at most {context} at once
- **Observed state shown:** tokens: {tokens}; context: {context}
- **Fix:** shorten the state or the question's options, or decide with a model whose context holds the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_question_longer_than_the_context_is_refused`

<a id="ster-decide-letter-outside"></a>

### `ster.decide.letter-outside`

- **What failed:** reading the answer under letter {letter}
- **Cause:** the tokenizer spells {letter} as token {token}, and the model scores only tokens 0 to {vocabulary} exclusive, so its tokenizer and weights are of different vocabularies
- **Observed state shown:** token: {token}; scored tokens: {vocabulary}
- **Fix:** load the checkpoint with the tokenizer it was trained with; its config's vocab_size counts every token the tokenizer can produce
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::a_letter_the_model_does_not_score_is_refused`

<a id="ster-decide-answer-unwritable"></a>

### `ster.decide.answer-unwritable`

- **What failed:** writing the decision to {path}
- **Cause:** the file cannot be written: {error}
- **Observed state shown:** path: {path}
- **Fix:** name an --output in a directory that exists and this account can write, or leave --output out to read the answer on standard output
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/request.sh::an_answer_that_cannot_be_written_is_refused`
