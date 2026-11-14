# How a calibration fails

Every declared way `ster calibrate` and its `ster request calibrate` twin
fail, with the two layers they read through: a calibration artifact (which
`ster decide --calibration` reads too) and a labelled example set (which
`ster tune decide` reads too). Reading the decision requests inside an
example set is on [decide.md](decide.md), loading the model on
[model.md](model.md). How a refusal reads and which status it exits with is
on [How Ster fails](../errors.md).

The bin count, the calibration and the example set are all checked before a
weight is mapped.

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/decide-calibration.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.decide.calibration`

read a calibration artifact, check its schema, temperature and model, and write one back

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.decide.calibration`: 4 declared cause(s) of its own.

<a id="ster-decide-calibration-calibration-missing"></a>

### `ster.decide.calibration.calibration-missing`

- **What failed:** reading calibration {path}
- **Cause:** no file is there: nothing at {path} is the calibration named
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name a calibration `ster calibrate --output {path}` wrote for this model (`ls -ln {parent}` shows what that directory holds), or leave --calibration out to read raw probabilities
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/calibration.sh::a_missing_calibration_is_refused`

<a id="ster-decide-calibration-calibration-schema"></a>

### `ster.decide.calibration.calibration-schema`

- **What failed:** reading calibration {path}
- **Cause:** it declares schema "{schema}", and this Ster reads {expected}
- **Observed state shown:** schema: "{schema}"
- **Fix:** fit it again with this Ster: `ster calibrate --examples <labelled.json> --output {path}`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/calibration.sh::a_calibration_of_another_schema_is_refused`

<a id="ster-decide-calibration-calibration-temperature"></a>

### `ster.decide.calibration.calibration-temperature`

- **What failed:** reading calibration {path}
- **Cause:** its temperature is {temperature}, and the logits are divided by it, so only a positive number is a temperature
- **Observed state shown:** temperature: {temperature}
- **Fix:** fit it again with `ster calibrate --examples <labelled.json> --output {path}`; a fitted temperature is always positive
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/calibration.sh::a_calibration_with_a_temperature_of_zero_is_refused`

<a id="ster-decide-calibration-calibration-model"></a>

### `ster.decide.calibration.calibration-model`

- **What failed:** applying calibration {path}
- **Cause:** it was fitted for model {fitted}, and a temperature fitted on one checkpoint says nothing about another
- **Observed state shown:** calibration model: {fitted}; loaded model: {model}
- **Fix:** fit a calibration for {model} with `ster calibrate --model {model} --examples <labelled.json> --output <file>`, or name the calibration fitted for it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/calibration.sh::a_calibration_for_another_model_is_refused`

## `ster.decide.examples`

read a labelled example set and check every request in it and every label against the options it names

Started through: run only by the functionalities that use it. Uses: `ster.decide.request`, `ster.file`. Failure catalogue `ster.decide.examples`: 5 declared cause(s) of its own.

<a id="ster-decide-examples-examples-missing"></a>

### `ster.decide.examples.examples-missing`

- **What failed:** reading labelled examples {path}
- **Cause:** no file is there: nothing at {path} is the labelled example set named
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name an existing labelled example set with --examples (`ls -ln {parent}` shows what that directory holds), or write one with `ster decisions import` or `ster decisions synthesize`
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/examples.sh::a_missing_example_set_is_refused`

<a id="ster-decide-examples-examples-none"></a>

### `ster.decide.examples.examples-none`

- **What failed:** checking the labelled examples
- **Cause:** the set holds no example, and a temperature is fitted on labelled examples
- **Observed state shown:** {source}: "examples": []
- **Fix:** add at least one example with a request and the answer a person gave to one of its questions
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/examples.sh::an_empty_example_set_is_refused`

<a id="ster-decide-examples-label-unasked"></a>

### `ster.decide.examples.label-unasked`

- **What failed:** checking labelled example {index}
- **Cause:** it labels question '{id}', which its request does not ask
- **Observed state shown:** questions asked: {asked}
- **Fix:** label one of the questions its request asks, or add question '{id}' to the request
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/examples.sh::a_label_for_a_question_not_asked_is_refused`

<a id="ster-decide-examples-label-not-option"></a>

### `ster.decide.examples.label-not-option`

- **What failed:** checking labelled example {index}
- **Cause:** it answers question '{id}' with {answer}, which is not one of its options
- **Observed state shown:** answer: {answer}; options: {options}
- **Fix:** answer a choice question with one of its option names, a score question with a level index from 0, and a noul question with true or false
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/examples.sh::a_label_that_is_not_an_option_is_refused`

<a id="ster-decide-examples-labels-none"></a>

### `ster.decide.examples.labels-none`

- **What failed:** checking the labelled examples
- **Cause:** none of its {count} examples labels a question, so there is nothing to fit on
- **Observed state shown:** examples: {count}; labelled questions: 0
- **Fix:** give at least one example an answers entry naming the option a person chose
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/examples.sh::an_example_set_without_labels_is_refused`

## `ster.decide.calibrate`

fit the temperature that makes a model's decision probabilities match labelled examples, and write it as a calibration

Started through: `ster calibrate`, `ster request calibrate`. Uses: `ster.decide.examples`, `ster.decide.calibration`, `ster.model.load`, `ster.decide`. Failure catalogue `ster.decide.calibrate`: 1 declared cause(s) of its own.

<a id="ster-decide-calibrate-bins-none"></a>

### `ster.decide.calibrate.bins-none`

- **What failed:** checking the calibration settings
- **Cause:** the expected calibration error is measured over equal-width confidence bins, and zero bins measure nothing
- **Observed state shown:** --ece-bins {bins}
- **Fix:** give --ece-bins a whole number of 1 or more, such as 10
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/decide/calibration.sh::zero_confidence_bins_are_refused`
