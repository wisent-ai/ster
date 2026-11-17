# How building and measuring labelled decision sets fails

Every declared way `ster decisions fetch`, `import`, `synthesize`, `split`
and `benchmark` fail: the question a fetched or imported set asks, what the
Hugging Face datasets server answers and the columns a dataset holds, the
rows of a JSONL export, a decision schema and what the writing model
answers, and the held-out fraction. Reading and writing their files is on
[file.md](file.md); reading a labelled set, a calibration and a model is on
[decide-calibration.md](decide-calibration.md) and
[model-load.md](model-load.md). How a refusal reads and which status it
exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/decide-pipeline.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.decisions.question`

check the id and instructions of the one choice question a fetched or imported set asks, before any row is read

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.decisions.question`: 2 declared cause(s) of its own.

<a id="ster-decisions-question-question-id-blank"></a>

### `ster.decisions.question.question-id-blank`

- **What failed:** checking the question every {source} example asks
- **Cause:** --question-id is blank, and every example answers its question under that id
- **Observed state shown:** --question-id {value}
- **Fix:** give --question-id the id the question is answered under, such as sentiment
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::a_blank_question_id_is_refused`

<a id="ster-decisions-question-instructions-blank"></a>

### `ster.decisions.question.instructions-blank`

- **What failed:** checking the question every {source} example asks
- **Cause:** --instructions is blank, and the model is asked the question in those words
- **Observed state shown:** --instructions {value}
- **Fix:** give --instructions the question the model answers about each text
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::blank_instructions_are_refused`

## `ster.decisions.fetch`

read a split's rows from the Hugging Face datasets server and turn its text column and class-label column, or each row's own question, choices and answer, into one labelled choice question per row

Started through: `ster decisions fetch`. Uses: `ster.decisions.question`, `ster.pairs.import.fields`, `ster.pairs.import.rows`, `ster.decide.request`, `ster.file`. Failure catalogue `ster.decisions.fetch`: 18 declared cause(s) of its own.

<a id="ster-decisions-fetch-count-none"></a>

### `ster.decisions.fetch.count-none`

- **What failed:** checking the fetch settings
- **Cause:** --count 0 asks for no row, so no labelled decision could be written
- **Observed state shown:** --count 0
- **Fix:** leave --count out to read every row of the split from --offset
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_count_of_zero_is_refused`

<a id="ster-decisions-fetch-server-unreachable"></a>

### `ster.decisions.fetch.server-unreachable`

- **What failed:** asking the datasets server for rows of {dataset}
- **Cause:** no answer came from https://datasets-server.huggingface.co: the connection was not made or broke (no network, a name the resolver does not know, a proxy or TLS failing)
- **Observed state shown:** GET {url}: the client's error kind {kind}; it answered: {error}
- **Fix:** run the command again once https://datasets-server.huggingface.co answers from this machine
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/decide/remote/fetch.sh::an_unreachable_datasets_server_is_refused`

<a id="ster-decisions-fetch-server-refused"></a>

### `ster.decisions.fetch.server-refused`

- **What failed:** asking the datasets server for rows of {dataset}
- **Cause:** the datasets server answered {status}: it serves no rows of {dataset} with config {config} and split {split} to this account (a dataset, config or split it does not hold, or a gated one)
- **Observed state shown:** GET {url} answered HTTP {status}; the server said: {said}
- **Fix:** name a dataset, --config and --split https://huggingface.co/datasets/{dataset} lists, or one this account may read
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::an_unknown_dataset_is_refused`

<a id="ster-decisions-fetch-server-limited"></a>

### `ster.decisions.fetch.server-limited`

- **What failed:** asking the datasets server for rows of {dataset}
- **Cause:** the datasets server answered 429: it limits how often it is asked, and this fetch passed the limit
- **Observed state shown:** GET {url} answered HTTP {status}; the server said: {said}
- **Fix:** run the command again after a while, or fetch fewer rows with --count
- **Code:** `rate_limit`, exit 69; retrying unchanged can help
- **Test:** `tests/decide/remote/fetch.sh::a_datasets_server_limiting_requests_is_refused`

<a id="ster-decisions-fetch-server-failing"></a>

### `ster.decisions.fetch.server-failing`

- **What failed:** asking the datasets server for rows of {dataset}
- **Cause:** the datasets server answered {status}: it failed on its side, so the request was right and the server could not serve it
- **Observed state shown:** GET {url} answered HTTP {status}; the server said: {said}
- **Fix:** run the command again once https://status.huggingface.co shows the datasets server serving
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/decide/remote/fetch.sh::a_datasets_server_failing_on_its_side_is_refused`

<a id="ster-decisions-fetch-page-unread"></a>

### `ster.decisions.fetch.page-unread`

- **What failed:** reading the datasets server's page of {dataset}
- **Cause:** the page broke off before it was read whole: the server closed the connection midway, or answered bytes that are not UTF-8 text
- **Observed state shown:** GET {url} answered HTTP {status}; error kind {kind}; the client answered: {error}
- **Fix:** run the command again; the rows are read from the start
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/decide/remote/fetch.sh::a_page_that_breaks_off_is_refused`

<a id="ster-decisions-fetch-page-not-json"></a>

### `ster.decisions.fetch.page-not-json`

- **What failed:** reading the datasets server's page of {dataset}
- **Cause:** the server answered text that is not JSON, where the rows route answers one JSON document
- **Observed state shown:** GET {url} answered HTTP 200 with {size} bytes: {page}; the JSON reader stopped at line {line}, column {column}: {error}
- **Fix:** run the command again; when it repeats, https://huggingface.co/docs/dataset-viewer says what the rows route answers now
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_page_that_is_not_json_is_refused`

<a id="ster-decisions-fetch-page-rowless"></a>

### `ster.decisions.fetch.page-rowless`

- **What failed:** reading the datasets server's page of {dataset} at offset {offset}
- **Cause:** the page carries no `rows` array, which the rows route answers with
- **Observed state shown:** the page's fields: {fields}
- **Fix:** run the command again; when it repeats, https://huggingface.co/docs/dataset-viewer says what the rows route answers now
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_page_without_rows_is_refused`

<a id="ster-decisions-fetch-page-featureless"></a>

### `ster.decisions.fetch.page-featureless`

- **What failed:** reading the columns of {dataset}
- **Cause:** the page carries no `features` array naming the dataset's columns, which the rows route answers with
- **Observed state shown:** the page's fields: {fields}
- **Fix:** run the command again; when it repeats, https://huggingface.co/docs/dataset-viewer says what the rows route answers now
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_page_without_features_is_refused`

<a id="ster-decisions-fetch-label-column-absent"></a>

### `ster.decisions.fetch.label-column-absent`

- **What failed:** reading the label column '{label}' of {dataset}
- **Cause:** the dataset has no column named '{label}'
- **Observed state shown:** the dataset's columns: {columns}
- **Fix:** name one of the dataset's class-label columns with --label-field
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_label_column_the_dataset_lacks_is_refused`

<a id="ster-decisions-fetch-label-not-class"></a>

### `ster.decisions.fetch.label-not-class`

- **What failed:** reading the label column '{label}' of {dataset}
- **Cause:** column '{label}' is no class label with named classes, so its values name no option
- **Observed state shown:** column '{label}' is {feature}
- **Fix:** name a ClassLabel column with --label-field; the dataset's columns are listed on https://huggingface.co/datasets/{dataset}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_label_column_that_is_not_a_class_label_is_refused`

<a id="ster-decisions-fetch-class-name-not-text"></a>

### `ster.decisions.fetch.class-name-not-text`

- **What failed:** reading the class names of column '{label}' of {dataset}
- **Cause:** column '{label}' names a class with something other than text, and an option is named by text
- **Observed state shown:** class {index}: {found}
- **Fix:** name another class-label column with --label-field, or the dataset whose classes are named
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_class_named_with_no_text_is_refused`

<a id="ster-decisions-fetch-classes-too-few"></a>

### `ster.decisions.fetch.classes-too-few`

- **What failed:** reading the class names of column '{label}' of {dataset}
- **Cause:** column '{label}' names {count} class(es), and a choice needs at least two options
- **Observed state shown:** classes: {classes}
- **Fix:** name a class-label column with two or more classes with --label-field
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_label_column_with_one_class_is_refused`

<a id="ster-decisions-fetch-classes-too-many"></a>

### `ster.decisions.fetch.classes-too-many`

- **What failed:** reading the class names of column '{label}' of {dataset}
- **Cause:** column '{label}' names {count} classes, and a choice is answered under one of {most} letters
- **Observed state shown:** classes: {count}; the most: {most}
- **Fix:** name a class-label column with at most {most} classes with --label-field
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_label_column_with_too_many_classes_is_refused`

<a id="ster-decisions-fetch-text-column-absent"></a>

### `ster.decisions.fetch.text-column-absent`

- **What failed:** reading row {index} of {dataset}
- **Cause:** the row has no column '{text}', so it carries no text to decide about
- **Observed state shown:** the row's columns: {columns}
- **Fix:** name one of the dataset's text columns with --text-field
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_text_column_the_dataset_lacks_is_refused`

<a id="ster-decisions-fetch-text-column-mistyped"></a>

### `ster.decisions.fetch.text-column-mistyped`

- **What failed:** reading row {index} of {dataset}
- **Cause:** column '{text}' holds {found} in this row, not text
- **Observed state shown:** row {index}: {text}: {found}
- **Fix:** name a column that holds text with --text-field
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_text_column_that_holds_no_text_is_refused`

<a id="ster-decisions-fetch-none-kept"></a>

### `ster.decisions.fetch.none-kept`

- **What failed:** turning rows {first}..{end} of {dataset} into labelled decisions
- **Cause:** no row read carried both a text and a label the dataset's classes name
- **Observed state shown:** rows read: {rows}; skipped: {skipped}; classes: {classes}
- **Fix:** fetch from another --offset or --split, or name the columns that hold the text and the label
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::rows_without_a_known_label_are_refused`

<a id="ster-decisions-fetch-rows-none"></a>

### `ster.decisions.fetch.rows-none`

- **What failed:** reading rows from {first} of {dataset}
- **Cause:** the split {split} holds no row from offset {first}, so there is no multiple-choice row to turn into a labelled decision
- **Observed state shown:** rows read: 0; --split {split}; --offset {first}
- **Fix:** fetch from a lower --offset, or name a --split that holds rows
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/fetch.sh::a_split_without_rows_from_the_offset_is_refused`

## `ster.decisions.import.rows`

read one line of a JSONL export holding context, options and label into one labelled choice, or name why it cannot be one

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.decisions.import.rows`: 13 declared cause(s) of its own.

<a id="ster-decisions-import-rows-row-not-json"></a>

### `ster.decisions.import.rows.row-not-json`

- **What failed:** importing line {line} of {path}
- **Cause:** line {line} is not JSON, and every line of the export is one {row_fields} object
- **Observed state shown:** line {line}: {excerpt}; the JSON reader stopped at column {column}: {error}
- **Fix:** write line {line} as one JSON object on one line, such as {row_example}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_line_that_is_not_json_is_refused`

<a id="ster-decisions-import-rows-row-not-object"></a>

### `ster.decisions.import.rows.row-not-object`

- **What failed:** importing line {line} of {path}
- **Cause:** line {line} is JSON, {found}, where a row is one object holding context, options and label
- **Observed state shown:** line {line}: {excerpt}
- **Fix:** write line {line} as one object, such as {row_example}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_line_that_is_not_an_object_is_refused`

<a id="ster-decisions-import-rows-row-field-missing"></a>

### `ster.decisions.import.rows.row-field-missing`

- **What failed:** importing line {line} of {path}
- **Cause:** line {line} has no {field}, and a row states its context, its options and the index of the right one
- **Observed state shown:** the fields it has: {fields}
- **Fix:** add {field} to line {line}, as in {row_example}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_row_without_a_field_is_refused`

<a id="ster-decisions-import-rows-row-context-not-text"></a>

### `ster.decisions.import.rows.row-context-not-text`

- **What failed:** importing line {line} of {path}
- **Cause:** the context of line {line} is {found}, where it is the text the question is asked about
- **Observed state shown:** line {line}: {excerpt}
- **Fix:** write the context of line {line} as text in double quotes
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_context_that_is_not_text_is_refused`

<a id="ster-decisions-import-rows-row-options-not-list"></a>

### `ster.decisions.import.rows.row-options-not-list`

- **What failed:** importing line {line} of {path}
- **Cause:** the options of line {line} are {found}, where they are a list of the texts to choose from
- **Observed state shown:** line {line}: {excerpt}
- **Fix:** write the options of line {line} as a list of two or more texts, such as ["calm", "rough"]
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::options_that_are_not_a_list_are_refused`

<a id="ster-decisions-import-rows-row-option-not-text"></a>

### `ster.decisions.import.rows.row-option-not-text`

- **What failed:** importing line {line} of {path}
- **Cause:** option {index} of line {line} is {found}, where every option is text
- **Observed state shown:** the options: {options}
- **Fix:** write option {index} of line {line} as text in double quotes
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::an_option_that_is_not_text_is_refused`

<a id="ster-decisions-import-rows-row-label-not-index"></a>

### `ster.decisions.import.rows.row-label-not-index`

- **What failed:** importing line {line} of {path}
- **Cause:** the label of line {line} is {found}, where it is the index of the right option, a whole number from 0
- **Observed state shown:** the options: {options}
- **Fix:** set the label of line {line} to the index of the right option, counted from 0
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_label_that_is_not_an_index_is_refused`

<a id="ster-decisions-import-rows-row-context-blank"></a>

### `ster.decisions.import.rows.row-context-blank`

- **What failed:** importing line {line} of {path}
- **Cause:** the context of line {line} is blank, so there is nothing to ask the question about
- **Observed state shown:** context: {context}
- **Fix:** write the text the question is asked about as the context of line {line}, or remove the line
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_blank_context_is_refused`

<a id="ster-decisions-import-rows-row-options-few"></a>

### `ster.decisions.import.rows.row-options-few`

- **What failed:** importing line {line} of {path}
- **Cause:** line {line} holds too few options to choose between, {count}, and a choice needs at least two
- **Observed state shown:** the options: {options}
- **Fix:** give line {line} at least two options, the right one among them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_row_with_one_option_is_refused`

<a id="ster-decisions-import-rows-row-options-many"></a>

### `ster.decisions.import.rows.row-options-many`

- **What failed:** importing line {line} of {path}
- **Cause:** line {line} has {count} options, and Ster labels at most {max}, one letter each
- **Observed state shown:** the options: {options}
- **Fix:** keep at most {max} options on line {line}, the right one among them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_row_with_too_many_options_is_refused`

<a id="ster-decisions-import-rows-row-option-blank"></a>

### `ster.decisions.import.rows.row-option-blank`

- **What failed:** importing line {line} of {path}
- **Cause:** option {index} of line {line} is blank, and a blank option cannot be told from no option
- **Observed state shown:** the options: {options}
- **Fix:** write option {index} of line {line}, or take it out and correct the label
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_blank_option_is_refused`

<a id="ster-decisions-import-rows-row-options-repeated"></a>

### `ster.decisions.import.rows.row-options-repeated`

- **What failed:** importing line {line} of {path}
- **Cause:** options {first} and {second} of line {line} carry the same text once trimmed, {text}, so the label cannot say which is right
- **Observed state shown:** the options: {options}
- **Fix:** make options {first} and {second} of line {line} differ, or keep one of them and correct the label
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::repeated_options_are_refused`

<a id="ster-decisions-import-rows-row-label-outside"></a>

### `ster.decisions.import.rows.row-label-outside`

- **What failed:** importing line {line} of {path}
- **Cause:** the label {label} of line {line} names no option: the row has {count} options, numbered from 0 to {last}
- **Observed state shown:** the options: {options}
- **Fix:** set the label of line {line} to the index of the right option, 0 to {last}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/import-rows.sh::a_label_past_the_options_is_refused`

## `ster.decisions.import`

read a JSONL export of context/options/label rows into one labelled choice question per row, reporting each refused row under its cause

Started through: `ster decisions import`. Uses: `ster.decisions.question`, `ster.decisions.import.rows`, `ster.decide.examples`, `ster.file`. Failure catalogue `ster.decisions.import`: 1 declared cause(s) of its own.

<a id="ster-decisions-import-rows-none"></a>

### `ster.decisions.import.rows-none`

- **What failed:** importing {path}
- **Cause:** the file holds no row: every line is blank
- **Observed state shown:** path: {path} ({state})
- **Fix:** name the JSONL export with one object of context, options and label per line
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::an_export_without_rows_is_refused`

## `ster.decisions.synthesize`

write states for every option of every question a schema asks through Stado's inference alternatives, each labelled with the option it was written for

Started through: `ster decisions synthesize`. Uses: `ster.decide.request`, `ster.inference`, `ster.decide.examples`, `ster.file`. Failure catalogue `ster.decisions.synthesize`: 5 declared cause(s) of its own.

<a id="ster-decisions-synthesize-per-option-none"></a>

### `ster.decisions.synthesize.per-option-none`

- **What failed:** checking the synthesis settings
- **Cause:** --per-option 0 asks for no state, so no labelled decision could be written
- **Observed state shown:** --per-option 0
- **Fix:** give --per-option the number of states to write for each option, 1 or more
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::no_state_per_option_is_refused`

<a id="ster-decisions-synthesize-retries-none"></a>

### `ster.decisions.synthesize.retries-none`

- **What failed:** checking the synthesis settings
- **Cause:** --retry-multiplier 0 allows no attempt, so no state could be written
- **Observed state shown:** --retry-multiplier 0
- **Fix:** give --retry-multiplier how many attempts each state may take, 1 or more
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::no_attempt_per_state_is_refused`

<a id="ster-decisions-synthesize-schema-domain-blank"></a>

### `ster.decisions.synthesize.schema-domain-blank`

- **What failed:** reading the decision schema {path}
- **Cause:** its domain is blank, and the writer is told what to write by it
- **Observed state shown:** domain: {domain}
- **Fix:** state in the schema's domain what the states are, such as "a customer support message to an online shoe store"
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::a_schema_without_a_domain_is_refused`

<a id="ster-decisions-synthesize-schema-questions-none"></a>

### `ster.decisions.synthesize.schema-questions-none`

- **What failed:** reading the decision schema {path}
- **Cause:** it asks no question, so no state would be labelled
- **Observed state shown:** questions: 0
- **Fix:** add a question to the schema's questions, as docs/guide/decisions.md gives one
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::a_schema_without_questions_is_refused`

<a id="ster-decisions-synthesize-none-written"></a>

### `ster.decisions.synthesize.none-written`

- **What failed:** synthesizing labelled decisions about {domain}
- **Cause:** the writing model answered no usable state: every answer was empty or repeated an earlier one
- **Observed state shown:** attempts: {attempts}; empty: {empty}; repeated: {repeated}; generator: {generator}
- **Fix:** raise --temperature or --retry-multiplier, or name another --generator-model
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::a_writer_that_writes_nothing_usable_is_refused`

## `ster.decisions.split`

part a labelled set into a training set and a held-out set by a seeded shuffle, and write both

Started through: `ster decisions split`. Uses: `ster.decide.examples`, `ster.file`. Failure catalogue `ster.decisions.split`: 1 declared cause(s) of its own.

<a id="ster-decisions-split-holdout-outside"></a>

### `ster.decisions.split.holdout-outside`

- **What failed:** checking the split settings
- **Cause:** --holdout {holdout} is no fraction above zero and below one, so it parts no example from the rest
- **Observed state shown:** --holdout {holdout}
- **Fix:** give --holdout the share of examples to hold out, such as 0.2
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/decide/remote/pipeline.sh::a_holdout_outside_zero_and_one_is_refused`

## `ster.decisions.benchmark`

answer every labelled question of a set with a model, unsteered and, when steering artifacts are named, steered, score accuracy, likelihood and calibration per question type, list every question's gold, prediction and option probabilities, and write the report

Started through: `ster decisions benchmark`. Uses: `ster.decide.calibrate`, `ster.decide.examples`, `ster.decide.calibration`, `ster.runtime.vectors`, `ster.model.load`, `ster.runtime.steer`, `ster.file`. Failure catalogue `ster.decisions.benchmark`: 1 declared cause(s) of its own.

<a id="ster-decisions-benchmark-prediction-unnamed"></a>

### `ster.decisions.benchmark.prediction-unnamed`

- **What failed:** reading question '{question}' of example {example}
- **Cause:** the model's probabilities put the most mass on option {picked}, and the question names {count} option(s), so that option names no answer
- **Observed state shown:** picked option: {picked}; options: {count}; probabilities: {probabilities}
- **Fix:** report this to Ster's maintainers with the example set and model: the read gave a probability for an option the question does not have
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/decide/steering.sh::a_prediction_outside_the_options_is_refused`
