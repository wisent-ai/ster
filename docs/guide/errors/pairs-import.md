# How importing a benchmark export fails

Every declared way `ster pairs import` and `ster request pairs/import` fail:
the flags, each checked before the export is read, and what the export yields.
Reading the export itself is on [pairs-export.md](pairs-export.md), and
writing the set on [pairs.md](pairs.md). How a refusal reads and which status
it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/pairs-import.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.pairs.import.rows`

read one row of a TruthfulQA, Do-Not-Answer, BiPO, LiveCodeBench, multiple-choice or preference (DPO) export into one pair, or name why it cannot be one

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.pairs.import.rows`: 22 declared cause(s) of its own.

<a id="ster-pairs-import-rows-question-blank"></a>

### `ster.pairs.import.rows.question-blank`

- **What failed:** importing {row} of {source}
- **Cause:** the {column} column of {row} is blank, and every pair asks a question
- **Observed state shown:** {cells}
- **Fix:** write the question of {row} in {source}, or remove the row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_row_without_a_question_is_skipped`

<a id="ster-pairs-import-rows-answer-blank"></a>

### `ster.pairs.import.rows.answer-blank`

- **What failed:** importing {row} of {source}
- **Cause:** the {column} column of {row} is blank, and it is the {side} side of the pair
- **Observed state shown:** {cells}
- **Fix:** write the {column} of {row} in {source}, or remove the row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_row_with_a_blank_answer_is_skipped`

<a id="ster-pairs-import-rows-incorrect-none"></a>

### `ster.pairs.import.rows.incorrect-none`

- **What failed:** importing {row} of {source}
- **Cause:** the Incorrect Answers column of {row} holds no answer once split at `;`, and the negative side is one of them
- **Observed state shown:** {cells}
- **Fix:** write at least one incorrect answer of {row}, answers parted by `;`, or remove the row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_row_without_an_incorrect_answer_is_skipped`

<a id="ster-pairs-import-rows-problem-id-missing"></a>

### `ster.pairs.import.rows.problem-id-missing`

- **What failed:** importing {row} of {source}
- **Cause:** {row} has no question_id text, which is what pairs a problem with its examples
- **Observed state shown:** question_id: {found}
- **Fix:** give {row} of {source} the question_id its examples carry
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_problem_without_an_id_is_skipped`

<a id="ster-pairs-import-rows-problem-content-blank"></a>

### `ster.pairs.import.rows.problem-content-blank`

- **What failed:** importing {row} of {source}
- **Cause:** {row} has no question_content text, and it is the question of the pair
- **Observed state shown:** question_content: {found}
- **Fix:** give {row} of {source} its question_content
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_problem_without_content_is_skipped`

<a id="ster-pairs-import-rows-example-missing"></a>

### `ster.pairs.import.rows.example-missing`

- **What failed:** importing {row} of {source}
- **Cause:** no question of {examples} carries the question_id of {row}, {id}, and the pair's code comes from its examples
- **Observed state shown:** questions in {examples}: {count}
- **Fix:** name the question_examples.json written with {source} through --examples, or add the examples of {id} to it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_problem_without_examples_is_skipped`

<a id="ster-pairs-import-rows-example-code-missing"></a>

### `ster.pairs.import.rows.example-code-missing`

- **What failed:** importing {row} of {source}
- **Cause:** the example of {id} in {examples} has no {missing} code, and the pair is its good code against its bad
- **Observed state shown:** good_example.code: {good}; bad_example.code: {bad}
- **Fix:** write the {missing} code of {id} in {examples}, or remove the example
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::an_example_without_code_is_skipped`

<a id="ster-pairs-import-rows-choice-question-missing"></a>

### `ster.pairs.import.rows.choice-question-missing`

- **What failed:** importing {row} of {source}
- **Cause:** {row} has no question text at {pointer}, where --question points
- **Observed state shown:** {pointer}: {found}
- **Fix:** point --question at the question of every row, or remove {row} from {source}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_choice_row_without_a_question_is_skipped`

<a id="ster-pairs-import-rows-choices-not-list"></a>

### `ster.pairs.import.rows.choices-not-list`

- **What failed:** importing {row} of {source}
- **Cause:** {row} holds no list at {pointer}, where --choices points
- **Observed state shown:** {pointer}: {found}
- **Fix:** point --choices at the list of choices of every row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::choices_that_are_not_a_list_are_skipped`

<a id="ster-pairs-import-rows-choice-not-text"></a>

### `ster.pairs.import.rows.choice-not-text`

- **What failed:** importing {row} of {source}
- **Cause:** choice {index} of {row} is {found}, where every choice is non-empty text
- **Observed state shown:** {pointer}: {choices}
- **Fix:** write choice {index} of {row} in {source}, or remove the row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_choice_that_is_not_text_is_skipped`

<a id="ster-pairs-import-rows-choices-few"></a>

### `ster.pairs.import.rows.choices-few`

- **What failed:** importing {row} of {source}
- **Cause:** {row} holds {count} choices at {pointer}, and a pair is the correct choice against another
- **Observed state shown:** {pointer}: {choices}
- **Fix:** give {row} at least two choices, or remove it from {source}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_row_with_one_choice_is_skipped`

<a id="ster-pairs-import-rows-answer-missing"></a>

### `ster.pairs.import.rows.answer-missing`

- **What failed:** importing {row} of {source}
- **Cause:** {row} has no answer at {pointer}, where --answer points
- **Observed state shown:** {pointer}: {found}
- **Fix:** point --answer at the answer of every row, or remove {row} from {source}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_row_without_an_answer_is_skipped`

<a id="ster-pairs-import-rows-answer-not-index"></a>

### `ster.pairs.import.rows.answer-not-index`

- **What failed:** importing {row} of {source}
- **Cause:** the answer of {row} at {pointer} is {found}, where --answer-form index reads a whole number from 0, or text holding one
- **Observed state shown:** {detail}
- **Fix:** give --answer-form the form the export's answers take (index, label or text), or correct the answer of {row}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::an_answer_that_is_not_an_index_is_skipped`

<a id="ster-pairs-import-rows-answer-index-outside"></a>

### `ster.pairs.import.rows.answer-index-outside`

- **What failed:** importing {row} of {source}
- **Cause:** the answer of {row} names choice {index}, and the row holds {count} choices, numbered from 0 to {last}
- **Observed state shown:** choices: {choices}
- **Fix:** correct the answer of {row}, or point --choices at the list its answer counts in
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::an_answer_past_the_choices_is_skipped`

<a id="ster-pairs-import-rows-labels-not-list"></a>

### `ster.pairs.import.rows.labels-not-list`

- **What failed:** importing {row} of {source}
- **Cause:** {row} holds no list at {pointer}, where --labels points
- **Observed state shown:** {pointer}: {found}
- **Fix:** point --labels at the list of labels of every row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::labels_that_are_not_a_list_are_skipped`

<a id="ster-pairs-import-rows-label-not-text"></a>

### `ster.pairs.import.rows.label-not-text`

- **What failed:** importing {row} of {source}
- **Cause:** label {index} of {row} is {found}, where every label is non-empty text
- **Observed state shown:** {pointer}: {labels}
- **Fix:** write label {index} of {row} in {source}, or remove the row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_label_that_is_not_text_is_skipped`

<a id="ster-pairs-import-rows-labels-misaligned"></a>

### `ster.pairs.import.rows.labels-misaligned`

- **What failed:** importing {row} of {source}
- **Cause:** {row} holds {labels} labels and {choices} choices, and --answer-form label reads the label at each choice's position
- **Observed state shown:** labels: {shown_labels}; choices: {shown_choices}
- **Fix:** give {row} one label per choice, or point --labels at the list that lines up with --choices
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::labels_that_do_not_line_up_are_skipped`

<a id="ster-pairs-import-rows-answer-label-not-text"></a>

### `ster.pairs.import.rows.answer-label-not-text`

- **What failed:** importing {row} of {source}
- **Cause:** the answer of {row} at {pointer} is {found}, where --answer-form label reads one of the row's labels as text
- **Observed state shown:** labels: {labels}
- **Fix:** give --answer-form the form the export's answers take (index, label or text), or correct the answer of {row}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::an_answer_that_is_not_a_label_is_skipped`

<a id="ster-pairs-import-rows-answer-label-unknown"></a>

### `ster.pairs.import.rows.answer-label-unknown`

- **What failed:** importing {row} of {source}
- **Cause:** the answer of {row}, {answer}, is none of its labels
- **Observed state shown:** labels: {labels}
- **Fix:** correct the answer of {row} to one of its labels, or point --labels at the list its answer names
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::an_answer_that_names_no_label_is_skipped`

<a id="ster-pairs-import-rows-answer-text-not-text"></a>

### `ster.pairs.import.rows.answer-text-not-text`

- **What failed:** importing {row} of {source}
- **Cause:** the answer of {row} at {pointer} is {found}, where --answer-form text reads the correct choice's text
- **Observed state shown:** choices: {choices}
- **Fix:** give --answer-form the form the export's answers take (index, label or text), or correct the answer of {row}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::an_answer_that_is_not_text_is_skipped`

<a id="ster-pairs-import-rows-answer-text-unknown"></a>

### `ster.pairs.import.rows.answer-text-unknown`

- **What failed:** importing {row} of {source}
- **Cause:** the answer of {row}, {answer}, is none of its choices
- **Observed state shown:** choices: {choices}
- **Fix:** correct the answer of {row} to the text of one of its choices
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::an_answer_that_names_no_choice_is_skipped`

<a id="ster-pairs-import-rows-preference-side-missing"></a>

### `ster.pairs.import.rows.preference-side-missing`

- **What failed:** importing {row} of {source}
- **Cause:** {row} holds no {field} text (a string, or a list of chat messages ending with the assistant's), and it is the {side} side of the pair
- **Observed state shown:** {found}
- **Fix:** write the {field} of {row} in {source}, or remove the row
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import-rows.sh::a_preference_row_without_a_side_is_skipped`

## `ster.pairs.import`

write a pair set from a TruthfulQA, Do-Not-Answer, LiveCodeBench or BiPO export, or from multiple-choice rows whose parts the caller locates; a row that cannot become a pair is reported under its cause, never repaired

Started through: `ster pairs import`, `ster request pairs/import`. Uses: `ster.pairs.import.rows`, `ster.pairs.export`, `ster.pairs.file`. Failure catalogue `ster.pairs.import`: 13 declared cause(s) of its own.

<a id="ster-pairs-import-benchmark-unknown"></a>

### `ster.pairs.import.benchmark-unknown`

- **What failed:** reading the benchmark name
- **Cause:** unknown benchmark {benchmark}: Ster imports no export by that name
- **Observed state shown:** --benchmark "{benchmark}"; benchmarks: {benchmarks}
- **Fix:** name one of them with --benchmark
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::an_unknown_benchmark_is_refused`

<a id="ster-pairs-import-choice-fields-not-applicable"></a>

### `ster.pairs.import.choice-fields-not-applicable`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --question, --choices, --answer, --answer-form and --labels apply only to --benchmark choices
- **Observed state shown:** given: {given}
- **Fix:** leave the row flags out for {benchmark}, or import multiple-choice rows with --benchmark choices
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::row_flags_on_another_benchmark_are_refused`

<a id="ster-pairs-import-choice-fields-missing"></a>

### `ster.pairs.import.choice-fields-missing`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --benchmark choices needs --question, --choices, --answer and --answer-form: where each part of a row sits
- **Observed state shown:** --benchmark {benchmark}; none of --question, --choices, --answer, --answer-form and --labels is given
- **Fix:** name each part's JSON pointer and the answer's form, e.g. --question /question --choices /choices/text --answer /answerKey --answer-form label --labels /choices/label
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::choices_without_row_flags_are_refused`

<a id="ster-pairs-import-choice-field-missing"></a>

### `ster.pairs.import.choice-field-missing`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** multiple-choice rows need {flag}: --question, --choices, --answer and --answer-form name where each part of a row sits
- **Observed state shown:** given: {given}
- **Fix:** add {flag}: a JSON pointer to that part of a row, or for --answer-form the form the answer takes (index, label or text)
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_missing_row_flag_is_refused`

<a id="ster-pairs-import-answer-form-unknown"></a>

### `ster.pairs.import.answer-form-unknown`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** unknown answer form {form}: choose index, label or text
- **Observed state shown:** --answer-form {form}
- **Fix:** give --answer-form index (a position counted from 0), label (one of --labels) or text (the choice itself)
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::an_unknown_answer_form_is_refused`

<a id="ster-pairs-import-pointer-invalid"></a>

### `ster.pairs.import.pointer-invalid`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** {flag} {pointer} is not a JSON pointer, which starts with / (for example /question or /choices/text)
- **Observed state shown:** {flag} {pointer}
- **Fix:** write {flag} as a JSON pointer (RFC 6901) from the row's root, e.g. /question
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_pointer_without_its_slash_is_refused`

<a id="ster-pairs-import-labels-missing"></a>

### `ster.pairs.import.labels-missing`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --answer-form label needs --labels, the list of labels the answer names
- **Observed state shown:** --answer-form label without --labels; the answer is read at {answer} and the choices at {choices}
- **Fix:** add --labels with the pointer to the row's labels, aligned with its choices, e.g. /choices/label
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::label_form_without_labels_is_refused`

<a id="ster-pairs-import-labels-not-applicable"></a>

### `ster.pairs.import.labels-not-applicable`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --labels applies only to --answer-form label
- **Observed state shown:** --answer-form {form} with --labels {labels}
- **Fix:** leave --labels out, or read the answer as a label with --answer-form label
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::labels_with_another_answer_form_are_refused`

<a id="ster-pairs-import-examples-missing"></a>

### `ster.pairs.import.examples-missing`

- **What failed:** importing livecodebench from {source}
- **Cause:** --benchmark livecodebench needs --examples, the question_examples.json holding each question's good and bad code
- **Observed state shown:** no --examples given; --source {source}
- **Fix:** add --examples with the question_examples.json published beside problems.json
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::livecodebench_without_examples_is_refused`

<a id="ster-pairs-import-examples-not-applicable"></a>

### `ster.pairs.import.examples-not-applicable`

- **What failed:** importing {benchmark} from {source}
- **Cause:** --examples applies only to --benchmark livecodebench
- **Observed state shown:** --benchmark {benchmark} with --examples {examples}
- **Fix:** leave --examples out for {benchmark}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::examples_on_another_benchmark_are_refused`

<a id="ster-pairs-import-count-zero"></a>

### `ster.pairs.import.count-zero`

- **What failed:** importing {benchmark} from {source}
- **Cause:** --count 0 keeps no pair, and a set without pairs is refused
- **Observed state shown:** --count {count}
- **Fix:** give --count a number above zero, or leave it out to keep every pair
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_count_of_zero_is_refused`

<a id="ster-pairs-import-count-exceeds"></a>

### `ster.pairs.import.count-exceeds`

- **What failed:** drawing {count} pairs from {source}
- **Cause:** {source} yields {pairs} pairs, fewer than --count {count}
- **Observed state shown:** rows: {rows}, rows skipped: {skipped}
- **Fix:** give --count {pairs} or fewer, or leave it out to keep every pair
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_count_above_the_pairs_is_refused`

<a id="ster-pairs-import-no-pairs"></a>

### `ster.pairs.import.no-pairs`

- **What failed:** importing {benchmark} from {source}
- **Cause:** the export holds no row, so no pair could be made, and a set without pairs is refused
- **Observed state shown:** rows: 0; {source} ({state})
- **Fix:** name the {benchmark} export itself, which holds one row per question, or for --benchmark choices check that it is JSON Lines or one JSON list of rows
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::an_export_without_a_usable_row_is_refused`
