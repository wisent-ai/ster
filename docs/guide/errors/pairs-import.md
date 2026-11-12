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
## `ster.pairs.import`

write a pair set from a TruthfulQA, Do-Not-Answer, LiveCodeBench or BiPO export, or from multiple-choice rows whose parts the caller locates; a row that cannot become a pair is reported, never repaired

Started through: `ster pairs import`, `ster request pairs/import`. Uses: `ster.pairs.export`, `ster.pairs.file`. Failure catalogue `ster.pairs.import`: 13 declared cause(s) of its own.

<a id="ster-pairs-import-benchmark-unknown"></a>

### `ster.pairs.import.benchmark-unknown`

- **What failed:** reading the benchmark name
- **Cause:** unknown benchmark {benchmark}: Ster imports no export by that name
- **Observed state shown:** benchmarks: truthfulqa, dna, livecodebench, bipo, choices
- **Fix:** name one of them with --benchmark
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::an_unknown_benchmark_is_refused`

<a id="ster-pairs-import-choice-fields-not-applicable"></a>

### `ster.pairs.import.choice-fields-not-applicable`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --question, --choices, --answer, --answer-form and --labels apply only to --benchmark choices
- **Observed state shown:** given: {given}
- **Fix:** leave the row flags out for {benchmark}, or import multiple-choice rows with --benchmark choices
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::row_flags_on_another_benchmark_are_refused`

<a id="ster-pairs-import-choice-fields-missing"></a>

### `ster.pairs.import.choice-fields-missing`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --benchmark choices needs --question, --choices, --answer and --answer-form: where each part of a row sits
- **Observed state shown:** none of the row flags is given
- **Fix:** name each part's JSON pointer and the answer's form, e.g. --question /question --choices /choices/text --answer /answerKey --answer-form label --labels /choices/label
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::choices_without_row_flags_are_refused`

<a id="ster-pairs-import-choice-field-missing"></a>

### `ster.pairs.import.choice-field-missing`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** multiple-choice rows need {flag}: --question, --choices, --answer and --answer-form name where each part of a row sits
- **Observed state shown:** given: {given}
- **Fix:** add {flag}: a JSON pointer to that part of a row, or for --answer-form the form the answer takes (index, label or text)
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_missing_row_flag_is_refused`

<a id="ster-pairs-import-answer-form-unknown"></a>

### `ster.pairs.import.answer-form-unknown`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** unknown answer form {form}: choose index, label or text
- **Observed state shown:** --answer-form {form}
- **Fix:** give --answer-form index (a position counted from 0), label (one of --labels) or text (the choice itself)
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::an_unknown_answer_form_is_refused`

<a id="ster-pairs-import-pointer-invalid"></a>

### `ster.pairs.import.pointer-invalid`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** {flag} {pointer} is not a JSON pointer, which starts with / (for example /question or /choices/text)
- **Observed state shown:** {flag} {pointer}
- **Fix:** write {flag} as a JSON pointer (RFC 6901) from the row's root, e.g. /question
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_pointer_without_its_slash_is_refused`

<a id="ster-pairs-import-labels-missing"></a>

### `ster.pairs.import.labels-missing`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --answer-form label needs --labels, the list of labels the answer names
- **Observed state shown:** --answer-form label without --labels
- **Fix:** add --labels with the pointer to the row's labels, aligned with its choices, e.g. /choices/label
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::label_form_without_labels_is_refused`

<a id="ster-pairs-import-labels-not-applicable"></a>

### `ster.pairs.import.labels-not-applicable`

- **What failed:** reading the row flags of --benchmark {benchmark}
- **Cause:** --labels applies only to --answer-form label
- **Observed state shown:** --answer-form {form} with --labels {labels}
- **Fix:** leave --labels out, or read the answer as a label with --answer-form label
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::labels_with_another_answer_form_are_refused`

<a id="ster-pairs-import-examples-missing"></a>

### `ster.pairs.import.examples-missing`

- **What failed:** importing livecodebench from {source}
- **Cause:** --benchmark livecodebench needs --examples, the question_examples.json holding each question's good and bad code
- **Observed state shown:** no --examples given
- **Fix:** add --examples with the question_examples.json published beside problems.json
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::livecodebench_without_examples_is_refused`

<a id="ster-pairs-import-examples-not-applicable"></a>

### `ster.pairs.import.examples-not-applicable`

- **What failed:** importing {benchmark} from {source}
- **Cause:** --examples applies only to --benchmark livecodebench
- **Observed state shown:** --benchmark {benchmark} with --examples {examples}
- **Fix:** leave --examples out for {benchmark}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::examples_on_another_benchmark_are_refused`

<a id="ster-pairs-import-count-zero"></a>

### `ster.pairs.import.count-zero`

- **What failed:** importing {benchmark} from {source}
- **Cause:** --count 0 keeps no pair, and a set without pairs is refused
- **Observed state shown:** --count 0
- **Fix:** give --count a number above zero, or leave it out to keep every pair
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_count_of_zero_is_refused`

<a id="ster-pairs-import-count-exceeds"></a>

### `ster.pairs.import.count-exceeds`

- **What failed:** drawing {count} pairs from {source}
- **Cause:** {source} yields {pairs} pairs, fewer than --count {count}
- **Observed state shown:** rows: {rows}, rows skipped: {skipped}
- **Fix:** give --count {pairs} or fewer, or leave it out to keep every pair
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_count_above_the_pairs_is_refused`

<a id="ster-pairs-import-no-pairs"></a>

### `ster.pairs.import.no-pairs`

- **What failed:** importing {benchmark} from {source}
- **Cause:** no row of {source} became a pair, and a set without pairs is refused
- **Observed state shown:** rows: {rows}; {first}
- **Fix:** name the {benchmark} export itself, or for --benchmark choices point the row flags at each part; a skipped row's reason says what it lacks
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::an_export_without_a_usable_row_is_refused`
