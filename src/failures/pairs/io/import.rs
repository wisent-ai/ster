//! How importing a pair set from a benchmark export fails: the flags, which
//! are checked before the export is read, and what the export yields.

use wisent_errors::{Cause, Code, Operation};

use super::FILE;
use super::export::EXPORT;
use super::rows::ROWS;

/// docs/guide/errors/pairs-import.md, where this functionality is listed.
const DOCS: &str = page!("pairs-import");

/// What every refusal of the row flags names as the step that failed.
const ROW_FLAGS: &str = "reading the row flags of --benchmark {benchmark}";

pub static BENCHMARK_UNKNOWN: Cause = Cause {
    id: "benchmark-unknown",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: "reading the benchmark name",
    cause: "unknown benchmark {benchmark}: Ster imports no export by that name",
    observed: "--benchmark \"{benchmark}\"; benchmarks: {benchmarks}",
    remedy: "name one of them with --benchmark",
    test: "tests/pairs/imports/import.sh::an_unknown_benchmark_is_refused",
};

pub static CHOICE_FIELDS_NOT_APPLICABLE: Cause = Cause {
    id: "choice-fields-not-applicable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: ROW_FLAGS,
    cause: "--question, --choices, --answer, --answer-form and --labels apply only to --benchmark choices",
    observed: "given: {given}",
    remedy: "leave the row flags out for {benchmark}, or import multiple-choice rows with --benchmark choices",
    test: "tests/pairs/imports/import.sh::row_flags_on_another_benchmark_are_refused",
};

pub static CHOICE_FIELDS_MISSING: Cause = Cause {
    id: "choice-fields-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: ROW_FLAGS,
    cause: "--benchmark choices needs --question, --choices, --answer and --answer-form: where each part of a row sits",
    observed: "--benchmark {benchmark}; none of --question, --choices, --answer, --answer-form and --labels is given",
    remedy: "name each part's JSON pointer and the answer's form, e.g. --question /question --choices /choices/text --answer /answerKey --answer-form label --labels /choices/label",
    test: "tests/pairs/imports/import.sh::choices_without_row_flags_are_refused",
};

pub static CHOICE_FIELD_MISSING: Cause = Cause {
    id: "choice-field-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: ROW_FLAGS,
    cause: "multiple-choice rows need {flag}: --question, --choices, --answer and --answer-form name where each part of a row sits",
    observed: "given: {given}",
    remedy: "add {flag}: a JSON pointer to that part of a row, or for --answer-form the form the answer takes (index, label or text)",
    test: "tests/pairs/imports/import.sh::a_missing_row_flag_is_refused",
};

pub static ANSWER_FORM_UNKNOWN: Cause = Cause {
    id: "answer-form-unknown",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: ROW_FLAGS,
    cause: "unknown answer form {form}: choose index, label or text",
    observed: "--answer-form {form}",
    remedy: "give --answer-form index (a position counted from 0), label (one of --labels) or text (the choice itself)",
    test: "tests/pairs/imports/import.sh::an_unknown_answer_form_is_refused",
};

pub static POINTER_INVALID: Cause = Cause {
    id: "pointer-invalid",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: ROW_FLAGS,
    cause: "{flag} {pointer} is not a JSON pointer, which starts with / (for example /question or /choices/text)",
    observed: "{flag} {pointer}",
    remedy: "write {flag} as a JSON pointer (RFC 6901) from the row's root, e.g. /question",
    test: "tests/pairs/imports/import.sh::a_pointer_without_its_slash_is_refused",
};

pub static LABELS_MISSING: Cause = Cause {
    id: "labels-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: ROW_FLAGS,
    cause: "--answer-form label needs --labels, the list of labels the answer names",
    observed: "--answer-form label without --labels; the answer is read at {answer} and the choices at {choices}",
    remedy: "add --labels with the pointer to the row's labels, aligned with its choices, e.g. /choices/label",
    test: "tests/pairs/imports/import.sh::label_form_without_labels_is_refused",
};

pub static LABELS_NOT_APPLICABLE: Cause = Cause {
    id: "labels-not-applicable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: ROW_FLAGS,
    cause: "--labels applies only to --answer-form label",
    observed: "--answer-form {form} with --labels {labels}",
    remedy: "leave --labels out, or read the answer as a label with --answer-form label",
    test: "tests/pairs/imports/import.sh::labels_with_another_answer_form_are_refused",
};

pub static EXAMPLES_MISSING: Cause = Cause {
    id: "examples-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "importing livecodebench from {source}",
    cause: "--benchmark livecodebench needs --examples, the question_examples.json holding each question's good and bad code",
    observed: "no --examples given; --source {source}",
    remedy: "add --examples with the question_examples.json published beside problems.json",
    test: "tests/pairs/imports/import.sh::livecodebench_without_examples_is_refused",
};

pub static EXAMPLES_NOT_APPLICABLE: Cause = Cause {
    id: "examples-not-applicable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "importing {benchmark} from {source}",
    cause: "--examples applies only to --benchmark livecodebench",
    observed: "--benchmark {benchmark} with --examples {examples}",
    remedy: "leave --examples out for {benchmark}",
    test: "tests/pairs/imports/import.sh::examples_on_another_benchmark_are_refused",
};

pub static COUNT_ZERO: Cause = Cause {
    id: "count-zero",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "importing {benchmark} from {source}",
    cause: "--count 0 keeps no pair, and a set without pairs is refused",
    observed: "--count {count}",
    remedy: "give --count a number above zero, or leave it out to keep every pair",
    test: "tests/pairs/imports/import.sh::a_count_of_zero_is_refused",
};

pub static COUNT_EXCEEDS: Cause = Cause {
    id: "count-exceeds",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "drawing {count} pairs from {source}",
    cause: "{source} yields {pairs} pairs, fewer than --count {count}",
    observed: "rows: {rows}, rows skipped: {skipped}",
    remedy: "give --count {pairs} or fewer, or leave it out to keep every pair",
    test: "tests/pairs/imports/import.sh::a_count_above_the_pairs_is_refused",
};

pub static NO_PAIRS: Cause = Cause {
    id: "no-pairs",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "importing {benchmark} from {source}",
    cause: "the export holds no row, so no pair could be made, and a set without pairs is refused",
    observed: "rows: 0; {source} ({state})",
    remedy: "name the {benchmark} export itself, which holds one row per question, or for --benchmark choices check that it is JSON Lines or one JSON list of rows",
    test: "tests/pairs/imports/import.sh::an_export_without_a_usable_row_is_refused",
};

/// A pair set read from a benchmark export.
pub static IMPORT: Operation = Operation {
    id: "ster.pairs.import",
    entries: &["ster pairs import", "ster request pairs/import"],
    uses: &[&ROWS, &EXPORT, &FILE],
    summary: "write a pair set from a TruthfulQA, Do-Not-Answer, LiveCodeBench or BiPO export, or from multiple-choice rows whose parts the caller locates; a row that cannot become a pair is reported under its cause, never repaired",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &BENCHMARK_UNKNOWN,
        &CHOICE_FIELDS_NOT_APPLICABLE,
        &CHOICE_FIELDS_MISSING,
        &CHOICE_FIELD_MISSING,
        &ANSWER_FORM_UNKNOWN,
        &POINTER_INVALID,
        &LABELS_MISSING,
        &LABELS_NOT_APPLICABLE,
        &EXAMPLES_MISSING,
        &EXAMPLES_NOT_APPLICABLE,
        &COUNT_ZERO,
        &COUNT_EXCEEDS,
        &NO_PAIRS,
    ],
};
