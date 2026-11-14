//! How a completion Ster asks of Stado's inference alternatives fails
//! (src/inference.rs): the route it names, starting Stado, and what Stado
//! answers. Writing a pair set with a hosted route uses it.

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/inference.md, where this functionality is listed.
const DOCS: &str = page!("inference");

pub static ROUTE_BLANK: Cause = Cause {
    id: "route-blank",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the route Ster asks Stado's inference alternatives for",
    cause: "the route is blank",
    observed: "--generator-model \"{route}\"",
    remedy: "name a Brama alias, a canonical provider/model route or a selector with --generator-model, e.g. wisent-backend/chat/primary",
    test: "tests/synthesis/inference.sh::a_blank_route_is_refused",
};

pub static STADO_BIN_MISSING: Cause = Cause {
    id: "stado-bin-missing",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting `{call}`",
    cause: "STADO_BIN names {stado}, where no file is, and STADO_BIN is the first place Stado says another program finds it (`stado --help`), so PATH and $HOME/.stado/bin are not looked at",
    observed: "STADO_BIN: {stado} ({state})",
    remedy: "set STADO_BIN to the path of the `stado` program (`command -v stado` prints it when one is on PATH), or unset it so PATH and then $STADO_BIN_DIR/stado (or $HOME/.stado/bin/stado when STADO_BIN_DIR is unset or empty) are looked at",
    test: "tests/synthesis/inference.sh::a_stado_bin_naming_nothing_is_refused",
};

pub static STADO_ABSENT: Cause = Cause {
    id: "stado-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting `{call}`",
    cause: "no `stado` is at any place Stado says another program finds it (`stado --help`): STADO_BIN is unset, no `stado` is on PATH and none is at $STADO_BIN_DIR/stado (or $HOME/.stado/bin/stado when STADO_BIN_DIR is unset or empty), and Stado hands every completion Ster asks to its inference alternatives",
    observed: "STADO_BIN: {program}; PATH: {path}; {place}: {state}",
    remedy: "install Stado with its install-stado.sh, which puts it in $STADO_BIN_DIR when that names a directory, else in $HOME/.stado/bin, or set STADO_BIN to the path of `stado`; `stado capabilities call --help` answers once it is found",
    test: "tests/synthesis/inference.sh::a_missing_stado_is_refused",
};

pub static STADO_NOT_EXECUTABLE: Cause = Cause {
    id: "stado-not-executable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting `{call}`",
    cause: "the `stado` Ster found through {through} may not be executed by this account: its permission bits or a directory before it keep this account out",
    observed: "{stado} ({state}); the system's error kind: {kind}; it answered: {error}",
    remedy: "make {stado} executable for this account (`chmod u+x {stado}` as its owner), or set STADO_BIN to the path of an executable `stado`",
    test: "tests/synthesis/inference.sh::a_stado_this_account_may_not_execute_is_refused",
};

pub static STADO_UNLAUNCHABLE: Cause = Cause {
    id: "stado-unlaunchable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting `{call}`",
    cause: "the system answered that the `stado` Ster found through {through} does not exist when Ster started it: it is a script whose interpreter is not installed, or it was removed after it was found",
    observed: "{stado} ({state}); its interpreter: {interpreter}; the system's error kind: {kind}; it answered: {error}",
    remedy: "reinstall Stado with its install-stado.sh, which writes the program itself rather than a script, or set STADO_BIN to the path of a `stado` that starts (`{stado} --help` shows whether it does)",
    test: "tests/synthesis/inference.sh::a_stado_whose_interpreter_is_missing_is_refused",
};

pub static CALL_UNFINISHED: Cause = Cause {
    id: "call-unfinished",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "handing the request to `{call}`",
    cause: "the pipe between Ster and Stado broke: Stado exited before it read the whole request, or its answer stopped coming",
    observed: "the system's error kind: {kind}; it answered: {error}",
    remedy: "run the command again; when it repeats, `stado capabilities resolve inference --consumer ster` says whether Stado itself answers",
    test: "tests/synthesis/inference.sh::a_call_whose_pipe_breaks_is_refused",
};

pub static UNANSWERED: Cause = Cause {
    id: "unanswered",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "asking Stado's inference alternatives for a completion from {route}",
    cause: "no inference alternative answered `{call}`: every alternative in Stado's order was unreachable or refused the request, so asking again later can help",
    observed: "`{call}` in {directory} exited {status}; Stado's cause {named}: {said}; its fix: {fix}",
    remedy: "follow Stado's fix: `stado capabilities attestation record inference --consumer ster <request> --checked <what was checked>` answers a request Stado kept, `stado capabilities resolve inference --consumer ster` names each alternative's cause, and `stado capabilities select inference --order <alternatives>` changes the order",
    test: "tests/synthesis/resolved.sh::nothing_answering_is_refused_and_the_request_kept",
};

pub static STADO_REFUSED: Cause = Cause {
    id: "stado-refused",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "asking Stado's inference alternatives for a completion from {route}",
    cause: "Stado refused `{call}`, and asking again unchanged will not help: it refuses the request's form, or does not serve ster as a consumer",
    observed: "`{call}` in {directory} exited {status}; Stado's cause {named}: {said}; its fix: {fix}",
    remedy: "follow Stado's fix; `stado capabilities call --help` says what a request must be, and `stado capabilities resolve inference --consumer ster` whether ster is a consumer it serves",
    test: "tests/synthesis/inference.sh::a_request_stado_refuses_is_refused",
};

pub static ANSWER_UNREADABLE: Cause = Cause {
    id: "answer-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading what `{call}` printed",
    cause: "it succeeded and printed no JSON, where `--json` promises one JSON object: the `stado` Ster ran is one without --json, or printed something else beside it",
    observed: "printed: {printed}; the JSON reader answered: {error}",
    remedy: "install a Stado whose `stado capabilities call --help` offers --json, and run the command again",
    test: "tests/synthesis/inference.sh::an_answer_that_is_not_stados_form_is_refused",
};

pub static ANSWER_FIELDLESS: Cause = Cause {
    id: "answer-fieldless",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading what `{call}` printed",
    cause: "it succeeded and printed a JSON document without the answer field `--json` promises: the `stado` Ster ran writes another envelope",
    observed: "printed: {printed}",
    remedy: "install a Stado whose `stado capabilities call --help` offers --json, and run the command again",
    test: "tests/synthesis/inference.sh::an_answer_without_its_answer_field_is_refused",
};

pub static COMPLETION_MALFORMED: Cause = Cause {
    id: "completion-malformed",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "reading the completion {provider} answered for {route}",
    cause: "the answer carries no assistant message at choices[0].message.content, which every chat completion holds",
    observed: "answer: {answer}",
    remedy: "ask another route with --generator-model, or put another alternative first with `stado capabilities select inference --order <alternatives>`",
    test: "tests/synthesis/inference.sh::a_completion_without_a_message_is_refused",
};

/// One completion asked of Stado's inference alternatives.
pub static INFERENCE: Operation = Operation {
    id: "ster.inference",
    entries: &[],
    uses: &[],
    summary: "hand one chat completion to `stado capabilities call inference --consumer ster --json`, which asks the first alternative in Stado's order that answers, and read the assistant text and the alternative that answered",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &ROUTE_BLANK,
        &STADO_BIN_MISSING,
        &STADO_ABSENT,
        &STADO_NOT_EXECUTABLE,
        &STADO_UNLAUNCHABLE,
        &CALL_UNFINISHED,
        &UNANSWERED,
        &STADO_REFUSED,
        &ANSWER_UNREADABLE,
        &ANSWER_FIELDLESS,
        &COMPLETION_MALFORMED,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&INFERENCE];
