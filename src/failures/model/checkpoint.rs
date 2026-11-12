//! How finding a checkpoint's files fails: a local directory, or a Hugging
//! Face Hub repository fetched into the Hub cache
//! (src/runtime/load/checkpoint/resolve.rs).

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

pub static DIRECTORY_MISSING: Cause = Cause {
    id: "directory-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "finding model {model}",
    cause: "it names a local path (it starts with /, ./, ../ or ~/) and no checkpoint directory is there: {state}",
    observed: "model: {model}",
    remedy: "name an existing checkpoint directory, or a Hugging Face model id such as org/name; `ster model fixture create` writes a local one",
    test: "tests/models/load.sh::a_local_path_that_is_no_directory_is_refused",
};

pub static DIRECTORY_UNLISTABLE: Cause = Cause {
    id: "directory-unlistable",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "listing checkpoint directory {path}",
    cause: "the system broke off listing the directory although this account may read it: it was removed or replaced while Ster listed it, or the disk failed",
    observed: "path: {path} ({state}); error kind {kind}; the system answered: {error}",
    remedy: "see what is at {path} with `ls -ln {path}`, and run the command again once it lists",
    test: "tests/models/load.sh::a_directory_listing_the_system_breaks_off_is_refused",
};

pub static CONFIG_ABSENT: Cause = Cause {
    id: "config-absent",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "finding the config of checkpoint {place}",
    cause: "no {file} is there",
    observed: "path: {path}",
    remedy: "use a checkpoint that holds its config.json (or Mistral's params.json) beside its weights, as `ster model fixture create` writes one",
    test: "tests/models/load.sh::a_directory_without_a_config_is_refused",
};

pub static TOKENIZER_ABSENT: Cause = Cause {
    id: "tokenizer-absent",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "finding the tokenizer of checkpoint {place}",
    cause: "{place} publishes no tokenizer: none of {files} is there",
    observed: "looked for: {files}",
    remedy: "use a checkpoint that ships its tokenizer.json, or write one beside the weights with `ster model tokenizer create`",
    test: "tests/models/load.sh::a_directory_without_a_tokenizer_is_refused",
};

pub static WEIGHTS_ABSENT: Cause = Cause {
    id: "weights-absent",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "finding the weights of checkpoint {place}",
    cause: "{place} holds no {kind} weights",
    observed: "layout: {layout}",
    remedy: "use a checkpoint that holds its weights: *.safetensors files (Mistral's consolidated*.safetensors beside params.json), or one .gguf file beside a Transformers config and tokenizer",
    test: "tests/models/load.sh::a_directory_without_weights_is_refused",
};

pub static HUB_CLIENT: Cause = Cause {
    id: "hub-client",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting the Hugging Face Hub client for {model}",
    cause: "the client's TLS connector could not be built: the system's TLS library or certificate store did not load",
    observed: "HF_HOME: {hf_home}; the client answered: {error}",
    remedy: "repair the system's TLS certificate store, which the client reads, or name a local checkpoint directory instead of a Hub id",
    test: "tests/models/hub.sh::a_hub_client_that_cannot_start_is_refused",
};

pub static REPOSITORY_UNKNOWN: Cause = Cause {
    id: "repository-unknown",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading model repository {model} on the Hugging Face Hub",
    cause: "the Hub answered {status}: no repository {model} at revision {revision} is visible to this account",
    observed: "GET {url} answered HTTP {status}: {error}",
    remedy: "name a repository and revision that exist on https://huggingface.co, or a local checkpoint directory; for a gated or private repository, store a Hugging Face token in $HF_HOME/token (default ~/.cache/huggingface/token)",
    test: "tests/models/hub.sh::an_unknown_repository_is_refused",
};

pub static HUB_UNREACHABLE: Cause = Cause {
    id: "hub-unreachable",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "fetching {what} from the Hugging Face Hub",
    cause: "no answer came from the Hub: the connection was not made or broke (no network, a name the resolver does not know, a proxy or TLS failing)",
    observed: "GET {url}: status: {status}; the client answered: {error}",
    remedy: "run the command again once https://huggingface.co answers from this machine; the files already fetched stay in the Hugging Face cache",
    test: "tests/models/hub.sh::an_unreachable_hub_is_refused",
};

pub static HUB_LIMITED: Cause = Cause {
    id: "hub-limited",
    code: Code::RateLimit,
    usage: false,
    exit: None,
    failed: "fetching {what} from the Hugging Face Hub",
    cause: "the Hub answered 429: it limits how often this account or address may ask, and this request passed the limit",
    observed: "GET {url} answered HTTP {status}: {error}",
    remedy: "run the command again after a while, or store a Hugging Face token in $HF_HOME/token, which the Hub allows more requests",
    test: "tests/models/hub.sh::a_hub_that_limits_requests_is_refused",
};

pub static HUB_FAILING: Cause = Cause {
    id: "hub-failing",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "fetching {what} from the Hugging Face Hub",
    cause: "the Hub answered {status}: it failed on its side, so the request was right and the Hub could not serve it",
    observed: "GET {url} answered HTTP {status}: {error}",
    remedy: "run the command again once https://status.huggingface.co shows the Hub serving; the files already fetched stay in the Hugging Face cache",
    test: "tests/models/hub.sh::a_hub_that_fails_on_its_side_is_refused",
};

pub static HUB_REFUSED: Cause = Cause {
    id: "hub-refused",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "fetching {what} from the Hugging Face Hub",
    cause: "the Hub answered {status}, which says this request itself is wrong: a revision of another form, a file the Hub serves no more, or an address the client built from a model id the Hub does not read",
    observed: "model: {model}; revision: {revision}; GET {url} answered HTTP {status}: {error}",
    remedy: "check the model id and --revision against https://huggingface.co/{model}, then run again",
    test: "tests/models/hub.sh::a_request_the_hub_refuses_is_refused",
};

pub static DOWNLOAD_UNCACHED: Cause = Cause {
    id: "download-uncached",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "fetching {what} from the Hugging Face Hub",
    cause: "the file arrived and could not be written into the Hugging Face cache {cache}: the cache directory is not writable for this account, or its disk is full",
    observed: "GET {url}; cache: {cache} ({cache_state}); the client answered: {error}",
    remedy: "let this account write {cache} and free space on its disk (`df -h {cache}`), or point HF_HOME at a writable directory",
    test: "tests/models/hub.sh::a_cache_the_client_cannot_write_is_refused",
};

/// Finding a checkpoint's config, tokenizer and weights.
pub static CHECKPOINT: Operation = Operation {
    id: "ster.model.checkpoint",
    entries: &[],
    uses: &[&super::super::file::FILE],
    summary: "find a checkpoint's config, tokenizer and weights in a local directory, or fetch them from a Hugging Face Hub repository into the Hub cache",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &DIRECTORY_MISSING,
        &DIRECTORY_UNLISTABLE,
        &CONFIG_ABSENT,
        &TOKENIZER_ABSENT,
        &WEIGHTS_ABSENT,
        &HUB_CLIENT,
        &REPOSITORY_UNKNOWN,
        &HUB_UNREACHABLE,
        &HUB_LIMITED,
        &HUB_FAILING,
        &HUB_REFUSED,
        &DOWNLOAD_UNCACHED,
    ],
};
