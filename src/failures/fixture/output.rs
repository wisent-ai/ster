//! How writing and verifying a fixture checkpoint fails (src/fixture): the
//! new output directory, its three files, the tensors and the forward pass
//! that proves Ster loads what was written.

use wisent_errors::{Cause, Code};

pub static OUTPUT_NOT_UTF8: Cause = Cause {
    id: "output-not-utf8",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing fixture output {path}",
    cause: "fixture output {path} cannot be passed to Ster's UTF-8 checkpoint loader: its name holds bytes that are not UTF-8",
    observed: "path: {path}",
    remedy: "give --output a directory whose path is UTF-8 text",
    test: "tests/models/fixture.sh::an_output_named_in_bytes_that_are_not_text_is_refused",
};

pub static OUTPUT_EXISTS: Cause = Cause {
    id: "output-exists",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing fixture output {path}",
    cause: "fixture output {path} already exists; choose a new directory; existing files were not changed",
    observed: "path: {path} ({state})",
    remedy: "give --output a directory that does not exist yet",
    test: "tests/models/fixture.sh::an_output_that_exists_is_refused",
};

pub static OUTPUT_UNINSPECTABLE: Cause = Cause {
    id: "output-uninspectable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "choosing fixture output {path}",
    cause: "Ster cannot tell whether {path} exists: this account may not look inside {parent}",
    observed: "{parent}: {parent_state}; error kind {kind}; the system answered: {error}",
    remedy: "give this account search access to {parent} (chmod u+x), or choose an output elsewhere",
    test: "tests/models/fixture.sh::an_output_in_a_closed_directory_is_refused",
};

pub static OUTPUT_DENIED: Cause = Cause {
    id: "output-denied",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "creating fixture {what} {path}",
    cause: "this account may not create {path}: the directory above it does not let it write",
    observed: "{parent}: {parent_state}; the system answered: {error}; nothing was written",
    remedy: "give this account write access to {parent} (chmod u+w), or choose an output elsewhere",
    test: "tests/models/fixture.sh::an_output_in_a_read_only_directory_is_refused",
};

pub static OUTPUT_UNCREATABLE: Cause = Cause {
    id: "output-uncreatable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "creating fixture {what} {path}",
    cause: "{path} cannot be created although this account may write beside it: a file stands where a directory of the path should be, or the disk is full or read-only",
    observed: "{parent}: {parent_state}; error kind {kind}; the system answered: {error}; nothing was written",
    remedy: "see what stands on the way with `ls -ln {parent}`, then choose an output whose parents are directories on a writable disk",
    test: "tests/models/fixture.sh::an_output_below_a_file_is_refused",
};

pub static FILE_UNWRITTEN: Cause = Cause {
    id: "file-unwritten",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "writing fixture {what} {path}",
    cause: "the disk refused the write after the output directory was created: it is full or read-only, or the directory was removed meanwhile",
    observed: "{bytes} bytes to write; error kind {kind}; the system answered: {error}",
    remedy: "free space on the disk holding {path}, or choose an output on another disk, and run the command again",
    test: "tests/models/fixture.sh::a_write_the_disk_refuses_is_refused",
};

pub static TENSOR_UNBUILT: Cause = Cause {
    id: "tensor-unbuilt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "building fixture tensor {name} of shape {shape}",
    cause: "this host cannot hold its {count} values: the declared widths ask for more memory than it has, or for more values than it can count",
    observed: "{name}: shape {shape}, {count} values; the allocator answered: {error}",
    remedy: "lower hidden_size, intermediate_size, vocab_size or the head counts in the config; a fixture is meant to be small",
    test: "tests/models/fixture.sh::a_tensor_past_the_host_is_refused",
};

pub static SAMPLE_NOT_FINITE: Cause = Cause {
    id: "sample-not-finite",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "building fixture tensor {name} of shape {shape}",
    cause: "the normal distribution it is drawn from gave a value past what f32 holds: its scale is too large",
    observed: "{name}: element {index} drew {value}",
    remedy: "lower the scale of the initialization {name} is drawn from in the declaration",
    test: "tests/models/fixture.sh::a_scale_past_f32_is_refused",
};

pub static VERIFICATION_WIDTH: Cause = Cause {
    id: "verification-width",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "verifying fixture {path} at layer {layer}",
    cause: "layer {layer} returned width {width}; Ster loaded hidden size {hidden}: the config Ster wrote and the loader that read it disagree on the width",
    observed: "layer {layer}: {width} values; hidden size loaded: {hidden}; the output was removed",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the declaration and the config: this is a defect of Ster, not of the declaration",
    test: "tests/models/fixture.sh::a_layer_of_another_width_is_refused",
};

pub static VERIFICATION_NOT_FINITE: Cause = Cause {
    id: "verification-not-finite",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "verifying fixture {path} at layer {layer}",
    cause: "layer {layer} returned non-finite activation {value} at element {index}: the declared initialization scales grow the activations past what f32 holds",
    observed: "layer {layer}, element {index}: {value}; {count} of {width} values not finite; the output was removed",
    remedy: "lower the initialization scales in the declaration (initialization.projection, embedding) and create the fixture again",
    test: "tests/models/fixture.sh::activations_that_overflow_are_refused",
};
